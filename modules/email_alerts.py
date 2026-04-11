"""
Email Alert Engine
==================
Sends email alerts when Live Intelligence generates HIGH urgency signals.
Supports two methods:
  1. Gmail SMTP (easiest — needs Gmail App Password, not your real password)
  2. SendGrid API (more robust for scale)

Configure via .env or Streamlit secrets:
  EMAIL_METHOD=gmail          (or sendgrid)
  GMAIL_ADDRESS=you@gmail.com
  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
  ALERT_RECIPIENT=you@gmail.com
  SENDGRID_API_KEY=SG.xxx    (if using sendgrid)
  SENDGRID_FROM=you@yourdomain.com
"""

import streamlit as st
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def _get_config() -> dict:
    """Read email config from Streamlit secrets or env."""
    def _get(key, default=""):
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except Exception:
            return os.getenv(key, default)

    return {
        "method":            _get("EMAIL_METHOD", "gmail"),
        "gmail_address":     _get("GMAIL_ADDRESS", ""),
        "gmail_app_password":_get("GMAIL_APP_PASSWORD", ""),
        "alert_recipient":   _get("ALERT_RECIPIENT", ""),
        "sendgrid_api_key":  _get("SENDGRID_API_KEY", ""),
        "sendgrid_from":     _get("SENDGRID_FROM", ""),
    }


def _build_html(article: dict, analysis: dict, recs: list) -> str:
    urgency      = analysis.get("urgency", "")
    market_impact= analysis.get("market_impact", "NEUTRAL")
    reasoning    = analysis.get("impact_reasoning", "")
    impact_color = {"BULLISH": "#4ade80", "BEARISH": "#f87171", "NEUTRAL": "#94a3b8"}.get(market_impact, "#94a3b8")

    recs_html = ""
    for rec in recs:
        action   = rec.get("action", "WATCH")
        ticker   = rec.get("ticker", "")
        conf     = rec.get("confidence", "")
        horizon  = rec.get("time_horizon", "")
        reason   = rec.get("reasoning", "")
        ac = {"BUY": "#4ade80", "SHORT": "#f87171", "HOLD": "#fbbf24", "WATCH": "#38bdf8"}.get(action, "#94a3b8")
        recs_html += f"""
        <tr>
            <td style="padding:8px;font-weight:700;color:{ac};">{ticker}</td>
            <td style="padding:8px;font-weight:700;color:{ac};">{action}</td>
            <td style="padding:8px;color:#6b8fad;">{conf}</td>
            <td style="padding:8px;color:#6b8fad;">{horizon}</td>
            <td style="padding:8px;color:#c9d8e8;">{reason}</td>
        </tr>"""

    sectors  = ", ".join(analysis.get("affected_sectors", []))
    key_risk = analysis.get("key_risk", "")
    now_str  = datetime.now().strftime("%B %d, %Y at %H:%M ET")

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#07111f;color:#c9d8e8;font-family:Arial,sans-serif;padding:24px;margin:0;">

  <div style="max-width:640px;margin:0 auto;">

    <div style="background:linear-gradient(135deg,#0d1b2a,#122338);border:1px solid #1e3a5f;
                border-radius:14px;padding:24px;margin-bottom:16px;">
      <div style="font-size:22px;font-weight:700;color:#f87171;margin-bottom:4px;">
        🔥 HIGH URGENCY ALERT — FinSight AI
      </div>
      <div style="color:#6b8fad;font-size:13px;">{now_str}</div>
    </div>

    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:20px;margin-bottom:16px;">
      <div style="color:#6b8fad;font-size:12px;margin-bottom:6px;">{article.get('source','')}</div>
      <div style="font-size:17px;font-weight:600;color:#e2e8f0;margin-bottom:12px;">
        {article.get('title','')}
      </div>
      <div style="display:inline-block;padding:4px 12px;border-radius:6px;
                  background:{impact_color}22;border:1px solid {impact_color}44;
                  color:{impact_color};font-weight:700;font-size:13px;margin-bottom:12px;">
        {market_impact}
      </div>
      <div style="color:#94a3b8;font-size:14px;line-height:1.6;">{reasoning}</div>
    </div>

    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:20px;margin-bottom:16px;">
      <div style="font-weight:700;color:#7dd3fc;margin-bottom:12px;">📋 Stock Recommendations</div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:1px solid #1e3a5f;">
            <th style="padding:8px;text-align:left;color:#6b8fad;font-size:12px;">TICKER</th>
            <th style="padding:8px;text-align:left;color:#6b8fad;font-size:12px;">ACTION</th>
            <th style="padding:8px;text-align:left;color:#6b8fad;font-size:12px;">CONF</th>
            <th style="padding:8px;text-align:left;color:#6b8fad;font-size:12px;">HORIZON</th>
            <th style="padding:8px;text-align:left;color:#6b8fad;font-size:12px;">REASONING</th>
          </tr>
        </thead>
        <tbody>{recs_html}</tbody>
      </table>
    </div>

    {"<div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:16px;margin-bottom:16px;'><span style='color:#6b8fad;font-size:12px;'>📂 Sectors: </span><span style='color:#c9d8e8;'>" + sectors + "</span></div>" if sectors else ""}
    {"<div style='background:#1a0a0a;border:1px solid #7f1d1d;border-radius:12px;padding:16px;margin-bottom:16px;'><span style='color:#f87171;font-weight:700;'>⚠️ Key Risk: </span><span style='color:#fca5a5;'>" + key_risk + "</span></div>" if key_risk else ""}

    <div style="margin-top:16px;">
      <a href="{article.get('url','#')}"
         style="background:linear-gradient(135deg,#0ea5e9,#2563eb);color:white;
                padding:10px 20px;border-radius:8px;text-decoration:none;font-weight:700;">
        🔗 Read Full Article
      </a>
    </div>

    <div style="margin-top:24px;padding:12px;border:1px solid #1e3a5f;border-radius:8px;
                color:#6b8fad;font-size:11px;line-height:1.6;">
      ⚠️ This alert is generated by FinSight AI for informational purposes only.
      It does not constitute financial advice. Always conduct your own due diligence.
    </div>

  </div>
</body>
</html>"""


def _send_gmail(config: dict, subject: str, html: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = config["gmail_address"]
        msg["To"]      = config["alert_recipient"]
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(config["gmail_address"], config["gmail_app_password"])
            server.sendmail(config["gmail_address"], config["alert_recipient"], msg.as_string())
        return True
    except Exception as e:
        st.warning(f"⚠️ Email send failed: {str(e)[:100]}")
        return False


def _send_sendgrid(config: dict, subject: str, html: str) -> bool:
    try:
        import requests
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {config['sendgrid_api_key']}",
                "Content-Type":  "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": config["alert_recipient"]}]}],
                "from":    {"email": config["sendgrid_from"]},
                "subject": subject,
                "content": [{"type": "text/html", "value": html}],
            },
            timeout=10,
        )
        return resp.status_code == 202
    except Exception as e:
        st.warning(f"⚠️ SendGrid send failed: {str(e)[:100]}")
        return False


def send_alert(article: dict, analysis: dict) -> bool:
    """
    Send a HIGH urgency alert email.
    Returns True if sent successfully.
    """
    config = _get_config()

    if not config["alert_recipient"]:
        return False  # Not configured — silently skip

    recs = analysis.get("recommendations", [])
    if not recs:
        return False

    tickers  = ", ".join(r.get("ticker", "") for r in recs[:3])
    impact   = analysis.get("market_impact", "")
    subject  = f"🔥 FinSight AI Alert: {impact} — {tickers} | {article.get('source','')}"
    html     = _build_html(article, analysis, recs)

    method = config["method"].lower()
    if method == "sendgrid" and config["sendgrid_api_key"]:
        return _send_sendgrid(config, subject, html)
    elif config["gmail_address"] and config["gmail_app_password"]:
        return _send_gmail(config, subject, html)
    return False


def is_configured() -> bool:
    config = _get_config()
    has_gmail    = bool(config["gmail_address"] and config["gmail_app_password"] and config["alert_recipient"])
    has_sendgrid = bool(config["sendgrid_api_key"] and config["sendgrid_from"] and config["alert_recipient"])
    return has_gmail or has_sendgrid


def render_email_setup():
    """Show email configuration instructions inside the app."""
    st.markdown('<div class="section-header">📧 Email Alert Setup</div>', unsafe_allow_html=True)

    configured = is_configured()
    if configured:
        st.success("✅ Email alerts are configured and active for HIGH urgency signals.")
    else:
        st.warning("⚠️ Email alerts not configured. Follow the steps below.")

    with st.expander("📋 How to set up email alerts (Gmail — recommended)", expanded=not configured):
        st.markdown("""
**Step 1 — Enable 2-Factor Authentication on your Google account**
Go to [myaccount.google.com/security](https://myaccount.google.com/security) → 2-Step Verification → Turn On

**Step 2 — Create a Gmail App Password**
Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
→ Select app: "Mail" → Select device: "Other" → type "FinSight AI" → Generate
→ Copy the 16-character password shown (e.g. `abcd efgh ijkl mnop`)

**Step 3 — Add to your `.env` file** (local) or **Streamlit secrets** (cloud)

For local `.env`:
```
EMAIL_METHOD=gmail
GMAIL_ADDRESS=youremail@gmail.com
GMAIL_APP_PASSWORD=abcd efgh ijkl mnop
ALERT_RECIPIENT=youremail@gmail.com
```

For Streamlit Cloud secrets (Settings → Secrets):
```toml
EMAIL_METHOD = "gmail"
GMAIL_ADDRESS = "youremail@gmail.com"
GMAIL_APP_PASSWORD = "abcd efgh ijkl mnop"
ALERT_RECIPIENT = "youremail@gmail.com"
```

**That's it.** Alerts will fire automatically whenever Live Intelligence
generates a HIGH urgency signal. You'll get a formatted email with the
article, market impact, and all stock recommendations.
        """)

    with st.expander("📋 Alternative: SendGrid (better for multiple recipients)"):
        st.markdown("""
**Step 1** — Sign up at [sendgrid.com](https://sendgrid.com) (free tier = 100 emails/day)

**Step 2** — Create an API key: Settings → API Keys → Create API Key → Full Access

**Step 3** — Verify a sender email in Sendgrid: Settings → Sender Authentication

**Step 4** — Add to `.env` or Streamlit secrets:
```
EMAIL_METHOD=sendgrid
SENDGRID_API_KEY=SG.your_key_here
SENDGRID_FROM=verified@yourdomain.com
ALERT_RECIPIENT=you@gmail.com
```
        """)

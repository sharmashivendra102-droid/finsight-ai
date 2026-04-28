"""
Signal History Log — Supabase backend
======================================
Replaces SQLite with Supabase (Postgres) so signal history
persists across Streamlit Cloud redeploys forever.

Supabase setup (one-time, takes 5 minutes):
  1. Go to supabase.com → New project (free tier)
  2. Go to SQL Editor → run this:

     CREATE TABLE signals (
       id            BIGSERIAL PRIMARY KEY,
       timestamp     TEXT NOT NULL,
       source        TEXT NOT NULL,
       ticker        TEXT NOT NULL,
       action        TEXT NOT NULL,
       confidence    TEXT NOT NULL,
       urgency       TEXT,
       market_impact TEXT,
       time_horizon  TEXT,
       reasoning     TEXT,
       article_title TEXT,
       article_url   TEXT,
       source_feed   TEXT
     );

  3. Go to Project Settings → API
     Copy: Project URL  → SUPABASE_URL
     Copy: anon/public key → SUPABASE_KEY

  4. Add to Streamlit Secrets (Settings → Secrets on share.streamlit.io):
     SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
     SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIs..."

  5. Add to requirements.txt:
     supabase>=2.0.0
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Client initialisation ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_client():
    """
    Returns a Supabase client.
    Falls back to None if credentials are missing — all functions
    degrade gracefully so the app never crashes without Supabase.
    """
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def _is_configured() -> bool:
    return _get_client() is not None


# ── Write functions ───────────────────────────────────────────────────────────

def log_signal(
    source:        str,
    ticker:        str,
    action:        str,
    confidence:    str,
    urgency:       str  = "",
    market_impact: str  = "",
    time_horizon:  str  = "",
    reasoning:     str  = "",
    article_title: str  = "",
    article_url:   str  = "",
    source_feed:   str  = "",
):
    """Insert a single signal row. Never raises — silently skips on error."""
    client = _get_client()
    if client is None:
        return

    try:
        client.table("signals").insert({
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":        source,
            "ticker":        ticker.upper(),
            "action":        action.upper(),
            "confidence":    confidence.upper(),
            "urgency":       urgency       or "",
            "market_impact": market_impact or "",
            "time_horizon":  time_horizon  or "",
            "reasoning":     (reasoning     or "")[:400],
            "article_title": (article_title or "")[:200],
            "article_url":   (article_url   or "")[:500],
            "source_feed":   source_feed   or "",
        }).execute()
    except Exception:
        pass  # never crash the calling code


def log_signals_from_live(article: dict, analysis: dict):
    for rec in analysis.get("recommendations", []):
        ticker = rec.get("ticker", "")
        if not ticker:
            continue
        log_signal(
            source        = "live_intelligence",
            ticker        = ticker,
            action        = rec.get("action",     "WATCH"),
            confidence    = rec.get("confidence", "LOW"),
            urgency       = analysis.get("urgency",       ""),
            market_impact = analysis.get("market_impact", ""),
            time_horizon  = rec.get("time_horizon", ""),
            reasoning     = rec.get("reasoning",    ""),
            article_title = article.get("title",    ""),
            article_url   = article.get("url",      ""),
            source_feed   = article.get("source",   ""),
        )


def log_signals_from_ticker(ticker_result: dict):
    ticker  = ticker_result.get("ticker", "")
    for sig in ticker_result.get("signals", []):
        log_signal(
            source        = "ticker_signals",
            ticker        = ticker,
            action        = sig.get("action",       "WATCH"),
            confidence    = sig.get("confidence",   "LOW"),
            urgency       = ticker_result.get("urgency",       ""),
            market_impact = ticker_result.get("market_impact", ""),
            time_horizon  = sig.get("time_horizon",  ""),
            reasoning     = sig.get("reasoning",     ""),
            article_title = sig.get("source_article",""),
            article_url   = "",
            source_feed   = "ticker_analysis",
        )


def log_signals_from_briefing(tickers_to_watch: list):
    for tw in tickers_to_watch:
        log_signal(
            source        = "market_briefing",
            ticker        = tw.get("ticker",    ""),
            action        = tw.get("action",    "WATCH"),
            confidence    = tw.get("confidence","LOW"),
            urgency       = "MEDIUM",
            market_impact = "",
            time_horizon  = "",
            reasoning     = tw.get("reasoning", ""),
            article_title = tw.get("catalyst",  ""),
            article_url   = "",
            source_feed   = "market_briefing",
        )


# ── Read functions ────────────────────────────────────────────────────────────

def get_signals_df(
    days_back:     int  = 30,
    source_filter: list = None,
    action_filter: list = None,
) -> pd.DataFrame:
    """Fetch signals as DataFrame with optional filters."""
    client = _get_client()
    if client is None:
        st.warning(
            "⚠️ **Supabase not configured.** Signal history will not persist across redeploys. "
            "See the Signal History tab for setup instructions."
        )
        return pd.DataFrame()

    try:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")

        query = (
            client.table("signals")
            .select("*")
            .gte("timestamp", cutoff)
            .order("timestamp", desc=True)
        )

        # Source filter
        if source_filter:
            query = query.in_("source", source_filter)

        # Action filter
        if action_filter:
            query = query.in_("action", action_filter)

        resp = query.limit(5000).execute()
        rows = resp.data if resp.data else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    except Exception as e:
        st.warning(f"⚠️ Could not load signal history: {str(e)[:100]}")
        return pd.DataFrame()


def get_signal_count() -> int:
    """Fast count for Command Center status strip."""
    client = _get_client()
    if client is None:
        return 0
    try:
        resp = client.table("signals").select("id", count="exact").execute()
        return resp.count or 0
    except Exception:
        return 0


def get_signal_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total       = len(df)
    by_act      = df["action"].value_counts().to_dict()
    by_conf     = df["confidence"].value_counts().to_dict()
    by_src      = df["source"].value_counts().to_dict()
    top_tickers = df["ticker"].value_counts().head(10).to_dict()
    high_conf   = df[df["confidence"] == "HIGH"]

    return {
        "total":         total,
        "by_action":     by_act,
        "by_confidence": by_conf,
        "by_source":     by_src,
        "top_tickers":   top_tickers,
        "high_conf_pct": round(len(high_conf) / total * 100, 1) if total > 0 else 0,
        "date_range":    f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}",
    }


def delete_all_signals():
    """Delete all rows from the signals table."""
    client = _get_client()
    if client is None:
        return
    try:
        # Supabase requires a filter — delete where id > 0 clears everything
        client.table("signals").delete().gt("id", 0).execute()
    except Exception as e:
        st.error(f"Could not clear signals: {str(e)[:100]}")


# ── UI ────────────────────────────────────────────────────────────────────────

def run_signal_history():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CARD   = "#0d1b2a"; BORDER = "#1e3a5f"
    MUTED  = "#6b8fad"; CYAN   = "#7dd3fc"
    TEXT   = "#c9d8e8"; GREEN  = "#4ade80"
    RED    = "#f87171"; AMBER  = "#fbbf24"
    BLUE   = "#38bdf8"

    # ── Supabase status ───────────────────────────────────────────────────
    if not _is_configured():
        st.markdown("""
        <div class="card card-accent-red">
          <div style="font-family:var(--font-mono);font-size:.8rem;font-weight:700;
                      color:var(--accent-red);margin-bottom:.6rem;">
            ⚠️ SUPABASE NOT CONFIGURED — Signal history will be lost on redeploy
          </div>
          <div style="font-size:.82rem;color:var(--text-muted);line-height:1.8;">
            <b>1.</b> Go to <a href="https://supabase.com" target="_blank" style="color:var(--accent-cyan);">supabase.com</a>
            → New project (free)<br>
            <b>2.</b> In SQL Editor run the CREATE TABLE command from this file's docstring<br>
            <b>3.</b> Copy Project URL + anon key from Settings → API<br>
            <b>4.</b> Add to Streamlit Secrets:<br>
            <code style="background:var(--bg-elevated);padding:.2rem .4rem;border-radius:4px;">
            SUPABASE_URL = "https://xxxx.supabase.co"<br>
            SUPABASE_KEY = "eyJ..."
            </code>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Persistent signal log — every signal from every tab is saved here.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Signals saved to Supabase (Postgres) — survives redeploys permanently.
        Download CSV any time for a local backup.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        days_back = st.selectbox(
            "Time period", [1, 7, 14, 30, 90, 365], index=3,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
    with f2:
        source_filter = st.multiselect(
            "Source",
            ["live_intelligence", "ticker_signals", "market_briefing",
             "strategy_signals", "auto_evaluator"],
            default=["live_intelligence", "ticker_signals", "market_briefing",
                     "strategy_signals", "auto_evaluator"],
        )
    with f3:
        action_filter = st.multiselect(
            "Action", ["BUY", "SHORT", "HOLD", "WATCH"],
            default=["BUY", "SHORT", "HOLD", "WATCH"],
        )
    with f4:
        ticker_search = st.text_input("Search ticker", placeholder="e.g. NVDA")

    df = get_signals_df(
        days_back     = days_back,
        source_filter = source_filter or None,
        action_filter = action_filter or None,
    )

    if ticker_search.strip():
        df = df[df["ticker"].str.upper() == ticker_search.strip().upper()]

    if df.empty:
        st.info(
            "No signals yet. Run Live News Feed, Ticker Signal Lookup, "
            "or Morning Briefing to start building your history."
        )
        return

    stats = get_signal_stats(df)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
    st.caption(f"Period: {stats.get('date_range', '—')}  ·  Filters applied")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">Total Signals</div>'
                    f'<div class="value">{stats["total"]}</div></div>', unsafe_allow_html=True)
    with m2:
        buys = stats["by_action"].get("BUY", 0)
        st.markdown(f'<div class="metric-box"><div class="label">BUY Signals</div>'
                    f'<div class="value" style="color:#4ade80;">{buys}</div></div>',
                    unsafe_allow_html=True)
    with m3:
        shorts = stats["by_action"].get("SHORT", 0)
        st.markdown(f'<div class="metric-box"><div class="label">SHORT Signals</div>'
                    f'<div class="value" style="color:#f87171;">{shorts}</div></div>',
                    unsafe_allow_html=True)
    with m4:
        hc = stats["by_confidence"].get("HIGH", 0)
        st.markdown(f'<div class="metric-box"><div class="label">HIGH Conf.</div>'
                    f'<div class="value" style="color:#38bdf8;">{hc}</div></div>',
                    unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-box"><div class="label">HIGH Conf %</div>'
                    f'<div class="value">{stats["high_conf_pct"]}%</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        act = stats["by_action"]
        if act:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(CARD)
            ax.set_facecolor(CARD)
            colors  = {"BUY": GREEN, "SHORT": RED, "HOLD": AMBER, "WATCH": BLUE}
            pc      = [colors.get(k, MUTED) for k in act.keys()]
            _, _, autotexts = ax.pie(
                act.values(), labels=act.keys(), autopct="%1.0f%%",
                colors=pc, startangle=90, textprops={"color": TEXT, "fontsize": 9}
            )
            for at in autotexts:
                at.set_color(CARD)
                at.set_fontsize(8)
            ax.set_title("Signal Breakdown", color=CYAN, fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        top = stats["top_tickers"]
        if top:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(CARD)
            ax.set_facecolor(CARD)
            ax.barh(list(top.keys())[::-1], list(top.values())[::-1],
                    color=BLUE, edgecolor=BORDER, linewidth=0.5)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title("Most Signalled Tickers", color=CYAN, fontsize=10)
            ax.set_xlabel("Count", color=MUTED, fontsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(axis="x", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Timeline ──────────────────────────────────────────────────────────
    if len(df) > 1:
        st.markdown('<div class="section-header">📅 Signal Timeline</div>', unsafe_allow_html=True)
        df_time        = df.copy()
        df_time["date"] = pd.to_datetime(df_time["timestamp"]).dt.date
        daily           = df_time.groupby("date").size().reset_index(name="count")

        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)
        ax.bar(pd.to_datetime(daily["date"]), daily["count"],
               color=BLUE, width=0.8, edgecolor=BORDER, linewidth=0.3)
        ax.set_title("Signals per Day", color=CYAN, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Full table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Signals</div>', unsafe_allow_html=True)
    display_cols = ["timestamp", "source", "ticker", "action", "confidence",
                    "urgency", "time_horizon", "reasoning", "article_title"]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()

    def ca(v): return {"BUY": "color:#4ade80", "SHORT": "color:#f87171",
                       "HOLD": "color:#fbbf24", "WATCH": "color:#38bdf8"}.get(v, "")
    def cc(v): return {"HIGH": "color:#38bdf8", "MEDIUM": "color:#fbbf24",
                       "LOW":  "color:#6b8fad"}.get(v, "")

    sty = display_df.style
    if "action"     in display_df.columns: sty = sty.map(ca, subset=["action"])
    if "confidence" in display_df.columns: sty = sty.map(cc, subset=["confidence"])
    st.dataframe(sty, use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download full history as CSV",
        data=csv,
        file_name=f"finsight_signals_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Danger zone ───────────────────────────────────────────────────────
    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Clear ALL signal history", key="clear_history"):
            delete_all_signals()
            st.success("Signal history cleared.")
            st.rerun()

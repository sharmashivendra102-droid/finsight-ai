"""
News Signal Accuracy Evaluator
================================
Evaluates ONLY news-based signals (Live Feed, Ticker Signals, Market Briefing)
against real price outcomes — matched to each signal's stated time horizon.

A signal is only evaluated when:
  (now - signal_time) >= horizon_days for that signal

Signals not yet ready are shown separately with a countdown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent.parent / "signal_history.db"

BLUE   = "#38bdf8"
CYAN   = "#7dd3fc"
AMBER  = "#fbbf24"
RED    = "#f87171"
GREEN  = "#4ade80"
CARD   = "#0d1b2a"
BORDER = "#1e3a5f"
TEXT   = "#c9d8e8"
MUTED  = "#6b8fad"

HORIZON_MAP = {
    "INTRADAY":          1,
    "SWING (1-5 DAYS)":  5,
    "SWING":             5,
    "MEDIUM (WEEKS)":    14,
    "MEDIUM":            14,
    "LONG (MONTHS)":     30,
    "LONG":              30,
}

def _horizon_days(horizon_str: str) -> int:
    if not horizon_str:
        return 5
    key = str(horizon_str).upper().strip()
    if key in HORIZON_MAP:
        return HORIZON_MAP[key]
    for k, v in HORIZON_MAP.items():
        if k in key:
            return v
    return 5

def _horizon_label(days: int) -> str:
    if days <= 1:  return "Intraday"
    if days <= 5:  return "Swing"
    if days <= 14: return "Medium"
    return "Long"


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.4)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_price(ticker: str, date_str: str) -> float | None:
    import yfinance as yf
    try:
        dt    = pd.to_datetime(date_str)
        start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        after = closes[closes.index.date >= dt.date()]
        return float(after.iloc[0]) if not after.empty else float(closes.iloc[-1])
    except Exception:
        return None


def _load_news_signals() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        df   = pd.read_sql_query("""
            SELECT * FROM signals
            WHERE source IN ('live_intelligence','ticker_signals','market_briefing')
            AND action IN ('BUY','SHORT')
            ORDER BY timestamp DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (ready_with_outcomes, not_ready).
    Only evaluates when (now - signal_time) >= signal's own horizon_days.
    """
    now     = datetime.now()
    ready   = []
    waiting = []
    total   = len(df)
    prog    = st.progress(0, text="Evaluating news signal outcomes…")

    for i, (_, row) in enumerate(df.iterrows()):
        prog.progress((i+1)/total,
                      text=f"Checking {row['ticker']} — {row['timestamp'][:10]} ({i+1}/{total})")

        sig_dt       = pd.to_datetime(row["timestamp"])
        horizon_days = _horizon_days(row.get("time_horizon", ""))
        days_elapsed = (now - sig_dt).days
        remaining    = horizon_days - days_elapsed

        base = {
            "timestamp":    row["timestamp"],
            "date":         row["timestamp"][:10],
            "ticker":       row["ticker"],
            "action":       row["action"],
            "confidence":   row["confidence"],
            "source":       row["source"],
            "time_horizon": row.get("time_horizon", ""),
            "horizon_days": horizon_days,
            "days_elapsed": days_elapsed,
            "reasoning":    str(row.get("reasoning", ""))[:100],
            "article":      str(row.get("article_title", ""))[:70],
            "urgency":      row.get("urgency", ""),
        }

        if days_elapsed < horizon_days:
            base["not_ready_reason"] = f"Wait {remaining} more day{'s' if remaining != 1 else ''} (horizon: {horizon_days}d, elapsed: {days_elapsed}d)"
            waiting.append(base)
            continue

        entry = _get_price(row["ticker"], row["timestamp"][:10])
        if entry is None or entry <= 0:
            base["not_ready_reason"] = "Entry price unavailable"
            waiting.append(base)
            continue

        out_date  = (sig_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        out_price = _get_price(row["ticker"], out_date)
        if out_price is None or out_price <= 0:
            base["not_ready_reason"] = "Outcome price unavailable"
            waiting.append(base)
            continue

        raw_ret = (out_price - entry) / entry * 100
        pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
        correct = int(pnl_ret > 0)

        ready.append({
            **base,
            "entry_price":  round(entry, 2),
            "out_price":    round(out_price, 2),
            "return_pct":   round(pnl_ret, 3),
            "correct":      correct,
            "outcome":      "✅ Correct" if correct else "❌ Incorrect",
        })

    prog.empty()
    return pd.DataFrame(ready), pd.DataFrame(waiting)


def _significance(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) < 5:
        return {"p": None, "t": None, "sig": False, "n": len(clean)}
    t, p = scipy_stats.ttest_1samp(clean, 0)
    return {"p": round(float(p), 4), "t": round(float(t), 3),
            "sig": bool(p < 0.05), "n": len(clean)}


def _ac(acc: float) -> str:
    return GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)


def run_news_signal_evaluator():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Did the Live Feed and Ticker Signals actually predict market direction?</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Reads every BUY/SHORT signal from Live News Feed, Ticker Signal Lookup, and Morning Briefing.
        Each signal is evaluated against the real price outcome — but <b>only once its own stated time
        horizon has passed</b>. A SWING signal needs 5 days. A MEDIUM signal needs 14 days. LONG needs 30.
        Signals not yet ready show a countdown to when they can be evaluated.
        </span>
    </div>
    """, unsafe_allow_html=True)

    df_raw = _load_news_signals()

    if df_raw.empty:
        st.warning("""
        **No news signals in the database yet.**

        To build your news signal track record:
        1. Go to **Live News Feed** → press ▶️ Start → leave running for 30+ minutes
        2. Go to **Ticker Signal Lookup** → analyse several tickers
        3. Come back here and click Evaluate

        Signals accumulate in the Signal History Log automatically.
        You need at least 30 evaluated signals for meaningful results.
        """)
        return

    directional = df_raw[df_raw["action"].isin(["BUY","SHORT"])]

    # ── Counts ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="label">Total News Signals</div><div class="value">{len(directional)}</div></div>', unsafe_allow_html=True)
    with c2:
        hc = len(directional[directional["confidence"] == "HIGH"])
        st.markdown(f'<div class="metric-box"><div class="label">HIGH Confidence</div><div class="value" style="color:{BLUE};">{hc}</div></div>', unsafe_allow_html=True)
    with c3:
        buys = len(directional[directional["action"] == "BUY"])
        st.markdown(f'<div class="metric-box"><div class="label">BUY Signals</div><div class="value" style="color:{GREEN};">{buys}</div></div>', unsafe_allow_html=True)
    with c4:
        shorts = len(directional[directional["action"] == "SHORT"])
        st.markdown(f'<div class="metric-box"><div class="label">SHORT Signals</div><div class="value" style="color:{RED};">{shorts}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Filter ─────────────────────────────────────────────────────────────
    with st.expander("🔧 Filter signals", expanded=False):
        ff1, ff2, ff3 = st.columns(3)
        with ff1:
            conf_f = st.multiselect("Confidence", ["HIGH","MEDIUM","LOW"],
                                    default=["HIGH","MEDIUM","LOW"], key="nse_conf")
        with ff2:
            src_f = st.multiselect("Source",
                                   ["live_intelligence","ticker_signals","market_briefing"],
                                   default=["live_intelligence","ticker_signals","market_briefing"],
                                   key="nse_src")
        with ff3:
            days_back = st.selectbox("Signals from last", [7,14,30,60,90,365],
                                     index=2, format_func=lambda x: f"{x} days", key="nse_days")

    filtered = directional.copy()
    if conf_f: filtered = filtered[filtered["confidence"].isin(conf_f)]
    if src_f:  filtered = filtered[filtered["source"].isin(src_f)]
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filtered = filtered[filtered["timestamp"] >= cutoff]

    if filtered.empty:
        st.info("No signals match your current filters.")
        return

    st.info(f"**{len(filtered)} signals** selected. Each will be evaluated against its own stated time horizon.")

    if st.button("🔬 Evaluate News Signal Accuracy", key="nse_run"):
        with st.spinner(f"Fetching prices for {len(filtered)} signals…"):
            ready_df, waiting_df = _evaluate(filtered)
        st.session_state["nse_ready"]   = ready_df
        st.session_state["nse_waiting"] = waiting_df
        st.session_state["nse_time"]    = datetime.now().strftime("%H:%M:%S")

    ready_df   = st.session_state.get("nse_ready",   pd.DataFrame())
    waiting_df = st.session_state.get("nse_waiting", pd.DataFrame())

    # ── Not-ready signals ──────────────────────────────────────────────────
    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not ready yet — horizon window hasn't passed", expanded=False):
            st.caption("These signals will become evaluable once enough time passes to match their stated time horizon.")
            show = ["date","ticker","action","confidence","time_horizon",
                    "horizon_days","days_elapsed","not_ready_reason","source"]
            show = [c for c in show if c in waiting_df.columns]
            def c_act(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            sty_w = waiting_df[show].style
            if "action" in show: sty_w = sty_w.map(c_act, subset=["action"])
            st.dataframe(sty_w, use_container_width=True, hide_index=True)

    if ready_df.empty:
        if not waiting_df.empty:
            st.warning(f"None of the {len(waiting_df)} signals have reached their evaluation window yet. "
                       f"Come back once their time horizons have passed.")
        return

    n_eval = len(ready_df)
    if n_eval < 10:
        st.warning(f"Only {n_eval} signals evaluated so far — need at least 30 for reliable conclusions. "
                   f"Keep running Live News Feed daily.")

    # ── Main accuracy banner ───────────────────────────────────────────────
    total = len(ready_df)
    corr  = int(ready_df["correct"].sum())
    acc   = corr / total * 100 if total > 0 else 0
    ac_c  = _ac(acc)
    label = "Strong Edge 🟢" if acc >= 60 else ("Marginal 🟡" if acc >= 50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {ac_c}44;border-radius:16px;
                padding:1.4rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;color:{ac_c};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1rem;margin-top:0.3rem;">News Signal Accuracy — {label}</div>
        <div style="color:#6b8fad;font-size:0.82rem;margin-top:0.3rem;">
            {corr} correct out of {total} evaluated · {len(waiting_df)} not ready yet
            · Run at {st.session_state.get('nse_time','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence breakdown — THE KEY TABLE ───────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level — Key Calibration Test</div>', unsafe_allow_html=True)
    st.caption("If HIGH confidence signals don't beat MEDIUM and LOW, the confidence scoring is not calibrated correctly.")

    conf_rows = []
    for conf in ["HIGH","MEDIUM","LOW"]:
        sub = ready_df[ready_df["confidence"] == conf]
        row_d = {"Confidence": conf, "Signals": len(sub)}
        for h_name, h_days in [("Intraday",1),("Swing",5),("Medium",14),("Long",30)]:
            sub_h = sub[sub["horizon_days"] == h_days]
            if len(sub_h) >= 3:
                row_d[f"Acc ({h_name})"] = round(sub_h["correct"].mean() * 100, 1)
                row_d[f"Ret ({h_name})"] = round(sub_h["return_pct"].mean(), 2)
            else:
                row_d[f"Acc ({h_name})"] = np.nan
                row_d[f"Ret ({h_name})"] = np.nan
        sig = _significance(sub["return_pct"])
        row_d["p-Value"] = sig["p"] if sig["p"] else np.nan
        row_d["Significant?"] = "✅ YES" if sig["sig"] else "❌ No"
        conf_rows.append(row_d)

    if conf_rows:
        cdf = pd.DataFrame(conf_rows)
        acc_cols = [c for c in cdf.columns if c.startswith("Acc")]
        ret_cols = [c for c in cdf.columns if c.startswith("Ret")]
        def ca(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{_ac(v)};font-weight:700"
        def cr(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        def cs(v): return f"color:{GREEN}" if "YES" in str(v) else f"color:{RED}"
        sty_c = cdf.style
        if acc_cols: sty_c = sty_c.map(ca, subset=acc_cols)
        if ret_cols: sty_c = sty_c.map(cr, subset=ret_cols)
        if "Significant?" in cdf.columns: sty_c = sty_c.map(cs, subset=["Significant?"])
        fmt = {c: "{:.1f}%" for c in acc_cols}
        fmt.update({c: "{:+.2f}%" for c in ret_cols})
        if "p-Value" in cdf.columns: fmt["p-Value"] = "{:.4f}"
        sty_c = sty_c.format(fmt, na_rep="—")
        st.dataframe(sty_c, use_container_width=True, hide_index=True)

    # ── By source ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📡 Accuracy by Source</div>', unsafe_allow_html=True)
    src_rows = []
    labels = {"live_intelligence":"Live News Feed","ticker_signals":"Ticker Signal Lookup",
               "market_briefing":"Morning Briefing"}
    for src in ready_df["source"].unique():
        sub = ready_df[ready_df["source"] == src]
        sig = _significance(sub["return_pct"])
        src_rows.append({
            "Source":       labels.get(src, src),
            "Signals":      len(sub),
            "Accuracy":     round(sub["correct"].mean()*100, 1) if len(sub)>=3 else np.nan,
            "Avg Return":   round(sub["return_pct"].mean(), 2) if len(sub)>=3 else np.nan,
            "p-Value":      sig["p"] if sig["p"] else np.nan,
            "Significant?": "✅ YES" if sig["sig"] else "❌ No",
        })
    if src_rows:
        sdf = pd.DataFrame(src_rows)
        def cac(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{_ac(v)}"
        def cre(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN}" if v>0 else f"color:{RED}"
        sty_s = sdf.style
        if "Accuracy"   in sdf.columns: sty_s = sty_s.map(cac, subset=["Accuracy"])
        if "Avg Return" in sdf.columns: sty_s = sty_s.map(cre, subset=["Avg Return"])
        sty_s = sty_s.format({"Accuracy":"{:.1f}%","Avg Return":"{:+.2f}%","p-Value":"{:.4f}"}, na_rep="—")
        st.dataframe(sty_s, use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        _style_fig(fig, ax)
        confs = ["HIGH","MEDIUM","LOW"]
        palette = {"HIGH": BLUE, "MEDIUM": AMBER, "LOW": MUTED}
        x = np.arange(4)
        w = 0.25
        h_names = ["Intraday","Swing","Medium","Long"]
        h_days  = [1, 5, 14, 30]
        for ji, conf in enumerate(confs):
            sub_c = ready_df[ready_df["confidence"] == conf]
            accs = []
            for hd in h_days:
                sub_h = sub_c[sub_c["horizon_days"] == hd]
                accs.append(sub_h["correct"].mean()*100 if len(sub_h)>=3 else np.nan)
            bars = ax.bar(x + ji*w, accs, w, label=conf,
                          color=palette[conf], edgecolor=BORDER, linewidth=0.3)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xticks(x + w)
        ax.set_xticklabels(h_names, color=MUTED, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("Accuracy by Confidence × Horizon", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        rets = ready_df["return_pct"].dropna()
        if not rets.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            _style_fig(fig, ax)
            ax.hist(rets, bins=15,
                    color=GREEN if rets.mean()>0 else RED,
                    edgecolor=BORDER, linewidth=0.3, alpha=0.8)
            ax.axvline(0,           color=MUTED, linewidth=1, linestyle="--")
            ax.axvline(rets.mean(), color=CYAN,  linewidth=1.5,
                       label=f"Mean {rets.mean():+.2f}%")
            ax.set_title("Return Distribution", color=CYAN, fontsize=9)
            ax.set_xlabel("Return % (horizon-matched)", color=MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Rolling accuracy ───────────────────────────────────────────────────
    if len(ready_df) >= 8:
        st.markdown('<div class="section-header">📅 Rolling Accuracy Over Time</div>', unsafe_allow_html=True)
        ev = ready_df.copy()
        ev["date_dt"] = pd.to_datetime(ev["date"])
        ev = ev.sort_values("date_dt")
        ev["rolling"] = ev["correct"].rolling(min(10, len(ev)), min_periods=3).mean() * 100

        fig, ax = plt.subplots(figsize=(12, 3))
        _style_fig(fig, ax)
        ax.plot(ev["date_dt"], ev["rolling"], color=BLUE, linewidth=1.5, label="Rolling accuracy")
        ax.fill_between(ev["date_dt"], ev["rolling"], 50,
                        where=ev["rolling"]>=50, alpha=0.1, color=GREEN)
        ax.fill_between(ev["date_dt"], ev["rolling"], 50,
                        where=ev["rolling"]<50,  alpha=0.1, color=RED)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% baseline")
        ax.axhline(55, color=AMBER, linewidth=0.8, linestyle=":", label="55% target")
        ax.set_ylim(20, 85)
        ax.set_ylabel("Rolling Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("News Signal Accuracy Over Time (horizon-matched)", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=20, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    disp_cols = ["date","ticker","action","confidence","time_horizon","horizon_days",
                 "entry_price","out_price","return_pct","outcome","source","article"]
    disp = ready_df[[c for c in disp_cols if c in ready_df.columns]].copy()
    def c_a(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
    def c_o(v): return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def c_r(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v>0 else f"color:{RED}"
    sty_d = disp.style
    if "action"     in disp.columns: sty_d = sty_d.map(c_a, subset=["action"])
    if "outcome"    in disp.columns: sty_d = sty_d.map(c_o, subset=["outcome"])
    if "return_pct" in disp.columns: sty_d = sty_d.map(c_r, subset=["return_pct"])
    sty_d = sty_d.format({
        "entry_price": "${:.2f}",
        "out_price":   "${:.2f}",
        "return_pct":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
    })
    st.dataframe(sty_d, use_container_width=True, hide_index=True)

    # ── Honest interpretation ──────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 What These Numbers Mean</div>', unsafe_allow_html=True)
    if total < 20:
        st.error(f"**{total} evaluated signals is not enough for reliable conclusions.** "
                 f"You need at least 30–50. Run Live News Feed for 30+ minutes daily and "
                 f"analyse tickers in Ticker Signal Lookup. Come back in 1–2 weeks.")
    else:
        sig_all = _significance(ready_df["return_pct"])
        hc_sub  = ready_df[ready_df["confidence"]=="HIGH"]
        hc_acc  = hc_sub["correct"].mean()*100 if len(hc_sub)>=5 else None

        if acc >= 60 and sig_all["sig"]:
            st.success(f"🟢 **{acc:.1f}% accuracy (p={sig_all['p']:.4f}) on {total} signals** — "
                       f"statistically significant edge. You can begin trading these signals with small real positions.")
        elif acc >= 55:
            st.info(f"🟡 **{acc:.1f}% accuracy** — modest edge. "
                    f"{'Statistically significant.' if sig_all['sig'] else f'Not yet significant (p={sig_all[chr(112)] or 0:.4f}). Need more signals.'}")
        else:
            st.warning(f"🔴 **{acc:.1f}% accuracy** — no meaningful edge yet. "
                       f"Try filtering to HIGH confidence only, or check if the time horizons match your expectations.")

        if hc_acc is not None:
            diff = hc_acc - acc
            if diff > 5:
                st.success(f"✅ HIGH confidence ({hc_acc:.1f}%) beats overall ({acc:.1f}%) by {diff:.1f}pp — confidence scoring is calibrated correctly.")
            else:
                st.warning(f"⚠️ HIGH confidence ({hc_acc:.1f}%) is not significantly better than overall ({acc:.1f}%). Confidence calibration may need improvement.")

    csv = ready_df.to_csv(index=False)
    st.download_button("⬇️ Download evaluation data as CSV", data=csv,
                       file_name=f"news_eval_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

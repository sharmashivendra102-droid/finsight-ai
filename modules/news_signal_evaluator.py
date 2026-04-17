"""
News Signal Accuracy Evaluator
================================
Evaluates ONLY news-based signals using eval_core (horizon-matched + supersession-aware).
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
def _fetch_price(ticker: str, date_str: str) -> float | None:
    import yfinance as yf
    try:
        dt    = pd.to_datetime(date_str)
        start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty: return None
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
            ORDER BY timestamp
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_all_ticker_signals() -> pd.DataFrame:
    """Load ALL signals (all sources) for the supersession timeline."""
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        df   = pd.read_sql_query(
            "SELECT * FROM signals WHERE action IN ('BUY','SHORT') ORDER BY timestamp",
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


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
    from modules.eval_core import evaluate_signals

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Did the news AI signals actually predict market direction?</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Evaluates Live News Feed, Ticker Signal Lookup, and Morning Briefing signals only.
        Each signal is evaluated at its own stated time horizon. If an opposing signal arrives
        before the horizon (e.g. a BUY signal after a SHORT), the original is closed at that
        earlier date — exactly like a real trader would. Signals not yet ready show a countdown.
        </span>
    </div>
    """, unsafe_allow_html=True)

    df_raw = _load_news_signals()

    if df_raw.empty:
        st.warning("""
        **No news signals in the database yet.**

        1. Go to **Live News Feed** → press ▶️ Start → leave running 30+ minutes
        2. Go to **Ticker Signal Lookup** → analyse several tickers
        3. Come back here after signals have had time to play out

        You need at least 30 evaluated signals for meaningful results.
        """)
        return

    directional = df_raw.copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="label">Total News Signals</div><div class="value">{len(directional)}</div></div>', unsafe_allow_html=True)
    with c2:
        hc = len(directional[directional["confidence"]=="HIGH"])
        st.markdown(f'<div class="metric-box"><div class="label">HIGH Confidence</div><div class="value" style="color:{BLUE};">{hc}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="label">BUY Signals</div><div class="value" style="color:{GREEN};">{len(directional[directional["action"]=="BUY"])}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="label">SHORT Signals</div><div class="value" style="color:{RED};">{len(directional[directional["action"]=="SHORT"])}</div></div>', unsafe_allow_html=True)
    st.markdown("")

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
            days_back = st.selectbox("From last", [7,14,30,60,90,365],
                                     index=2, format_func=lambda x: f"{x} days", key="nse_days")

    filtered = directional.copy()
    if conf_f: filtered = filtered[filtered["confidence"].isin(conf_f)]
    if src_f:  filtered = filtered[filtered["source"].isin(src_f)]
    cutoff = (datetime.now()-timedelta(days=days_back)).strftime("%Y-%m-%d")
    filtered = filtered[filtered["timestamp"]>=cutoff]

    if filtered.empty:
        st.info("No signals match your filters.")
        return

    st.info(f"**{len(filtered)} signals** selected. Each evaluated at its own time horizon, closed early if an opposing signal arrives.")

    if not st.button("🔬 Evaluate News Signal Accuracy", key="nse_run"):
        if st.session_state.get("nse_ready") is None:
            return
    else:
        df_full = _load_all_ticker_signals()
        prog    = st.progress(0, text="Evaluating…")
        def _cb(i, total, ticker, date):
            prog.progress((i+1)/total, text=f"Checking {ticker} — {date} ({i+1}/{total})")
        ready_df, waiting_df = evaluate_signals(
            df                       = filtered,
            fetch_price_fn           = _fetch_price,
            all_signals_for_timeline = df_full,
            progress_callback        = _cb,
        )
        prog.empty()
        st.session_state["nse_ready"]   = ready_df
        st.session_state["nse_waiting"] = waiting_df
        st.session_state["nse_time"]    = datetime.now().strftime("%H:%M:%S")

    ready_df   = st.session_state.get("nse_ready",   pd.DataFrame())
    waiting_df = st.session_state.get("nse_waiting", pd.DataFrame())
    sup_count  = int(ready_df["superseded"].sum()) if not ready_df.empty and "superseded" in ready_df.columns else 0

    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not ready yet", expanded=False):
            st.caption("Horizon window hasn't passed. These are not counted in accuracy.")
            show = [c for c in ["date","ticker","action","confidence","time_horizon",
                                 "days_elapsed","not_ready_reason"] if c in waiting_df.columns]
            def ca(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            sty_w = waiting_df[show].style
            if "action" in show: sty_w = sty_w.map(ca, subset=["action"])
            st.dataframe(sty_w, use_container_width=True, hide_index=True)

    if ready_df.empty:
        st.warning("No signals have reached their evaluation window yet.")
        return

    if sup_count > 0:
        st.info(f"ℹ️ **{sup_count} signal(s) were closed early** because an opposing signal arrived before the original horizon. "
                f"This is the correct behaviour — a BUY signal after a SHORT means the thesis has changed.")

    n_eval = len(ready_df)
    if n_eval < 10:
        st.warning(f"Only {n_eval} signals evaluated. Need at least 30 for reliable conclusions.")

    # ── Accuracy banner ────────────────────────────────────────────────────
    total = len(ready_df)
    corr  = int(ready_df["correct"].sum())
    acc   = corr/total*100 if total>0 else 0
    ac_c  = _ac(acc)
    label = "Strong Edge 🟢" if acc>=60 else ("Marginal 🟡" if acc>=50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {ac_c}44;border-radius:16px;
                padding:1.4rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;color:{ac_c};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1rem;margin-top:0.3rem;">News Signal Accuracy — {label}</div>
        <div style="color:#6b8fad;font-size:0.82rem;margin-top:0.3rem;">
            {corr} correct out of {total} · {sup_count} closed early by reversal · {len(waiting_df)} pending
            · {st.session_state.get("nse_time","")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence table ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level</div>', unsafe_allow_html=True)
    st.caption("HIGH confidence signals should beat MEDIUM and LOW. If not, the confidence calibration needs improvement.")

    conf_rows = []
    for conf in ["HIGH","MEDIUM","LOW"]:
        sub = ready_df[ready_df["confidence"]==conf]
        sig = _significance(sub["return_pct"]) if len(sub)>=5 else {"p":None,"sig":False}
        conf_rows.append({
            "Confidence":  conf,
            "Signals":     len(sub),
            "Accuracy":    round(sub["correct"].mean()*100,1) if len(sub)>=3 else np.nan,
            "Avg Return":  round(sub["return_pct"].mean(),2)  if len(sub)>=3 else np.nan,
            "Early Exits": int(sub["superseded"].sum()) if "superseded" in sub.columns else 0,
            "p-Value":     sig["p"] if sig["p"] else np.nan,
            "Significant?":"✅ YES" if sig["sig"] else "❌ No",
        })

    cdf = pd.DataFrame(conf_rows)
    def ca2(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{_ac(v)};font-weight:700"
    def cr2(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{GREEN}" if v>0 else f"color:{RED}"
    def cs2(v): return f"color:{GREEN}" if "YES" in str(v) else f"color:{RED}"
    sty_c = cdf.style
    if "Accuracy"    in cdf.columns: sty_c = sty_c.map(ca2, subset=["Accuracy"])
    if "Avg Return"  in cdf.columns: sty_c = sty_c.map(cr2, subset=["Avg Return"])
    if "Significant?"in cdf.columns: sty_c = sty_c.map(cs2, subset=["Significant?"])
    sty_c = sty_c.format({"Accuracy":"{:.1f}%","Avg Return":"{:+.2f}%","p-Value":"{:.4f}"}, na_rep="—")
    st.dataframe(sty_c, use_container_width=True, hide_index=True)

    # ── Source breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📡 Accuracy by Source</div>', unsafe_allow_html=True)
    src_labels = {"live_intelligence":"Live News Feed","ticker_signals":"Ticker Signal Lookup",
                  "market_briefing":"Morning Briefing"}
    src_rows = []
    for src in ready_df["source"].unique():
        sub = ready_df[ready_df["source"]==src]
        sig = _significance(sub["return_pct"])
        src_rows.append({
            "Source":       src_labels.get(src, src),
            "Signals":      len(sub),
            "Accuracy":     round(sub["correct"].mean()*100,1) if len(sub)>=3 else np.nan,
            "Avg Return":   round(sub["return_pct"].mean(),2) if len(sub)>=3 else np.nan,
            "p-Value":      sig["p"] if sig["p"] else np.nan,
            "Significant?": "✅ YES" if sig["sig"] else "❌ No",
        })
    if src_rows:
        sdf = pd.DataFrame(src_rows)
        sty_s = sdf.style
        if "Accuracy"   in sdf: sty_s = sty_s.map(ca2, subset=["Accuracy"])
        if "Avg Return" in sdf: sty_s = sty_s.map(cr2, subset=["Avg Return"])
        sty_s = sty_s.format({"Accuracy":"{:.1f}%","Avg Return":"{:+.2f}%","p-Value":"{:.4f}"}, na_rep="—")
        st.dataframe(sty_s, use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(5,3.5))
        _style_fig(fig, ax)
        confs   = ["HIGH","MEDIUM","LOW"]
        palette = {"HIGH":BLUE,"MEDIUM":AMBER,"LOW":MUTED}
        accs2   = [ready_df[ready_df["confidence"]==c]["correct"].mean()*100
                   if len(ready_df[ready_df["confidence"]==c])>=3 else 0 for c in confs]
        ns2     = [len(ready_df[ready_df["confidence"]==c]) for c in confs]
        bcolors = [GREEN if a>=60 else (AMBER if a>=50 else RED) for a in accs2]
        bars2   = ax.bar(confs, accs2, color=bcolors, edgecolor=BORDER, linewidth=0.4)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(0,100); ax.set_title("Accuracy by Confidence", color=CYAN, fontsize=9)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        for bar, a, n in zip(bars2, accs2, ns2):
            if n>=3: ax.text(bar.get_x()+bar.get_width()/2, a+1,
                             f"{a:.0f}%\n(n={n})", ha="center", color=TEXT, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        rets = ready_df["return_pct"].dropna()
        if not rets.empty:
            fig, ax = plt.subplots(figsize=(5,3.5))
            _style_fig(fig, ax)
            ax.hist(rets, bins=15, color=GREEN if rets.mean()>0 else RED,
                    edgecolor=BORDER, linewidth=0.3, alpha=0.8)
            ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
            ax.axvline(rets.mean(), color=CYAN, linewidth=1.5,
                       label=f"Mean {rets.mean():+.2f}%")
            ax.set_title("Return Distribution", color=CYAN, fontsize=9)
            ax.set_xlabel("Return % (horizon-matched, reversal-aware)", color=MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Rolling accuracy ───────────────────────────────────────────────────
    if len(ready_df) >= 8:
        st.markdown('<div class="section-header">📅 Rolling Accuracy Over Time</div>', unsafe_allow_html=True)
        ev = ready_df.copy()
        ev["date_dt"] = pd.to_datetime(ev["date"])
        ev = ev.sort_values("date_dt")
        ev["rolling"] = ev["correct"].rolling(min(10,len(ev)), min_periods=3).mean()*100
        fig, ax = plt.subplots(figsize=(12,3))
        _style_fig(fig, ax)
        ax.plot(ev["date_dt"], ev["rolling"], color=BLUE, linewidth=1.5)
        ax.fill_between(ev["date_dt"], ev["rolling"], 50,
                        where=ev["rolling"]>=50, alpha=0.1, color=GREEN)
        ax.fill_between(ev["date_dt"], ev["rolling"], 50,
                        where=ev["rolling"]<50, alpha=0.1, color=RED)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% baseline")
        ax.axhline(55, color=AMBER, linewidth=0.8, linestyle=":", label="55% target")
        ax.set_ylim(20,85); ax.set_ylabel("Rolling Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("News Signal Accuracy Over Time (reversal-aware, horizon-matched)", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=20, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    disp_cols = ["date","ticker","action","confidence","time_horizon","held_days",
                 "entry_price","exit_price","return_pct","outcome","superseded","exit_reason","article"]
    disp = ready_df[[c for c in disp_cols if c in ready_df.columns]].copy()
    def c_a(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
    def c_o(v): return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def c_r(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v>0 else f"color:{RED}"
    def c_s(v): return f"color:{AMBER}" if v else ""
    sty_d = disp.style
    if "action"     in disp.columns: sty_d = sty_d.map(c_a, subset=["action"])
    if "outcome"    in disp.columns: sty_d = sty_d.map(c_o, subset=["outcome"])
    if "return_pct" in disp.columns: sty_d = sty_d.map(c_r, subset=["return_pct"])
    if "superseded" in disp.columns: sty_d = sty_d.map(c_s, subset=["superseded"])
    sty_d = sty_d.format({
        "entry_price": "${:.2f}",
        "exit_price":  "${:.2f}",
        "return_pct":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
    })
    st.dataframe(sty_d, use_container_width=True, hide_index=True)

    # ── Honest interpretation ──────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Interpretation</div>', unsafe_allow_html=True)
    if total < 20:
        st.error(f"**{total} evaluated signals is not enough.** Need 30–50 minimum. "
                 f"Run Live News Feed daily and analyse tickers regularly.")
    else:
        sig_all = _significance(ready_df["return_pct"])
        hc_sub  = ready_df[ready_df["confidence"]=="HIGH"]
        hc_acc  = hc_sub["correct"].mean()*100 if len(hc_sub)>=5 else None
        if acc>=60 and sig_all["sig"]:
            st.success(f"🟢 **{acc:.1f}% accuracy (p={sig_all['p']:.4f}) on {total} signals** — statistically significant edge.")
        elif acc>=55:
            st.info(f"🟡 **{acc:.1f}% accuracy** — modest edge. "
                    f"{'Statistically significant.' if sig_all['sig'] else f'Not yet significant (p={sig_all[chr(112)]:.4f}). Need more signals.'}")
        else:
            st.warning(f"🔴 **{acc:.1f}% accuracy** — no meaningful edge yet. "
                       f"Try filtering to HIGH confidence only.")
        if hc_acc is not None:
            diff = hc_acc - acc
            if diff > 5:
                st.success(f"✅ HIGH confidence ({hc_acc:.1f}%) beats overall ({acc:.1f}%) by {diff:.1f}pp — confidence calibrated correctly.")
            else:
                st.warning(f"⚠️ HIGH confidence ({hc_acc:.1f}%) not significantly better than overall ({acc:.1f}%). Confidence calibration may need improvement.")

    csv = ready_df.to_csv(index=False)
    st.download_button("⬇️ Download as CSV", data=csv,
                       file_name=f"news_eval_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

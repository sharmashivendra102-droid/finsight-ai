"""
Signal Performance Tracker
============================
Evaluates every BUY/SHORT signal across ALL sources.
Uses eval_core for horizon-matching and signal supersession detection.
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
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        after = closes[closes.index.date >= dt.date()]
        return float(after.iloc[0]) if not after.empty else float(closes.iloc[-1])
    except Exception:
        return None


def _load_all_signals(days_back: int, src_filter=None, act_filter=None) -> pd.DataFrame:
    try:
        conn   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_sql_query(
            "SELECT * FROM signals WHERE timestamp >= ? AND action IN ('BUY','SHORT') ORDER BY timestamp",
            conn, params=(cutoff,)
        )
        conn.close()
    except Exception:
        return pd.DataFrame()
    if src_filter: df = df[df["source"].isin(src_filter)]
    if act_filter: df = df[df["action"].isin(act_filter)]
    return df


def _load_full_timeline() -> pd.DataFrame:
    """Load ALL signals (not filtered by date) for supersession detection."""
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


def run_signal_performance():
    from modules.eval_core import evaluate_signals, horizon_label, get_horizon_days

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Complete signal track record — every source, horizon-matched, supersession-aware.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Each signal is evaluated at its own stated time horizon. If a newer opposing signal
        (e.g. a BUY after a SHORT) arrives before the horizon, the original signal is closed
        at that earlier date — just like a real trader would. Signals not yet ready show a
        countdown. HOLD and WATCH are excluded.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        days_back = st.selectbox("Signals from last", [7,14,30,60,90,365],
                                 index=2, format_func=lambda x: f"{x} days", key="sp_days")
    with f2:
        src_filter = st.multiselect(
            "Source",
            ["live_intelligence","ticker_signals","market_briefing",
             "strategy_signals","auto_evaluator"],
            default=["live_intelligence","ticker_signals","market_briefing",
                     "strategy_signals","auto_evaluator"],
            key="sp_src"
        )
    with f3:
        act_filter = st.multiselect("Action", ["BUY","SHORT"],
                                    default=["BUY","SHORT"], key="sp_act")

    if not st.button("🔄 Evaluate All Signals", key="sp_run"):
        if st.session_state.get("sp_ready") is None:
            st.info("Click **Evaluate All Signals** to compute accuracy against real market outcomes.")
            return
    else:
        df_raw  = _load_all_signals(days_back, src_filter or None, act_filter or None)
        df_full = _load_full_timeline()

        if df_raw.empty:
            st.warning("No signals found. Run Live News Feed and Ticker Signals to build history.")
            return

        prog = st.progress(0, text="Starting evaluation…")
        def _cb(i, total, ticker, date):
            prog.progress((i+1)/total, text=f"Checking {ticker} — {date} ({i+1}/{total})")

        ready_df, waiting_df = evaluate_signals(
            df                      = df_raw,
            fetch_price_fn          = _fetch_price,
            all_signals_for_timeline= df_full,
            progress_callback       = _cb,
        )
        prog.empty()
        st.session_state["sp_ready"]   = ready_df
        st.session_state["sp_waiting"] = waiting_df

    ready_df   = st.session_state.get("sp_ready",   pd.DataFrame())
    waiting_df = st.session_state.get("sp_waiting", pd.DataFrame())

    # ── Not ready ──────────────────────────────────────────────────────────
    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not ready for evaluation yet", expanded=False):
            st.caption("These haven't reached their time horizon window yet.")
            show = [c for c in ["date","ticker","action","confidence","time_horizon",
                                 "days_elapsed","not_ready_reason","source"]
                    if c in waiting_df.columns]
            def ca(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            sty = waiting_df[show].style
            if "action" in show: sty = sty.map(ca, subset=["action"])
            st.dataframe(sty, use_container_width=True, hide_index=True)

    if ready_df.empty:
        st.warning("No signals ready for evaluation yet.")
        return

    # ── Supersession note ──────────────────────────────────────────────────
    sup_count = int(ready_df["superseded"].sum()) if "superseded" in ready_df.columns else 0
    if sup_count > 0:
        st.info(f"ℹ️ **{sup_count} signal(s) were closed early** because an opposing signal arrived "
                f"before the original horizon — evaluated at that earlier exit date. "
                f"This gives a more realistic picture of actual trading performance.")

    # ── Accuracy banner ────────────────────────────────────────────────────
    total = len(ready_df)
    corr  = int(ready_df["correct"].sum())
    acc   = corr / total * 100 if total > 0 else 0
    ac    = GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)
    label = "Strong Edge 🟢" if acc >= 60 else ("Marginal 🟡" if acc >= 50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {ac}44;border-radius:16px;
                padding:1.4rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;color:{ac};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1rem;margin-top:0.3rem;">Overall Signal Accuracy — {label}</div>
        <div style="color:#6b8fad;font-size:0.82rem;margin-top:0.3rem;">
            {corr} correct out of {total} evaluated · {sup_count} closed early by reversal · {len(waiting_df)} not ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    if total < 20:
        st.warning(f"⚠️ {total} evaluated signals is below the 20-signal minimum. Keep running Live Feed and Ticker Signals.")

    # ── Key metrics ────────────────────────────────────────────────────────
    wins  = ready_df[ready_df["correct"]==1]["return_pct"]
    loses = ready_df[ready_df["correct"]==0]["return_pct"]
    avg_win  = float(wins.mean())  if not wins.empty  else 0
    avg_loss = float(loses.mean()) if not loses.empty else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    rr_c = GREEN if rr >= 2 else (AMBER if rr >= 1 else RED)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.markdown(f'<div class="metric-box"><div class="label">Accuracy</div><div class="value" style="color:{ac};">{acc:.1f}%</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-box"><div class="label">Avg Win</div><div class="value" style="color:{GREEN};">+{avg_win:.1f}%</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-box"><div class="label">Avg Loss</div><div class="value" style="color:{RED};">{avg_loss:.1f}%</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-box"><div class="label">Win/Loss Ratio</div><div class="value" style="color:{rr_c};">{rr:.2f}</div></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="metric-box"><div class="label">Evaluated</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    with m6: st.markdown(f'<div class="metric-box"><div class="label">Early Exits</div><div class="value" style="color:{AMBER};">{sup_count}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Normal vs superseded comparison ───────────────────────────────────
    if "superseded" in ready_df.columns and sup_count > 0:
        st.markdown('<div class="section-header">🔄 Normal Exit vs Early Exit (Signal Reversal)</div>', unsafe_allow_html=True)
        st.caption("Comparing signals held to their full horizon vs signals closed early due to an opposing signal.")
        nc1, nc2 = st.columns(2)
        normal = ready_df[ready_df["superseded"] == False]
        super_ = ready_df[ready_df["superseded"] == True]
        with nc1:
            n_acc = normal["correct"].mean()*100 if len(normal)>=3 else None
            nc = GREEN if n_acc and n_acc>=60 else (AMBER if n_acc and n_acc>=50 else RED)
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="color:#6b8fad;font-size:0.75rem;font-weight:700;margin-bottom:0.3rem;">HELD FULL HORIZON</div>
                <div style="font-size:1.8rem;font-weight:700;color:{nc if n_acc else MUTED};">
                    {f"{n_acc:.1f}%" if n_acc else "—"}
                </div>
                <div style="color:#6b8fad;font-size:0.75rem;">n={len(normal)}</div>
            </div>""", unsafe_allow_html=True)
        with nc2:
            s_acc = super_["correct"].mean()*100 if len(super_)>=3 else None
            sc = GREEN if s_acc and s_acc>=60 else (AMBER if s_acc and s_acc>=50 else RED)
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #fbbf2444;border-radius:12px;padding:1rem;text-align:center;">
                <div style="color:#fbbf24;font-size:0.75rem;font-weight:700;margin-bottom:0.3rem;">CLOSED EARLY (REVERSAL)</div>
                <div style="font-size:1.8rem;font-weight:700;color:{sc if s_acc else MUTED};">
                    {f"{s_acc:.1f}%" if s_acc else "—"}
                </div>
                <div style="color:#6b8fad;font-size:0.75rem;">n={len(super_)}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("If 'Closed Early' accuracy is higher, the reversal signals are improving returns by cutting losing trades early.")
        st.markdown("")

    # ── By confidence ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level</div>', unsafe_allow_html=True)
    cc = st.columns(3)
    conf_colors = {"HIGH": BLUE, "MEDIUM": AMBER, "LOW": MUTED}
    for i, conf in enumerate(["HIGH","MEDIUM","LOW"]):
        sub = ready_df[ready_df["confidence"]==conf]
        color = conf_colors[conf]
        with cc[i]:
            if len(sub) >= 3:
                a = sub["correct"].mean()*100
                ac2 = GREEN if a>=60 else (AMBER if a>=50 else RED)
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {color}44;border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:{color};font-weight:700;">{conf}</div>
                    <div style="font-size:1.8rem;font-weight:700;color:{ac2};">{a:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.72rem;">{int(sub["correct"].sum())}/{len(sub)} correct</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {color}22;border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:{color};font-weight:700;">{conf}</div>
                    <div style="font-size:1.1rem;color:{MUTED};">Need {max(0,3-len(sub))} more</div>
                    <div style="color:#6b8fad;font-size:0.72rem;">{len(sub)} so far</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── By horizon ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⏱ Accuracy by Time Horizon</div>', unsafe_allow_html=True)
    hh = st.columns(4)
    for i, (lbl, days) in enumerate([("Intraday",1),("Swing",5),("Medium",14),("Long",30)]):
        sub = ready_df[ready_df["horizon_days"]==days]
        with hh[i]:
            if len(sub) >= 3:
                a = sub["correct"].mean()*100
                ac3 = GREEN if a>=60 else (AMBER if a>=50 else RED)
                st.markdown(f'<div class="metric-box"><div class="label">{lbl}</div><div class="value" style="color:{ac3};">{a:.1f}%</div><div style="color:#6b8fad;font-size:0.7rem;">n={len(sub)}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-box"><div class="label">{lbl}</div><div class="value" style="color:{MUTED};">—</div><div style="color:#6b8fad;font-size:0.7rem;">n={len(sub)}</div></div>', unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        _style_fig(fig, ax)
        confs = ["HIGH","MEDIUM","LOW"]
        accs  = [ready_df[ready_df["confidence"]==c]["correct"].mean()*100
                 if len(ready_df[ready_df["confidence"]==c])>=3 else 0 for c in confs]
        ns    = [len(ready_df[ready_df["confidence"]==c]) for c in confs]
        bcs   = [GREEN if a>=60 else (AMBER if a>=50 else RED) for a in accs]
        bars  = ax.bar(confs, accs, color=bcs, edgecolor=BORDER, linewidth=0.4)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(0, 100); ax.set_title("Accuracy by Confidence", color=CYAN, fontsize=9)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        for bar, a, n in zip(bars, accs, ns):
            if n >= 3:
                ax.text(bar.get_x()+bar.get_width()/2, a+1,
                        f"{a:.0f}%\n(n={n})", ha="center", color=TEXT, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        rets = ready_df["return_pct"].dropna()
        if not rets.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            _style_fig(fig, ax)
            ax.hist(rets, bins=15, color=GREEN if rets.mean()>0 else RED,
                    edgecolor=BORDER, linewidth=0.4, alpha=0.8)
            ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
            ax.axvline(rets.mean(), color=CYAN, linewidth=1.5,
                       label=f"Mean {rets.mean():+.2f}%")
            ax.set_title("Return Distribution", color=CYAN, fontsize=9)
            ax.set_xlabel("Return %", color=MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Best / Worst ───────────────────────────────────────────────────────
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="section-header">🏆 Best Signals</div>', unsafe_allow_html=True)
        for _, r in ready_df.nlargest(3,"return_pct").iterrows():
            ac4 = GREEN if r["action"]=="BUY" else RED
            sup_tag = " ⚡ early exit" if r.get("superseded") else ""
            st.markdown(f"""
            <div style="background:#041a10;border:1px solid #166534;border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{"🟢" if r["action"]=="BUY" else "🔴"} {r["ticker"]} {r["action"]}</b>
                <span style="color:{GREEN};font-weight:700;float:right;">+{r["return_pct"]:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.73rem;">{r["confidence"]} · {r["time_horizon"]} · {r["date"][:10]}{sup_tag}</span>
            </div>""", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-header">💀 Worst Signals</div>', unsafe_allow_html=True)
        for _, r in ready_df.nsmallest(3,"return_pct").iterrows():
            ac4 = GREEN if r["action"]=="BUY" else RED
            sup_tag = " ⚡ early exit" if r.get("superseded") else ""
            st.markdown(f"""
            <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{"🟢" if r["action"]=="BUY" else "🔴"} {r["ticker"]} {r["action"]}</b>
                <span style="color:{RED};font-weight:700;float:right;">{r["return_pct"]:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.73rem;">{r["confidence"]} · {r["time_horizon"]} · {r["date"][:10]}{sup_tag}</span>
            </div>""", unsafe_allow_html=True)

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    disp_cols = ["date","ticker","action","confidence","time_horizon","held_days",
                 "entry_price","exit_price","return_pct","outcome","superseded","exit_reason","source"]
    disp = ready_df[[c for c in disp_cols if c in ready_df.columns]].copy()

    def c_a(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
    def c_o(v): return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def c_r(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v>0 else f"color:{RED}"
    def c_s(v): return f"color:{AMBER}" if v else ""

    sty = disp.style
    if "action"    in disp.columns: sty = sty.map(c_a, subset=["action"])
    if "outcome"   in disp.columns: sty = sty.map(c_o, subset=["outcome"])
    if "return_pct"in disp.columns: sty = sty.map(c_r, subset=["return_pct"])
    if "superseded"in disp.columns: sty = sty.map(c_s, subset=["superseded"])
    sty = sty.format({
        "entry_price": "${:.2f}",
        "exit_price":  "${:.2f}",
        "return_pct":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
    })
    st.dataframe(sty, use_container_width=True, hide_index=True)

    csv = ready_df.to_csv(index=False)
    st.download_button("⬇️ Download as CSV", data=csv,
                       file_name=f"signal_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

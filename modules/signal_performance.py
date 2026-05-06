"""
Signal Performance Tracker — uses eval_core with trading-day counting,
supersession, and black swan detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BLUE=  "#38bdf8"; CYAN=  "#7dd3fc"; AMBER= "#fbbf24"
RED=   "#f87171"; GREEN= "#4ade80"; CARD=  "#0d1b2a"
BORDER="#1e3a5f"; TEXT=  "#c9d8e8"; MUTED= "#6b8fad"


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)


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


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlc(ticker: str, entry_date: str, exit_date: str,
                lookback_days: int = 60):
    import yfinance as yf
    try:
        start = (pd.to_datetime(entry_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end   = (pd.to_datetime(exit_date)  + timedelta(days=3)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist if not hist.empty else None
    except Exception:
        return None


def _load_signals(days_back, src_filter=None, act_filter=None):
    from modules.signal_history import get_signals_df
    return get_signals_df(
        days_back=days_back,
        source_filter=src_filter,
        action_filter=act_filter
    )


def _load_full():
    from modules.signal_history import get_signals_df
    return get_signals_df(days_back=3650, action_filter=["BUY", "SHORT"])


def _mb(label, value, color=None):
    color = color or BLUE
    st.markdown(f'<div class="metric-box"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color};">{value}</div></div>',
                unsafe_allow_html=True)


def run_signal_performance():
    from modules.eval_core import evaluate_signals, get_horizon_trading_days

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Complete signal track record — trading-day accurate, reversal-aware, shock-filtered.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        <b>Trading days not calendar days</b> — INTRADAY = next trading day close,
        SWING = 5 trading days. <b>Reversal-aware</b> — if an opposing signal arrives
        before the horizon, the trade closes at that earlier date.
        <b>Black swan filter</b> — signals where an external shock moved the stock
        3× its normal daily range are flagged as "inconclusive" and excluded from
        accuracy stats (shown separately so you can review them).
        </span>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        days_back  = st.selectbox("Signals from last", [7,14,30,60,90,365],
                                   index=2, format_func=lambda x: f"{x} days", key="sp_days")
    with f2:
        src_filter = st.multiselect(
            "Source",
            ["live_intelligence","ticker_signals","market_briefing","strategy_signals","auto_evaluator"],
            default=["live_intelligence","ticker_signals","market_briefing","strategy_signals","auto_evaluator"],
            key="sp_src")
    with f3:
        act_filter = st.multiselect("Action", ["BUY","SHORT"],
                                    default=["BUY","SHORT"], key="sp_act")

    detect_shocks = st.checkbox(
        "🌪️ Detect and exclude black swan / external shock events",
        value=True,
        help="Flags signals where the stock moved 3× its normal daily range during the eval window. "
             "These are marked 'inconclusive' and excluded from accuracy stats. "
             "Uncheck to include them (they'll count as correct or incorrect)."
    )

    if not st.button("🔄 Evaluate All Signals", key="sp_run"):
        if st.session_state.get("sp_ready") is None:
            st.info("Click **Evaluate All Signals** to compute accuracy against real market outcomes.")
            return
    else:
        df_raw  = _load_signals(days_back, src_filter or None, act_filter or None)
        df_full = _load_full()
        if df_raw.empty:
            st.warning("No signals found in Supabase. Run Live News Feed and Ticker Signals to build history.")
            return

        prog = st.progress(0)
        def _cb(i, total, ticker, date):
            prog.progress((i+1)/total, text=f"Checking {ticker} — {date} ({i+1}/{total})")

        ready_df, waiting_df, inconclusive_df = evaluate_signals(
            df                       = df_raw,
            fetch_price_fn           = _fetch_price,
            fetch_ohlc_fn            = _fetch_ohlc,
            all_signals_for_timeline = df_full,
            progress_callback        = _cb,
        )
        prog.empty()
        st.session_state.update({
            "sp_ready": ready_df, "sp_waiting": waiting_df,
            "sp_inconclusive": inconclusive_df
        })

    ready_df        = st.session_state.get("sp_ready",        pd.DataFrame())
    waiting_df      = st.session_state.get("sp_waiting",      pd.DataFrame())
    inconclusive_df = st.session_state.get("sp_inconclusive", pd.DataFrame())

    # ── Waiting ────────────────────────────────────────────────────────────
    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not ready yet", expanded=False):
            st.caption("Haven't reached their trading-day horizon yet.")
            show = [c for c in ["date","ticker","action","confidence","time_horizon",
                                 "td_elapsed","not_ready_reason","source"] if c in waiting_df.columns]
            def ca(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            sty = waiting_df[show].style
            if "action" in show: sty = sty.map(ca, subset=["action"])
            st.dataframe(sty, use_container_width=True, hide_index=True)

    # ── Inconclusive ───────────────────────────────────────────────────────
    if not inconclusive_df.empty:
        with st.expander(f"🌪️ {len(inconclusive_df)} signal(s) excluded — external shock detected", expanded=False):
            st.caption("These signals coincided with an abnormal market move (3× normal daily range). "
                       "They are NOT counted in the accuracy stats — the original thesis may have been correct "
                       "but was overridden by an unpredictable external event.")
            show = [c for c in ["date","ticker","action","confidence","time_horizon",
                                 "entry_price","exit_price","return_pct","spike_reason"] if c in inconclusive_df.columns]
            def ca2(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            def cr2(v):
                if pd.isna(v): return ""
                return f"color:{GREEN}" if v>0 else f"color:{RED}"
            sty2 = inconclusive_df[show].style
            if "action"     in show: sty2 = sty2.map(ca2, subset=["action"])
            if "return_pct" in show: sty2 = sty2.map(cr2, subset=["return_pct"])
            sty2 = sty2.format({
                "entry_price": "${:.2f}", "exit_price": "${:.2f}",
                "return_pct":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
            })
            st.dataframe(sty2, use_container_width=True, hide_index=True)

    if ready_df.empty:
        st.warning("No signals ready for evaluation yet — none have reached their trading-day horizon.")
        return

    # ── Reversal note ──────────────────────────────────────────────────────
    sup_count = int(ready_df["superseded"].sum()) if "superseded" in ready_df.columns else 0
    if sup_count > 0:
        st.info(f"ℹ️ {sup_count} signal(s) were closed early by an opposing signal — evaluated at that earlier exit date.")

    # ── Accuracy banner ────────────────────────────────────────────────────
    total = len(ready_df)
    corr  = int(ready_df["correct"].sum())
    acc   = corr/total*100 if total>0 else 0
    ac    = GREEN if acc>=60 else (AMBER if acc>=50 else RED)
    label = "Strong Edge 🟢" if acc>=60 else ("Marginal 🟡" if acc>=50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {ac}44;border-radius:16px;
                padding:1.4rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;color:{ac};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1rem;margin-top:0.3rem;">Overall Signal Accuracy — {label}</div>
        <div style="color:#6b8fad;font-size:0.82rem;margin-top:0.3rem;">
            {corr} correct · {total-corr} incorrect · {sup_count} early exits ·
            {len(inconclusive_df)} excluded (external shock) · {len(waiting_df)} pending
        </div>
    </div>""", unsafe_allow_html=True)

    if total < 20:
        st.warning(f"⚠️ {total} evaluated signals — need 20+ for reliable conclusions.")

    # ── Metrics ────────────────────────────────────────────────────────────
    wins  = ready_df[ready_df["correct"]==1]["return_pct"]
    loses = ready_df[ready_df["correct"]==0]["return_pct"]
    aw    = float(wins.mean())  if not wins.empty  else 0
    al    = float(loses.mean()) if not loses.empty else 0
    rr    = abs(aw/al) if al!=0 else 0
    rr_c  = GREEN if rr>=2 else (AMBER if rr>=1 else RED)

    has_mfe = "mfe" in ready_df.columns and ready_df["mfe"].notna().any()
    mfe_threshold = st.slider(
        "📐 MFE threshold — min % move to count as 'directionally correct'",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="sp_mfe_thresh",
    ) if has_mfe else 0.5

    if has_mfe:
        dir_mask  = ready_df["mfe"].notna() & (ready_df["mfe"] >= mfe_threshold)
        dir_acc   = dir_mask.mean() * 100
        dir_ac_c  = GREEN if dir_acc>=60 else (AMBER if dir_acc>=50 else RED)
    else:
        dir_acc   = None

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: _mb("Exit Accuracy",   f"{acc:.1f}%",  ac)
    with m2: _mb("Avg Win",         f"+{aw:.1f}%",  GREEN)
    with m3: _mb("Avg Loss",        f"{al:.1f}%",   RED)
    with m4: _mb("Win/Loss Ratio",  f"{rr:.2f}",    rr_c)
    with m5: _mb("Evaluated",       str(total),     BLUE)
    with m6:
        if dir_acc is not None:
            _mb("Dir. Accuracy", f"{dir_acc:.1f}%", dir_ac_c)
        else:
            _mb("Excl. Shock", str(len(inconclusive_df)), AMBER)

    st.markdown("")

    # ── MFE Excursion section ──────────────────────────────────────────────
    if has_mfe:
        st.markdown('<div class="section-header">📐 Excursion Analysis (MFE / MAE)</div>', unsafe_allow_html=True)
        exc_df = ready_df[ready_df["mfe"].notna()].copy()
        if len(exc_df) >= 3:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            _style_fig(fig, list(axes))

            ax1 = axes[0]
            dot_colors = [GREEN if c else RED for c in exc_df["correct"]]
            ax1.scatter(exc_df["mfe"], exc_df["return_pct"],
                        c=dot_colors, alpha=0.75, edgecolors=BORDER, linewidth=0.5, s=55)
            ax1.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
            ax1.axvline(mfe_threshold, color=AMBER, linewidth=1, linestyle=":",
                        label=f"MFE ≥ {mfe_threshold:.1f}%")
            ymin = exc_df["return_pct"].min()
            ax1.fill_betweenx([ymin * 1.05, 0],
                              mfe_threshold, exc_df["mfe"].max() * 1.1,
                              alpha=0.05, color=AMBER)
            ax1.set_title("MFE vs Final Return %", color=CYAN, fontsize=9)
            ax1.set_xlabel("MFE % (best move in signal direction)", color=MUTED, fontsize=8)
            ax1.set_ylabel("Final Return %", color=MUTED, fontsize=8)
            from matplotlib.lines import Line2D
            ax1.legend(handles=[
                Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN, markersize=7, label='Correct exit'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor=RED,   markersize=7, label='Incorrect exit'),
            ], fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)

            ax2 = axes[1]
            grps = ["Correct", "Incorrect"]
            corr_s = exc_df[exc_df["correct"]==1]
            incr_s = exc_df[exc_df["correct"]==0]
            avg_mfe_v = [corr_s["mfe"].mean() if len(corr_s) else 0, incr_s["mfe"].mean() if len(incr_s) else 0]
            avg_mae_v = [corr_s["mae"].mean() if len(corr_s) else 0, incr_s["mae"].mean() if len(incr_s) else 0]
            x = np.arange(len(grps))
            w = 0.35
            ax2.bar(x - w/2, avg_mfe_v, w, color=GREEN, edgecolor=BORDER, linewidth=0.4, label="Avg MFE", alpha=0.85)
            ax2.bar(x + w/2, avg_mae_v, w, color=RED,   edgecolor=BORDER, linewidth=0.4, label="Avg MAE", alpha=0.85)
            ax2.set_xticks(x); ax2.set_xticklabels(grps, color=TEXT, fontsize=9)
            ax2.set_title("Avg MFE vs MAE: Correct vs Incorrect Exits", color=CYAN, fontsize=9)
            ax2.set_ylabel("% move", color=MUTED, fontsize=8)
            ax2.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            if len(incr_s) > 0:
                timing_victims = int((incr_s["mfe"] >= mfe_threshold).sum())
                if timing_victims > 0:
                    st.warning(
                        f"⚡ **{timing_victims} of {len(incr_s)} 'incorrect' signals ({timing_victims/len(incr_s)*100:.0f}%) "
                        f"were directionally right** — price moved ≥{mfe_threshold:.1f}% favourably but reversed before exit. "
                        f"True directional accuracy: **{dir_acc:.1f}%** vs exit accuracy **{acc:.1f}%**."
                    )

    # ── Normal vs early-exit comparison ───────────────────────────────────
    if "superseded" in ready_df.columns and sup_count > 0:
        st.markdown('<div class="section-header">🔄 Full Horizon vs Early Exit (Reversal)</div>', unsafe_allow_html=True)
        norm = ready_df[ready_df["superseded"]==False]
        sup  = ready_df[ready_df["superseded"]==True]
        nc1, nc2 = st.columns(2)
        with nc1:
            na = norm["correct"].mean()*100 if len(norm)>=3 else None
            nc_c = GREEN if na and na>=60 else (AMBER if na and na>=50 else RED)
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="color:#6b8fad;font-size:0.72rem;font-weight:700;">HELD FULL HORIZON</div>
                <div style="font-size:1.8rem;font-weight:700;color:{nc_c if na else MUTED};">{f"{na:.1f}%" if na else "—"}</div>
                <div style="color:#6b8fad;font-size:0.72rem;">n={len(norm)}</div>
            </div>""", unsafe_allow_html=True)
        with nc2:
            sa = sup["correct"].mean()*100 if len(sup)>=3 else None
            sc_c = GREEN if sa and sa>=60 else (AMBER if sa and sa>=50 else RED)
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #fbbf2444;border-radius:12px;padding:1rem;text-align:center;">
                <div style="color:#fbbf24;font-size:0.72rem;font-weight:700;">CLOSED EARLY (REVERSAL)</div>
                <div style="font-size:1.8rem;font-weight:700;color:{sc_c if sa else MUTED};">{f"{sa:.1f}%" if sa else "—"}</div>
                <div style="color:#6b8fad;font-size:0.72rem;">n={len(sup)}</div>
            </div>""", unsafe_allow_html=True)

    # ── By confidence ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level</div>', unsafe_allow_html=True)
    cc = st.columns(3)
    for i, conf in enumerate(["HIGH","MEDIUM","LOW"]):
        sub = ready_df[ready_df["confidence"]==conf]
        color = {0:BLUE,1:AMBER,2:MUTED}[i]
        with cc[i]:
            if len(sub)>=3:
                a  = sub["correct"].mean()*100
                ac2= GREEN if a>=60 else (AMBER if a>=50 else RED)
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

    # ── By horizon ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⏱ Accuracy by Time Horizon</div>', unsafe_allow_html=True)
    hh = st.columns(4)
    for i, (lbl, td) in enumerate([("Intraday",1),("Swing",5),("Medium",14),("Long",30)]):
        sub = ready_df[ready_df["horizon_td"]==td]
        with hh[i]:
            if len(sub)>=3:
                a = sub["correct"].mean()*100
                ac3 = GREEN if a>=60 else (AMBER if a>=50 else RED)
                st.markdown(f'<div class="metric-box"><div class="label">{lbl} ({td}td)</div>'
                            f'<div class="value" style="color:{ac3};">{a:.1f}%</div>'
                            f'<div style="color:#6b8fad;font-size:0.7rem;">n={len(sub)}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-box"><div class="label">{lbl}</div>'
                            f'<div class="value" style="color:{MUTED};">—</div>'
                            f'<div style="color:#6b8fad;font-size:0.7rem;">n={len(sub)}</div></div>',
                            unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────────────────
    rets = ready_df["return_pct"].dropna()
    if not rets.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        _style_fig(fig, ax)
        ax.hist(rets, bins=20, color=GREEN if rets.mean()>0 else RED,
                edgecolor=BORDER, linewidth=0.3, alpha=0.8)
        ax.axvline(0,          color=MUTED, linewidth=1, linestyle="--")
        ax.axvline(rets.mean(), color=CYAN, linewidth=1.5,
                   label=f"Mean {rets.mean():+.2f}%")
        ax.set_title("Return Distribution (external shocks excluded)", color=CYAN, fontsize=9)
        ax.set_xlabel("Return %", color=MUTED, fontsize=8)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Best / Worst ───────────────────────────────────────────────────────
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="section-header">🏆 Best Signals</div>', unsafe_allow_html=True)
        for _, r in ready_df.nlargest(3,"return_pct").iterrows():
            ac4 = GREEN if r["action"]=="BUY" else RED
            tag = " ⚡" if r.get("superseded") else ""
            st.markdown(f"""<div style="background:#041a10;border:1px solid #166534;border-radius:8px;
                padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{"🟢" if r["action"]=="BUY" else "🔴"} {r["ticker"]} {r["action"]}</b>
                <span style="color:{GREEN};font-weight:700;float:right;">+{r["return_pct"]:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.73rem;">{r["confidence"]} · {r["time_horizon"]} · {r["date"][:10]}{tag}</span>
                </div>""", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-header">💀 Worst Signals</div>', unsafe_allow_html=True)
        for _, r in ready_df.nsmallest(3,"return_pct").iterrows():
            ac4 = GREEN if r["action"]=="BUY" else RED
            tag = " ⚡" if r.get("superseded") else ""
            st.markdown(f"""<div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;
                padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{"🟢" if r["action"]=="BUY" else "🔴"} {r["ticker"]} {r["action"]}</b>
                <span style="color:{RED};font-weight:700;float:right;">{r["return_pct"]:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.73rem;">{r["confidence"]} · {r["time_horizon"]} · {r["date"][:10]}{tag}</span>
                </div>""", unsafe_allow_html=True)

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    disp_cols = ["date","entry_date","ticker","action","confidence","time_horizon","held_td",
                 "entry_price","exit_price","return_pct","mfe","mae","excursion_ratio",
                 "directional_correct","outcome","superseded","source"]
    disp = ready_df[[c for c in disp_cols if c in ready_df.columns]].copy()
    def c_a(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
    def c_o(v): return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def c_r(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v>0 else f"color:{RED}"
    def c_dc(v): return f"color:{GREEN}" if v else f"color:{MUTED}"
    sty = disp.style
    if "action"             in disp.columns: sty = sty.map(c_a,  subset=["action"])
    if "outcome"            in disp.columns: sty = sty.map(c_o,  subset=["outcome"])
    if "return_pct"         in disp.columns: sty = sty.map(c_r,  subset=["return_pct"])
    if "mfe"                in disp.columns: sty = sty.map(c_r,  subset=["mfe"])
    if "directional_correct" in disp.columns: sty = sty.map(c_dc, subset=["directional_correct"])
    sty = sty.format({
        "entry_price":     "${:.2f}",
        "exit_price":      "${:.2f}",
        "return_pct":      lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
        "mfe":             lambda x: f"+{x:.2f}%" if pd.notna(x) else "—",
        "mae":             lambda x: f"-{x:.2f}%" if pd.notna(x) else "—",
        "excursion_ratio": lambda x: f"{x:.2f}"   if pd.notna(x) else "—",
    })
    st.dataframe(sty, use_container_width=True, hide_index=True)

    csv = ready_df.to_csv(index=False)
    st.download_button("⬇️ Download as CSV (includes MFE/MAE)", data=csv,
                       file_name=f"signal_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

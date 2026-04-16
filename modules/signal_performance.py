"""
Signal Performance Tracker
============================
Evaluates every BUY/SHORT signal against real price outcomes.
Critically: each signal is only evaluated when enough time has passed
to match its stated time horizon. Signals not ready yet are shown
separately and clearly labeled.

Horizon matching:
  INTRADAY        → needs 1 day to pass
  SWING (1-5 days)→ needs 5 days to pass
  MEDIUM (weeks)  → needs 14 days to pass
  LONG (months)   → needs 30 days to pass
  Unknown         → needs 5 days (conservative default)
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
PURPLE = "#a78bfa"

# ── Horizon parsing ────────────────────────────────────────────────────────
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
    if days <= 1:  return "Intraday (1d)"
    if days <= 5:  return "Swing (5d)"
    if days <= 14: return "Medium (14d)"
    return "Long (30d)"


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


def _load_signals(days_back: int, source_filter: list = None,
                  action_filter: list = None) -> pd.DataFrame:
    try:
        conn   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
        df     = pd.read_sql_query(
            "SELECT * FROM signals WHERE timestamp >= ? AND action IN ('BUY','SHORT') ORDER BY timestamp DESC",
            conn, params=(cutoff,)
        )
        conn.close()
    except Exception:
        return pd.DataFrame()

    if source_filter and df.empty is False:
        df = df[df["source"].isin(source_filter)]
    if action_filter and df.empty is False:
        df = df[df["action"].isin(action_filter)]
    return df


def _evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split signals into:
      - ready_df:   horizon window has passed → evaluate
      - waiting_df: horizon window not yet passed → not ready

    Returns (ready_df_with_outcomes, waiting_df)
    """
    now     = datetime.now()
    ready   = []
    waiting = []

    total = len(df)
    prog  = st.progress(0, text="Evaluating signals…")

    for i, (_, row) in enumerate(df.iterrows()):
        prog.progress((i + 1) / total,
                      text=f"Checking {row['ticker']} — {row['timestamp'][:10]} ({i+1}/{total})")

        sig_dt       = pd.to_datetime(row["timestamp"])
        horizon_days = _horizon_days(row.get("time_horizon", ""))
        days_elapsed = (now - sig_dt).days

        base = {
            "timestamp":     row["timestamp"],
            "date":          row["timestamp"][:10],
            "ticker":        row["ticker"],
            "action":        row["action"],
            "confidence":    row["confidence"],
            "source":        row["source"],
            "time_horizon":  row.get("time_horizon", ""),
            "horizon_days":  horizon_days,
            "days_elapsed":  days_elapsed,
            "reasoning":     str(row.get("reasoning", ""))[:100],
            "article":       str(row.get("article_title", ""))[:70],
        }

        if days_elapsed < horizon_days:
            # Not enough time has passed — do not evaluate
            remaining = horizon_days - days_elapsed
            base["ready_in"] = f"{remaining} day{'s' if remaining != 1 else ''}"
            waiting.append(base)
            continue

        # Fetch entry and outcome prices
        entry_price = _fetch_price(row["ticker"], row["timestamp"][:10])
        if entry_price is None or entry_price <= 0:
            waiting.append({**base, "ready_in": "price data unavailable"})
            continue

        outcome_date  = (sig_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        outcome_price = _fetch_price(row["ticker"], outcome_date)
        if outcome_price is None or outcome_price <= 0:
            waiting.append({**base, "ready_in": "outcome price unavailable"})
            continue

        raw_ret = (outcome_price - entry_price) / entry_price * 100
        # For SHORT: profit when price falls
        pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
        correct = int(pnl_ret > 0)

        ready.append({
            **base,
            "entry_price":   round(entry_price, 2),
            "outcome_price": round(outcome_price, 2),
            "return_pct":    round(pnl_ret, 3),
            "correct":       correct,
            "outcome":       "✅ Correct" if correct else "❌ Incorrect",
        })

    prog.empty()
    return pd.DataFrame(ready), pd.DataFrame(waiting)


def run_signal_performance():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Every signal evaluated against real price outcomes — matched to its stated time horizon.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        A signal is only evaluated once enough time has passed to match its time horizon.
        INTRADAY signals need 1 day. SWING needs 5 days. MEDIUM needs 14. LONG needs 30.
        Signals that haven't reached their window yet are shown separately as "Not Ready."
        HOLD and WATCH signals are excluded — they make no directional prediction.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        days_back = st.selectbox("Signals from last", [7, 14, 30, 60, 90, 365],
                                 index=2, format_func=lambda x: f"{x} days", key="sp_days")
    with f2:
        src_filter = st.multiselect(
            "Source",
            ["live_intelligence", "ticker_signals", "market_briefing",
             "strategy_signals", "auto_evaluator"],
            default=["live_intelligence", "ticker_signals", "market_briefing",
                     "strategy_signals", "auto_evaluator"],
            key="sp_src"
        )
    with f3:
        act_filter = st.multiselect("Action", ["BUY", "SHORT"],
                                    default=["BUY", "SHORT"], key="sp_act")

    run_btn = st.button("🔄 Evaluate All Signals", key="sp_run")

    if not run_btn and st.session_state.get("sp_ready") is None:
        st.info("Click **Evaluate All Signals** to compute accuracy against real market outcomes.")
        return

    if run_btn:
        df_raw = _load_signals(days_back, src_filter or None, act_filter or None)
        if df_raw.empty:
            st.warning("No signals found matching your filters. Run Live News Feed and Ticker Signals to build history.")
            return
        with st.spinner(f"Fetching prices for {len(df_raw)} signals…"):
            ready_df, waiting_df = _evaluate(df_raw)
        st.session_state["sp_ready"]   = ready_df
        st.session_state["sp_waiting"] = waiting_df

    ready_df   = st.session_state.get("sp_ready",   pd.DataFrame())
    waiting_df = st.session_state.get("sp_waiting", pd.DataFrame())

    # ── Waiting / not ready signals ────────────────────────────────────────
    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not ready for evaluation yet", expanded=False):
            st.caption("These signals haven't reached their time horizon window. They'll be evaluable once enough time passes.")
            show_cols = ["date", "ticker", "action", "confidence", "time_horizon",
                         "horizon_days", "days_elapsed", "ready_in", "source"]
            show_cols = [c for c in show_cols if c in waiting_df.columns]

            def c_act(v): return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
            sty = waiting_df[show_cols].style.map(c_act, subset=["action"]) \
                      if "action" in show_cols else waiting_df[show_cols].style
            st.dataframe(sty, use_container_width=True, hide_index=True)

    if ready_df.empty:
        st.warning("No signals are ready for evaluation yet — none have reached their stated time horizon. "
                   "Check back once signals have had time to play out.")
        return

    # ── Accuracy banner ────────────────────────────────────────────────────
    total  = len(ready_df)
    corr   = int(ready_df["correct"].sum())
    acc    = corr / total * 100 if total > 0 else 0
    ac     = GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)
    label  = "Strong Edge 🟢" if acc >= 60 else ("Marginal 🟡" if acc >= 50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {ac}44;border-radius:16px;
                padding:1.4rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;color:{ac};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1rem;margin-top:0.3rem;">Overall Signal Accuracy — {label}</div>
        <div style="color:#6b8fad;font-size:0.82rem;margin-top:0.3rem;">
            {corr} correct out of {total} evaluated · {len(waiting_df)} not ready yet
        </div>
    </div>
    """, unsafe_allow_html=True)

    if total < 20:
        st.warning(f"⚠️ {total} evaluated signals is below the 20-signal minimum for reliable conclusions. Keep running Live Feed and Ticker Signals.")

    # ── Metrics row ────────────────────────────────────────────────────────
    wins  = ready_df[ready_df["correct"] == 1]["return_pct"]
    loses = ready_df[ready_df["correct"] == 0]["return_pct"]
    avg_win  = float(wins.mean())  if not wins.empty  else 0
    avg_loss = float(loses.mean()) if not loses.empty else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(f'<div class="metric-box"><div class="label">Accuracy</div><div class="value" style="color:{ac};">{acc:.1f}%</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-box"><div class="label">Avg Win</div><div class="value" style="color:{GREEN};">+{avg_win:.1f}%</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-box"><div class="label">Avg Loss</div><div class="value" style="color:{RED};">{avg_loss:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        rr_c = GREEN if rr >= 2 else (AMBER if rr >= 1 else RED)
        st.markdown(f'<div class="metric-box"><div class="label">Win/Loss Ratio</div><div class="value" style="color:{rr_c};">{rr:.2f}</div></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="metric-box"><div class="label">Evaluated</div><div class="value">{total}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Breakdown by confidence ────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level</div>', unsafe_allow_html=True)
    st.caption("HIGH confidence signals should consistently beat MEDIUM and LOW. If they don't, the confidence scoring needs tuning.")
    cc = st.columns(3)
    conf_colors = {"HIGH": BLUE, "MEDIUM": AMBER, "LOW": MUTED}
    for i, conf in enumerate(["HIGH", "MEDIUM", "LOW"]):
        sub = ready_df[ready_df["confidence"] == conf]
        color = conf_colors[conf]
        with cc[i]:
            if len(sub) >= 3:
                a = sub["correct"].mean() * 100
                ac2 = GREEN if a >= 60 else (AMBER if a >= 50 else RED)
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {color}44;border-radius:12px;
                            padding:1rem;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                color:{color};font-weight:700;margin-bottom:0.3rem;">{conf}</div>
                    <div style="font-size:1.8rem;font-weight:700;color:{ac2};">{a:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.75rem;">{int(sub['correct'].sum())}/{len(sub)} correct</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {color}22;border-radius:12px;
                            padding:1rem;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{color};font-weight:700;">{conf}</div>
                    <div style="font-size:1.1rem;color:{MUTED};">Need {max(0, 3-len(sub))} more</div>
                    <div style="color:#6b8fad;font-size:0.72rem;">{len(sub)} signal(s) so far</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Breakdown by horizon bucket ────────────────────────────────────────
    st.markdown('<div class="section-header">⏱ Accuracy by Time Horizon</div>', unsafe_allow_html=True)
    st.caption("Each signal is matched to its own stated horizon — these are not arbitrary windows.")
    ready_df["horizon_label"] = ready_df["horizon_days"].apply(_horizon_label)
    hh = st.columns(4)
    for i, (label, days) in enumerate([("Intraday (1d)", 1), ("Swing (5d)", 5),
                                        ("Medium (14d)", 14), ("Long (30d)", 30)]):
        sub = ready_df[ready_df["horizon_days"] == days]
        with hh[i]:
            if len(sub) >= 3:
                a = sub["correct"].mean() * 100
                ac3 = GREEN if a >= 60 else (AMBER if a >= 50 else RED)
                st.markdown(f"""
                <div class="metric-box">
                    <div class="label">{label}</div>
                    <div class="value" style="color:{ac3};">{a:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.72rem;">n={len(sub)}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="label">{label}</div>
                    <div class="value" style="color:{MUTED};">—</div>
                    <div style="color:#6b8fad;font-size:0.72rem;">n={len(sub)} (need 3+)</div>
                </div>""", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        _style_fig(fig, ax)
        confs = ["HIGH", "MEDIUM", "LOW"]
        accs  = []
        ns    = []
        for c in confs:
            sub = ready_df[ready_df["confidence"] == c]
            accs.append(sub["correct"].mean() * 100 if len(sub) >= 3 else 0)
            ns.append(len(sub))
        bcs = [GREEN if a >= 60 else (AMBER if a >= 50 else RED) for a in accs]
        bars = ax.bar(confs, accs, color=bcs, edgecolor=BORDER, linewidth=0.4)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% (random)")
        ax.set_ylim(0, 100)
        ax.set_title("Accuracy by Confidence", color=CYAN, fontsize=9)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        for bar, acc_v, n in zip(bars, accs, ns):
            if n >= 3:
                ax.text(bar.get_x() + bar.get_width()/2, acc_v + 1,
                        f"{acc_v:.0f}%\n(n={n})", ha="center", color=TEXT, fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 5,
                        f"n={n}", ha="center", color=MUTED, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        rets = ready_df["return_pct"].dropna()
        if not rets.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            _style_fig(fig, ax)
            ax.hist(rets, bins=15, color=GREEN if rets.mean() > 0 else RED,
                    edgecolor=BORDER, linewidth=0.4, alpha=0.8)
            ax.axvline(0,          color=MUTED, linewidth=1, linestyle="--")
            ax.axvline(rets.mean(), color=CYAN,  linewidth=1.5,
                       label=f"Mean {rets.mean():+.2f}%")
            ax.set_title("Return Distribution", color=CYAN, fontsize=9)
            ax.set_xlabel("Return %", color=MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Best / worst ───────────────────────────────────────────────────────
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="section-header">🏆 Best Signals</div>', unsafe_allow_html=True)
        best = ready_df.nlargest(3, "return_pct")[["ticker","action","confidence","time_horizon","return_pct","date"]]
        for _, r in best.iterrows():
            ac4 = GREEN if r["action"] == "BUY" else RED
            dot = "🟢" if r["action"] == "BUY" else "🔴"
            st.markdown(f"""
            <div style="background:#041a10;border:1px solid #166534;border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{dot} {r['ticker']} {r['action']}</b>
                <span style="color:{GREEN};font-weight:700;float:right;">+{r['return_pct']:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.75rem;">{r['confidence']} · {r['time_horizon']} · {str(r['date'])[:10]}</span>
            </div>""", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-header">💀 Worst Signals</div>', unsafe_allow_html=True)
        worst = ready_df.nsmallest(3, "return_pct")[["ticker","action","confidence","time_horizon","return_pct","date"]]
        for _, r in worst.iterrows():
            ac4 = GREEN if r["action"] == "BUY" else RED
            dot = "🟢" if r["action"] == "BUY" else "🔴"
            st.markdown(f"""
            <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.4rem;">
                <b style="color:{ac4};">{dot} {r['ticker']} {r['action']}</b>
                <span style="color:{RED};font-weight:700;float:right;">{r['return_pct']:.1f}%</span><br>
                <span style="color:#6b8fad;font-size:0.75rem;">{r['confidence']} · {r['time_horizon']} · {str(r['date'])[:10]}</span>
            </div>""", unsafe_allow_html=True)

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    display_cols = ["date","ticker","action","confidence","time_horizon",
                    "entry_price","outcome_price","return_pct","outcome","source","reasoning"]
    display = ready_df[[c for c in display_cols if c in ready_df.columns]].copy()

    def c_act(v):  return f"color:{GREEN}" if v=="BUY"        else f"color:{RED}"
    def c_out(v):  return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def c_ret(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v > 0 else f"color:{RED}"

    sty = display.style
    if "action"  in display.columns: sty = sty.map(c_act, subset=["action"])
    if "outcome" in display.columns: sty = sty.map(c_out, subset=["outcome"])
    if "return_pct" in display.columns: sty = sty.map(c_ret, subset=["return_pct"])
    sty = sty.format({
        "entry_price":   "${:.2f}",
        "outcome_price": "${:.2f}",
        "return_pct":    lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
    })
    st.dataframe(sty, use_container_width=True, hide_index=True)

    csv = ready_df.to_csv(index=False)
    st.download_button("⬇️ Download evaluated signals as CSV", data=csv,
                       file_name=f"signal_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

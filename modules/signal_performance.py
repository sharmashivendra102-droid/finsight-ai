"""
Signal Performance Tracker
===========================
Fetches price at signal time and compares to price N days later.
BUY correct = price went up. SHORT correct = price went down.
HOLD/WATCH excluded from accuracy (no directional bet).
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
def _fetch_price_on_date(ticker: str, target_date: str) -> float | None:
    """Get closing price for a ticker on or after a given date."""
    import yfinance as yf
    try:
        dt    = pd.to_datetime(target_date)
        start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        # Get closest price on or after signal date
        after = closes[closes.index.date >= dt.date()]
        if not after.empty:
            return float(after.iloc[0])
        return float(closes.iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_current_price(ticker: str) -> float | None:
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        closes = hist["Close"].dropna()
        return float(closes.iloc[-1]) if not closes.empty else None
    except Exception:
        return None


def _load_signals(days_back: int = 30) -> pd.DataFrame:
    try:
        conn   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
        df     = pd.read_sql_query(
            "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp DESC",
            conn, params=(cutoff,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _evaluate_signals(df: pd.DataFrame, outcome_days: int) -> pd.DataFrame:
    """
    For each BUY/SHORT signal, fetch entry price and outcome price.
    Determine correct/incorrect based on direction.
    """
    if df.empty:
        return pd.DataFrame()

    # Only evaluate directional signals
    directional = df[df["action"].isin(["BUY", "SHORT"])].copy()
    if directional.empty:
        return pd.DataFrame()

    results = []
    cutoff_date = datetime.now() - timedelta(days=outcome_days)

    progress = st.progress(0, text="Evaluating signal outcomes…")
    total = len(directional)

    for i, (_, row) in enumerate(directional.iterrows()):
        progress.progress((i + 1) / total,
                          text=f"Checking {row['ticker']} signal from {row['timestamp'][:10]}…")

        sig_dt = pd.to_datetime(row["timestamp"])
        ticker = row["ticker"]

        # Entry price — price on signal date
        entry_price = _fetch_price_on_date(ticker, row["timestamp"][:10])

        # Outcome — only evaluate if signal is old enough
        if sig_dt > cutoff_date:
            outcome_status = "pending"
            outcome_price  = None
            return_pct     = None
            correct        = None
        else:
            outcome_date  = (sig_dt + timedelta(days=outcome_days)).strftime("%Y-%m-%d")
            outcome_price = _fetch_price_on_date(ticker, outcome_date)

            if entry_price and outcome_price and entry_price > 0:
                return_pct = (outcome_price - entry_price) / entry_price * 100
                if row["action"] == "BUY":
                    correct = return_pct > 0
                else:  # SHORT
                    correct = return_pct < 0
                outcome_status = "correct" if correct else "incorrect"
            else:
                outcome_status = "no_data"
                return_pct     = None
                correct        = None

        results.append({
            "timestamp":      row["timestamp"],
            "ticker":         ticker,
            "action":         row["action"],
            "confidence":     row["confidence"],
            "source":         row["source"],
            "reasoning":      row["reasoning"][:80] if row.get("reasoning") else "",
            "article":        row["article_title"][:60] if row.get("article_title") else "",
            "entry_price":    entry_price,
            "outcome_price":  outcome_price,
            "return_pct":     return_pct,
            "correct":        correct,
            "outcome_status": outcome_status,
        })

    progress.empty()
    return pd.DataFrame(results)


def _compute_stats(perf_df: pd.DataFrame) -> dict:
    if perf_df.empty:
        return {}

    evaluated = perf_df[perf_df["outcome_status"].isin(["correct", "incorrect"])]
    pending   = perf_df[perf_df["outcome_status"] == "pending"]

    if evaluated.empty:
        return {"pending_only": True, "pending_count": len(pending)}

    total_eval = len(evaluated)
    correct    = evaluated["correct"].sum()
    accuracy   = correct / total_eval * 100 if total_eval > 0 else 0

    # By confidence
    conf_stats = {}
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        sub = evaluated[evaluated["confidence"] == conf]
        if len(sub) > 0:
            conf_stats[conf] = {
                "total":    len(sub),
                "correct":  int(sub["correct"].sum()),
                "accuracy": sub["correct"].mean() * 100,
            }

    # By action
    action_stats = {}
    for action in ["BUY", "SHORT"]:
        sub = evaluated[evaluated["action"] == action]
        if len(sub) > 0:
            action_stats[action] = {
                "total":    len(sub),
                "correct":  int(sub["correct"].sum()),
                "accuracy": sub["correct"].mean() * 100,
            }

    # By source
    source_stats = {}
    for src in evaluated["source"].unique():
        sub = evaluated[evaluated["source"] == src]
        if len(sub) > 0:
            source_stats[src] = {
                "total":    len(sub),
                "accuracy": sub["correct"].mean() * 100,
            }

    # Average return on correct signals
    correct_rets  = evaluated[evaluated["correct"] == True]["return_pct"].dropna()
    incorrect_rets= evaluated[evaluated["correct"] == False]["return_pct"].dropna()
    avg_win  = float(correct_rets.abs().mean())  if not correct_rets.empty  else 0
    avg_loss = float(incorrect_rets.abs().mean()) if not incorrect_rets.empty else 0

    # Best/worst signals
    best  = evaluated.nlargest(3,  "return_pct")[["ticker","action","return_pct","confidence","timestamp"]]
    worst = evaluated.nsmallest(3, "return_pct")[["ticker","action","return_pct","confidence","timestamp"]]

    return {
        "total_evaluated": total_eval,
        "total_correct":   int(correct),
        "accuracy":        accuracy,
        "pending_count":   len(pending),
        "conf_stats":      conf_stats,
        "action_stats":    action_stats,
        "source_stats":    source_stats,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "best_signals":    best,
        "worst_signals":   worst,
    }


def run_signal_performance():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Signal accuracy tracked automatically — every BUY and SHORT signal is evaluated against real price outcomes.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Entry price is recorded at signal time. Outcome is checked after your chosen evaluation window.
        HOLD and WATCH signals are excluded (no directional bet to evaluate).
        The more signals accumulate over time, the more statistically meaningful these numbers become.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        days_back = st.selectbox(
            "Signals from last…",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"{x} days"
        )
    with c2:
        outcome_days = st.selectbox(
            "Evaluate outcome after…",
            [1, 3, 5, 7, 14],
            index=1,
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )
    with c3:
        conf_filter = st.multiselect(
            "Confidence levels",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )

    run_btn = st.button("🔄 Evaluate Signal Performance", key="run_perf",
                        use_container_width=False)

    if not run_btn and not st.session_state.get("perf_results"):
        st.info("👆 Click **Evaluate Signal Performance** to analyse your signal track record. "
                "This fetches historical prices for every signal — may take 30–60 seconds.")
        return

    if run_btn:
        df_raw = _load_signals(days_back=days_back)
        if df_raw.empty:
            st.warning("No signals in the database yet. Run Live Intelligence or Ticker Signals first to build history.")
            return

        if conf_filter:
            df_raw = df_raw[df_raw["confidence"].isin(conf_filter)]

        perf_df = _evaluate_signals(df_raw, outcome_days)
        st.session_state["perf_results"]     = perf_df
        st.session_state["perf_stats"]       = _compute_stats(perf_df)
        st.session_state["perf_outcome_days"]= outcome_days
        st.session_state["perf_days_back"]   = days_back

    perf_df     = st.session_state.get("perf_results", pd.DataFrame())
    stats       = st.session_state.get("perf_stats", {})
    outcome_days= st.session_state.get("perf_outcome_days", 3)
    days_back   = st.session_state.get("perf_days_back", 30)

    if perf_df.empty or not stats:
        st.info("No evaluated signals yet.")
        return

    if stats.get("pending_only"):
        st.warning(f"All {stats['pending_count']} signals are less than {outcome_days} day(s) old — "
                   f"outcomes not yet available. Check back after the evaluation window passes.")
        return

    # ── Headline accuracy banner ───────────────────────────────────────────
    acc   = stats["accuracy"]
    total = stats["total_evaluated"]
    corr  = stats["total_correct"]

    acc_color = GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)
    acc_label = "Strong Edge 🟢" if acc >= 60 else ("Marginal 🟡" if acc >= 50 else "Below Random 🔴")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {acc_color}44;border-radius:16px;
                padding:1.5rem 2rem;margin:1rem 0;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:3rem;
                    font-weight:700;color:{acc_color};">{acc:.1f}%</div>
        <div style="color:#c9d8e8;font-size:1.1rem;margin-top:0.3rem;">
            Overall Signal Accuracy — {acc_label}
        </div>
        <div style="color:#6b8fad;font-size:0.85rem;margin-top:0.4rem;">
            {corr} correct out of {total} evaluated BUY/SHORT signals
            · {outcome_days}-day outcome window
            · Last {days_back} days of signals
        </div>
        {"<div style='color:#fbbf24;font-size:0.8rem;margin-top:0.4rem;'>⏳ " + str(stats['pending_count']) + " signals still pending (too recent to evaluate)</div>" if stats.get('pending_count', 0) > 0 else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">Avg Win</div><div class="value" style="color:{GREEN};">+{stats["avg_win"]:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="label">Avg Loss</div><div class="value" style="color:{RED};">-{stats["avg_loss"]:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        pf = stats["avg_win"] / stats["avg_loss"] if stats["avg_loss"] > 0 else 0
        pf_color = GREEN if pf > 1 else RED
        st.markdown(f'<div class="metric-box"><div class="label">Win/Loss Ratio</div><div class="value" style="color:{pf_color};">{pf:.2f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="label">Pending</div><div class="value" style="color:{AMBER};">{stats.get("pending_count", 0)}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Accuracy by confidence ─────────────────────────────────────────────
    conf_stats = stats.get("conf_stats", {})
    if conf_stats:
        st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level</div>', unsafe_allow_html=True)
        st.caption("This is the most important table — HIGH confidence signals should have the highest accuracy.")

        cs_cols = st.columns(len(conf_stats))
        conf_colors = {"HIGH": BLUE, "MEDIUM": AMBER, "LOW": MUTED}
        for i, (conf, cs) in enumerate(conf_stats.items()):
            color = conf_colors.get(conf, MUTED)
            acc_c = cs["accuracy"]
            acc_col = GREEN if acc_c >= 60 else (AMBER if acc_c >= 50 else RED)
            with cs_cols[i]:
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {color}44;border-radius:12px;
                            padding:1rem;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{color};
                                font-weight:700;margin-bottom:0.4rem;">{conf} CONFIDENCE</div>
                    <div style="font-size:2rem;font-weight:700;color:{acc_col};">{acc_c:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.8rem;">{cs['correct']}/{cs['total']} correct</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    evaluated = perf_df[perf_df["outcome_status"].isin(["correct", "incorrect"])]

    if not evaluated.empty:
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="section-header">📊 Accuracy by Action Type</div>', unsafe_allow_html=True)
            action_stats = stats.get("action_stats", {})
            if action_stats:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                _style_fig(fig, ax)
                acts  = list(action_stats.keys())
                accs  = [action_stats[a]["accuracy"] for a in acts]
                tots  = [action_stats[a]["total"] for a in acts]
                bcolors = [GREEN if a >= 55 else (AMBER if a >= 50 else RED) for a in accs]
                bars = ax.bar(acts, accs, color=bcolors, edgecolor=BORDER, linewidth=0.5)
                ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% (random)")
                ax.set_ylim(0, 100)
                ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
                ax.set_title("Accuracy by Signal Type", color=CYAN, fontsize=10)
                ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
                for bar, acc_v, tot in zip(bars, accs, tots):
                    ax.text(bar.get_x() + bar.get_width()/2, acc_v + 1,
                            f"{acc_v:.0f}%\n(n={tot})", ha="center", color=TEXT, fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with ch2:
            st.markdown('<div class="section-header">📈 Return Distribution</div>', unsafe_allow_html=True)
            rets = evaluated["return_pct"].dropna()
            if not rets.empty:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                _style_fig(fig, ax)
                colors_hist = [GREEN if r >= 0 else RED for r in rets]
                ax.bar(range(len(rets)), sorted(rets), color=[GREEN if r >= 0 else RED
                       for r in sorted(rets)], edgecolor=BORDER, linewidth=0.3, width=0.8)
                ax.axhline(0, color=MUTED, linewidth=0.8)
                ax.set_title(f"Return % per Signal ({outcome_days}d)", color=CYAN, fontsize=10)
                ax.set_ylabel("Return %", color=MUTED, fontsize=8)
                ax.set_xlabel("Signals (sorted)", color=MUTED, fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Accuracy over time ─────────────────────────────────────────────────
    if len(evaluated) >= 5:
        st.markdown('<div class="section-header">📅 Rolling Accuracy Over Time</div>', unsafe_allow_html=True)
        ev_time = evaluated.copy()
        ev_time["date"]    = pd.to_datetime(ev_time["timestamp"]).dt.date
        ev_time["correct_int"] = ev_time["correct"].astype(int)
        daily_acc = ev_time.groupby("date").agg(
            accuracy=("correct_int", "mean"),
            count=("correct_int", "count")
        ).reset_index()
        daily_acc["accuracy"] *= 100

        fig, ax = plt.subplots(figsize=(12, 3))
        _style_fig(fig, ax)
        ax.bar(pd.to_datetime(daily_acc["date"]), daily_acc["accuracy"],
               color=[GREEN if v >= 50 else RED for v in daily_acc["accuracy"]],
               width=0.8, edgecolor=BORDER, linewidth=0.3)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% baseline")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Daily Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("Signal Accuracy by Day", color=CYAN, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Best / Worst signals ───────────────────────────────────────────────
    if not evaluated.empty:
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="section-header">🏆 Best Signals</div>', unsafe_allow_html=True)
            best = stats.get("best_signals")
            if best is not None and not best.empty:
                for _, r in best.iterrows():
                    ret = r["return_pct"]
                    sign = "+" if ret >= 0 else ""
                    action = r["action"]
                    dot = "🟢" if action == "BUY" else "🔴"
                    st.markdown(f"""
                    <div style="background:#041a10;border:1px solid #166534;border-radius:8px;
                                padding:0.6rem 1rem;margin-bottom:0.4rem;">
                        <b style="color:#4ade80;">{dot} {r['ticker']} {action}</b>
                        <span style="color:#4ade80;font-weight:700;float:right;">{sign}{ret:.1f}%</span><br>
                        <span style="color:#6b8fad;font-size:0.78rem;">{r['confidence']} conf · {str(r['timestamp'])[:10]}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with b2:
            st.markdown('<div class="section-header">💀 Worst Signals</div>', unsafe_allow_html=True)
            worst = stats.get("worst_signals")
            if worst is not None and not worst.empty:
                for _, r in worst.iterrows():
                    ret = r["return_pct"]
                    sign = "+" if ret >= 0 else ""
                    action = r["action"]
                    dot = "🟢" if action == "BUY" else "🔴"
                    st.markdown(f"""
                    <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;
                                padding:0.6rem 1rem;margin-bottom:0.4rem;">
                        <b style="color:#f87171;">{dot} {r['ticker']} {action}</b>
                        <span style="color:#f87171;font-weight:700;float:right;">{sign}{ret:.1f}%</span><br>
                        <span style="color:#6b8fad;font-size:0.78rem;">{r['confidence']} conf · {str(r['timestamp'])[:10]}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Full evaluated table ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    display_cols = ["timestamp", "ticker", "action", "confidence", "source",
                    "entry_price", "outcome_price", "return_pct", "outcome_status", "reasoning"]
    display = perf_df[[c for c in display_cols if c in perf_df.columns]].copy()

    def c_status(v):
        return {
            "correct":   f"color:{GREEN}",
            "incorrect": f"color:{RED}",
            "pending":   f"color:{AMBER}",
            "no_data":   f"color:{MUTED}",
        }.get(v, "")
    def c_ret(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v >= 0 else f"color:{RED}"
    def c_act(v):
        return {"BUY": f"color:{GREEN}", "SHORT": f"color:{RED}"}.get(v, "")

    styler = display.style
    if "outcome_status" in display.columns: styler = styler.map(c_status, subset=["outcome_status"])
    if "return_pct"     in display.columns: styler = styler.map(c_ret,    subset=["return_pct"])
    if "action"         in display.columns: styler = styler.map(c_act,    subset=["action"])
    styler = styler.format({
        "entry_price":   lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
        "outcome_price": lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
        "return_pct":    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Accuracy interpretation ────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 What These Numbers Mean</div>', unsafe_allow_html=True)
    acc = stats["accuracy"]
    if total < 20:
        st.warning(f"⚠️ Only {total} evaluated signals — statistically insufficient. Need 50+ signals for meaningful conclusions. Keep running the app.")
    elif acc >= 65:
        st.success(f"🟢 {acc:.1f}% accuracy over {total} signals is genuinely strong. Professional quant funds target 55–60%. If this holds with 100+ signals, you have real edge.")
    elif acc >= 55:
        st.info(f"🟡 {acc:.1f}% accuracy over {total} signals shows modest edge above random. Build more signal history to confirm.")
    elif acc >= 50:
        st.warning(f"🟡 {acc:.1f}% — marginally above random. Not tradeable with conviction yet. Review which sources/confidence levels perform best.")
    else:
        st.error(f"🔴 {acc:.1f}% — below random. Something is wrong with signal quality or the evaluation window doesn't match the time horizon. Try a longer outcome window.")

    csv = perf_df.to_csv(index=False)
    st.download_button("⬇️ Download performance data as CSV",
                       data=csv,
                       file_name=f"signal_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

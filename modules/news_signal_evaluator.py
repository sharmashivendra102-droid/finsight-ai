"""
News Signal Evaluator
======================
Tests whether HIGH/MEDIUM confidence signals from Live Feed and Ticker Signals
actually predicted market direction correctly.

How it works:
- Reads every news-based signal stored in signal_history.db
- Fetches the stock price at signal time
- Fetches the price N days later
- Determines correct/incorrect based on BUY (price went up) or SHORT (price went down)
- Breaks results down by confidence level, source, ticker, and time horizon
- Runs statistical significance tests

This is real evaluation on real signals the app actually generated.
The longer the app has been running, the more data and the more reliable the results.
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
PURPLE = "#a78bfa"

HORIZONS = [1, 3, 5, 10]


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


def _load_news_signals() -> pd.DataFrame:
    """Load all news-based signals from the database."""
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        df = pd.read_sql_query("""
            SELECT * FROM signals
            WHERE source IN ('live_intelligence', 'ticker_signals', 'market_briefing')
            AND action IN ('BUY', 'SHORT')
            ORDER BY timestamp DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_price_on_date(ticker: str, date_str: str) -> float | None:
    """Get closing price on or immediately after a given date."""
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


def _evaluate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, compute return at each horizon.
    Only evaluates signals old enough for the horizon to have passed.
    """
    now     = datetime.now()
    results = []
    total   = len(df)

    prog = st.progress(0, text="Evaluating news signal outcomes…")

    for i, (_, row) in enumerate(df.iterrows()):
        prog.progress((i + 1) / total,
                      text=f"Checking {row['ticker']} — {row['timestamp'][:10]}  ({i+1}/{total})")

        sig_dt      = pd.to_datetime(row["timestamp"])
        entry_price = _get_price_on_date(row["ticker"], row["timestamp"][:10])

        if entry_price is None or entry_price <= 0:
            continue

        out_row = {
            "timestamp":   row["timestamp"],
            "date":        row["timestamp"][:10],
            "ticker":      row["ticker"],
            "action":      row["action"],
            "confidence":  row["confidence"],
            "source":      row["source"],
            "urgency":     row.get("urgency", ""),
            "reasoning":   str(row.get("reasoning", ""))[:120],
            "article":     str(row.get("article_title", ""))[:80],
            "entry_price": entry_price,
        }

        for h in HORIZONS:
            days_elapsed = (now - sig_dt).days
            if days_elapsed < h:
                out_row[f"ret_{h}d"]     = np.nan
                out_row[f"correct_{h}d"] = np.nan
                out_row[f"status_{h}d"]  = "pending"
            else:
                future_date  = (sig_dt + timedelta(days=h)).strftime("%Y-%m-%d")
                future_price = _get_price_on_date(row["ticker"], future_date)
                if future_price is None or future_price <= 0:
                    out_row[f"ret_{h}d"]     = np.nan
                    out_row[f"correct_{h}d"] = np.nan
                    out_row[f"status_{h}d"]  = "no_data"
                else:
                    raw_ret = (future_price - entry_price) / entry_price * 100
                    pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
                    correct = int(pnl_ret > 0)
                    out_row[f"ret_{h}d"]     = round(pnl_ret, 3)
                    out_row[f"correct_{h}d"] = correct
                    out_row[f"status_{h}d"]  = "correct" if correct else "incorrect"

        results.append(out_row)

    prog.empty()
    return pd.DataFrame(results)


def _significance(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) < 5:
        return {"p": None, "t": None, "sig": False, "n": len(clean)}
    t, p = scipy_stats.ttest_1samp(clean, 0)
    return {"p": round(float(p), 4), "t": round(float(t), 3),
            "sig": bool(p < 0.05), "n": len(clean)}


def _accuracy_color(acc: float) -> str:
    return GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)


def run_news_signal_evaluator():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">This evaluates every news-based signal the app has ever generated against real market outcomes.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        It reads your Signal History, fetches the stock price at the exact time each signal was generated,
        then checks what the price was 1, 3, 5, and 10 days later. BUY signals are correct if the price
        went UP. SHORT signals are correct if the price went DOWN. The longer you run the Live Feed and
        Ticker Signals tabs, the more data you'll have here and the more reliable these results become.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load signals ───────────────────────────────────────────────────────
    df_raw = _load_news_signals()

    if df_raw.empty:
        st.warning("""
        **No news signals in the database yet.**

        To build your news signal track record:
        1. Go to **Live Feed** and press **▶️ Start** — leave it running for at least 30 minutes
        2. Go to **Ticker Signals** and analyse several tickers
        3. Come back here and click Evaluate

        The more you run those tabs, the more meaningful this analysis becomes.
        Real edge takes real time to prove — there's no shortcut here.
        """)
        return

    directional = df_raw[df_raw["action"].isin(["BUY", "SHORT"])]
    total_available = len(directional)

    # ── Summary counts ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="label">Total News Signals</div><div class="value">{total_available}</div></div>', unsafe_allow_html=True)
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

    # ── Filter controls ────────────────────────────────────────────────────
    with st.expander("🔧 Filter signals to evaluate", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            conf_filter = st.multiselect("Confidence", ["HIGH", "MEDIUM", "LOW"],
                                         default=["HIGH", "MEDIUM", "LOW"],
                                         key="nse_conf")
        with f2:
            src_filter = st.multiselect("Source", ["live_intelligence", "ticker_signals", "market_briefing"],
                                        default=["live_intelligence", "ticker_signals", "market_briefing"],
                                        key="nse_src")
        with f3:
            days_back = st.selectbox("Signals from last", [7, 14, 30, 60, 90, 365],
                                     index=2, format_func=lambda x: f"{x} days",
                                     key="nse_days")

    filtered = directional.copy()
    if conf_filter: filtered = filtered[filtered["confidence"].isin(conf_filter)]
    if src_filter:  filtered = filtered[filtered["source"].isin(src_filter)]
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filtered = filtered[filtered["timestamp"] >= cutoff]

    st.info(f"**{len(filtered)} signals** selected for evaluation. "
            f"Fetching prices takes ~1–2 seconds per signal.")

    if st.button("🔬 Evaluate News Signal Accuracy", key="nse_run",
                 use_container_width=False):
        if filtered.empty:
            st.error("No signals match your filters.")
        else:
            with st.spinner(f"Fetching prices for {len(filtered)} signals…"):
                results = _evaluate_all(filtered)
            st.session_state["nse_results"] = results
            st.session_state["nse_run_time"] = datetime.now().strftime("%H:%M:%S")

    results = st.session_state.get("nse_results", pd.DataFrame())
    if results.empty:
        return

    # ── How many are evaluated vs pending ─────────────────────────────────
    evaluated_mask = results["status_5d"].isin(["correct", "incorrect"])
    evaluated  = results[evaluated_mask]
    pending    = results[~evaluated_mask]

    st.caption(f"Evaluated: {len(evaluated)} · Pending (too recent): {len(pending)} · "
               f"Run time: {st.session_state.get('nse_run_time','')}")

    if len(evaluated) == 0:
        st.warning("All signals are too recent to evaluate yet — outcomes haven't had time to play out. "
                   "Come back after a few days.")
        return

    if len(evaluated) < 10:
        st.warning(f"Only {len(evaluated)} signals have been evaluated. "
                   f"Results are not statistically meaningful yet — you need at least 30–50 signals. "
                   f"Keep running the Live Feed and Ticker Signals tabs daily.")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN ACCURACY DISPLAY
    # ══════════════════════════════════════════════════════════════════════

    # ── Big accuracy banner ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Overall Accuracy by Time Horizon</div>', unsafe_allow_html=True)

    h_cols = st.columns(len(HORIZONS))
    for i, h in enumerate(HORIZONS):
        col_c = f"correct_{h}d"
        col_r = f"ret_{h}d"
        sub   = results.dropna(subset=[col_c])
        with h_cols[i]:
            if len(sub) >= 3:
                acc   = sub[col_c].mean() * 100
                sig   = _significance(sub[col_r])
                ac    = _accuracy_color(acc)
                p_str = f"p={sig['p']:.3f} {'✅' if sig['sig'] else '❌'}" if sig["p"] else ""
                st.markdown(f"""
                <div class="metric-box">
                    <div class="label">{h}-Day Hold</div>
                    <div class="value" style="color:{ac};">{acc:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.7rem;margin-top:2px;">{p_str}</div>
                    <div style="color:#6b8fad;font-size:0.7rem;">n={len(sub)}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-box"><div class="label">{h}d</div><div class="value" style="color:{MUTED};">—</div><div style="color:{MUTED};font-size:0.7rem;">need more</div></div>', unsafe_allow_html=True)

    st.caption("✅ p<0.05 = statistically significant | n = number of evaluated signals")
    st.markdown("")

    # ── Confidence breakdown ── THE KEY TABLE ─────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Confidence Level — The Most Important Table</div>', unsafe_allow_html=True)
    st.caption("If the system is well-calibrated, HIGH confidence signals should have higher accuracy than MEDIUM, which should beat LOW.")

    conf_data = []
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        sub_conf = results[results["confidence"] == conf]
        row_data = {"Confidence": conf, "Total Signals": len(sub_conf)}
        for h in HORIZONS:
            col_c = f"correct_{h}d"
            col_r = f"ret_{h}d"
            ev    = sub_conf.dropna(subset=[col_c])
            if len(ev) >= 3:
                acc = ev[col_c].mean() * 100
                sig = _significance(ev[col_r])
                row_data[f"Acc {h}d"]   = round(acc, 1)
                row_data[f"Avg Ret {h}d"] = round(ev[col_r].mean(), 2)
                row_data[f"p {h}d"]     = sig["p"] if sig["p"] else np.nan
            else:
                row_data[f"Acc {h}d"]     = np.nan
                row_data[f"Avg Ret {h}d"] = np.nan
                row_data[f"p {h}d"]       = np.nan
        conf_data.append(row_data)

    conf_df = pd.DataFrame(conf_data)
    acc_cols = [c for c in conf_df.columns if c.startswith("Acc")]
    ret_cols = [c for c in conf_df.columns if c.startswith("Avg Ret")]

    def c_acc(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{_accuracy_color(v)};font-weight:700"
    def c_ret(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{GREEN}" if v > 0 else f"color:{RED}"
    def c_p(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{GREEN}" if v < 0.05 else f"color:{MUTED}"

    styler = conf_df.style
    if acc_cols: styler = styler.map(c_acc, subset=acc_cols)
    if ret_cols: styler = styler.map(c_ret, subset=ret_cols)
    p_cols = [c for c in conf_df.columns if c.startswith("p ")]
    if p_cols: styler = styler.map(c_p, subset=p_cols)
    fmt = {c: "{:.1f}%" for c in acc_cols}
    fmt.update({c: "{:+.2f}%" for c in ret_cols})
    fmt.update({c: "{:.4f}" for c in p_cols})
    styler = styler.format(fmt, na_rep="—")
    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Source breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📡 Accuracy by Source</div>', unsafe_allow_html=True)
    src_data = []
    source_labels = {
        "live_intelligence": "Live Feed",
        "ticker_signals":    "Ticker Signals",
        "market_briefing":   "Market Briefing",
    }
    for src in results["source"].unique():
        sub_src = results[results["source"] == src]
        row = {"Source": source_labels.get(src, src), "Signals": len(sub_src)}
        for h in [3, 5]:
            col_c = f"correct_{h}d"
            col_r = f"ret_{h}d"
            ev = sub_src.dropna(subset=[col_c])
            if len(ev) >= 3:
                row[f"Acc {h}d"]     = round(ev[col_c].mean() * 100, 1)
                row[f"Avg Ret {h}d"] = round(ev[col_r].mean(), 2)
            else:
                row[f"Acc {h}d"]     = np.nan
                row[f"Avg Ret {h}d"] = np.nan
        src_data.append(row)

    if src_data:
        src_df = pd.DataFrame(src_data)
        a_cols = [c for c in src_df.columns if c.startswith("Acc")]
        r_cols = [c for c in src_df.columns if c.startswith("Avg")]
        styler2 = src_df.style
        if a_cols: styler2 = styler2.map(c_acc, subset=a_cols)
        if r_cols: styler2 = styler2.map(c_ret, subset=r_cols)
        styler2 = styler2.format(
            {c: "{:.1f}%" for c in a_cols} | {c: "{:+.2f}%" for c in r_cols},
            na_rep="—"
        )
        st.dataframe(styler2, use_container_width=True, hide_index=True)

    # ── Top tickers ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Best Tickers for News Signals (5-day hold)</div>', unsafe_allow_html=True)
    ticker_data = []
    for ticker in results["ticker"].unique():
        sub_t = results[results["ticker"] == ticker].dropna(subset=["correct_5d"])
        if len(sub_t) < 3:
            continue
        sig = _significance(sub_t["ret_5d"])
        ticker_data.append({
            "Ticker":      ticker,
            "Signals":     len(sub_t),
            "Accuracy":    sub_t["correct_5d"].mean() * 100,
            "Avg Return":  sub_t["ret_5d"].mean(),
            "p-Value":     sig["p"] if sig["p"] else np.nan,
            "Significant": "✅" if sig["sig"] else "❌",
        })

    if ticker_data:
        tk_df = pd.DataFrame(ticker_data).sort_values("Accuracy", ascending=False)
        styler3 = tk_df.style
        styler3 = styler3.map(c_acc, subset=["Accuracy"])
        styler3 = styler3.map(c_ret, subset=["Avg Return"])
        styler3 = styler3.format({
            "Accuracy":   "{:.1f}%",
            "Avg Return": "{:+.2f}%",
            "p-Value":    "{:.4f}",
        }, na_rep="—")
        st.dataframe(styler3, use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        # Accuracy by confidence × horizon
        st.markdown('<div class="section-header">📊 Accuracy: HIGH vs MEDIUM vs LOW</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        _style_fig(fig, ax)
        palette = {"HIGH": BLUE, "MEDIUM": AMBER, "LOW": MUTED}
        x = np.arange(len(HORIZONS))
        w = 0.25
        for j, conf in enumerate(["HIGH", "MEDIUM", "LOW"]):
            accs = []
            for h in HORIZONS:
                sub = results[results["confidence"] == conf].dropna(subset=[f"correct_{h}d"])
                accs.append(sub[f"correct_{h}d"].mean() * 100 if len(sub) >= 3 else np.nan)
            bars = ax.bar(x + j * w, accs, w, label=conf,
                          color=palette[conf], edgecolor=BORDER, linewidth=0.4)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{h}d" for h in HORIZONS], color=MUTED, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("Accuracy by Confidence Level", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with ch2:
        # Return distribution
        st.markdown('<div class="section-header">📈 Return Distribution (5-day hold)</div>', unsafe_allow_html=True)
        rets = results["ret_5d"].dropna()
        if not rets.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig, ax)
            ax.hist(rets, bins=20,
                    color=[GREEN if rets.mean() > 0 else RED][0],
                    edgecolor=BORDER, linewidth=0.4, alpha=0.8)
            ax.axvline(0,          color=MUTED, linewidth=1,   linestyle="--")
            ax.axvline(rets.mean(), color=CYAN,  linewidth=1.5, linestyle="-",
                       label=f"Mean: {rets.mean():+.2f}%")
            ax.set_xlabel("Return %", color=MUTED, fontsize=8)
            ax.set_ylabel("Count", color=MUTED, fontsize=8)
            ax.set_title("Distribution of 5-Day Returns", color=CYAN, fontsize=9)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Accuracy over time ─────────────────────────────────────────────────
    ev_5d = results.dropna(subset=["correct_5d"])
    if len(ev_5d) >= 5:
        st.markdown('<div class="section-header">📅 Accuracy Over Time (5-day hold, rolling)</div>', unsafe_allow_html=True)
        ev_5d = ev_5d.copy()
        ev_5d["date_dt"] = pd.to_datetime(ev_5d["date"])
        ev_5d = ev_5d.sort_values("date_dt")
        ev_5d["rolling_acc"] = ev_5d["correct_5d"].rolling(10, min_periods=3).mean() * 100

        fig, ax = plt.subplots(figsize=(12, 3))
        _style_fig(fig, ax)
        ax.plot(ev_5d["date_dt"], ev_5d["rolling_acc"],
                color=BLUE, linewidth=1.5, label="10-signal rolling accuracy")
        ax.fill_between(ev_5d["date_dt"], ev_5d["rolling_acc"], 50,
                        where=ev_5d["rolling_acc"] >= 50, alpha=0.1, color=GREEN)
        ax.fill_between(ev_5d["date_dt"], ev_5d["rolling_acc"], 50,
                        where=ev_5d["rolling_acc"] < 50, alpha=0.1, color=RED)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% baseline")
        ax.axhline(55, color=AMBER, linewidth=0.8, linestyle=":", label="55% target")
        ax.set_ylim(20, 85)
        ax.set_ylabel("Rolling Accuracy %", color=MUTED, fontsize=8)
        ax.set_title("Rolling 10-Signal Accuracy Over Time", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=20, fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Full signal table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>', unsafe_allow_html=True)
    display_cols = ["date", "ticker", "action", "confidence", "source",
                    "entry_price", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
                    "correct_5d", "article"]
    display = results[[c for c in display_cols if c in results.columns]].copy()

    def c_act(v):  return f"color:{GREEN}" if v=="BUY" else f"color:{RED}"
    def c_cor(v):
        if pd.isna(v): return f"color:{MUTED}"
        return f"color:{GREEN}" if v==1 else f"color:{RED}"
    def c_ret(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v > 0 else f"color:{RED}"

    sty = display.style
    if "action"     in display.columns: sty = sty.map(c_act, subset=["action"])
    if "correct_5d" in display.columns: sty = sty.map(c_cor, subset=["correct_5d"])
    for r in ["ret_1d", "ret_3d", "ret_5d", "ret_10d"]:
        if r in display.columns: sty = sty.map(c_ret, subset=[r])

    fmt = {"entry_price": "${:.2f}"}
    for r in ["ret_1d", "ret_3d", "ret_5d", "ret_10d"]:
        if r in display.columns: fmt[r] = lambda x: f"{x:+.2f}%" if pd.notna(x) else "pending"
    sty = sty.format(fmt, na_rep="pending")
    st.dataframe(sty, use_container_width=True, hide_index=True)

    # ── Honest interpretation ──────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 How to Interpret These Results</div>', unsafe_allow_html=True)

    n_eval = len(results.dropna(subset=["correct_5d"]))
    if n_eval < 20:
        st.error(f"""
        **{n_eval} evaluated signals is not enough for reliable conclusions.**

        You need at least 30–50 evaluated signals before these numbers mean anything.
        Here's what to do:
        - Run **Live Feed** for 30+ minutes every day for 2–3 weeks
        - Run **Ticker Signals** on 5–10 tickers per day
        - Check back here weekly

        This is not a limitation of the app — it's statistics. Even the best quant funds
        need hundreds of signals before they trust a result.
        """)
    else:
        overall_acc_5d = results.dropna(subset=["correct_5d"])["correct_5d"].mean() * 100
        hc = results[results["confidence"] == "HIGH"].dropna(subset=["correct_5d"])
        hc_acc = hc["correct_5d"].mean() * 100 if len(hc) >= 3 else None
        sig = _significance(results.dropna(subset=["ret_5d"])["ret_5d"])

        if overall_acc_5d >= 60 and sig["sig"]:
            st.success(f"🟢 **{overall_acc_5d:.1f}% overall accuracy on {n_eval} signals (p={sig['p']:.4f})** — this is a statistically significant edge. "
                       f"You can begin trading these signals with real capital, starting with small position sizes.")
        elif overall_acc_5d >= 55:
            st.info(f"🟡 **{overall_acc_5d:.1f}% accuracy** — modest edge. "
                    f"{'Statistically significant — keep scaling up signal volume.' if sig['sig'] else 'Not yet statistically significant (p={:.4f}). Need more signals.'.format(sig['p'] or 0)}")
        else:
            st.warning(f"🔴 **{overall_acc_5d:.1f}% accuracy** — no meaningful edge detected yet on news signals. "
                       f"This could mean: not enough signals (n={n_eval}), wrong time horizons, or the news signals genuinely don't predict direction reliably. "
                       f"Try filtering to HIGH confidence only and checking the 1-day horizon.")

        if hc_acc is not None:
            if hc_acc > overall_acc_5d + 5:
                st.success(f"✅ HIGH confidence signals ({hc_acc:.1f}%) beat overall ({overall_acc_5d:.1f}%) by "
                           f"{hc_acc - overall_acc_5d:.1f}pp — the confidence scoring is working correctly.")
            else:
                st.warning(f"⚠️ HIGH confidence signals ({hc_acc:.1f}%) are not significantly better than overall ({overall_acc_5d:.1f}%). "
                           f"The confidence calibration may need improvement.")

    csv = results.to_csv(index=False)
    st.download_button("⬇️ Download evaluation data as CSV",
                       data=csv,
                       file_name=f"news_signal_eval_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

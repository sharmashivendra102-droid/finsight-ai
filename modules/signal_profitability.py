"""
Signal Profitability Engine
============================
Turns raw signal history into a self-improving intelligence layer.

For every evaluated signal it computes:
  - Raw return
  - Volatility-adjusted return (signal Sharpe)
  - Market regime at signal time (bull/bear/sideways via SPY 50MA)
  - Signal category (news-LLM, RSI, MA, momentum, ML-weekly)
  - Hold duration vs planned duration

Then aggregates to answer:
  - Which signal TYPE makes the most money?
  - Which signal type works best in each REGIME?
  - Which TICKERS respond best to which signal types?
  - Which CONFIDENCE LEVEL actually predicts returns?
  - Auto-ranking: kill bad strategies, weight good ones

The output directly feeds signal weighting so future signals
from better-performing sources get displayed more prominently.
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

BLUE   = "#38bdf8"; CYAN   = "#7dd3fc"; AMBER  = "#fbbf24"
RED    = "#f87171"; GREEN  = "#4ade80"; CARD   = "#0d1b2a"
BORDER = "#1e3a5f"; TEXT   = "#c9d8e8"; MUTED  = "#6b8fad"
PURPLE = "#a78bfa"; ORANGE = "#fb923c"


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)


# ── Regime detection ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _get_spy_regime_series(start_date: str) -> pd.Series:
    """
    Returns daily regime series: 'bull', 'bear', 'sideways'
    Bull  = SPY price > 50-day MA
    Bear  = SPY price < 50-day MA by more than 2%
    Sideways = within 2% of 50-day MA
    """
    import yfinance as yf
    try:
        hist = yf.Ticker("SPY").history(start=start_date, auto_adjust=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        closes = hist["Close"].dropna()
        ma50   = closes.rolling(50).mean()
        diff   = (closes - ma50) / ma50 * 100

        regime = pd.Series("sideways", index=closes.index)
        regime[diff > 2]  = "bull"
        regime[diff < -2] = "bear"
        return regime
    except Exception:
        return pd.Series(dtype=str)


def _get_regime_on_date(date_str: str, regime_series: pd.Series) -> str:
    try:
        dt = pd.Timestamp(date_str).normalize()
        if regime_series.empty:
            return "unknown"
        # Find closest date
        available = regime_series.index[regime_series.index <= dt]
        if available.empty:
            return "unknown"
        return regime_series.loc[available[-1]]
    except Exception:
        return "unknown"


# ── Signal category mapping ───────────────────────────────────────────────────

def _categorise_source(source: str, time_horizon: str) -> str:
    """Map raw source string to a clean signal category."""
    s = str(source).lower()
    h = str(time_horizon).upper()

    if s in ("live_intelligence", "ticker_signals", "market_briefing"):
        return "News LLM"
    elif s == "strategy_signals":
        if "RSI" in h or "REVERSION" in h:
            return "RSI Mean Reversion"
        elif "MA" in h or "CROSSOVER" in h or "MOVING" in h:
            return "MA Crossover"
        elif "MOMENTUM" in h:
            return "Momentum"
        elif "ML" in h or "WEEKLY" in h or "RF" in h:
            return "ML Weekly"
        else:
            return "Strategy Signal"
    elif s == "auto_evaluator":
        return "Historical Simulation"
    else:
        return "Other"


# ── Load and enrich evaluated signals ────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _load_evaluated_signals() -> pd.DataFrame:
    """Load all signals that have been evaluated (have return_pct)."""
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        df   = pd.read_sql_query("""
            SELECT * FROM signals
            WHERE action IN ('BUY','SHORT')
            ORDER BY timestamp
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _enrich_with_outcomes(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, fetch entry and exit prices and compute outcomes.
    Uses eval_core for horizon matching and supersession.
    """
    if df_raw.empty:
        return pd.DataFrame()

    from modules.eval_core import evaluate_signals
    import yfinance as yf

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_price(ticker: str, date_str: str) -> float | None:
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
    def _fetch_ohlc_spe(ticker: str, entry_date: str, exit_date: str, lookback_days: int = 60):
        try:
            start = (pd.to_datetime(entry_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end   = (pd.to_datetime(exit_date)  + timedelta(days=3)).strftime("%Y-%m-%d")
            hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            return hist if not hist.empty else None
        except Exception:
            return None

    ready_df, _, _ = evaluate_signals(
        df                       = df_raw,
        fetch_price_fn           = _fetch_price,
        fetch_ohlc_fn            = _fetch_ohlc_spe,   # enables MFE/MAE in profitability analysis
        all_signals_for_timeline = df_raw,
        progress_callback        = None,
    )
    return ready_df


def _compute_volatility_adjusted_return(row: pd.Series,
                                         volatility_map: dict) -> float | None:
    """
    Signal Sharpe = return / ticker_volatility
    Uses annualised daily vol fetched separately.
    """
    ret = row.get("return_pct")
    if ret is None or np.isnan(ret):
        return None
    ticker_vol = volatility_map.get(row["ticker"], None)
    if ticker_vol is None or ticker_vol == 0:
        return None
    return round(ret / ticker_vol, 4)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_ticker_volatility(tickers: tuple) -> dict:
    """Fetch annualised daily volatility for each ticker."""
    import yfinance as yf
    result = {}
    for ticker in tickers:
        try:
            hist   = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
            closes = hist["Close"].dropna()
            if len(closes) < 10:
                continue
            vol = float(closes.pct_change().dropna().std() * (252 ** 0.5) * 100)
            result[ticker] = vol
        except Exception:
            continue
    return result


# ── Core analytics ────────────────────────────────────────────────────────────

def _build_profitability_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by signal_category × regime and compute:
    - count, accuracy, avg_return, vol_adj_return, avg_hold_days
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for cat in df["signal_category"].unique():
        for regime in ["bull", "bear", "sideways", "unknown"]:
            sub = df[(df["signal_category"] == cat) &
                     (df["regime"] == regime)]
            if len(sub) < 3:
                continue
            avg_ret = sub["return_pct"].mean()
            va_ret  = sub["vol_adj_return"].dropna().mean() if "vol_adj_return" in sub.columns else np.nan
            rows.append({
                "Signal Type":      cat,
                "Regime":           regime.capitalize(),
                "Signals":          len(sub),
                "Accuracy %":       round(sub["correct"].mean() * 100, 1),
                "Avg Return %":     round(avg_ret, 2),
                "Vol-Adj Return":   round(va_ret, 3) if not np.isnan(va_ret) else np.nan,
                "Best Ticker":      sub.groupby("ticker")["return_pct"].mean().idxmax()
                                    if len(sub) > 1 else sub["ticker"].iloc[0],
                "Score":            round(avg_ret * sub["correct"].mean(), 3),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return result


def _rank_strategies(prof_df: pd.DataFrame) -> pd.DataFrame:
    """Auto-rank signal types by composite score across all regimes."""
    if prof_df.empty:
        return pd.DataFrame()

    ranked = prof_df.groupby("Signal Type").agg(
        Total_Signals    = ("Signals", "sum"),
        Avg_Accuracy     = ("Accuracy %", "mean"),
        Avg_Return       = ("Avg Return %", "mean"),
        Avg_Vol_Adj      = ("Vol-Adj Return", "mean"),
        Composite_Score  = ("Score", "mean"),
    ).reset_index().sort_values("Composite_Score", ascending=False)

    ranked["Rank"]       = range(1, len(ranked) + 1)
    ranked["Verdict"]    = ranked["Composite_Score"].apply(
        lambda x: "✅ Keep" if x > 0.5 else ("⚠️ Watch" if x > 0 else "❌ Kill")
    )
    return ranked


# ── Visualisations ────────────────────────────────────────────────────────────

def _plot_regime_heatmap(prof_df: pd.DataFrame):
    """Heatmap: signal type × regime → avg return."""
    if prof_df.empty:
        return None

    pivot = prof_df.pivot_table(
        values="Avg Return %",
        index="Signal Type",
        columns="Regime",
        aggfunc="mean"
    ).fillna(0)

    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2),
                                    max(3, len(pivot)*0.9)))
    _style_fig(fig, ax)

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, color=TEXT, fontsize=9)
    ax.set_yticklabels(pivot.index,   color=TEXT, fontsize=9)
    ax.set_title("Avg Return % — Signal Type × Market Regime", color=CYAN, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color="white" if abs(val) > 3 else TEXT, fontsize=8, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    return fig


def _plot_return_by_confidence(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _style_fig(fig, list(axes))

    # Accuracy by confidence
    ax1 = axes[0]
    confs = ["HIGH", "MEDIUM", "LOW"]
    accs  = [df[df["confidence"]==c]["correct"].mean()*100
             if len(df[df["confidence"]==c])>=3 else np.nan for c in confs]
    colors= [GREEN if (a or 0)>=60 else (AMBER if (a or 0)>=50 else RED) for a in accs]
    bars  = ax1.bar(confs, [a or 0 for a in accs], color=colors, edgecolor=BORDER, linewidth=0.4)
    ax1.axhline(50, color=MUTED, linewidth=1, linestyle="--")
    ax1.set_ylim(0, 100)
    ax1.set_title("Accuracy by Confidence", color=CYAN, fontsize=9)
    ax1.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
    for bar, a in zip(bars, accs):
        if a: ax1.text(bar.get_x()+bar.get_width()/2, (a or 0)+1,
                       f"{a:.1f}%", ha="center", color=TEXT, fontsize=8)

    # Avg return by confidence
    ax2 = axes[1]
    rets = [df[df["confidence"]==c]["return_pct"].mean()
            if len(df[df["confidence"]==c])>=3 else 0 for c in confs]
    rc   = [GREEN if r>=0 else RED for r in rets]
    ax2.bar(confs, rets, color=rc, edgecolor=BORDER, linewidth=0.4)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_title("Avg Return % by Confidence", color=CYAN, fontsize=9)
    ax2.set_ylabel("Return %", color=MUTED, fontsize=8)

    fig.tight_layout()
    return fig


def _plot_strategy_ranking(ranked: pd.DataFrame):
    if ranked.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, len(ranked)*0.7)))
    _style_fig(fig, ax)

    colors = [GREEN if s > 0.5 else (AMBER if s > 0 else RED)
              for s in ranked["Composite_Score"]]
    bars = ax.barh(ranked["Signal Type"][::-1],
                   ranked["Composite_Score"][::-1],
                   color=colors[::-1], edgecolor=BORDER, linewidth=0.4)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Strategy Composite Score (return × accuracy)", color=CYAN, fontsize=10)
    ax.set_xlabel("Composite Score", color=MUTED, fontsize=8)

    for bar, score in zip(bars, ranked["Composite_Score"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:+.2f}", va="center", color=TEXT, fontsize=8)

    fig.tight_layout()
    return fig


def _plot_equity_curves_by_type(df: pd.DataFrame):
    """Cumulative return curves split by signal category."""
    if df.empty or "signal_category" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    _style_fig(fig, ax)

    palette = [BLUE, AMBER, GREEN, RED, PURPLE, ORANGE, CYAN]
    df_sorted = df.sort_values("date")

    for i, cat in enumerate(df_sorted["signal_category"].unique()):
        sub = df_sorted[df_sorted["signal_category"] == cat].copy()
        if len(sub) < 3:
            continue
        sub["cum_ret"] = (1 + sub["return_pct"] / 100).cumprod()
        ax.plot(pd.to_datetime(sub["date"]),
                (sub["cum_ret"] - 1) * 100,
                color=palette[i % len(palette)],
                linewidth=1.5, label=cat, alpha=0.9)

    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_title("Cumulative Return by Signal Type", color=CYAN, fontsize=10)
    ax.set_ylabel("Cumulative Return %", color=MUTED, fontsize=8)
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
              loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=20, fontsize=7)
    fig.tight_layout()
    return fig


# ── Main entry point ──────────────────────────────────────────────────────────

def run_signal_profitability():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Signal Profitability Engine — self-improving trading intelligence.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Analyses every evaluated signal across regime (bull/bear/sideways), signal type, confidence,
        and ticker. Computes volatility-adjusted returns and auto-ranks which strategies actually make money.
        Signals from better-performing categories are shown with higher weighting.
        The more signal history you build, the more accurate this becomes.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not st.button("🧠 Compute Signal Profitability", key="spe_run"):
        if st.session_state.get("spe_data") is not None:
            _display(st.session_state["spe_data"])
        else:
            st.info("Click **Compute Signal Profitability** to analyse your signal history. "
                    "Requires at least 10 evaluated signals across different types.")
        return

    # ── Load raw signals ───────────────────────────────────────────────────
    df_raw = _load_evaluated_signals()
    if df_raw.empty:
        st.warning("No signals found. Run Live News Feed and other signal tabs first.")
        return

    # ── Evaluate outcomes ──────────────────────────────────────────────────
    with st.spinner("📊 Evaluating signal outcomes…"):
        df_eval = _enrich_with_outcomes(df_raw)

    if df_eval.empty:
        st.warning("No signals have reached their evaluation window yet. Come back once signals have aged past their time horizon.")
        return

    n = len(df_eval)
    if n < 10:
        st.warning(f"Only {n} evaluated signals. Need at least 10 for meaningful profitability analysis.")

    # ── Fetch regime series ────────────────────────────────────────────────
    earliest = df_eval["date"].min() if "date" in df_eval.columns else "2024-01-01"
    with st.spinner("📡 Fetching market regime data…"):
        regime_series = _get_spy_regime_series(
            (pd.Timestamp(earliest) - timedelta(days=60)).strftime("%Y-%m-%d")
        )

    # ── Fetch volatility ───────────────────────────────────────────────────
    tickers = tuple(df_eval["ticker"].unique().tolist())
    with st.spinner("📡 Fetching ticker volatility…"):
        vol_map = _get_ticker_volatility(tickers)

    # ── Enrich ────────────────────────────────────────────────────────────
    df_eval["regime"]          = df_eval["date"].apply(
        lambda d: _get_regime_on_date(d, regime_series))
    df_eval["signal_category"] = df_eval.apply(
        lambda r: _categorise_source(r.get("source",""), r.get("time_horizon","")), axis=1)
    df_eval["vol_adj_return"]  = df_eval.apply(
        lambda r: _compute_volatility_adjusted_return(r, vol_map), axis=1)

    # ── Build analytics ────────────────────────────────────────────────────
    prof_df  = _build_profitability_table(df_eval)
    ranked   = _rank_strategies(prof_df)

    data = {
        "df_eval": df_eval,
        "prof_df": prof_df,
        "ranked":  ranked,
        "vol_map": vol_map,
        "n":       n,
    }
    st.session_state["spe_data"] = data
    _display(data)


def _display(data: dict):
    df_eval = data["df_eval"]
    prof_df = data["prof_df"]
    ranked  = data["ranked"]
    n       = data["n"]

    st.caption(f"Based on {n} evaluated signals · {df_eval['signal_category'].nunique()} signal types · "
               f"{df_eval['ticker'].nunique()} tickers · "
               f"Regimes: {df_eval['regime'].value_counts().to_dict()}")

    # ── Strategy ranking ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Strategy Auto-Ranking</div>', unsafe_allow_html=True)
    st.caption("Composite Score = Avg Return × Accuracy. Positive = worth keeping. Negative = kill it.")

    if not ranked.empty:
        def c_verdict(v): return f"color:{GREEN}" if "Keep" in str(v) else (f"color:{AMBER}" if "Watch" in str(v) else f"color:{RED}")
        def c_ret(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v>0 else f"color:{RED}"
        def c_acc(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v>=60 else (f"color:{AMBER}" if v>=50 else f"color:{RED}")
        def c_score(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v>0.5 else (f"color:{AMBER}" if v>0 else f"color:{RED}")

        disp = ranked.rename(columns={
            "Signal Type": "Signal Type", "Total_Signals": "Signals",
            "Avg_Accuracy": "Accuracy %", "Avg_Return": "Avg Return %",
            "Avg_Vol_Adj": "Vol-Adj Return", "Composite_Score": "Score",
            "Rank": "Rank", "Verdict": "Verdict"
        })
        sty = disp.style
        if "Verdict"     in disp.columns: sty = sty.map(c_verdict, subset=["Verdict"])
        if "Avg Return %" in disp.columns: sty = sty.map(c_ret, subset=["Avg Return %"])
        if "Accuracy %"  in disp.columns: sty = sty.map(c_acc, subset=["Accuracy %"])
        if "Score"       in disp.columns: sty = sty.map(c_score, subset=["Score"])
        sty = sty.format({
            "Accuracy %":    "{:.1f}%",
            "Avg Return %":  "{:+.2f}%",
            "Vol-Adj Return":"{:+.3f}",
            "Score":         "{:+.3f}",
        }, na_rep="—")
        st.dataframe(sty, use_container_width=True, hide_index=True)

        fig_rank = _plot_strategy_ranking(ranked)
        if fig_rank:
            st.pyplot(fig_rank); plt.close(fig_rank)

    # ── Best signal type in plain English ─────────────────────────────────
    if not ranked.empty:
        best  = ranked.iloc[0]
        worst = ranked.iloc[-1]
        st.markdown('<div class="section-header">💡 Plain-English Intelligence</div>', unsafe_allow_html=True)

        insights = []

        if best["Composite_Score"] > 0.5:
            insights.append(f"🟢 **{best['Signal Type']}** is your best performing signal type with a composite score of {best['Composite_Score']:+.2f}. Average return {best['Avg_Return']:+.1f}% at {best['Avg_Accuracy']:.0f}% accuracy.")
        if worst["Composite_Score"] < 0:
            insights.append(f"🔴 **{worst['Signal Type']}** is actively losing — composite score {worst['Composite_Score']:+.2f}. Consider disabling or inverting this signal type.")

        # Regime insight
        if not prof_df.empty:
            bull_rows = prof_df[prof_df["Regime"] == "Bull"]
            bear_rows = prof_df[prof_df["Regime"] == "Bear"]
            if not bull_rows.empty:
                best_bull = bull_rows.loc[bull_rows["Score"].idxmax()]
                insights.append(f"📈 **In bull markets**, {best_bull['Signal Type']} performs best: {best_bull['Avg Return %']:+.1f}% avg return.")
            if not bear_rows.empty:
                best_bear = bear_rows.loc[bear_rows["Score"].idxmax()]
                insights.append(f"📉 **In bear markets**, {best_bear['Signal Type']} performs best: {best_bear['Avg Return %']:+.1f}% avg return.")

        # Confidence calibration
        hc = df_eval[df_eval["confidence"]=="HIGH"]
        lc = df_eval[df_eval["confidence"]=="LOW"]
        if len(hc)>=3 and len(lc)>=3:
            hc_ret = hc["return_pct"].mean()
            lc_ret = lc["return_pct"].mean()
            if hc_ret > lc_ret + 1:
                insights.append(f"✅ **Confidence scoring is working**: HIGH confidence signals return {hc_ret:+.1f}% vs {lc_ret:+.1f}% for LOW confidence — a {hc_ret-lc_ret:.1f}pp edge.")
            else:
                insights.append(f"⚠️ **Confidence scoring may need re-calibration**: HIGH confidence ({hc_ret:+.1f}%) is not significantly better than LOW confidence ({lc_ret:+.1f}%).")

        for insight in insights:
            st.markdown(insight)

    # ── Regime heatmap ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌡️ Return Heatmap — Signal Type × Market Regime</div>', unsafe_allow_html=True)
    st.caption("Which signals work in bull markets? Which work in bear markets? This is how you adapt.")
    fig_hm = _plot_regime_heatmap(prof_df)
    if fig_hm:
        st.pyplot(fig_hm); plt.close(fig_hm)

    # ── Confidence breakdown ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Return & Accuracy by Confidence Level</div>', unsafe_allow_html=True)
    fig_conf = _plot_return_by_confidence(df_eval)
    st.pyplot(fig_conf); plt.close(fig_conf)

    # ── Equity curves ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Cumulative Return by Signal Type</div>', unsafe_allow_html=True)
    fig_eq = _plot_equity_curves_by_type(df_eval)
    if fig_eq:
        st.pyplot(fig_eq); plt.close(fig_eq)

    # ── Per-regime detail ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Full Profitability Breakdown</div>', unsafe_allow_html=True)
    if not prof_df.empty:
        def c_ret2(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v>0 else f"color:{RED}"
        sty2 = prof_df.style.map(c_ret2, subset=["Avg Return %","Score"])
        sty2 = sty2.format({
            "Accuracy %":   "{:.1f}%",
            "Avg Return %": "{:+.2f}%",
            "Vol-Adj Return": "{:+.3f}",
            "Score":        "{:+.3f}",
        }, na_rep="—")
        st.dataframe(sty2, use_container_width=True, hide_index=True)

    # ── Volatility-adjusted returns ────────────────────────────────────────
    if "vol_adj_return" in df_eval.columns and df_eval["vol_adj_return"].notna().any():
        st.markdown('<div class="section-header">⚡ Volatility-Adjusted Returns (Signal Sharpe)</div>', unsafe_allow_html=True)
        st.caption("Return divided by ticker's annualised volatility. Comparable across assets with different risk levels.")
        va = df_eval.groupby("signal_category")["vol_adj_return"].agg(["mean","count"]).reset_index()
        va.columns = ["Signal Type", "Avg Vol-Adj Return", "Signals"]
        va = va[va["Signals"] >= 3].sort_values("Avg Vol-Adj Return", ascending=False)
        if not va.empty:
            fig_va, ax_va = plt.subplots(figsize=(8, 3))
            _style_fig(fig_va, ax_va)
            vc = [GREEN if v>0 else RED for v in va["Avg Vol-Adj Return"]]
            ax_va.barh(va["Signal Type"], va["Avg Vol-Adj Return"],
                       color=vc, edgecolor=BORDER, linewidth=0.4)
            ax_va.axvline(0, color=MUTED, linewidth=0.8)
            ax_va.set_title("Volatility-Adjusted Return by Signal Type", color=CYAN, fontsize=9)
            ax_va.set_xlabel("Return / Volatility", color=MUTED, fontsize=8)
            fig_va.tight_layout()
            st.pyplot(fig_va); plt.close(fig_va)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:1rem;padding:0.8rem;border:1px solid #1e3a5f;border-radius:8px;
                color:#6b8fad;font-size:0.78rem;">
        ⚠️ Past signal performance does not guarantee future returns. Market regimes change.
        All figures are based on historical simulated outcomes. Not financial advice.
    </div>
    """, unsafe_allow_html=True)

    csv = df_eval.to_csv(index=False)
    st.download_button("⬇️ Download enriched signal data as CSV", data=csv,
                       file_name=f"signal_profitability_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

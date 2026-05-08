"""
Signal Profitability Engine
============================
FIXED: reads from Supabase instead of SQLite so data persists across redeploys.
FIXED: uses directional accuracy (MFE-based) for Accuracy % and Composite Score,
       not exit-price accuracy. Directional = price moved in signal's favour at
       any point during the hold window (even if it reversed by formal exit).
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

BLUE   = "#38bdf8"; CYAN   = "#7dd3fc"; AMBER  = "#fbbf24"
RED    = "#f87171"; GREEN  = "#4ade80"; CARD   = "#0d1b2a"
BORDER = "#1e3a5f"; TEXT   = "#c9d8e8"; MUTED  = "#6b8fad"
PURPLE = "#a78bfa"; ORANGE = "#fb923c"

# Default MFE threshold — price must move at least this far in the signal's
# direction (intraday, using High/Low) to count as "directionally correct".
DEFAULT_MFE_THRESHOLD = 0.5  # %


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
        available = regime_series.index[regime_series.index <= dt]
        if available.empty:
            return "unknown"
        return regime_series.loc[available[-1]]
    except Exception:
        return "unknown"


# ── Signal category mapping ───────────────────────────────────────────────────

def _categorise_source(source: str, time_horizon: str) -> str:
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


# ── Load from Supabase ────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _load_evaluated_signals() -> pd.DataFrame:
    """Load all BUY/SHORT signals from Supabase."""
    try:
        from modules.signal_history import get_signals_df
        df = get_signals_df(days_back=3650, action_filter=["BUY", "SHORT"])
        if df.empty:
            return pd.DataFrame()
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not load signals from Supabase: {e}")
        return pd.DataFrame()


def _enrich_with_outcomes(df_raw: pd.DataFrame) -> pd.DataFrame:
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
        fetch_ohlc_fn            = _fetch_ohlc_spe,
        all_signals_for_timeline = df_raw,
        progress_callback        = None,
    )
    return ready_df


def _compute_volatility_adjusted_return(row: pd.Series, volatility_map: dict) -> float | None:
    ret = row.get("return_pct")
    if ret is None or np.isnan(ret):
        return None
    ticker_vol = volatility_map.get(row["ticker"], None)
    if ticker_vol is None or ticker_vol == 0:
        return None
    return round(ret / ticker_vol, 4)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_ticker_volatility(tickers: tuple) -> dict:
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

def _resolve_directional_correct(df: pd.DataFrame, mfe_threshold: float) -> pd.Series:
    """
    Return a boolean Series representing directional correctness.

    If MFE data is available: directional = (mfe >= mfe_threshold).
    If not (OHLC unavailable for some rows): fall back to exit-price correct.

    This means the composite score and accuracy % always reflect the most
    generous but still real measure of whether the signal was right.
    """
    if "mfe" in df.columns and df["mfe"].notna().any():
        # Where MFE is available use it; where it's null fall back to correct
        dir_correct = df["mfe"].apply(
            lambda x: (x >= mfe_threshold) if pd.notna(x) else np.nan
        )
        # Fill NaN positions with the exit-accuracy fallback
        if "correct" in df.columns:
            dir_correct = dir_correct.fillna(df["correct"].astype(float))
        return dir_correct.astype(float)
    elif "correct" in df.columns:
        return df["correct"].astype(float)
    else:
        return pd.Series(np.nan, index=df.index)


def _build_profitability_table(df: pd.DataFrame, mfe_threshold: float) -> pd.DataFrame:
    """
    Build per-(signal_type × regime) profitability table.

    Accuracy % and Score are computed from DIRECTIONAL accuracy (MFE-based),
    not from exit-price accuracy.  This correctly captures whether the signal
    thesis was validated by the market, independent of exact exit timing.
    """
    if df.empty:
        return pd.DataFrame()

    # Pre-compute directional correct column into df so group-by can use it
    df = df.copy()
    df["_dir_correct"] = _resolve_directional_correct(df, mfe_threshold)

    rows = []
    for cat in df["signal_category"].unique():
        for regime in ["bull", "bear", "sideways", "unknown"]:
            sub = df[(df["signal_category"] == cat) & (df["regime"] == regime)]
            if len(sub) < 3:
                continue

            avg_ret      = sub["return_pct"].mean()
            dir_acc      = sub["_dir_correct"].mean()          # directional accuracy 0–1
            exit_acc     = sub["correct"].mean() if "correct" in sub.columns else np.nan
            va_ret       = sub["vol_adj_return"].dropna().mean() if "vol_adj_return" in sub.columns else np.nan

            rows.append({
                "Signal Type":       cat,
                "Regime":            regime.capitalize(),
                "Signals":           len(sub),
                # PRIMARY: directional accuracy
                "Accuracy %":        round(dir_acc * 100, 1),
                "Exit Accuracy %":   round(exit_acc * 100, 1) if not np.isnan(exit_acc) else np.nan,
                "Avg Return %":      round(avg_ret, 2),
                "Vol-Adj Return":    round(va_ret, 3) if not np.isnan(va_ret) else np.nan,
                "Best Ticker":       sub.groupby("ticker")["return_pct"].mean().idxmax()
                                     if len(sub) > 1 else sub["ticker"].iloc[0],
                # Composite score = directional accuracy × avg return
                "Score":             round(avg_ret * dir_acc, 3),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False)


def _rank_strategies(prof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(type × regime) table up to per-type ranking.
    All accuracy metrics are directional.
    """
    if prof_df.empty:
        return pd.DataFrame()

    ranked = prof_df.groupby("Signal Type").agg(
        Total_Signals       = ("Signals",         "sum"),
        Dir_Accuracy        = ("Accuracy %",       "mean"),   # directional
        Exit_Accuracy       = ("Exit Accuracy %",  "mean"),   # for reference
        Avg_Return          = ("Avg Return %",     "mean"),
        Avg_Vol_Adj         = ("Vol-Adj Return",   "mean"),
        Composite_Score     = ("Score",            "mean"),
    ).reset_index().sort_values("Composite_Score", ascending=False)

    ranked["Rank"] = range(1, len(ranked) + 1)
    ranked["Verdict"] = ranked["Composite_Score"].apply(
        lambda x: "✅ Keep" if x > 0.5 else ("⚠️ Watch" if x > 0 else "❌ Kill")
    )
    return ranked


# ── Visualisations ────────────────────────────────────────────────────────────

def _plot_regime_heatmap(prof_df: pd.DataFrame):
    """Heatmap uses Avg Return % — regime rows × signal-type cols."""
    if prof_df.empty:
        return None
    pivot = prof_df.pivot_table(
        values="Avg Return %", index="Signal Type", columns="Regime", aggfunc="mean"
    ).fillna(0)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), max(3, len(pivot)*0.9)))
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
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        color="white" if abs(val) > 3 else TEXT, fontsize=8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    return fig


def _plot_accuracy_comparison(df: pd.DataFrame, mfe_threshold: float):
    """
    Side-by-side: directional accuracy vs exit accuracy by confidence level.
    Makes it immediately clear why directional is the right primary metric.
    """
    confs   = ["HIGH", "MEDIUM", "LOW"]
    dir_acc = []
    exit_acc= []

    df = df.copy()
    df["_dir_correct"] = _resolve_directional_correct(df, mfe_threshold)

    for conf in confs:
        sub = df[df["confidence"] == conf]
        if len(sub) >= 3:
            dir_acc.append(sub["_dir_correct"].mean() * 100)
            exit_acc.append(sub["correct"].mean() * 100 if "correct" in sub.columns else np.nan)
        else:
            dir_acc.append(np.nan)
            exit_acc.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _style_fig(fig, list(axes))

    ax1 = axes[0]
    x   = np.arange(len(confs))
    w   = 0.35
    dir_vals  = [v if not np.isnan(v) else 0 for v in dir_acc]
    exit_vals = [v if not np.isnan(v) else 0 for v in exit_acc]
    ax1.bar(x - w/2, dir_vals,  w, color=BLUE,  label="Directional (MFE)", edgecolor=BORDER, linewidth=0.4, alpha=0.9)
    ax1.bar(x + w/2, exit_vals, w, color=MUTED, label="Exit-price",        edgecolor=BORDER, linewidth=0.4, alpha=0.7)
    ax1.axhline(50, color=AMBER, linewidth=1, linestyle="--")
    ax1.set_xticks(x); ax1.set_xticklabels(confs, color=TEXT, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_title("Directional vs Exit Accuracy by Confidence", color=CYAN, fontsize=9)
    ax1.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
    ax1.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    for i, (dv, ev) in enumerate(zip(dir_vals, exit_vals)):
        if dv: ax1.text(i - w/2, dv + 1, f"{dv:.0f}%", ha="center", color=TEXT, fontsize=8)
        if ev: ax1.text(i + w/2, ev + 1, f"{ev:.0f}%", ha="center", color=TEXT, fontsize=8)

    ax2 = axes[1]
    rets = [df[df["confidence"]==c]["return_pct"].mean()
            if len(df[df["confidence"]==c]) >= 3 else 0 for c in confs]
    rc   = [GREEN if r >= 0 else RED for r in rets]
    ax2.bar(confs, rets, color=rc, edgecolor=BORDER, linewidth=0.4)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_title("Avg Return % by Confidence", color=CYAN, fontsize=9)
    ax2.set_ylabel("Return %", color=MUTED, fontsize=8)
    ax2.tick_params(colors=MUTED, labelsize=9)
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
    ax.set_title("Strategy Composite Score  (directional accuracy × avg return)", color=CYAN, fontsize=10)
    ax.set_xlabel("Composite Score", color=MUTED, fontsize=8)
    for bar, score in zip(bars, ranked["Composite_Score"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:+.2f}", va="center", color=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_equity_curves_by_type(df: pd.DataFrame):
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
        Accuracy % and Composite Score use <b>directional accuracy</b> (MFE-based) — whether price
        moved in the signal's direction at any point during the hold, using intraday High/Low data.
        This captures the true signal quality, independent of exact exit timing. Exit-price accuracy
        is shown separately for reference.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # MFE threshold slider — shown before the run button so it affects the compute
    mfe_threshold = st.slider(
        "📐 MFE threshold — min % intraday move to count as 'directionally correct'",
        min_value=0.1, max_value=5.0, value=DEFAULT_MFE_THRESHOLD, step=0.1,
        key="spe_mfe_thresh",
        help=(
            "Price must move at least this far in the signal's direction (using intraday High/Low) "
            "to be counted as directionally correct. 0.5% is a reasonable default. "
            "Higher = stricter. All Accuracy % and Composite Scores use this threshold."
        ),
    )

    if not st.button("🧠 Compute Signal Profitability", key="spe_run"):
        if st.session_state.get("spe_data") is not None:
            _display(st.session_state["spe_data"], mfe_threshold)
        else:
            st.info("Click **Compute Signal Profitability** to analyse your signal history. "
                    "Requires at least 10 evaluated signals across different types.")
        return

    # ── Load raw signals from Supabase ─────────────────────────────────────
    df_raw = _load_evaluated_signals()
    if df_raw.empty:
        st.warning("No signals found in Supabase. Run Live News Feed and other signal tabs first.")
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

    # Check whether MFE data is actually available
    has_mfe = "mfe" in df_eval.columns and df_eval["mfe"].notna().any()
    if not has_mfe:
        st.warning(
            "⚠️ No MFE (intraday High/Low) data available — OHLC fetch may have failed for these tickers. "
            "Falling back to exit-price accuracy for all metrics. "
            "Directional and exit accuracy will be identical until OHLC data is available."
        )

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
    prof_df = _build_profitability_table(df_eval, mfe_threshold)
    ranked  = _rank_strategies(prof_df)

    data = {
        "df_eval":  df_eval,
        "prof_df":  prof_df,
        "ranked":   ranked,
        "vol_map":  vol_map,
        "n":        n,
        "has_mfe":  has_mfe,
    }
    st.session_state["spe_data"] = data
    _display(data, mfe_threshold)


def _display(data: dict, mfe_threshold: float = DEFAULT_MFE_THRESHOLD):
    df_eval = data["df_eval"]
    prof_df = data["prof_df"]
    ranked  = data["ranked"]
    n       = data["n"]
    has_mfe = data.get("has_mfe", False)

    # Re-compute directional column so the display reflects the current slider value
    df_eval = df_eval.copy()
    df_eval["_dir_correct"] = _resolve_directional_correct(df_eval, mfe_threshold)

    dir_acc_overall  = df_eval["_dir_correct"].mean() * 100
    exit_acc_overall = df_eval["correct"].mean() * 100 if "correct" in df_eval.columns else np.nan
    dir_ac_c = GREEN if dir_acc_overall >= 60 else (AMBER if dir_acc_overall >= 50 else RED)

    st.caption(
        f"Based on {n} evaluated signals · "
        f"{df_eval['signal_category'].nunique()} signal types · "
        f"{df_eval['ticker'].nunique()} tickers · "
        f"Regimes: {df_eval['regime'].value_counts().to_dict()}"
    )

    # ── Overall accuracy banner ────────────────────────────────────────────
    ban1, ban2 = st.columns([3, 2])
    with ban1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                    border:2px solid {dir_ac_c}55;border-radius:16px;
                    padding:1.4rem 2rem;margin-bottom:1rem;text-align:center;">
            <div style="color:#6b8fad;font-size:0.72rem;font-family:'Space Mono',monospace;
                        letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;">
                Overall Directional Accuracy (MFE ≥ {mfe_threshold:.1f}%)
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:3rem;font-weight:700;
                        color:{dir_ac_c};line-height:1;">{dir_acc_overall:.1f}%</div>
            <div style="margin-top:.7rem;color:#8ba3c1;font-size:.75rem;">
                Exit-price accuracy (reference only):
                <b style="color:{MUTED};">{exit_acc_overall:.1f}%</b>
            </div>
        </div>""", unsafe_allow_html=True)
    with ban2:
        st.markdown(f"""
        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;
                    padding:1.1rem 1.2rem;font-size:.81rem;color:#8ba3c1;line-height:1.7;">
            <b style="color:#7dd3fc;">Accuracy % = Directional</b><br>
            Price moved ≥ {mfe_threshold:.1f}% in the signal's direction at any
            point during the hold (intraday High/Low).<br><br>
            <b>Composite Score = directional accuracy × avg return.</b><br>
            Positive = keep. Negative = reconsider or kill.
        </div>""", unsafe_allow_html=True)

    # ── Strategy ranking ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Strategy Auto-Ranking  (directional accuracy)</div>', unsafe_allow_html=True)
    st.caption("Composite Score = Avg Return × Directional Accuracy. Positive = worth keeping. Negative = kill it.")

    if not ranked.empty:
        def c_verdict(v): return f"color:{GREEN}" if "Keep" in str(v) else (f"color:{AMBER}" if "Watch" in str(v) else f"color:{RED}")
        def c_ret(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        def c_acc(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v >= 60 else (f"color:{AMBER}" if v >= 50 else f"color:{RED}")
        def c_score(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v > 0.5 else (f"color:{AMBER}" if v > 0 else f"color:{RED}")

        disp = ranked.rename(columns={
            "Total_Signals":   "Signals",
            "Dir_Accuracy":    "Dir. Accuracy %",   # PRIMARY
            "Exit_Accuracy":   "Exit Accuracy %",   # reference
            "Avg_Return":      "Avg Return %",
            "Avg_Vol_Adj":     "Vol-Adj Return",
            "Composite_Score": "Score",
        })
        sty = disp.style
        if "Verdict"          in disp.columns: sty = sty.map(c_verdict, subset=["Verdict"])
        if "Avg Return %"     in disp.columns: sty = sty.map(c_ret,     subset=["Avg Return %"])
        if "Dir. Accuracy %"  in disp.columns: sty = sty.map(c_acc,     subset=["Dir. Accuracy %"])
        if "Exit Accuracy %"  in disp.columns: sty = sty.map(c_acc,     subset=["Exit Accuracy %"])
        if "Score"            in disp.columns: sty = sty.map(c_score,   subset=["Score"])
        sty = sty.format({
            "Dir. Accuracy %":  "{:.1f}%",
            "Exit Accuracy %":  "{:.1f}%",
            "Avg Return %":     "{:+.2f}%",
            "Vol-Adj Return":   "{:+.3f}",
            "Score":            "{:+.3f}",
        }, na_rep="—")
        st.dataframe(sty, use_container_width=True, hide_index=True)

        fig_rank = _plot_strategy_ranking(ranked)
        if fig_rank:
            st.pyplot(fig_rank); plt.close(fig_rank)

    # ── Plain-English insights ─────────────────────────────────────────────
    if not ranked.empty:
        best  = ranked.iloc[0]
        worst = ranked.iloc[-1]
        st.markdown('<div class="section-header">💡 Plain-English Intelligence</div>', unsafe_allow_html=True)
        insights = []
        if best["Composite_Score"] > 0.5:
            insights.append(
                f"🟢 **{best['Signal Type']}** is your best performing signal type — "
                f"composite score {best['Composite_Score']:+.2f}. "
                f"Directional accuracy {best['Dir_Accuracy']:.0f}%, avg return {best['Avg_Return']:+.1f}%."
            )
        if worst["Composite_Score"] < 0:
            insights.append(
                f"🔴 **{worst['Signal Type']}** is drag — composite score {worst['Composite_Score']:+.2f}. "
                f"Consider disabling or inverting this signal type."
            )
        if not prof_df.empty:
            bull_rows = prof_df[prof_df["Regime"] == "Bull"]
            bear_rows = prof_df[prof_df["Regime"] == "Bear"]
            if not bull_rows.empty:
                best_bull = bull_rows.loc[bull_rows["Score"].idxmax()]
                insights.append(
                    f"📈 **In bull markets**, {best_bull['Signal Type']} performs best: "
                    f"{best_bull['Accuracy %']:.0f}% directional accuracy, {best_bull['Avg Return %']:+.1f}% avg return."
                )
            if not bear_rows.empty:
                best_bear = bear_rows.loc[bear_rows["Score"].idxmax()]
                insights.append(
                    f"📉 **In bear markets**, {best_bear['Signal Type']} performs best: "
                    f"{best_bear['Accuracy %']:.0f}% directional accuracy, {best_bear['Avg Return %']:+.1f}% avg return."
                )
        hc = df_eval[df_eval["confidence"] == "HIGH"]
        lc = df_eval[df_eval["confidence"] == "LOW"]
        if len(hc) >= 3 and len(lc) >= 3:
            hc_dir = hc["_dir_correct"].mean() * 100
            lc_dir = lc["_dir_correct"].mean() * 100
            if hc_dir > lc_dir + 5:
                insights.append(
                    f"✅ **Confidence scoring works**: HIGH directional accuracy {hc_dir:.0f}% "
                    f"vs {lc_dir:.0f}% for LOW."
                )
            else:
                insights.append(
                    f"⚠️ **Confidence may need re-calibration**: HIGH ({hc_dir:.0f}%) is not "
                    f"significantly better than LOW ({lc_dir:.0f}%) on directional accuracy."
                )
        for insight in insights:
            st.markdown(insight)

    # ── Regime heatmap ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌡️ Return Heatmap — Signal Type × Market Regime</div>', unsafe_allow_html=True)
    fig_hm = _plot_regime_heatmap(prof_df)
    if fig_hm:
        st.pyplot(fig_hm); plt.close(fig_hm)

    # ── Accuracy comparison: directional vs exit ───────────────────────────
    st.markdown('<div class="section-header">🎯 Directional vs Exit Accuracy by Confidence</div>', unsafe_allow_html=True)
    st.caption(
        f"Blue = directional (price moved ≥{mfe_threshold:.1f}% in signal direction, intraday). "
        "Grey = exit-price accuracy at fixed horizon. The gap shows how much timing cost you."
    )
    fig_conf = _plot_accuracy_comparison(df_eval, mfe_threshold)
    st.pyplot(fig_conf); plt.close(fig_conf)

    # ── Equity curves ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Cumulative Return by Signal Type</div>', unsafe_allow_html=True)
    fig_eq = _plot_equity_curves_by_type(df_eval)
    if fig_eq:
        st.pyplot(fig_eq); plt.close(fig_eq)

    # ── Full breakdown table ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Full Profitability Breakdown</div>', unsafe_allow_html=True)
    st.caption("Accuracy % = directional. Exit Accuracy % shown for reference.")
    if not prof_df.empty:
        def c_ret2(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        def c_acc2(v):
            if pd.isna(v): return ""
            return f"color:{GREEN}" if v >= 60 else (f"color:{AMBER}" if v >= 50 else f"color:{RED}")
        sty2 = prof_df.style
        for col in ["Accuracy %", "Exit Accuracy %"]:
            if col in prof_df.columns: sty2 = sty2.map(c_acc2, subset=[col])
        for col in ["Avg Return %", "Score"]:
            if col in prof_df.columns: sty2 = sty2.map(c_ret2, subset=[col])
        sty2 = sty2.format({
            "Accuracy %":      "{:.1f}%",
            "Exit Accuracy %": "{:.1f}%",
            "Avg Return %":    "{:+.2f}%",
            "Vol-Adj Return":  "{:+.3f}",
            "Score":           "{:+.3f}",
        }, na_rep="—")
        st.dataframe(sty2, use_container_width=True, hide_index=True)

    # ── Vol-adjusted returns ───────────────────────────────────────────────
    if "vol_adj_return" in df_eval.columns and df_eval["vol_adj_return"].notna().any():
        st.markdown('<div class="section-header">⚡ Volatility-Adjusted Returns (Signal Sharpe)</div>', unsafe_allow_html=True)
        va = df_eval.groupby("signal_category")["vol_adj_return"].agg(["mean","count"]).reset_index()
        va.columns = ["Signal Type", "Avg Vol-Adj Return", "Signals"]
        va = va[va["Signals"] >= 3].sort_values("Avg Vol-Adj Return", ascending=False)
        if not va.empty:
            fig_va, ax_va = plt.subplots(figsize=(8, 3))
            _style_fig(fig_va, ax_va)
            vc = [GREEN if v > 0 else RED for v in va["Avg Vol-Adj Return"]]
            ax_va.barh(va["Signal Type"], va["Avg Vol-Adj Return"],
                       color=vc, edgecolor=BORDER, linewidth=0.4)
            ax_va.axvline(0, color=MUTED, linewidth=0.8)
            ax_va.set_title("Volatility-Adjusted Return by Signal Type", color=CYAN, fontsize=9)
            ax_va.set_xlabel("Return / Volatility", color=MUTED, fontsize=8)
            fig_va.tight_layout()
            st.pyplot(fig_va); plt.close(fig_va)

    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.8rem;border:1px solid #1e3a5f;border-radius:8px;
                color:#6b8fad;font-size:0.78rem;">
        ⚠️ Past signal performance does not guarantee future returns. Not financial advice.<br>
        Directional accuracy uses MFE ≥ {mfe_threshold:.1f}% threshold (intraday High/Low). 
        Adjust the slider at the top to see how threshold choice affects the ranking.
    </div>
    """, unsafe_allow_html=True)

    csv = df_eval.to_csv(index=False)
    st.download_button("⬇️ Download enriched signal data as CSV", data=csv,
                       file_name=f"signal_profitability_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

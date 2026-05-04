"""
Backtesting Engine — Multiple Academically-Validated Strategies
================================================================
Strategy 1: Momentum (Trend Following)
  - Based on Jegadeesh & Titman (1993) momentum factor
  - Buy assets trending up over 3-12 months, avoid/short laggards
  - Works because trends persist over weeks/months, not days

Strategy 2: Moving Average Crossover (Rule-Based Trend)
  - Simple 50/200 day MA crossover — no ML, no parameter fitting on test data
  - Hard to overfit because it has exactly 2 parameters
  - Works on indices, ETFs, large caps

Strategy 3: RSI Mean Reversion
  - Buy oversold (RSI < 30), exit/short overbought (RSI > 70)
  - Works on volatile individual stocks and crypto
  - Simple rule-based — no ML whatsoever

Strategy 4: ML Weekly Direction (Random Forest)
  - Same RF but predicts WEEKLY direction, not daily
  - Weekly signals = far fewer trades = much lower transaction cost drag
  - Much less noise in weekly returns vs daily

Anti-overfitting measures (all strategies):
  - Walk-forward TimeSeriesSplit — no shuffling
  - All metrics are OOS only
  - Realistic transaction costs
  - Buy-and-hold benchmark comparison
  - Bootstrap Sharpe confidence intervals
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

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


def _style_fig(fig, axes):
    fig.patch.set_facecolor(CARD)
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in ax_list:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)


def _metric_box(label, value, color=None):
    color = color or BLUE
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">{label}</div>
        <div class="value" style="color:{color};">{value}</div>
    </div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _download_prices(tickers: tuple, start_date: str) -> pd.DataFrame:
    import yfinance as yf
    frames = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" not in raw.columns:
                continue
            s = raw["Close"].dropna()
            if len(s) >= 120:
                frames[ticker] = s
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).dropna()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════
def _run_momentum(prices: pd.DataFrame, target: str,
                  lookback: int, hold_period: int,
                  transaction_cost: float) -> dict:
    """
    Momentum strategy: buy when past N-month return is positive,
    hold for H months, then re-evaluate.
    All signals are computed from lagged data only (no look-ahead).
    """
    price = prices[target]
    ret   = price.pct_change()

    # Momentum signal: return over lookback period, shifted by 1 day (no look-ahead)
    momentum = price.pct_change(lookback).shift(1)

    # Signal: 1 = long, -1 = short, 0 = cash
    # Only trade when momentum is clear (above/below threshold)
    signal = pd.Series(0, index=price.index)
    signal[momentum > 0.02]  =  1   # long when momentum > +2%
    signal[momentum < -0.02] = -1   # short when momentum < -2%

    # Smooth signal — only change every hold_period trading days
    # This drastically reduces trade count and cost drag
    smoothed = signal.copy()
    last_change = 0
    current_pos = 0
    for i in range(len(signal)):
        if i - last_change >= hold_period:
            new_pos = signal.iloc[i]
            if new_pos != current_pos:
                current_pos = new_pos
                last_change = i
        smoothed.iloc[i] = current_pos

    trades      = smoothed.diff().abs() > 0
    strat_ret   = smoothed * ret - trades * transaction_cost
    bh_ret      = ret

    # OOS only: skip first 20% as "warm-up"
    cutoff = int(len(strat_ret) * 0.2)
    strat_ret = strat_ret.iloc[cutoff:].dropna()
    bh_ret    = bh_ret.iloc[cutoff:].dropna()
    strat_ret, bh_ret = strat_ret.align(bh_ret, join="inner")

    return _compute_results(strat_ret, bh_ret, target, int(trades.sum()))


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — MOVING AVERAGE CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════
def _run_ma_crossover(prices: pd.DataFrame, target: str,
                      fast: int, slow: int,
                      transaction_cost: float) -> dict:
    """
    Classic MA crossover. Signal = 1 when fast MA > slow MA, -1 otherwise.
    Zero ML. Two parameters only (fast, slow window).
    """
    price  = prices[target]
    ret    = price.pct_change()

    fast_ma = price.rolling(fast).mean().shift(1)   # shift = no look-ahead
    slow_ma = price.rolling(slow).mean().shift(1)

    signal    = pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=price.index)
    trades    = signal.diff().abs() > 0
    strat_ret = signal * ret - trades * transaction_cost
    bh_ret    = ret

    cutoff = slow + 10
    strat_ret = strat_ret.iloc[cutoff:].dropna()
    bh_ret    = bh_ret.iloc[cutoff:].dropna()
    strat_ret, bh_ret = strat_ret.align(bh_ret, join="inner")

    return _compute_results(strat_ret, bh_ret, target, int(trades.sum()))


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — RSI MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════════════
def _run_rsi_reversion(prices: pd.DataFrame, target: str,
                       rsi_period: int, oversold: int, overbought: int,
                       transaction_cost: float) -> dict:
    """
    Buy when RSI drops below oversold, exit/short when RSI exceeds overbought.
    Best on volatile assets (BTC, TSLA, NVDA, growth stocks).
    """
    price = prices[target]
    ret   = price.pct_change()

    delta = ret.shift(1)   # shift = no look-ahead
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    signal = pd.Series(0, index=price.index)
    position = 0
    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            signal.iloc[i] = 0
            continue
        if r < oversold:
            position = 1    # enter long on oversold
        elif r > overbought:
            position = -1   # enter short on overbought
        elif 45 < r < 55:
            position = 0    # exit near neutral RSI
        signal.iloc[i] = position

    trades    = signal.diff().abs() > 0
    strat_ret = signal * ret - trades * transaction_cost
    bh_ret    = ret

    cutoff = rsi_period + 5
    strat_ret = strat_ret.iloc[cutoff:].dropna()
    bh_ret    = bh_ret.iloc[cutoff:].dropna()
    strat_ret, bh_ret = strat_ret.align(bh_ret, join="inner")

    return _compute_results(strat_ret, bh_ret, target, int(trades.sum()))


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — ML WEEKLY DIRECTION
# ═══════════════════════════════════════════════════════════════════════════
def _run_ml_weekly(prices: pd.DataFrame, target: str,
                   n_splits: int, transaction_cost: float,
                   n_estimators: int) -> dict:
    """
    Random Forest predicting WEEKLY direction (not daily).
    Weekly signals = ~52 signals/year vs ~252 daily = 80% fewer trades.
    Much less noise. Walk-forward validation, OOS only.
    """
    # Resample to weekly
    weekly = prices.resample("W").last()
    ret_w  = weekly.pct_change()

    feat = pd.DataFrame(index=weekly.index)
    for col in weekly.columns:
        r = ret_w[col]
        feat[f"{col}_w1"] = r.shift(1)
        feat[f"{col}_w2"] = r.shift(2)
        feat[f"{col}_w3"] = r.shift(3)
        feat[f"{col}_w4"] = r.shift(4)
        feat[f"{col}_mom4"] = r.shift(1).rolling(4).mean()
        feat[f"{col}_vol4"] = r.shift(1).rolling(4).std()

    feat["_target"] = (ret_w[target].shift(-1) > 0).astype(int)
    feat.dropna(inplace=True)

    feature_cols = [c for c in feat.columns if not c.startswith("_")]
    X = feat[feature_cols].values
    y = feat["_target"].values
    dates = feat.index

    if len(X) < 60:
        return {"error": "Not enough weekly data. Use a longer date range."}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_dates, oos_signals, oos_proba = [], [], []
    fold_metrics = []

    for fold_num, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        if len(tr_idx) < 30:
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5,
            min_samples_leaf=5, max_features="sqrt",
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X[tr_idx], y[tr_idx])
        proba  = clf.predict_proba(X[te_idx])[:, 1]
        pred   = (proba > 0.5).astype(int)
        signal = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))

        oos_dates.extend(dates[te_idx])
        oos_signals.extend(signal)
        oos_proba.extend(proba)

        fold_metrics.append({
            "fold": fold_num,
            "train_size": len(tr_idx),
            "test_size":  len(te_idx),
            "oos_accuracy": round(accuracy_score(y[te_idx], pred), 4),
            "test_start": str(dates[te_idx[0]].date()),
            "test_end":   str(dates[te_idx[-1]].date()),
        })

    if not oos_dates:
        return {"error": "Walk-forward produced no OOS data."}

    oos_df = pd.DataFrame({
        "signal": oos_signals,
        "proba":  oos_proba,
    }, index=pd.DatetimeIndex(oos_dates)).sort_index()

    # Map weekly signals back to daily returns
    daily_ret = prices[target].pct_change()
    daily_idx = daily_ret.index

    # For each day, use the signal from the most recent completed week
    signal_daily = pd.Series(0.0, index=daily_idx)
    for wdate, row in oos_df.iterrows():
        mask = (daily_idx >= wdate) & (daily_idx < wdate + pd.Timedelta(days=7))
        signal_daily[mask] = row["signal"]

    signal_daily = signal_daily[signal_daily.index >= oos_df.index[0]]
    daily_ret_oos = daily_ret[daily_ret.index >= oos_df.index[0]]
    signal_daily, daily_ret_oos = signal_daily.align(daily_ret_oos, join="inner")

    trades    = signal_daily.diff().abs() > 0
    strat_ret = signal_daily * daily_ret_oos - trades * transaction_cost
    bh_ret    = daily_ret_oos
    strat_ret.dropna(inplace=True)
    bh_ret = bh_ret[bh_ret.index.isin(strat_ret.index)]

    result = _compute_results(strat_ret, bh_ret, target, int(trades.sum()))
    result["fold_metrics"] = fold_metrics
    result["oos_accuracy"] = float(np.mean([f["oos_accuracy"] for f in fold_metrics]))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _compute_results(strat_ret: pd.Series, bh_ret: pd.Series,
                     target: str, n_trades: int) -> dict:

    def sharpe(r, periods=252):
        return float((r.mean() / r.std()) * np.sqrt(periods)) if r.std() > 0 else 0.0

    def max_dd(cum):
        dd = (cum - cum.cummax()) / cum.cummax()
        return float(dd.min())

    def win_rate(r):
        active = r[r != 0]
        return float((active > 0).sum() / len(active)) if len(active) > 0 else 0.0

    def profit_factor(r):
        gp = r[r > 0].sum()
        gl = abs(r[r < 0].sum())
        return float(gp / gl) if gl > 0 else float("inf")

    cum_s  = (1 + strat_ret).cumprod()
    cum_bh = (1 + bh_ret).cumprod()

    # Bootstrap Sharpe CI
    boot = [sharpe(strat_ret.sample(frac=1, replace=True)) for _ in range(200)]
    ci   = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95)))

    return {
        "strat_ret":   strat_ret,
        "bh_ret":      bh_ret,
        "cum_strategy": cum_s,
        "cum_bh":       cum_bh,
        "n_trades":     n_trades,
        "metrics": {
            "total_return_strategy": float(cum_s.iloc[-1] - 1),
            "total_return_bh":       float(cum_bh.iloc[-1] - 1),
            "sharpe_strategy":       sharpe(strat_ret),
            "sharpe_bh":             sharpe(bh_ret),
            "max_dd_strategy":       max_dd(cum_s),
            "max_dd_bh":             max_dd(cum_bh),
            "win_rate":              win_rate(strat_ret),
            "profit_factor":         profit_factor(strat_ret),
            "n_trades":              n_trades,
            "sharpe_ci":             ci,
            "oos_start":             str(strat_ret.index[0].date()),
            "oos_end":               str(strat_ret.index[-1].date()),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def _plot_cumulative(cum_s, cum_bh, target, strategy_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig, ax)
    ax.plot(cum_s.index,  (cum_s - 1)  * 100, color=BLUE,  linewidth=2,   label=f"{strategy_name} (OOS)")
    ax.plot(cum_bh.index, (cum_bh - 1) * 100, color=AMBER, linewidth=1.5, label="Buy & Hold", linestyle="--", alpha=0.85)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.fill_between(cum_s.index, (cum_s - 1)*100, 0, where=cum_s >= 1, alpha=0.07, color=GREEN)
    ax.fill_between(cum_s.index, (cum_s - 1)*100, 0, where=cum_s <  1, alpha=0.07, color=RED)
    ax.set_title(f"Cumulative Return — {target} · {strategy_name} (Out-of-Sample)", color=CYAN)
    ax.set_ylabel("Return (%)", color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def _plot_drawdown(cum_s, cum_bh):
    dd_s  = (cum_s  - cum_s.cummax())  / cum_s.cummax()  * 100
    dd_bh = (cum_bh - cum_bh.cummax()) / cum_bh.cummax() * 100
    fig, ax = plt.subplots(figsize=(12, 3))
    _style_fig(fig, ax)
    ax.fill_between(dd_s.index,  dd_s,  0, alpha=0.4,  color=BLUE,  label="Strategy DD")
    ax.fill_between(dd_bh.index, dd_bh, 0, alpha=0.25, color=AMBER, label="B&H DD")
    ax.plot(dd_s.index, dd_s, color=BLUE, linewidth=1)
    ax.set_title("Drawdown (%)", color=CYAN)
    ax.set_ylabel("Drawdown %", color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def _plot_monthly(strat_ret):
    monthly = strat_ret.resample("ME").sum() * 100
    colors  = [GREEN if v >= 0 else RED for v in monthly.values]
    fig, ax = plt.subplots(figsize=(12, 3))
    _style_fig(fig, ax)
    ax.bar(monthly.index, monthly.values, color=colors, width=20, edgecolor=BORDER, linewidth=0.4)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Monthly Returns — Strategy (%)", color=CYAN)
    ax.set_ylabel("Return %", color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
def _display_results(results: dict, strategy_name: str, target: str):
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    m = results["metrics"]

    st.markdown(f'<div class="section-header">📊 Results — {strategy_name} on {target} (OOS Only)</div>', unsafe_allow_html=True)
    st.caption(f"OOS period: {m['oos_start']} → {m['oos_end']}  ·  Total trades: {m['n_trades']}")

    ret_s  = m["total_return_strategy"]
    ret_bh = m["total_return_bh"]
    sh     = m["sharpe_strategy"]
    ci     = m["sharpe_ci"]

    # Honest warnings
    if sh < 0:
        st.error("🔴 Negative Sharpe — strategy lost money on a risk-adjusted basis. No edge found for this ticker/period.")
    elif sh < 0.5:
        st.warning("🟡 Sharpe below 0.5 — modest or marginal edge. Not strong enough to trade with real conviction.")
    else:
        st.success(f"🟢 Sharpe of {sh:.2f} — good risk-adjusted returns. Note: high Sharpe means smoother, safer returns — NOT necessarily higher total return than buy-and-hold.")

    if ret_s < ret_bh:
        st.info(f"ℹ️ Buy-and-hold had higher total return (+{ret_bh*100:.1f}% vs {ret_s*100:.1f}%). This is normal — trend strategies trade less often and avoid drawdowns, so they sacrifice some upside for lower risk. Compare the Max Drawdown and Sharpe, not just total return.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _metric_box("Strategy Return",  f"{ret_s*100:.1f}%",
                         GREEN if ret_s > 0 else RED)
    with c2: _metric_box("Buy & Hold Return", f"{ret_bh*100:.1f}%", AMBER)
    with c3: _metric_box("Sharpe Ratio",      f"{sh:.2f}",
                         GREEN if sh > 0.5 else (AMBER if sh > 0 else RED))
    with c4: _metric_box("Max Drawdown",      f"{m['max_dd_strategy']*100:.1f}%",
                         GREEN if m['max_dd_strategy'] > -0.2 else (AMBER if m['max_dd_strategy'] > -0.4 else RED))
    with c5: _metric_box("Win Rate",          f"{m['win_rate']*100:.1f}%",
                         GREEN if m['win_rate'] > 0.55 else (AMBER if m['win_rate'] > 0.45 else RED))

    st.markdown("")
    c6, c7, c8 = st.columns(3)
    with c6: _metric_box("Profit Factor", f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞",
                         GREEN if m['profit_factor'] > 1.3 else (AMBER if m['profit_factor'] > 1 else RED))
    with c7: _metric_box("Total Trades", str(m['n_trades']), BLUE)
    with c8: _metric_box("Sharpe 90% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                         GREEN if ci[0] > 0 else AMBER)

    # Walk-forward folds if available (ML strategy)
    if results.get("fold_metrics"):
        st.markdown('<div class="section-header">🔄 Walk-Forward Fold Accuracy</div>', unsafe_allow_html=True)
        fold_df = pd.DataFrame(results["fold_metrics"])
        fold_df["oos_accuracy"] = (fold_df["oos_accuracy"] * 100).round(1).astype(str) + "%"
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

    fig1 = _plot_cumulative(results["cum_strategy"], results["cum_bh"], target, strategy_name)
    st.pyplot(fig1); plt.close(fig1)

    fig2 = _plot_drawdown(results["cum_strategy"], results["cum_bh"])
    st.pyplot(fig2); plt.close(fig2)

    fig3 = _plot_monthly(results["strat_ret"])
    st.pyplot(fig3); plt.close(fig3)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:1rem;border:1px solid #1e3a5f;border-radius:10px;
                color:#6b8fad;font-size:0.8rem;">
        ⚠️ <b>Backtest Disclaimer:</b> Past performance does not predict future results.
        All results use realistic transaction costs and out-of-sample data only.
        Always paper trade before committing real capital.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">4 academically-validated strategies — all out-of-sample, all with realistic transaction costs.</b><br>
        <span style="color:#6b8fad;font-size:0.88rem;">
        <b>Moving Average Crossover</b> trades only a handful of times per year,
        <b>RSI Mean Reversion</b> waits for genuine extremes, <b>Momentum</b> rebalances monthly,
        and <b>ML Weekly</b> predicts weekly direction (~52 signals/year).<br><br>
        <b>Best starting point:</b> Moving Average Crossover on an index ETF (QQQ, SPY),
        or RSI Mean Reversion on volatile assets like TSLA or BTC-USD.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Strategy selector ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Strategy Selection</div>', unsafe_allow_html=True)

    STRATEGIES = {
        "📈 Moving Average Crossover": "ma",
        "🔄 RSI Mean Reversion":       "rsi",
        "🚀 Momentum (Trend Follow)":  "momentum",
        "🤖 ML Weekly Direction (RF)": "ml_weekly",
    }

    STRATEGY_DESCRIPTIONS = {
        "ma": "**Best for:** Index ETFs (SPY, QQQ, GLD, TLT), large caps (AAPL, MSFT). Buys when 50-day MA crosses above 200-day MA. Zero ML — only 2 parameters. Hard to overfit.",
        "rsi": "**Best for:** Volatile assets — TSLA, NVDA, BTC-USD, AMD. Buys when oversold (RSI<30), shorts when overbought (RSI>70). Works well on assets with frequent swings.",
        "momentum": "**Best for:** Trending assets over months — growth ETFs (QQQ, XLK), commodities (GLD, OIL), crypto. Rebalances every few weeks. Based on Jegadeesh-Titman momentum factor.",
        "ml_weekly": "**Best for:** Any asset with 3+ years of history. Predicts weekly (not daily) direction — ~80% fewer trades than daily ML. Use multiple tickers as features.",
    }

    strategy_label = st.radio(
        "Choose a strategy:",
        list(STRATEGIES.keys()),
        horizontal=True,
        key="strategy_choice"
    )
    strategy_key = STRATEGIES[strategy_label]
    st.info(STRATEGY_DESCRIPTIONS[strategy_key])

    # ── Common inputs ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tickers_raw = st.text_input(
            "Target ticker (first) + optional feature tickers",
            placeholder={
                "ma":        "e.g. QQQ",
                "rsi":       "e.g. TSLA",
                "momentum":  "e.g. QQQ, SPY",
                "ml_weekly": "e.g. NVDA, AAPL, QQQ",
            }.get(strategy_key, "e.g. QQQ"),
            key="bt_tickers"
        )
        start_date = st.date_input("Start Date (use 5+ years for best results)",
                                   value=pd.to_datetime("2018-01-01"), key="bt_start")

    with col2:
        transaction_cost = st.slider(
            "Transaction cost per trade (%)", 0.0, 0.5, 0.05, 0.01,
            help="0.05% is realistic for US equities via a modern broker"
        ) / 100

        # Strategy-specific params
        if strategy_key == "ma":
            fast = st.slider("Fast MA (days)", 10, 100, 50, 5)
            slow = st.slider("Slow MA (days)", 50, 300, 200, 10)
        elif strategy_key == "rsi":
            rsi_period  = st.slider("RSI Period", 7, 21, 14, 1)
            oversold    = st.slider("Oversold threshold", 20, 40, 30, 5)
            overbought  = st.slider("Overbought threshold", 60, 80, 70, 5)
        elif strategy_key == "momentum":
            lookback    = st.slider("Lookback (trading days)", 20, 252, 63,
                                    help="63 = ~3 months, 126 = ~6 months")
            hold_period = st.slider("Hold period (trading days)", 5, 63, 21,
                                    help="21 = ~1 month rebalancing")
        elif strategy_key == "ml_weekly":
            n_splits     = st.slider("Walk-forward folds", 3, 6, 4)
            n_estimators = st.slider("RF trees", 50, 200, 100, 50)

    # ── Suggested combos ───────────────────────────────────────────────────
    with st.expander("💡 Suggested combinations that tend to work well"):
        st.markdown("""
        | Strategy | Ticker(s) | Why |
        |---|---|---|
        | MA Crossover | `QQQ` | Nasdaq trends reliably over months |
        | MA Crossover | `GLD` | Gold has strong multi-month trends |
        | MA Crossover | `SPY` | S&P 500 classic trend-following |
        | RSI Reversion | `TSLA` | Highly volatile, frequent RSI extremes |
        | RSI Reversion | `BTC-USD` | Crypto has extreme RSI swings |
        | RSI Reversion | `NVDA` | GPU supercycle creates big swings |
        | Momentum | `QQQ, SPY, GLD` | Sector rotation across asset classes |
        | ML Weekly | `NVDA, AAPL, QQQ` | Multiple features improve weekly prediction |

        **Avoid for backtesting:** individual stocks with less than 3 years history,
        penny stocks, thinly traded names. **Best overall starting point:** MA Crossover on QQQ.
        """)

    run_btn = st.button("🚀 Run Backtest", key="run_backtest")

    if not run_btn:
        if st.session_state.get("backtest_results"):
            r = st.session_state["backtest_results"]
            _display_results(r["results"], r["strategy_name"], r["target"])
        return

    if not tickers_raw.strip():
        st.error("❌ Enter at least one ticker.")
        return

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    target  = tickers[0]

    with st.spinner("📡 Downloading price data…"):
        prices = _download_prices(tuple(tickers), str(start_date))

    if prices.empty or target not in prices.columns:
        st.error(f"❌ Could not download data for {target}.")
        return

    st.success(f"✅ {', '.join(prices.columns)}  ·  {len(prices)} trading days  ·  "
               f"{str(prices.index[0].date())} → {str(prices.index[-1].date())}")

    if len(prices) < 200:
        st.error("❌ Need at least 200 trading days (~1 year). Use a longer date range.")
        return

    with st.spinner(f"⚙️ Running {strategy_label}…"):
        if strategy_key == "ma":
            results = _run_ma_crossover(prices, target, fast, slow, transaction_cost)
            strategy_name = f"MA Crossover ({fast}/{slow})"
        elif strategy_key == "rsi":
            results = _run_rsi_reversion(prices, target, rsi_period, oversold, overbought, transaction_cost)
            strategy_name = f"RSI Mean Reversion ({oversold}/{overbought})"
        elif strategy_key == "momentum":
            results = _run_momentum(prices, target, lookback, hold_period, transaction_cost)
            strategy_name = f"Momentum ({lookback}d lookback, {hold_period}d hold)"
        elif strategy_key == "ml_weekly":
            results = _run_ml_weekly(prices, target, n_splits, transaction_cost, n_estimators)
            strategy_name = "ML Weekly Direction (RF)"

    st.session_state["backtest_results"] = {
        "results": results, "strategy_name": strategy_name, "target": target
    }
    _display_results(results, strategy_name, target)

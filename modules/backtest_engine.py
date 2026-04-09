"""
Backtesting Engine — Walk-Forward Validation
=============================================
Anti-overfitting measures:
  1. TimeSeriesSplit (expanding window) — never shuffles time-series data
  2. All reported metrics are OUT-OF-SAMPLE only
  3. Realistic transaction costs (slippage + commission)
  4. Compared against buy-and-hold benchmark
  5. No parameter tuning on test folds
  6. Confidence intervals via bootstrap on OOS returns
  7. Clear labelling of what was trained vs what was tested
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
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# ── Style constants ──────────────────────────────────────────────────────────
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


def _build_features(prices: pd.DataFrame, target_ticker: str,
                    rolling_window: int = 20) -> pd.DataFrame:
    """
    Build feature matrix from price data.
    NO look-ahead: all features are lagged by at least 1 period.
    """
    feat = pd.DataFrame(index=prices.index)

    returns = prices.pct_change()

    for col in prices.columns:
        r = returns[col]
        # Lagged returns (1, 3, 5 days) — all safe, no look-ahead
        feat[f"{col}_ret1"]  = r.shift(1)
        feat[f"{col}_ret3"]  = r.shift(1).rolling(3).mean()
        feat[f"{col}_ret5"]  = r.shift(1).rolling(5).mean()
        # Rolling volatility
        feat[f"{col}_vol20"] = r.shift(1).rolling(rolling_window).std()
        # RSI proxy (momentum)
        delta = r.shift(1)
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feat[f"{col}_rsi"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Pairwise rolling correlations (lagged)
    from itertools import combinations
    cols = list(prices.columns)
    for c1, c2 in combinations(cols, 2):
        feat[f"corr_{c1}_{c2}"] = (
            returns[c1].shift(1).rolling(rolling_window).corr(returns[c2].shift(1))
        )

    # Target: next-day return direction for target_ticker (1 = up, 0 = down)
    # This is what we're trying to predict OUT-OF-SAMPLE
    feat["_target"] = (returns[target_ticker].shift(-1) > 0).astype(int)

    feat.dropna(inplace=True)
    return feat


def _walk_forward_backtest(
    feat_df: pd.DataFrame,
    target_ticker: str,
    prices: pd.DataFrame,
    n_splits: int,
    transaction_cost: float,
    n_estimators: int,
) -> dict:
    """
    Walk-forward (expanding window) backtest using TimeSeriesSplit.
    Only OOS predictions are used for performance calculation.
    """
    feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
    X = feat_df[feature_cols].values
    y = feat_df["_target"].values
    dates = feat_df.index

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oos_dates   = []
    oos_signals = []   # 1 = long, -1 = short, 0 = no trade
    oos_actual  = []
    oos_proba   = []
    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        # Minimum training size guard
        if len(train_idx) < 60:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_test = dates[test_idx]

        # Train — NO parameter tuning, fixed hyperparameters
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,           # deliberate depth cap to reduce overfitting
            min_samples_leaf=10,   # requires meaningful support
            max_features="sqrt",   # standard RF regularisation
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)

        proba     = clf.predict_proba(X_test)[:, 1]   # prob of up
        pred      = clf.predict(X_test)

        # Signal: only trade when model is confident (prob > 0.55 or < 0.45)
        # This reduces noise trades
        signal = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))

        oos_dates.extend(d_test)
        oos_signals.extend(signal)
        oos_actual.extend(y_test)
        oos_proba.extend(proba)

        # Per-fold accuracy on OOS data
        fold_acc = accuracy_score(y_test, pred)
        fold_metrics.append({
            "fold":        fold_num,
            "train_size":  len(train_idx),
            "test_size":   len(test_idx),
            "oos_accuracy": round(fold_acc, 4),
            "train_start": str(dates[train_idx[0]].date()),
            "train_end":   str(dates[train_idx[-1]].date()),
            "test_start":  str(dates[test_idx[0]].date()),
            "test_end":    str(dates[test_idx[-1]].date()),
        })

    if not oos_dates:
        return {}

    # ── Compute strategy returns ──────────────────────────────────────────
    oos_df = pd.DataFrame({
        "date":   oos_dates,
        "signal": oos_signals,
        "actual": oos_actual,
        "proba":  oos_proba,
    }).set_index("date").sort_index()

    # Get actual daily returns for target ticker aligned to OOS dates
    ticker_returns = prices[target_ticker].pct_change()
    oos_df["daily_ret"] = ticker_returns.reindex(oos_df.index).shift(-1)  # next-day return
    oos_df.dropna(subset=["daily_ret"], inplace=True)

    # Strategy return = signal * daily_return - transaction_cost when signal changes
    signal_series  = oos_df["signal"]
    trades         = signal_series.diff().abs() > 0
    strategy_ret   = (oos_df["signal"] * oos_df["daily_ret"]) - (trades * transaction_cost)

    # Buy and hold benchmark
    bh_ret = oos_df["daily_ret"]

    # Cumulative returns
    cum_strategy = (1 + strategy_ret).cumprod()
    cum_bh       = (1 + bh_ret).cumprod()

    # ── Performance metrics ────────────────────────────────────────────────
    def sharpe(returns, periods=252):
        if returns.std() == 0:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(periods))

    def max_drawdown(cum_ret):
        roll_max = cum_ret.cummax()
        dd       = (cum_ret - roll_max) / roll_max
        return float(dd.min())

    def win_rate(returns):
        wins = (returns > 0).sum()
        total = (returns != 0).sum()
        return float(wins / total) if total > 0 else 0.0

    def profit_factor(returns):
        gross_profit = returns[returns > 0].sum()
        gross_loss   = abs(returns[returns < 0].sum())
        return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    active_trades = strategy_ret[oos_df["signal"] != 0]
    n_trades      = int(trades.sum())

    # Bootstrap confidence interval for Sharpe (100 iterations, quick)
    bootstrap_sharpes = []
    for _ in range(100):
        sample = strategy_ret.sample(frac=1.0, replace=True)
        bootstrap_sharpes.append(sharpe(sample))
    sharpe_ci_low  = float(np.percentile(bootstrap_sharpes, 5))
    sharpe_ci_high = float(np.percentile(bootstrap_sharpes, 95))

    overall_acc = accuracy_score(oos_actual, [1 if p > 0.5 else 0 for p in oos_proba])

    return {
        "oos_df":          oos_df,
        "strategy_ret":    strategy_ret,
        "bh_ret":          bh_ret,
        "cum_strategy":    cum_strategy,
        "cum_bh":          cum_bh,
        "fold_metrics":    fold_metrics,
        "metrics": {
            "total_return_strategy": float(cum_strategy.iloc[-1] - 1),
            "total_return_bh":       float(cum_bh.iloc[-1] - 1),
            "sharpe_strategy":       sharpe(strategy_ret),
            "sharpe_bh":             sharpe(bh_ret),
            "max_dd_strategy":       max_drawdown(cum_strategy),
            "max_dd_bh":             max_drawdown(cum_bh),
            "win_rate":              win_rate(active_trades),
            "profit_factor":         profit_factor(active_trades),
            "n_trades":              n_trades,
            "oos_accuracy":          overall_acc,
            "sharpe_ci":             (sharpe_ci_low, sharpe_ci_high),
            "oos_period_start":      str(oos_df.index[0].date()),
            "oos_period_end":        str(oos_df.index[-1].date()),
        }
    }


def _plot_cumulative(cum_strategy, cum_bh, target_ticker):
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig, ax)
    ax.plot(cum_strategy.index, (cum_strategy - 1) * 100,
            color=BLUE, linewidth=2, label="Strategy (OOS)")
    ax.plot(cum_bh.index, (cum_bh - 1) * 100,
            color=AMBER, linewidth=1.5, linestyle="--", label="Buy & Hold", alpha=0.85)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.fill_between(cum_strategy.index, (cum_strategy - 1) * 100, 0,
                    where=(cum_strategy >= 1), alpha=0.07, color=GREEN)
    ax.fill_between(cum_strategy.index, (cum_strategy - 1) * 100, 0,
                    where=(cum_strategy < 1), alpha=0.07, color=RED)
    ax.set_title(f"Cumulative Return — {target_ticker} (Out-of-Sample Only)", color=CYAN)
    ax.set_ylabel("Return (%)", color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def _plot_drawdown(cum_strategy, cum_bh):
    dd_strat = (cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax() * 100
    dd_bh    = (cum_bh    - cum_bh.cummax())    / cum_bh.cummax()    * 100

    fig, ax = plt.subplots(figsize=(12, 3))
    _style_fig(fig, ax)
    ax.fill_between(dd_strat.index, dd_strat, 0, alpha=0.4, color=BLUE, label="Strategy DD")
    ax.fill_between(dd_bh.index,    dd_bh,    0, alpha=0.25, color=AMBER, label="B&H DD")
    ax.plot(dd_strat.index, dd_strat, color=BLUE, linewidth=1)
    ax.set_title("Drawdown (%)", color=CYAN)
    ax.set_ylabel("Drawdown %", color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def _plot_monthly_returns(strategy_ret):
    monthly = strategy_ret.resample("ME").sum() * 100
    colors  = [GREEN if v >= 0 else RED for v in monthly.values]
    fig, ax = plt.subplots(figsize=(12, 3))
    _style_fig(fig, ax)
    ax.bar(monthly.index, monthly.values, color=colors, width=20, edgecolor=BORDER, linewidth=0.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Monthly Returns — Strategy (%)", color=CYAN)
    ax.set_ylabel("Return %", color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def _plot_fold_accuracy(fold_metrics):
    folds = [f["fold"] for f in fold_metrics]
    accs  = [f["oos_accuracy"] * 100 for f in fold_metrics]
    colors = [GREEN if a >= 55 else (AMBER if a >= 50 else RED) for a in accs]

    fig, ax = plt.subplots(figsize=(8, 3))
    _style_fig(fig, ax)
    ax.bar(folds, accs, color=colors, edgecolor=BORDER, linewidth=0.5)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle="--", label="50% (random)")
    ax.axhline(55, color=AMBER, linewidth=0.8, linestyle=":", label="55% (target)")
    ax.set_title("OOS Accuracy Per Fold (Walk-Forward)", color=CYAN)
    ax.set_xlabel("Fold", color=MUTED)
    ax.set_ylabel("Accuracy %", color=MUTED)
    ax.set_xticks(folds)
    ax.set_ylim(40, 75)
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def run_backtest():
    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">How this backtester avoids overfitting:</b><br>
        <span style="color:#6b8fad;font-size:0.88rem;">
        ① Walk-forward expanding windows (TimeSeriesSplit) — no data shuffling<br>
        ② All displayed metrics are <b>out-of-sample only</b> — training data never appears in results<br>
        ③ Fixed model hyperparameters — no parameter optimisation on test folds<br>
        ④ Realistic transaction costs applied on every trade<br>
        ⑤ Bootstrap confidence intervals on Sharpe ratio<br>
        ⑥ Buy-and-hold benchmark shown for honest comparison
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Inputs ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Backtest Setup</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tickers_raw = st.text_input(
            "Tickers (comma-separated, first ticker is the prediction target)",
            placeholder="e.g. TSLA, NVDA, AMD, ^GSPC",
            key="bt_tickers"
        )
        start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"), key="bt_start")

    with col2:
        n_splits         = st.slider("Walk-forward folds", 3, 8, 5,
                                     help="More folds = more OOS data but smaller training windows")
        transaction_cost = st.slider("Transaction cost per trade (%)", 0.0, 0.5, 0.1, 0.05,
                                     help="Covers commission + slippage. 0.1% is conservative for US equities") / 100
        n_estimators     = st.slider("RF trees", 50, 200, 100, 50, key="bt_rf")

    run_btn = st.button("🚀 Run Backtest", key="run_backtest")

    if not run_btn:
        if st.session_state.get("backtest_results"):
            _display_results(st.session_state["backtest_results"])
        return

    if not tickers_raw.strip():
        st.error("❌ Enter at least one ticker.")
        return

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    target  = tickers[0]

    if len(tickers) < 2:
        st.warning("⚠️ Adding ^GSPC as a correlation feature. Enter multiple tickers for richer features.")
        tickers = tickers + ["^GSPC"]

    # ── Download data ──────────────────────────────────────────────────────
    with st.spinner("📡 Downloading price data…"):
        prices = _download_prices(tuple(tickers), str(start_date))

    if prices.empty or target not in prices.columns:
        st.error(f"❌ Could not download data for {target}. Check the ticker and try again.")
        return

    valid = list(prices.columns)
    st.success(f"✅ Downloaded: {', '.join(valid)}  ·  {len(prices)} trading days")

    if len(prices) < 252:
        st.error("❌ Need at least 1 year of data for meaningful backtesting.")
        return

    # ── Build features ─────────────────────────────────────────────────────
    with st.spinner("🔧 Building feature matrix…"):
        feat_df = _build_features(prices, target)

    st.caption(f"Feature matrix: {len(feat_df)} rows × {len([c for c in feat_df.columns if not c.startswith('_')])} features")

    # ── Walk-forward backtest ──────────────────────────────────────────────
    with st.spinner(f"🔄 Running {n_splits}-fold walk-forward validation…"):
        results = _walk_forward_backtest(
            feat_df, target, prices, n_splits, transaction_cost, n_estimators
        )

    if not results:
        st.error("❌ Backtest failed — not enough data for the number of folds requested. Try fewer folds or a longer date range.")
        return

    results["target"] = target
    results["tickers"] = valid
    st.session_state["backtest_results"] = results
    _display_results(results)


def _display_results(results: dict):
    m        = results["metrics"]
    target   = results["target"]

    st.markdown('<div class="section-header">📊 Performance Summary (Out-of-Sample Only)</div>', unsafe_allow_html=True)
    st.caption(f"OOS period: {m['oos_period_start']} → {m['oos_period_end']}")

    # ── Warning banner if results are weak ────────────────────────────────
    if m["sharpe_strategy"] < 0.3:
        st.warning("⚠️ Sharpe ratio below 0.3 — strategy does not show meaningful edge over this period. This is an honest result, not a bug.")
    elif m["oos_accuracy"] < 0.50:
        st.warning("⚠️ OOS accuracy below 50% — model is not predicting direction reliably. Consider different tickers or a longer history.")

    # ── Top metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ret_s  = m["total_return_strategy"]
    ret_bh = m["total_return_bh"]
    ret_color = GREEN if ret_s > ret_bh else RED

    with c1: _metric_box("Strategy Return", f"{ret_s*100:.1f}%", ret_color)
    with c2: _metric_box("Buy & Hold Return", f"{ret_bh*100:.1f}%", AMBER)
    with c3: _metric_box("Sharpe Ratio", f"{m['sharpe_strategy']:.2f}",
                         GREEN if m['sharpe_strategy'] > 0.5 else (AMBER if m['sharpe_strategy'] > 0 else RED))
    with c4: _metric_box("Max Drawdown", f"{m['max_dd_strategy']*100:.1f}%",
                         GREEN if m['max_dd_strategy'] > -0.15 else (AMBER if m['max_dd_strategy'] > -0.3 else RED))
    with c5: _metric_box("Win Rate", f"{m['win_rate']*100:.1f}%",
                         GREEN if m['win_rate'] > 0.55 else (AMBER if m['win_rate'] > 0.45 else RED))
    with c6: _metric_box("OOS Accuracy", f"{m['oos_accuracy']*100:.1f}%",
                         GREEN if m['oos_accuracy'] > 0.55 else (AMBER if m['oos_accuracy'] > 0.50 else RED))

    st.markdown("")

    c7, c8, c9 = st.columns(3)
    with c7: _metric_box("Profit Factor", f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞",
                         GREEN if m['profit_factor'] > 1.3 else (AMBER if m['profit_factor'] > 1.0 else RED))
    with c8: _metric_box("Total Trades", str(m['n_trades']), BLUE)
    with c9:
        ci = m["sharpe_ci"]
        _metric_box("Sharpe 90% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]", PURPLE)

    # ── Interpretation ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Honest Interpretation</div>', unsafe_allow_html=True)

    alpha = (m["total_return_strategy"] - m["total_return_bh"])
    interp_lines = []

    if m["sharpe_strategy"] > 0.7 and m["win_rate"] > 0.54:
        interp_lines.append(f"✅ **Strong signal detected** — Sharpe of {m['sharpe_strategy']:.2f} with {m['win_rate']*100:.1f}% win rate on OOS data suggests genuine predictive edge.")
    elif m["sharpe_strategy"] > 0.3:
        interp_lines.append(f"🟡 **Modest signal** — Sharpe of {m['sharpe_strategy']:.2f} shows some edge but not strong enough to trade with high conviction.")
    else:
        interp_lines.append(f"🔴 **Weak or no signal** — Sharpe of {m['sharpe_strategy']:.2f}. The model is not finding a reliable pattern for {target} over this period.")

    if alpha > 0:
        interp_lines.append(f"✅ **Outperformed buy-and-hold by {alpha*100:.1f}%** over the OOS period.")
    else:
        interp_lines.append(f"🔴 **Underperformed buy-and-hold by {abs(alpha)*100:.1f}%** — simply holding {target} would have done better.")

    if m["max_dd_strategy"] < -0.25:
        interp_lines.append(f"⚠️ **Max drawdown of {m['max_dd_strategy']*100:.1f}%** — significant. Ensure position sizing accounts for this.")

    ci = m["sharpe_ci"]
    if ci[0] < 0:
        interp_lines.append(f"⚠️ **Sharpe confidence interval includes negative values [{ci[0]:.2f}, {ci[1]:.2f}]** — results may not be statistically robust. Collect more data.")
    else:
        interp_lines.append(f"✅ **Sharpe CI [{ci[0]:.2f}, {ci[1]:.2f}] is entirely positive** — results are statistically more reliable.")

    for line in interp_lines:
        st.markdown(line)

    # ── Charts ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Cumulative Return</div>', unsafe_allow_html=True)
    fig1 = _plot_cumulative(results["cum_strategy"], results["cum_bh"], target)
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown('<div class="section-header">📉 Drawdown</div>', unsafe_allow_html=True)
    fig2 = _plot_drawdown(results["cum_strategy"], results["cum_bh"])
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown('<div class="section-header">📅 Monthly Returns</div>', unsafe_allow_html=True)
    fig3 = _plot_monthly_returns(results["strategy_ret"])
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Walk-forward fold detail ───────────────────────────────────────────
    st.markdown('<div class="section-header">🔄 Walk-Forward Fold Breakdown</div>', unsafe_allow_html=True)
    st.caption("Each fold trains on all data up to that point and tests on the next unseen period only.")

    fig4 = _plot_fold_accuracy(results["fold_metrics"])
    st.pyplot(fig4)
    plt.close(fig4)

    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df["oos_accuracy"] = (fold_df["oos_accuracy"] * 100).round(1).astype(str) + "%"
    st.dataframe(fold_df, use_container_width=True, hide_index=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;border:1px solid #1e3a5f;border-radius:10px;
                color:#6b8fad;font-size:0.8rem;">
        ⚠️ <b>Backtest Disclaimer:</b> Past performance does not predict future results.
        This backtest uses historical data and makes simplifying assumptions
        (daily execution, no market impact, constant transaction costs).
        Real trading involves slippage, liquidity constraints, and regime changes
        not captured here. Always paper trade before committing real capital.
    </div>
    """, unsafe_allow_html=True)

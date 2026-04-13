"""
Automated Strategy Evaluator
==============================
Runs a rigorous 3-week (or any period) historical simulation.

Key design principles:
  1. Walk-forward only — at each point in time, only data BEFORE that point is used
  2. Signals are generated exactly as the live system would generate them
  3. Outcome measured at multiple horizons: 1d, 3d, 5d, 10d, 21d
  4. Statistical significance tested (not just accuracy %)
  5. Multiple tickers and strategies tested in one run
  6. Every simulated signal saved to the database for the Performance tab to use

This answers: "Which strategy, on which ticker, at which confidence level,
over which time horizon, has a statistically meaningful edge?"
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

OUTCOME_HORIZONS = [1, 3, 5, 10, 21]   # days


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
def _fetch_prices(ticker: str, start_date: str) -> pd.Series | None:
    import yfinance as yf
    try:
        raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes if len(closes) >= 60 else None
    except Exception:
        return None


# ── Signal generators (walk-forward safe) ────────────────────────────────────

def _generate_ma_signals(closes: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Returns daily signal series: 1=BUY, -1=SHORT, 0=HOLD.
    At each point t, uses only closes up to and including t.
    """
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()
    signal  = pd.Series(0, index=closes.index)
    signal[fast_ma > slow_ma] =  1
    signal[fast_ma < slow_ma] = -1
    return signal.shift(1)  # shift: signal computed from yesterday's data, traded today


def _generate_rsi_signals(closes: pd.Series, period: int,
                           oversold: int, overbought: int) -> pd.Series:
    ret  = closes.pct_change()
    gain = ret.clip(lower=0).rolling(period).mean()
    loss = (-ret.clip(upper=0)).rolling(period).mean()
    rsi  = 100 - (100 / (1 + gain / (loss + 1e-9)))

    position = pd.Series(0, index=closes.index)
    pos = 0
    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            position.iloc[i] = 0
            continue
        if r < oversold:
            pos = 1
        elif r > overbought:
            pos = -1
        elif 45 < r < 55:
            pos = 0
        position.iloc[i] = pos

    return position.shift(1)


def _generate_momentum_signals(closes: pd.Series, lookback: int) -> pd.Series:
    mom = closes.pct_change(lookback)
    signal = pd.Series(0, index=closes.index)
    signal[mom > 0.02]  =  1
    signal[mom < -0.02] = -1
    return signal.shift(1)


def _generate_ml_signals(closes: pd.Series, n_splits: int = 4,
                          n_estimators: int = 100) -> pd.Series:
    """Walk-forward ML signal on weekly data, mapped back to daily."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit

    weekly = closes.resample("W").last()
    ret_w  = weekly.pct_change()
    feat   = pd.DataFrame(index=weekly.index)

    for lag in [1, 2, 3, 4]:
        feat[f"ret_w{lag}"]  = ret_w.shift(lag)
    feat["mom4"] = ret_w.shift(1).rolling(4).mean()
    feat["vol4"] = ret_w.shift(1).rolling(4).std()
    feat["_target"] = (ret_w.shift(-1) > 0).astype(int)
    feat.dropna(inplace=True)

    if len(feat) < 30:
        return pd.Series(0, index=closes.index)

    feature_cols = [c for c in feat.columns if not c.startswith("_")]
    X = feat[feature_cols].values
    y = feat["_target"].values
    dates = feat.index

    tscv = TimeSeriesSplit(n_splits=n_splits)
    weekly_signals = pd.Series(np.nan, index=dates)

    for tr_idx, te_idx in tscv.split(X):
        if len(tr_idx) < 20:
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5,
            min_samples_leaf=5, max_features="sqrt",
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[te_idx])[:, 1]
        for j, idx in enumerate(te_idx):
            sig = 1 if proba[j] > 0.55 else (-1 if proba[j] < 0.45 else 0)
            weekly_signals.iloc[idx] = sig

    # Map weekly signals to daily (forward fill)
    daily_signal = pd.Series(np.nan, index=closes.index)
    for wdate, sig in weekly_signals.dropna().items():
        mask = (closes.index >= wdate) & (closes.index < wdate + pd.Timedelta(days=7))
        daily_signal[mask] = sig

    return daily_signal.fillna(0).shift(1)


# ── Core evaluation engine ────────────────────────────────────────────────────

def _evaluate_signals_historically(
    closes: pd.Series,
    signals: pd.Series,
    ticker: str,
    strategy_name: str,
    eval_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    For each signal in the evaluation window, compute outcomes at all horizons.
    Returns a DataFrame of all signal-outcome pairs.
    """
    ret = closes.pct_change()

    # Only evaluate signals within the evaluation window
    eval_signals = signals[signals.index >= eval_start].dropna()

    rows = []
    for date, sig in eval_signals.items():
        if sig == 0:
            continue
        if date not in closes.index:
            continue

        entry_price = float(closes.loc[date])
        action = "BUY" if sig > 0 else "SHORT"

        row = {
            "date":          date.strftime("%Y-%m-%d"),
            "ticker":        ticker,
            "strategy":      strategy_name,
            "action":        action,
            "entry_price":   entry_price,
        }

        # Compute outcome at each horizon
        for h in OUTCOME_HORIZONS:
            future_dates = closes.index[closes.index > date]
            if len(future_dates) < h:
                row[f"ret_{h}d"] = np.nan
                row[f"correct_{h}d"] = np.nan
            else:
                future_date  = future_dates[h - 1]
                future_price = float(closes.loc[future_date])
                pct_ret      = (future_price - entry_price) / entry_price * 100
                if action == "SHORT":
                    pct_ret = -pct_ret   # SHORT profits from price falling
                row[f"ret_{h}d"]     = round(pct_ret, 3)
                row[f"correct_{h}d"] = int(pct_ret > 0)

        rows.append(row)

    return pd.DataFrame(rows)


def _compute_significance(returns: pd.Series) -> dict:
    """T-test: is the mean return significantly different from zero?"""
    if len(returns) < 5:
        return {"t_stat": None, "p_value": None, "significant": False}
    t_stat, p_value = scipy_stats.ttest_1samp(returns.dropna(), 0)
    return {
        "t_stat":      round(float(t_stat), 3),
        "p_value":     round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
    }


def _save_simulated_signals(df: pd.DataFrame):
    """Save simulated signals to signal_history.db so Performance tab can evaluate them."""
    if df.empty:
        return
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR IGNORE INTO signals
                (timestamp, source, ticker, action, confidence, urgency,
                 market_impact, time_horizon, reasoning, article_title,
                 article_url, source_feed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["date"] + " 09:30:00",
                "auto_evaluator",
                row["ticker"],
                row["action"],
                "MEDIUM",
                "MEDIUM",
                "BULLISH" if row["action"] == "BUY" else "BEARISH",
                "SWING (1-5 days)",
                f"Simulated {row['strategy']} signal",
                f"Auto-eval: {row['strategy']}",
                "",
                "historical_simulation",
            ))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── Display helpers ───────────────────────────────────────────────────────────

def _accuracy_badge(acc: float, n: int) -> str:
    if n < 5:
        return f"<span style='color:{MUTED};font-size:0.8rem;'>n={n} (too few)</span>"
    color = GREEN if acc >= 60 else (AMBER if acc >= 50 else RED)
    label = "Strong" if acc >= 60 else ("Marginal" if acc >= 50 else "Weak")
    return f"<span style='color:{color};font-weight:700;'>{acc:.1f}% ({label}, n={n})</span>"


def _render_summary_table(all_results: pd.DataFrame, horizon: int):
    """Render accuracy/return summary grouped by strategy × ticker."""
    col_name = f"correct_{horizon}d"
    ret_col  = f"ret_{horizon}d"

    if col_name not in all_results.columns:
        return

    grp = all_results.dropna(subset=[col_name]).groupby(["strategy", "ticker"]).agg(
        signals    = (col_name, "count"),
        accuracy   = (col_name, "mean"),
        avg_return = (ret_col,  "mean"),
        std_return = (ret_col,  "std"),
    ).reset_index()
    grp["accuracy"]   = grp["accuracy"]   * 100
    grp["avg_return"] = grp["avg_return"].round(2)

    if grp.empty:
        st.info("Not enough evaluated signals yet.")
        return

    def c_acc(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v >= 60 else (f"color:{AMBER}" if v >= 50 else f"color:{RED}")
    def c_ret(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v > 0 else f"color:{RED}"

    styler = grp.style
    if "accuracy"   in grp.columns: styler = styler.map(c_acc, subset=["accuracy"])
    if "avg_return" in grp.columns: styler = styler.map(c_ret, subset=["avg_return"])
    styler = styler.format({
        "accuracy":   "{:.1f}%",
        "avg_return": "{:+.2f}%",
        "std_return": "{:.2f}%",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _render_heatmap(all_results: pd.DataFrame):
    """Accuracy heatmap: strategies (rows) × horizons (cols)."""
    rows = []
    for strat in all_results["strategy"].unique():
        row_data = {"Strategy": strat}
        for h in OUTCOME_HORIZONS:
            col = f"correct_{h}d"
            sub = all_results[all_results["strategy"] == strat][col].dropna()
            if len(sub) >= 5:
                row_data[f"{h}d"] = sub.mean() * 100
            else:
                row_data[f"{h}d"] = np.nan
        rows.append(row_data)

    if not rows:
        return

    hm_df = pd.DataFrame(rows).set_index("Strategy")
    h_cols = [c for c in hm_df.columns if hm_df[c].notna().any()]
    if not h_cols:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.8)))
    _style_fig(fig, ax)

    data    = hm_df[h_cols].values.astype(float)
    im      = ax.imshow(data, cmap="RdYlGn", vmin=40, vmax=70, aspect="auto")

    ax.set_xticks(range(len(h_cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([f"{c} out" for c in h_cols], color=TEXT, fontsize=8)
    ax.set_yticklabels([r["Strategy"] for r in rows], color=TEXT, fontsize=8)
    ax.set_title("Accuracy % by Strategy × Time Horizon", color=CYAN, fontsize=10)

    for i in range(len(rows)):
        for j in range(len(h_cols)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color="white" if val < 45 or val > 65 else "#1a1a1a", fontsize=9,
                        fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_equity_curves(all_results: pd.DataFrame, closes_map: dict, horizon: int = 5):
    """Plot equity curves for each strategy × ticker combo."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _style_fig(fig, ax)

    palette = [BLUE, AMBER, GREEN, RED, PURPLE, CYAN, "#fb923c", "#a3e635"]
    idx = 0

    for strat in all_results["strategy"].unique():
        for ticker in all_results["ticker"].unique():
            sub = all_results[
                (all_results["strategy"] == strat) &
                (all_results["ticker"]   == ticker)
            ].copy()

            ret_col = f"ret_{horizon}d"
            sub = sub.dropna(subset=[ret_col])
            if len(sub) < 3:
                continue

            sub["date_dt"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date_dt")
            cum  = (1 + sub[ret_col] / 100).cumprod()

            label = f"{ticker} · {strat}"
            ax.plot(sub["date_dt"], (cum - 1) * 100,
                    color=palette[idx % len(palette)],
                    linewidth=1.5, label=label, alpha=0.85)
            idx += 1

    # SPY benchmark
    for ticker, closes in closes_map.items():
        if ticker == "SPY":
            bh = closes.pct_change().dropna()
            eval_start = all_results["date"].min() if not all_results.empty else None
            if eval_start:
                bh = bh[bh.index >= pd.Timestamp(eval_start)]
                cum_bh = (1 + bh).cumprod()
                ax.plot(cum_bh.index, (cum_bh - 1) * 100,
                        color=MUTED, linewidth=1.2, linestyle=":",
                        label="SPY B&H", alpha=0.7)
            break

    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_title(f"Simulated Equity Curves ({horizon}-day hold)", color=CYAN)
    ax.set_ylabel("Cumulative Return %", color=MUTED)
    ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
              loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=20, fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_signal_calendar(all_results: pd.DataFrame):
    """Daily signal count chart coloured by net accuracy."""
    if all_results.empty:
        return

    all_results = all_results.copy()
    all_results["date_dt"] = pd.to_datetime(all_results["date"])
    col = "correct_5d"
    if col not in all_results.columns:
        return

    daily = all_results.dropna(subset=[col]).groupby("date_dt").agg(
        signals  = (col, "count"),
        accuracy = (col, "mean")
    ).reset_index()
    daily["accuracy"] *= 100

    fig, ax = plt.subplots(figsize=(12, 2.5))
    _style_fig(fig, ax)

    bar_colors = [GREEN if a >= 55 else (AMBER if a >= 50 else RED)
                  for a in daily["accuracy"]]
    ax.bar(daily["date_dt"], daily["signals"], color=bar_colors,
           edgecolor=BORDER, linewidth=0.3, width=0.8)
    ax.set_title("Daily Signal Count (green=>55% acc, amber=50-55%, red=<50%)", color=CYAN, fontsize=9)
    ax.set_ylabel("Signals", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=20, fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════

def run_auto_evaluator():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Walk-forward historical simulation — the closest thing to a real 3-week test without waiting.</b><br>
        <span style="color:#6b8fad;font-size:0.88rem;">
        Runs your chosen strategies on real historical data using a strict walk-forward method:
        at each point in time, only past data is used to generate the signal.
        Then it checks what actually happened at 1, 3, 5, 10, and 21-day horizons.
        Statistical significance is tested — you see not just accuracy %, but whether
        that accuracy is likely to be real edge or just luck.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="border-color:#7f1d1d;">
        <b style="color:#f87171;">What this cannot simulate:</b><br>
        <span style="color:#6b8fad;font-size:0.85rem;">
        News-based signals (Live Feed, Ticker Signals) require real-time news archives we don't have access to.
        Those tabs need real calendar time to build a track record.
        This evaluator covers strategy-based signals: MA Crossover, RSI, Momentum, and ML Weekly.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Configuration ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Simulation Setup</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        tickers_raw = st.text_input(
            "Tickers to test (comma-separated)",
            value="QQQ, TSLA, NVDA, BTC-USD, GLD, SPY",
            key="ae_tickers",
            help="The simulation runs all selected strategies on each ticker."
        )
        sim_weeks = st.slider(
            "Simulation window (weeks)", 2, 52, 12,
            help="How many weeks of history to simulate. 12 weeks = ~3 months of signal data."
        )
        training_weeks = st.slider(
            "Training warmup (weeks)", 26, 104, 52,
            help="Weeks of data before the eval window — used to train models and compute MAs."
        )

    with c2:
        st.markdown("**Strategies to include:**")
        use_ma  = st.checkbox("📈 MA Crossover (50/200)", value=True)
        use_rsi = st.checkbox("🔄 RSI Mean Reversion (30/70)", value=True)
        use_mom = st.checkbox("🚀 Momentum (63d)", value=True)
        use_ml  = st.checkbox("🤖 ML Weekly (RF)", value=True)

        st.markdown("**MA parameters:**")
        ma_fast = st.slider("Fast MA", 10, 100, 50, 5, key="ae_fast")
        ma_slow = st.slider("Slow MA", 50, 300, 200, 10, key="ae_slow")

    run_btn = st.button("🚀 Run Full Simulation", key="ae_run",
                        use_container_width=False)

    if not run_btn and not st.session_state.get("ae_results") is not None:
        st.info("Configure your simulation above and click **Run Full Simulation**. "
                "Expected runtime: 30–120 seconds depending on ticker count.")
        return

    if run_btn:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if not tickers:
            st.error("❌ Enter at least one ticker.")
            return

        total_weeks  = training_weeks + sim_weeks
        start_date   = (datetime.now() - timedelta(weeks=total_weeks)).strftime("%Y-%m-%d")
        eval_start   = pd.Timestamp(datetime.now() - timedelta(weeks=sim_weeks))

        # ── Fetch all price data ───────────────────────────────────────────
        with st.spinner("📡 Fetching historical price data…"):
            closes_map = {}
            failed     = []
            prog = st.progress(0, text="Downloading prices…")
            for i, ticker in enumerate(tickers):
                closes = _fetch_prices(ticker, start_date)
                if closes is not None:
                    closes_map[ticker] = closes
                else:
                    failed.append(ticker)
                prog.progress((i + 1) / len(tickers),
                              text=f"Downloaded {i+1}/{len(tickers)}: {ticker}")
            prog.empty()

        if failed:
            for f in failed:
                st.warning(f"⚠️ Skipped {f} — no data")

        if not closes_map:
            st.error("❌ No valid data fetched.")
            return

        st.success(f"✅ Loaded {len(closes_map)} tickers · "
                   f"Training window: {training_weeks}w · "
                   f"Evaluation window: {sim_weeks}w "
                   f"({eval_start.strftime('%Y-%m-%d')} → today)")

        # ── Generate signals for all strategy × ticker combos ─────────────
        all_results_list = []
        strategies_used  = []

        strategy_progress = st.progress(0, text="Generating signals…")
        step = 0
        total_steps = sum([use_ma, use_rsi, use_mom, use_ml]) * len(closes_map)

        for ticker, closes in closes_map.items():
            ret = closes.pct_change()

            if use_ma:
                name = f"MA {ma_fast}/{ma_slow}"
                sigs = _generate_ma_signals(closes, ma_fast, ma_slow)
                df   = _evaluate_signals_historically(closes, sigs, ticker, name, eval_start)
                if not df.empty:
                    all_results_list.append(df)
                    if name not in strategies_used:
                        strategies_used.append(name)
                step += 1
                strategy_progress.progress(step / total_steps,
                                           text=f"MA signals: {ticker}")

            if use_rsi:
                name = "RSI 30/70"
                sigs = _generate_rsi_signals(closes, 14, 30, 70)
                df   = _evaluate_signals_historically(closes, sigs, ticker, name, eval_start)
                if not df.empty:
                    all_results_list.append(df)
                    if name not in strategies_used:
                        strategies_used.append(name)
                step += 1
                strategy_progress.progress(step / total_steps,
                                           text=f"RSI signals: {ticker}")

            if use_mom:
                name = "Momentum 63d"
                sigs = _generate_momentum_signals(closes, 63)
                df   = _evaluate_signals_historically(closes, sigs, ticker, name, eval_start)
                if not df.empty:
                    all_results_list.append(df)
                    if name not in strategies_used:
                        strategies_used.append(name)
                step += 1
                strategy_progress.progress(step / total_steps,
                                           text=f"Momentum signals: {ticker}")

            if use_ml:
                name = "ML Weekly RF"
                sigs = _generate_ml_signals(closes, n_splits=4, n_estimators=100)
                df   = _evaluate_signals_historically(closes, sigs, ticker, name, eval_start)
                if not df.empty:
                    all_results_list.append(df)
                    if name not in strategies_used:
                        strategies_used.append(name)
                step += 1
                strategy_progress.progress(step / total_steps,
                                           text=f"ML signals: {ticker}")

        strategy_progress.empty()

        if not all_results_list:
            st.error("❌ No signals generated. Try a longer simulation window or different tickers.")
            return

        all_results = pd.concat(all_results_list, ignore_index=True)

        # Save simulated signals to database
        _save_simulated_signals(all_results)

        st.session_state["ae_results"]   = all_results
        st.session_state["ae_closes"]    = closes_map
        st.session_state["ae_eval_start"]= eval_start
        st.session_state["ae_sim_weeks"] = sim_weeks

    # ── Display results ────────────────────────────────────────────────────
    all_results = st.session_state.get("ae_results", pd.DataFrame())
    closes_map  = st.session_state.get("ae_closes", {})

    if all_results.empty:
        st.warning("No results to display.")
        return

    total_signals = len(all_results)
    tickers_used  = all_results["ticker"].unique().tolist()
    strats_used   = all_results["strategy"].unique().tolist()

    st.markdown(f'<div class="section-header">📊 Results — {total_signals:,} signals evaluated across {len(tickers_used)} tickers × {len(strats_used)} strategies</div>', unsafe_allow_html=True)

    # ── Top-line accuracy by horizon ───────────────────────────────────────
    st.markdown("**Overall accuracy across all signals, by time horizon:**")
    h_cols = st.columns(len(OUTCOME_HORIZONS))
    for i, h in enumerate(OUTCOME_HORIZONS):
        col = f"correct_{h}d"
        sub = all_results[col].dropna()
        with h_cols[i]:
            if len(sub) >= 5:
                acc   = sub.mean() * 100
                ac    = GREEN if acc >= 55 else (AMBER if acc >= 50 else RED)
                sig   = _compute_significance(all_results[f"ret_{h}d"].dropna())
                p_str = f"p={sig['p_value']:.3f} {'✅' if sig['significant'] else '❌'}" if sig["p_value"] is not None else ""
                st.markdown(f"""
                <div class="metric-box">
                    <div class="label">{h}-Day Hold</div>
                    <div class="value" style="color:{ac};">{acc:.1f}%</div>
                    <div style="color:#6b8fad;font-size:0.7rem;">{p_str}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-box"><div class="label">{h}d Hold</div><div class="value" style="color:{MUTED};">N/A</div></div>', unsafe_allow_html=True)

    st.caption("✅ p<0.05 = statistically significant (less than 5% chance of being random luck)")
    st.markdown("")

    # ── Accuracy heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy Heatmap — Strategy × Time Horizon</div>', unsafe_allow_html=True)
    st.caption("Green = above 55% (meaningful edge), Yellow = 50-55% (marginal), Red = below 50% (no edge)")
    _render_heatmap(all_results)

    # ── Best horizon selector ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Detailed Results by Horizon</div>', unsafe_allow_html=True)
    selected_horizon = st.select_slider(
        "Show results for this outcome window:",
        options=OUTCOME_HORIZONS,
        value=5,
        format_func=lambda x: f"{x}-day hold",
        key="ae_horizon"
    )
    _render_summary_table(all_results, selected_horizon)

    # ── Equity curves ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Simulated Equity Curves</div>', unsafe_allow_html=True)
    _render_equity_curves(all_results, closes_map, selected_horizon)

    # ── Signal calendar ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Daily Signal Activity</div>', unsafe_allow_html=True)
    _render_signal_calendar(all_results)

    # ── Statistical significance by strategy ──────────────────────────────
    st.markdown('<div class="section-header">🔬 Statistical Significance</div>', unsafe_allow_html=True)
    st.caption("p < 0.05 means there is less than 5% chance the edge is random. This is the key test.")

    sig_rows = []
    for strat in all_results["strategy"].unique():
        sub_strat = all_results[all_results["strategy"] == strat]
        for h in [3, 5, 10]:
            ret_col = f"ret_{h}d"
            rets    = sub_strat[ret_col].dropna()
            if len(rets) < 5:
                continue
            sig = _compute_significance(rets)
            sig_rows.append({
                "Strategy":       strat,
                "Horizon":        f"{h}d",
                "Signals":        len(rets),
                "Avg Return":     f"{rets.mean():+.2f}%",
                "T-Statistic":    f"{sig['t_stat']:.3f}" if sig["t_stat"] else "N/A",
                "P-Value":        f"{sig['p_value']:.4f}" if sig["p_value"] else "N/A",
                "Significant?":   "✅ YES" if sig["significant"] else "❌ No",
            })

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        def c_sig(v): return f"color:{GREEN}" if "YES" in str(v) else f"color:{RED}"
        st.dataframe(
            sig_df.style.map(c_sig, subset=["Significant?"]),
            use_container_width=True, hide_index=True
        )

    # ── Best signals found ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Best Performing Combinations Found</div>', unsafe_allow_html=True)

    combos = []
    for strat in all_results["strategy"].unique():
        for ticker in all_results["ticker"].unique():
            for h in [3, 5, 10]:
                col = f"correct_{h}d"
                ret_col = f"ret_{h}d"
                sub = all_results[
                    (all_results["strategy"] == strat) &
                    (all_results["ticker"]   == ticker)
                ][col].dropna()
                rets = all_results[
                    (all_results["strategy"] == strat) &
                    (all_results["ticker"]   == ticker)
                ][ret_col].dropna()
                if len(sub) < 5:
                    continue
                sig = _compute_significance(rets)
                combos.append({
                    "Ticker":       ticker,
                    "Strategy":     strat,
                    "Horizon":      f"{h}d",
                    "Signals":      len(sub),
                    "Accuracy":     sub.mean() * 100,
                    "Avg Return":   rets.mean(),
                    "p-Value":      sig["p_value"] if sig["p_value"] else 1.0,
                    "Significant":  sig["significant"],
                })

    if combos:
        combos_df = pd.DataFrame(combos)
        # Rank by: significant AND accuracy
        best = combos_df[combos_df["Significant"] == True].sort_values(
            "Accuracy", ascending=False
        ).head(10)

        if best.empty:
            st.info("No statistically significant combinations found yet. Try a longer simulation window.")
            # Show best non-significant
            best = combos_df.sort_values("Accuracy", ascending=False).head(5)
            st.caption("Showing top 5 by accuracy (none are statistically significant yet):")
        else:
            st.caption(f"Found {len(best)} statistically significant combinations. "
                       f"These are your most tradeable setups:")

        def c_acc_v(v): return f"color:{GREEN}" if v >= 60 else (f"color:{AMBER}" if v >= 50 else f"color:{RED}")
        def c_ret_v(v): return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        def c_sig_v(v): return f"color:{GREEN}" if v else f"color:{MUTED}"

        display_best = best[["Ticker","Strategy","Horizon","Signals","Accuracy","Avg Return","p-Value","Significant"]].copy()
        styler = display_best.style
        if "Accuracy"   in display_best: styler = styler.map(c_acc_v, subset=["Accuracy"])
        if "Avg Return" in display_best: styler = styler.map(c_ret_v, subset=["Avg Return"])
        if "Significant"in display_best: styler = styler.map(c_sig_v, subset=["Significant"])
        styler = styler.format({
            "Accuracy":   "{:.1f}%",
            "Avg Return": "{:+.2f}%",
            "p-Value":    "{:.4f}",
        })
        st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Interpretation ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 How to Read These Results</div>', unsafe_allow_html=True)
    st.markdown("""
    **Accuracy %** — what % of signals pointed in the right direction.
    - Above 60%: strong, tradeable edge
    - 55–60%: meaningful edge, size positions accordingly
    - 50–55%: marginal, don't rely on this alone
    - Below 50%: no edge (or inverse signal!)

    **P-Value** — statistical significance test.
    - p < 0.05 means there is less than 5% chance the accuracy is random luck
    - p < 0.01 is even stronger
    - A high accuracy % with p > 0.05 means you don't have enough signals yet to be confident

    **Avg Return** — average % gain/loss per signal at the given horizon.
    - Positive avg return + significant p-value = the combination most worth trading

    **What to do with this:**
    Focus on the combinations in the "Best Performing Combinations" table that are both
    statistically significant AND show positive average return.
    Use those specific (strategy, ticker, horizon) combinations in the **Live Strategy** tab
    to generate today's forward signal.
    """)

    st.markdown(f"""
    <div style="margin-top:1rem;padding:1rem;border:1px solid #1e3a5f;border-radius:10px;
                color:#6b8fad;font-size:0.8rem;">
        ⚠️ <b>Disclaimer:</b> Historical simulation does not guarantee future performance.
        Market regimes change — a strategy that worked well in the past may not work in the future.
        Always paper trade before committing real capital. Position size according to your risk tolerance.
        These results are based on {total_signals:,} simulated signals across {len(tickers_used)} tickers
        from {st.session_state.get('ae_eval_start', pd.Timestamp.now()).strftime('%Y-%m-%d')} to today.
    </div>
    """, unsafe_allow_html=True)

    # Download
    csv = all_results.to_csv(index=False)
    st.download_button(
        "⬇️ Download full simulation data as CSV",
        data=csv,
        file_name=f"simulation_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

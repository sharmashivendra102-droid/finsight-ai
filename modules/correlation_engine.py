"""
Correlation Engine
- Downloads price data via yfinance
- Computes pairwise correlations
- Trains RandomForest to predict next-step rolling correlation
- Saves artifacts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations
import pickle
import io

import warnings
warnings.filterwarnings("ignore")


# ── Colour palette matching dark theme ───────────────────────────────────────
BLUE   = "#38bdf8"
CYAN   = "#7dd3fc"
AMBER  = "#fbbf24"
RED    = "#f87171"
GREEN  = "#4ade80"
BG     = "#07111f"
CARD   = "#0d1b2a"
BORDER = "#1e3a5f"
TEXT   = "#c9d8e8"
MUTED  = "#6b8fad"


def _style_fig(fig, ax_list):
    """Apply dark theme to matplotlib figure."""
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)


@st.cache_data(show_spinner=False)
def _download_data(tickers: list, start_date: str) -> pd.DataFrame:
    """Download adjusted close prices, skipping bad tickers."""
    import yfinance as yf

    valid_frames = {}
    failed = []

    prog = st.progress(0, text="Downloading market data…")
    for i, ticker in enumerate(tickers):
        try:
            raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if raw.empty:
                failed.append((ticker, "no data returned"))
                continue

            # yfinance can return MultiIndex or flat columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if "Close" not in raw.columns:
                failed.append((ticker, "missing Close column"))
                continue

            series = raw["Close"].dropna()
            if len(series) < 60:
                failed.append((ticker, f"only {len(series)} rows (need ≥60)"))
                continue

            valid_frames[ticker] = series

        except Exception as e:
            failed.append((ticker, str(e)))

        prog.progress((i + 1) / len(tickers), text=f"Downloaded {i+1}/{len(tickers)} tickers…")

    prog.empty()
    return valid_frames, failed


def run_correlation_analysis(
    tickers: list,
    start_date: str,
    rolling_window: int,
    n_estimators: int,
    train_split: float,
):
    # ── 1. Download data ───────────────────────────────────────────────────
    with st.spinner("📡 Fetching price data from Yahoo Finance…"):
        valid_frames, failed = _download_data(tickers, start_date)

    if failed:
        for ticker, reason in failed:
            st.warning(f"⚠️ Skipped **{ticker}**: {reason}")

    if len(valid_frames) < 2:
        st.error("❌ Need at least 2 valid tickers to compute correlations.")
        return

    # ── 2. Align & combine ─────────────────────────────────────────────────
    price_df = pd.DataFrame(valid_frames).dropna()
    price_df.index = pd.to_datetime(price_df.index)
    price_df.sort_index(inplace=True)

    # Save raw data
    price_df.to_csv("market_data.csv")

    valid_tickers = list(price_df.columns)
    st.success(f"✅ Valid tickers: {', '.join(f'**{t}**' for t in valid_tickers)}")
    st.caption(f"Date range: {price_df.index[0].date()} → {price_df.index[-1].date()}  |  {len(price_df)} trading days")

    # ── 3. Pairwise correlation table ──────────────────────────────────────
    st.markdown('<div class="section-header">Pairwise Correlations</div>', unsafe_allow_html=True)

    pairs = list(combinations(valid_tickers, 2))
    corr_rows = []
    for t1, t2 in pairs:
        c = price_df[t1].corr(price_df[t2])
        corr_rows.append({"Ticker 1": t1, "Ticker 2": t2, "Correlation": round(c, 4)})

    corr_df = pd.DataFrame(corr_rows).sort_values("Correlation", ascending=False)

    # Colour helper
    def _colour_corr(val):
        if val > 0.7:
            return f"color: {GREEN}"
        elif val > 0.3:
            return f"color: {AMBER}"
        elif val > -0.3:
            return f"color: {TEXT}"
        else:
            return f"color: {RED}"

    st.dataframe(
        corr_df.style.map(_colour_corr, subset=["Correlation"]).format({"Correlation": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Heatmap
    if len(valid_tickers) >= 3:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        full_corr = price_df.corr()
        fig_h, ax_h = plt.subplots(figsize=(max(5, len(valid_tickers)), max(4, len(valid_tickers) - 1)))
        _style_fig(fig_h, ax_h)

        import matplotlib.colors as mcolors
        cmap = plt.cm.coolwarm

        im = ax_h.imshow(full_corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax_h.set_xticks(range(len(valid_tickers)))
        ax_h.set_yticks(range(len(valid_tickers)))
        ax_h.set_xticklabels(valid_tickers, rotation=45, ha="right", color=TEXT, fontsize=9)
        ax_h.set_yticklabels(valid_tickers, color=TEXT, fontsize=9)
        ax_h.set_title("Correlation Matrix", color=CYAN, fontsize=11)

        for i in range(len(valid_tickers)):
            for j in range(len(valid_tickers)):
                val = full_corr.iloc[i, j]
                ax_h.text(j, i, f"{val:.2f}", ha="center", va="center",
                          color="white" if abs(val) > 0.5 else TEXT, fontsize=7)

        cb = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    # ── 4. Rolling correlations ────────────────────────────────────────────
    st.markdown('<div class="section-header">Rolling Correlations</div>', unsafe_allow_html=True)

    returns_df = price_df.pct_change().dropna()
    roll_corr_dict = {}

    for t1, t2 in pairs:
        col_name = f"{t1}_{t2}"
        rc = returns_df[t1].rolling(rolling_window).corr(returns_df[t2])
        roll_corr_dict[col_name] = rc

    roll_df = pd.DataFrame(roll_corr_dict).dropna()

    # Plot rolling correlations (up to 6 pairs for readability)
    n_pairs_plot = min(6, len(pairs))
    cols_to_plot = list(roll_df.columns)[:n_pairs_plot]

    palette = [BLUE, AMBER, GREEN, RED, CYAN, "#a78bfa"]
    fig_r, ax_r = plt.subplots(figsize=(12, 5))
    _style_fig(fig_r, ax_r)

    for idx, col in enumerate(cols_to_plot):
        ax_r.plot(roll_df.index, roll_df[col], label=col.replace("_", " / "),
                  color=palette[idx % len(palette)], linewidth=1.5, alpha=0.9)

    ax_r.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_r.set_title(f"{rolling_window}-Day Rolling Correlation", color=CYAN)
    ax_r.set_ylabel("Correlation", color=MUTED)
    ax_r.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER,
                labelcolor=TEXT, framealpha=0.8)
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_r.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    fig_r.tight_layout()
    st.pyplot(fig_r)
    plt.close(fig_r)

    # ── 5. ML — Next-step correlation prediction ───────────────────────────
    st.markdown('<div class="section-header">🤖 AI Correlation Predictor (Random Forest)</div>', unsafe_allow_html=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    if len(pairs) < 1:
        st.info("Need at least one pair for prediction.")
        return

    # Use first pair as target; all pairs as features
    target_col = list(roll_df.columns)[0]
    feature_cols = [c for c in roll_df.columns]

    ml_df = roll_df[feature_cols].copy()
    ml_df["Next_Corr"] = ml_df[target_col].shift(-1)
    ml_df.dropna(inplace=True)

    X = ml_df[feature_cols].values
    y = ml_df["Next_Corr"].values
    dates = ml_df.index

    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    if len(X_train) < 30 or len(X_test) < 10:
        st.warning("⚠️ Insufficient data for ML model (need more history).")
        return

    with st.spinner("Training Random Forest…"):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

    # Save model
    with open("correlation_predictor.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Save training data
    ml_df.to_csv("ai_training_data.csv")

    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">MSE</div>
            <div class="value">{mse:.5f}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">R² Score</div>
            <div class="value">{r2:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Target Pair</div>
            <div class="value" style="font-size:1rem;">{target_col.replace("_", " / ")}</div>
        </div>""", unsafe_allow_html=True)

    # Actual vs Predicted plot
    fig_ml, ax_ml = plt.subplots(figsize=(12, 4))
    _style_fig(fig_ml, ax_ml)
    ax_ml.plot(dates_test, y_test, label="Actual", color=BLUE, linewidth=1.5)
    ax_ml.plot(dates_test, y_pred, label="Predicted", color=AMBER,
               linewidth=1.5, linestyle="--", alpha=0.85)
    ax_ml.fill_between(dates_test, y_test, y_pred, alpha=0.1, color=RED)
    ax_ml.set_title(f"Actual vs Predicted — {target_col.replace('_', ' / ')}", color=CYAN)
    ax_ml.set_ylabel("Correlation", color=MUTED)
    ax_ml.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.8)
    ax_ml.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ml.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig_ml.tight_layout()
    st.pyplot(fig_ml)
    plt.close(fig_ml)

    st.success("✅ Model saved to `correlation_predictor.pkl` · Training data saved to `ai_training_data.csv` · Market data saved to `market_data.csv`")

    # Store results in session for Portfolio Risk tab
    st.session_state["corr_df"] = corr_df
    st.session_state["roll_df"] = roll_df
    st.session_state["valid_tickers"] = valid_tickers

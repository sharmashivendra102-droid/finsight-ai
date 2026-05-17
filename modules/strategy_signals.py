"""
Strategy Live Signals
======================
Takes the same logic from the backtester and applies it to CURRENT market data
to generate actionable forward-looking trade signals.

The user picks a strategy + tickers, and the system tells them:
- Current signal (BUY / SHORT / HOLD)
- Why (the exact indicator values driving it)
- What price level would flip the signal
- Suggested stop-loss and take-profit levels
- How this strategy has performed historically on this ticker (mini backtest)

ADDED: Signal Accuracy Tracker section
- Loads every BUY/SHORT from Supabase (source=strategy_signals)
- Evaluates against real prices via eval_core (horizon-matched, MFE/MAE, black-swan-filtered)
- Shows exit accuracy, directional accuracy, variance, Sharpe, per-strategy/ticker tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
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


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data(ticker: str, years: int = 3) -> pd.Series | None:
    try:
        start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes if len(closes) >= 50 else None
    except Exception:
        return None


# ── Strategy signal computers ──────────────────────────────────────────────

def _ma_signal(closes: pd.Series, fast: int, slow: int) -> dict:
    fast_ma  = closes.rolling(fast).mean()
    slow_ma  = closes.rolling(slow).mean()

    current_price = float(closes.iloc[-1])
    fast_now      = float(fast_ma.iloc[-1])
    slow_now      = float(slow_ma.iloc[-1])
    fast_prev     = float(fast_ma.iloc[-2])
    slow_prev     = float(slow_ma.iloc[-2])

    currently_above = fast_now > slow_now
    was_above       = fast_prev > slow_prev
    just_crossed    = currently_above != was_above

    if currently_above:
        signal    = "BUY"
        color     = GREEN
        rationale = f"{fast}-day MA (${fast_now:.2f}) is above the {slow}-day MA (${slow_now:.2f}) — bullish trend."
        if just_crossed:
            rationale += f" ⚡ **Golden Cross just occurred** — this is a fresh entry signal."
    else:
        signal    = "SHORT"
        color     = RED
        rationale = f"{fast}-day MA (${fast_now:.2f}) is below the {slow}-day MA (${slow_now:.2f}) — bearish trend."
        if just_crossed:
            rationale += f" ⚡ **Death Cross just occurred** — this is a fresh short signal."

    days_in_fast_ma = min(fast, len(closes))
    old_sum  = closes.iloc[-days_in_fast_ma:-1].sum()
    flip_price = slow_now * fast - old_sum
    flip_price = max(0, flip_price)

    dist_from_fast = (current_price - fast_now) / fast_now * 100
    dist_from_slow = (current_price - slow_now) / slow_now * 100

    if signal == "BUY":
        stop_loss   = slow_now * 0.98
        take_profit = current_price * 1.10
    else:
        stop_loss   = slow_now * 1.02
        take_profit = current_price * 0.90

    return {
        "signal":         signal,
        "color":          color,
        "rationale":      rationale,
        "current_price":  current_price,
        "fast_ma":        fast_now,
        "slow_ma":        slow_now,
        "flip_price":     flip_price,
        "stop_loss":      stop_loss,
        "take_profit":    take_profit,
        "just_crossed":   just_crossed,
        "dist_from_fast": dist_from_fast,
        "dist_from_slow": dist_from_slow,
        "fast_series":    fast_ma,
        "slow_series":    slow_ma,
        "closes":         closes,
        "strategy":       f"MA Crossover ({fast}/{slow})",
    }


def _rsi_signal(closes: pd.Series, period: int, oversold: int, overbought: int) -> dict:
    delta = closes.pct_change()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    current_price = float(closes.iloc[-1])
    current_rsi   = float(rsi.iloc[-1])
    prev_rsi      = float(rsi.iloc[-2])

    if current_rsi < oversold:
        signal    = "BUY"
        color     = GREEN
        rationale = (f"RSI is {current_rsi:.1f} — **deeply oversold** (below {oversold}). "
                     f"Mean reversion suggests a bounce is likely.")
        urgency   = "HIGH" if current_rsi < oversold - 5 else "MEDIUM"
    elif current_rsi > overbought:
        signal    = "SHORT"
        color     = RED
        rationale = (f"RSI is {current_rsi:.1f} — **overbought** (above {overbought}). "
                     f"Mean reversion suggests a pullback is likely.")
        urgency   = "HIGH" if current_rsi > overbought + 5 else "MEDIUM"
    else:
        signal    = "HOLD"
        color     = AMBER
        urgency   = "LOW"
        if 45 < current_rsi < 55:
            rationale = (f"RSI is {current_rsi:.1f} — neutral zone. No clear signal. "
                         f"Wait for RSI to reach <{oversold} (BUY) or >{overbought} (SHORT).")
        elif current_rsi < 55:
            rationale = (f"RSI is {current_rsi:.1f} — recovering from oversold but not there yet.")
        else:
            rationale = (f"RSI is {current_rsi:.1f} — elevated but not yet overbought.")

    buy_trigger   = f"RSI drops below {oversold}"
    short_trigger = f"RSI rises above {overbought}"
    exit_trigger  = f"RSI returns to 45–55 neutral zone"

    stop_loss   = current_price * 0.93 if signal == "BUY" else current_price * 1.07
    take_profit = current_price * 1.12 if signal == "BUY" else current_price * 0.88

    return {
        "signal":           signal,
        "color":            color,
        "rationale":        rationale,
        "urgency":          urgency,
        "current_price":    current_price,
        "rsi":              current_rsi,
        "prev_rsi":         prev_rsi,
        "rsi_trend":        "rising" if current_rsi > prev_rsi else "falling",
        "buy_trigger":      buy_trigger,
        "short_trigger":    short_trigger,
        "exit_trigger":     exit_trigger,
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "rsi_series":       rsi,
        "closes":           closes,
        "strategy":         f"RSI Mean Reversion ({oversold}/{overbought})",
    }


def _momentum_signal(closes: pd.Series, lookback: int, hold_period: int) -> dict:
    ret      = closes.pct_change()
    momentum = closes.pct_change(lookback)

    current_price    = float(closes.iloc[-1])
    current_momentum = float(momentum.iloc[-1])
    prev_momentum    = float(momentum.iloc[-2])

    hist_mom  = momentum.dropna()
    percentile = float((hist_mom < current_momentum).mean() * 100)

    if current_momentum > 0.02:
        signal    = "BUY"
        color     = GREEN
        strength  = "strong" if current_momentum > 0.10 else "moderate"
        rationale = (f"{lookback}-day momentum is **+{current_momentum*100:.1f}%** ({strength} positive trend). "
                     f"This is in the {percentile:.0f}th percentile of historical readings.")
    elif current_momentum < -0.02:
        signal    = "SHORT"
        color     = RED
        strength  = "strong" if current_momentum < -0.10 else "moderate"
        rationale = (f"{lookback}-day momentum is **{current_momentum*100:.1f}%** ({strength} negative trend). "
                     f"This is in the {percentile:.0f}th percentile of historical readings.")
    else:
        signal    = "HOLD"
        color     = AMBER
        rationale = (f"{lookback}-day momentum is {current_momentum*100:.1f}% — "
                     f"too close to zero for a clear signal.")

    stop_loss   = current_price * 0.95 if signal == "BUY" else current_price * 1.05
    take_profit = current_price * 1.15 if signal == "BUY" else current_price * 0.85

    return {
        "signal":             signal,
        "color":              color,
        "rationale":          rationale,
        "current_price":      current_price,
        "momentum_pct":       current_momentum * 100,
        "momentum_percentile": percentile,
        "hold_period":        hold_period,
        "stop_loss":          stop_loss,
        "take_profit":        take_profit,
        "closes":             closes,
        "momentum_series":    momentum,
        "strategy":           f"Momentum ({lookback}d lookback)",
    }


def _ml_weekly_signal(closes: pd.Series, feature_closes: dict, n_estimators: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    all_closes = {"target": closes}
    all_closes.update(feature_closes)
    weekly = pd.DataFrame({k: v.resample("W").last() for k, v in all_closes.items()}).dropna()

    if len(weekly) < 40:
        return {"error": "Need more data. Try a longer date range or different tickers."}

    ret_w = weekly.pct_change()
    feat  = pd.DataFrame(index=weekly.index)
    for col in weekly.columns:
        r = ret_w[col]
        feat[f"{col}_w1"]   = r.shift(1)
        feat[f"{col}_w2"]   = r.shift(2)
        feat[f"{col}_w3"]   = r.shift(3)
        feat[f"{col}_w4"]   = r.shift(4)
        feat[f"{col}_mom4"] = r.shift(1).rolling(4).mean()
        feat[f"{col}_vol4"] = r.shift(1).rolling(4).std()

    feat["_target"] = (ret_w["target"].shift(-1) > 0).astype(int)
    feat.dropna(inplace=True)

    if len(feat) < 30:
        return {"error": "Insufficient data after feature construction."}

    feature_cols = [c for c in feat.columns if not c.startswith("_")]
    X = feat[feature_cols].values
    y = feat["_target"].values

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X, y)

    last_features = feat[feature_cols].iloc[-1].values.reshape(1, -1)
    proba = float(clf.predict_proba(last_features)[0][1])

    current_price = float(closes.iloc[-1])

    if proba > 0.60:
        signal     = "BUY"
        color      = GREEN
        confidence = "HIGH" if proba > 0.70 else "MEDIUM"
        rationale  = (f"Model assigns **{proba*100:.0f}% probability of UP** next week. "
                      f"Confidence: {confidence}.")
    elif proba < 0.40:
        signal     = "SHORT"
        color      = RED
        confidence = "HIGH" if proba < 0.30 else "MEDIUM"
        rationale  = (f"Model assigns **{(1-proba)*100:.0f}% probability of DOWN** next week. "
                      f"Confidence: {confidence}.")
    else:
        signal     = "HOLD"
        color      = AMBER
        confidence = "LOW"
        rationale  = (f"Model probability is {proba*100:.0f}% — too close to 50% for a confident signal.")

    importances  = pd.Series(clf.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(5)

    stop_loss   = current_price * 0.95 if signal == "BUY" else current_price * 1.05
    take_profit = current_price * 1.10 if signal == "BUY" else current_price * 0.90

    return {
        "signal":        signal,
        "color":         color,
        "confidence":    confidence,
        "rationale":     rationale,
        "proba_up":      proba,
        "current_price": current_price,
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
        "top_features":  top_features,
        "closes":        closes,
        "strategy":      "ML Weekly Direction (RF)",
    }


def _render_signal_card(ticker: str, result: dict):
    signal  = result.get("signal", "HOLD")
    color   = result.get("color", AMBER)
    current = result.get("current_price", 0)
    sl      = result.get("stop_loss", 0)
    tp      = result.get("take_profit", 0)

    emoji  = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️"}.get(signal, "➡️")
    risk   = abs(current - sl) / current * 100 if current > 0 else 0
    reward = abs(tp - current) / current * 100 if current > 0 else 0
    rr     = reward / risk if risk > 0 else 0

    st.markdown("---")

    s1, s2, s3 = st.columns([2, 2, 3])
    with s1:
        st.markdown(f"""
        <div style="background:{color}18;border:2px solid {color}44;border-radius:12px;
                    padding:1rem;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:2.5rem;
                        font-weight:700;color:{color};">{signal}</div>
            <div style="color:#c9d8e8;font-size:1.1rem;font-weight:600;">`{ticker}`</div>
            <div style="color:#6b8fad;font-size:0.8rem;">{result.get('strategy','')}</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        rr_color = GREEN if rr >= 2 else (AMBER if rr >= 1 else RED)
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:0.4rem;">
            <div class="label">Current Price</div>
            <div class="value">${current:.2f}</div>
        </div>
        <div class="metric-box" style="margin-bottom:0.4rem;">
            <div class="label">Stop Loss</div>
            <div class="value" style="color:{RED};">${sl:.2f} <span style="font-size:0.8rem;">(-{risk:.1f}%)</span></div>
        </div>
        <div class="metric-box">
            <div class="label">Take Profit</div>
            <div class="value" style="color:{GREEN};">${tp:.2f} <span style="font-size:0.8rem;">(+{reward:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;
                    padding:1rem;height:100%;">
            <div style="color:{CYAN};font-weight:700;font-size:0.85rem;margin-bottom:0.5rem;">
                📋 Signal Rationale
            </div>
            <div style="color:#c9d8e8;font-size:0.88rem;line-height:1.6;">
                {result.get('rationale','')}
            </div>
            <div style="margin-top:0.8rem;color:#6b8fad;font-size:0.78rem;">
                Risk/Reward Ratio:
                <span style="color:{rr_color};font-weight:700;">{rr:.1f}:1</span>
                {"✅ Good" if rr >= 2 else ("⚠️ Marginal" if rr >= 1 else "❌ Poor")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # MA-specific
    if "fast_ma" in result:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.markdown(f'<div class="metric-box"><div class="label">Fast MA</div><div class="value" style="font-size:1.1rem;">${result["fast_ma"]:.2f}</div></div>', unsafe_allow_html=True)
        with d2: st.markdown(f'<div class="metric-box"><div class="label">Slow MA</div><div class="value" style="font-size:1.1rem;">${result["slow_ma"]:.2f}</div></div>', unsafe_allow_html=True)
        with d3: st.markdown(f'<div class="metric-box"><div class="label">Signal Flip At</div><div class="value" style="font-size:1.1rem;color:{AMBER};">${result["flip_price"]:.2f}</div></div>', unsafe_allow_html=True)
        with d4:
            jc = result.get("just_crossed", False)
            st.markdown(f'<div class="metric-box"><div class="label">Just Crossed?</div><div class="value" style="font-size:1rem;color:{"#4ade80" if jc else "#6b8fad"};">{"YES ⚡" if jc else "No"}</div></div>', unsafe_allow_html=True)

    # RSI-specific
    if "rsi" in result:
        rsi_val   = result["rsi"]
        rsi_color = RED if rsi_val > 70 else (GREEN if rsi_val < 30 else MUTED)
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.markdown(f'<div class="metric-box"><div class="label">Current RSI</div><div class="value" style="color:{rsi_color};">{rsi_val:.1f}</div></div>', unsafe_allow_html=True)
        with d2: st.markdown(f'<div class="metric-box"><div class="label">RSI Trend</div><div class="value" style="font-size:1rem;">{"📈 Rising" if result["rsi_trend"]=="rising" else "📉 Falling"}</div></div>', unsafe_allow_html=True)
        with d3: st.markdown(f'<div class="metric-box"><div class="label">BUY Triggers At</div><div class="value" style="font-size:0.9rem;color:{GREEN};">{result["buy_trigger"]}</div></div>', unsafe_allow_html=True)
        with d4: st.markdown(f'<div class="metric-box"><div class="label">SHORT Triggers At</div><div class="value" style="font-size:0.9rem;color:{RED};">{result["short_trigger"]}</div></div>', unsafe_allow_html=True)

    # Momentum-specific
    if "momentum_pct" in result:
        mom = result["momentum_pct"]
        mc  = GREEN if mom > 0 else RED
        d1, d2, d3 = st.columns(3)
        with d1: st.markdown(f'<div class="metric-box"><div class="label">Momentum</div><div class="value" style="color:{mc};">{mom:+.1f}%</div></div>', unsafe_allow_html=True)
        with d2: st.markdown(f'<div class="metric-box"><div class="label">Historical Percentile</div><div class="value">{result["momentum_percentile"]:.0f}th</div></div>', unsafe_allow_html=True)
        with d3: st.markdown(f'<div class="metric-box"><div class="label">Suggested Hold</div><div class="value" style="font-size:1rem;">{result["hold_period"]} days</div></div>', unsafe_allow_html=True)

    # ML-specific
    if "proba_up" in result:
        proba = result["proba_up"]
        pc    = GREEN if proba > 0.6 else (RED if proba < 0.4 else AMBER)
        d1, d2, d3 = st.columns(3)
        with d1: st.markdown(f'<div class="metric-box"><div class="label">P(Up Next Week)</div><div class="value" style="color:{pc};">{proba*100:.0f}%</div></div>', unsafe_allow_html=True)
        with d2: st.markdown(f'<div class="metric-box"><div class="label">P(Down Next Week)</div><div class="value" style="color:{RED if proba<0.4 else MUTED};">{(1-proba)*100:.0f}%</div></div>', unsafe_allow_html=True)
        with d3: st.markdown(f'<div class="metric-box"><div class="label">Model Confidence</div><div class="value">{result.get("confidence","LOW")}</div></div>', unsafe_allow_html=True)

        top_feat = result.get("top_features")
        if top_feat is not None:
            st.markdown("**🔍 Top features driving this prediction:**")
            feat_cols = st.columns(5)
            for i, (feat_name, imp) in enumerate(top_feat.items()):
                with feat_cols[i]:
                    st.metric(feat_name.replace("_", " "), f"{imp*100:.1f}%")

    # Price chart
    closes = result.get("closes")
    if closes is not None:
        display_closes = closes.tail(252)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                  gridspec_kw={"height_ratios": [3, 1]})
        _style_fig(fig, list(axes))

        ax1, ax2 = axes

        ax1.plot(display_closes.index, display_closes.values,
                 color=BLUE, linewidth=1.5, label="Price")

        if "fast_series" in result:
            fs = result["fast_series"].tail(252)
            ss = result["slow_series"].tail(252)
            ax1.plot(fs.index, fs.values, color=AMBER, linewidth=1, linestyle="--",
                     label="Fast MA", alpha=0.8)
            ax1.plot(ss.index, ss.values, color=RED, linewidth=1, linestyle="--",
                     label="Slow MA", alpha=0.8)

        if signal in ("BUY", "SHORT"):
            ax1.axhline(sl, color=RED,   linewidth=0.8, linestyle=":", alpha=0.7, label=f"Stop ${sl:.2f}")
            ax1.axhline(tp, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label=f"Target ${tp:.2f}")

        ax1.axhline(float(closes.iloc[-1]), color=CYAN, linewidth=1, alpha=0.5)
        ax1.set_title(f"{ticker} — {result.get('strategy','')} · Current Signal: {signal}", color=CYAN)
        ax1.set_ylabel("Price ($)", color=MUTED)
        ax1.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        if "rsi_series" in result:
            rsi_disp = result["rsi_series"].tail(252)
            ax2.plot(rsi_disp.index, rsi_disp.values, color=PURPLE, linewidth=1.2, label="RSI")
            ax2.axhline(70, color=RED,   linewidth=0.7, linestyle="--", alpha=0.6)
            ax2.axhline(30, color=GREEN, linewidth=0.7, linestyle="--", alpha=0.6)
            ax2.axhline(50, color=MUTED, linewidth=0.5, linestyle=":", alpha=0.4)
            ax2.fill_between(rsi_disp.index, rsi_disp.values, 30,
                             where=rsi_disp.values <= 30, alpha=0.15, color=GREEN)
            ax2.fill_between(rsi_disp.index, rsi_disp.values, 70,
                             where=rsi_disp.values >= 70, alpha=0.15, color=RED)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI", color=MUTED)
            ax2.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        elif "momentum_series" in result:
            mom_disp = result["momentum_series"].tail(252) * 100
            mcolors  = [GREEN if v >= 0 else RED for v in mom_disp.values]
            ax2.bar(mom_disp.index, mom_disp.values, color=mcolors, width=2, alpha=0.7)
            ax2.axhline(0,  color=MUTED, linewidth=0.8)
            ax2.axhline(2,  color=GREEN, linewidth=0.6, linestyle="--", alpha=0.5)
            ax2.axhline(-2, color=RED,   linewidth=0.6, linestyle="--", alpha=0.5)
            ax2.set_ylabel("Momentum %", color=MUTED)
        else:
            ax2.set_visible(False)

        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if signal in ("BUY", "SHORT"):
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#122338);
                    border:1px solid {color}33;border-radius:12px;padding:1rem 1.4rem;margin-top:0.5rem;">
            <div style="color:{CYAN};font-weight:700;font-size:0.88rem;margin-bottom:0.6rem;">
                📋 Trade Plan Summary — {ticker}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;font-size:0.83rem;color:#c9d8e8;">
                <div><span style="color:#6b8fad;">Direction:</span> <b style="color:{color};">{signal}</b></div>
                <div><span style="color:#6b8fad;">Entry near:</span> <b>${current:.2f}</b></div>
                <div><span style="color:#6b8fad;">Strategy:</span> <b>{result.get('strategy','')}</b></div>
                <div><span style="color:#6b8fad;">Stop Loss:</span> <b style="color:{RED};">${sl:.2f} (-{risk:.1f}%)</b></div>
                <div><span style="color:#6b8fad;">Take Profit:</span> <b style="color:{GREEN};">${tp:.2f} (+{reward:.1f}%)</b></div>
                <div><span style="color:#6b8fad;">Risk/Reward:</span> <b style="color:{rr_color};">{rr:.1f}:1</b></div>
            </div>
            <div style="margin-top:0.8rem;color:#6b8fad;font-size:0.75rem;">
                ⚠️ Systematic signal — not financial advice. Paper trade first.
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            from modules.signal_history import log_signal
            log_signal(
                source        = "strategy_signals",
                ticker        = ticker,
                action        = signal,
                confidence    = result.get("confidence", "MEDIUM"),
                urgency       = "MEDIUM",
                market_impact = "BULLISH" if signal == "BUY" else "BEARISH",
                time_horizon  = "SWING (1-5 days)" if "RSI" in result.get("strategy","") else "MEDIUM (weeks)",
                reasoning     = result.get("rationale","")[:300],
                article_title = f"Strategy: {result.get('strategy','')}",
                article_url   = "",
                source_feed   = "backtest_strategy",
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ACCURACY TRACKER — loads from Supabase, evaluates with eval_core
# ══════════════════════════════════════════════════════════════════════════════

def _load_strategy_signals_from_supabase() -> pd.DataFrame:
    try:
        from modules.signal_history import get_signals_df
        df = get_signals_df(
            days_back=3650,
            source_filter=["strategy_signals"],
            action_filter=["BUY", "SHORT"],
        )
        return df.sort_values("timestamp").reset_index(drop=True) if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Could not load strategy signals from Supabase: {e}")
        return pd.DataFrame()


def _load_all_signals_for_timeline() -> pd.DataFrame:
    try:
        from modules.signal_history import get_signals_df
        df = get_signals_df(days_back=3650, action_filter=["BUY", "SHORT"])
        return df.sort_values("timestamp").reset_index(drop=True) if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _eval_fetch_price(ticker: str, date_str: str) -> float | None:
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
def _eval_fetch_ohlc(ticker: str, entry_date: str, exit_date: str,
                     lookback_days: int = 60):
    try:
        start = (pd.to_datetime(entry_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end   = (pd.to_datetime(exit_date)  + timedelta(days=3)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist if not hist.empty else None
    except Exception:
        return None


def _extract_strategy_label(row) -> str:
    title   = str(row.get("article_title", "") or "")
    reason  = str(row.get("reasoning", "") or "")
    combined = title + " " + reason
    if "MA Crossover" in combined: return "MA Crossover"
    if "RSI"          in combined: return "RSI Mean Reversion"
    if "Momentum"     in combined: return "Momentum"
    if "ML" in combined or "RF" in combined or "Weekly" in combined: return "ML Weekly"
    return "Strategy Signal"


def _render_strategy_accuracy():
    """
    Full accuracy dashboard for Live Trading Signals stored in Supabase:
      - Exit accuracy, directional accuracy (MFE), variance, Sharpe
      - Per-strategy and per-ticker breakdowns
      - Return distribution, cumulative P&L, rolling accuracy, box plots, MFE scatter
    """
    from modules.eval_core import evaluate_signals
    from scipy import stats as scipy_stats

    st.markdown('<div class="section-header">📊 Signal Accuracy Tracker</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">How accurate are your Live Strategy signals against real market prices?</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Evaluates every BUY/SHORT from this tab stored in Supabase. Uses the same evaluation
        engine as Signal Performance: next-day OPEN entry, horizon-matched exit, intraday
        MFE/MAE for directional accuracy, and automatic black-swan exclusion.
        </span>
    </div>
    """, unsafe_allow_html=True)

    mfe_thresh = st.slider(
        "📐 MFE threshold — min % intraday move to count as directionally correct",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1,
        key="ss_acc_mfe",
    )

    if not st.button("🔬 Evaluate Strategy Signal Accuracy", key="ss_acc_run"):
        cached = st.session_state.get("ss_acc_ready")
        if cached is not None:
            _render_acc_results(
                cached,
                st.session_state.get("ss_acc_waiting", pd.DataFrame()),
                st.session_state.get("ss_acc_inconclusive", pd.DataFrame()),
                mfe_thresh,
            )
        else:
            st.info("Click **Evaluate Strategy Signal Accuracy** to analyse past signals. "
                    "You need signals that have aged past their time horizon.")
        return

    df_raw  = _load_strategy_signals_from_supabase()
    df_full = _load_all_signals_for_timeline()

    if df_raw.empty:
        st.warning("No strategy signals in Supabase yet. Generate some signals above and "
                   "wait for their time horizon to pass (SWING = 5 trading days).")
        return

    prog = st.progress(0, text="Evaluating against real market prices…")
    def _cb(i, total, ticker, date):
        prog.progress((i + 1) / total,
                      text=f"Checking {ticker} — {date} ({i+1}/{total})")

    ready_df, waiting_df, inconclusive_df = evaluate_signals(
        df                       = df_raw,
        fetch_price_fn           = _eval_fetch_price,
        fetch_ohlc_fn            = _eval_fetch_ohlc,
        all_signals_for_timeline = df_full,
        progress_callback        = _cb,
    )
    prog.empty()

    st.session_state["ss_acc_ready"]        = ready_df
    st.session_state["ss_acc_waiting"]      = waiting_df
    st.session_state["ss_acc_inconclusive"] = inconclusive_df

    _render_acc_results(ready_df, waiting_df, inconclusive_df, mfe_thresh)


def _render_acc_results(ready_df: pd.DataFrame, waiting_df: pd.DataFrame,
                         inconclusive_df: pd.DataFrame, mfe_thresh: float):
    from scipy import stats as scipy_stats

    # ── Callouts ───────────────────────────────────────────────────────────
    if not waiting_df.empty:
        with st.expander(f"⏳ {len(waiting_df)} signal(s) not yet evaluable", expanded=False):
            st.caption("Horizon window hasn't elapsed — not counted in accuracy.")
            show = [c for c in ["date","ticker","action","confidence","time_horizon",
                                 "td_elapsed","not_ready_reason"] if c in waiting_df.columns]
            st.dataframe(waiting_df[show], use_container_width=True, hide_index=True)

    if not inconclusive_df.empty:
        with st.expander(f"🌪️ {len(inconclusive_df)} excluded — external shock",
                         expanded=False):
            st.caption("Stock moved 3× normal range. Excluded from accuracy stats.")
            show = [c for c in ["date","ticker","action","return_pct","spike_reason"]
                    if c in inconclusive_df.columns]
            st.dataframe(inconclusive_df[show], use_container_width=True, hide_index=True)

    if ready_df.empty:
        st.warning("No signals have reached their evaluation window yet.")
        return

    n    = len(ready_df)
    corr = int(ready_df["correct"].sum())
    acc  = corr / n * 100

    has_mfe  = "mfe" in ready_df.columns and ready_df["mfe"].notna().any()
    dir_mask = (ready_df["mfe"].notna() & (ready_df["mfe"] >= mfe_thresh)
                if has_mfe else pd.Series(False, index=ready_df.index))
    dir_acc  = dir_mask.mean() * 100 if has_mfe else None
    dir_n    = int(dir_mask.sum())

    rets         = ready_df["return_pct"].dropna()
    avg_ret      = float(rets.mean())  if not rets.empty else 0.0
    std_ret      = float(rets.std())   if len(rets) >= 2 else 0.0
    variance     = std_ret ** 2
    sharpe       = (avg_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0.0
    win_rate     = float((rets > 0).mean() * 100) if not rets.empty else 0.0
    profit_factor= (rets[rets > 0].sum() / abs(rets[rets < 0].sum())
                    if rets[rets < 0].sum() != 0 else float("inf"))
    max_win  = float(rets.max()) if not rets.empty else 0.0
    max_loss = float(rets.min()) if not rets.empty else 0.0

    if len(rets) >= 5:
        t_stat, p_val = scipy_stats.ttest_1samp(rets, 0)
        p_str  = f"p = {p_val:.4f}"
        sig_str = "✅ Statistically significant" if p_val < 0.05 else "❌ Not significant (yet)"
    else:
        p_val, p_str, sig_str = None, "n < 5", "—"

    def _ac(v):  return GREEN if v >= 60 else (AMBER if v >= 50 else RED)
    def _rc(v):  return GREEN if v > 0  else RED

    # ── Big banner ─────────────────────────────────────────────────────────
    dir_color = _ac(dir_acc) if dir_acc is not None else _ac(acc)
    banner_cols = st.columns(4 if dir_acc is not None else 3)

    def _banner_cell(col, title, value, sub, color):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                        border:2px solid {color}44;border-radius:14px;
                        padding:1.1rem 1rem;text-align:center;margin-bottom:.5rem;">
                <div style="color:#6b8fad;font-size:.63rem;font-family:'Space Mono',monospace;
                            letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem;">
                    {title}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:2.2rem;
                            font-weight:700;color:{color};line-height:1.1;">{value}</div>
                <div style="color:#6b8fad;font-size:.68rem;margin-top:.25rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    if dir_acc is not None:
        _banner_cell(banner_cols[0], "Directional Accuracy",
                     f"{dir_acc:.1f}%", f"{dir_n}/{n} · MFE ≥ {mfe_thresh:.1f}%",
                     dir_color)
    _banner_cell(banner_cols[1 if dir_acc is not None else 0],
                 "Exit Accuracy", f"{acc:.1f}%", f"{corr}/{n} correct at horizon",
                 _ac(acc))
    _banner_cell(banner_cols[2 if dir_acc is not None else 1],
                 "Avg Return", f"{avg_ret:+.2f}%", sig_str, _rc(avg_ret))
    _banner_cell(banner_cols[3 if dir_acc is not None else 2],
                 "Std Dev / Variance",
                 f"{std_ret:.2f}%", f"variance = {variance:.3f}", TEXT)

    # ── Metric row ─────────────────────────────────────────────────────────
    st.markdown("")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, lbl, val, clr in [
        (m1, "Win Rate",        f"{win_rate:.1f}%",
         _ac(win_rate)),
        (m2, "Profit Factor",   f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
         GREEN if profit_factor >= 1.5 else (AMBER if profit_factor >= 1 else RED)),
        (m3, "Sharpe (rough)",  f"{sharpe:.2f}",
         GREEN if sharpe >= 1 else (AMBER if sharpe >= 0 else RED)),
        (m4, "Best Signal",     f"+{max_win:.2f}%",  GREEN),
        (m5, "Worst Signal",    f"{max_loss:.2f}%",  RED),
        (m6, "Evaluated",       str(n),              BLUE),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="label">{lbl}</div>'
                f'<div class="value" style="color:{clr};">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
                padding:.6rem 1rem;margin:.4rem 0;font-size:.78rem;color:#8ba3c1;">
        Statistical test: <b>{p_str}</b> — {sig_str}
        {f" · t-stat = {scipy_stats.ttest_1samp(rets, 0)[0]:.3f}" if len(rets)>=5 else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ───────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        # Return distribution + normal fit
        fig, ax = plt.subplots(figsize=(6, 4))
        _style_fig(fig, ax)
        if len(rets) >= 3:
            n_bins = max(8, n // 3)
            _, bins, patches = ax.hist(
                rets, bins=n_bins, color=BLUE,
                edgecolor=BORDER, linewidth=0.4, alpha=0.75, density=True,
            )
            for patch, left in zip(patches, bins[:-1]):
                patch.set_facecolor(GREEN if left >= 0 else RED)
                patch.set_alpha(0.75)
            if std_ret > 0:
                from scipy.stats import norm as _norm
                x_n = np.linspace(float(rets.min()), float(rets.max()), 200)
                ax.plot(x_n, _norm.pdf(x_n, avg_ret, std_ret),
                        color=CYAN, linewidth=2, linestyle="--", label="Normal fit")
            ax.axvline(0,       color=MUTED, linewidth=1,   linestyle="--")
            ax.axvline(avg_ret, color=AMBER, linewidth=1.5, label=f"Mean {avg_ret:+.2f}%")
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.set_title("Return Distribution (density)", color=CYAN, fontsize=9)
        ax.set_xlabel("Signal Return %", color=MUTED, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        # Cumulative P&L
        sorted_df = ready_df.sort_values("date").copy()
        cum = (1 + sorted_df["return_pct"].fillna(0) / 100).cumprod()
        fig, ax = plt.subplots(figsize=(6, 4))
        _style_fig(fig, ax)
        ax.plot(range(len(cum)), (cum - 1) * 100,
                color=BLUE, linewidth=2, label="Cumulative P&L")
        ax.fill_between(range(len(cum)), (cum - 1) * 100, 0,
                        where=(cum >= 1), alpha=0.08, color=GREEN)
        ax.fill_between(range(len(cum)), (cum - 1) * 100, 0,
                        where=(cum < 1),  alpha=0.08, color=RED)
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title("Cumulative P&L (chronological)", color=CYAN, fontsize=9)
        ax.set_xlabel("Signal #", color=MUTED, fontsize=8)
        ax.set_ylabel("Cumulative Return %", color=MUTED, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Rolling accuracy ───────────────────────────────────────────────────
    if n >= 8:
        st.markdown('<div class="section-header">📅 Rolling Accuracy</div>',
                    unsafe_allow_html=True)
        rv = ready_df.copy()
        rv["date_dt"] = pd.to_datetime(rv["date"])
        rv = rv.sort_values("date_dt")
        window = min(10, n)
        rv["rolling_acc"] = rv["correct"].rolling(window, min_periods=3).mean() * 100

        fig, ax = plt.subplots(figsize=(12, 3))
        _style_fig(fig, ax)
        ax.plot(rv["date_dt"], rv["rolling_acc"], color=BLUE, linewidth=1.8)
        ax.fill_between(rv["date_dt"], rv["rolling_acc"], 50,
                        where=rv["rolling_acc"] >= 50, alpha=0.10, color=GREEN)
        ax.fill_between(rv["date_dt"], rv["rolling_acc"], 50,
                        where=rv["rolling_acc"] < 50,  alpha=0.10, color=RED)
        ax.axhline(50, color=MUTED, linewidth=1,   linestyle="--", label="50% baseline")
        ax.axhline(60, color=GREEN, linewidth=0.8, linestyle=":",  label="60% target")
        ax.set_ylim(10, 95)
        ax.set_ylabel("Accuracy %", color=MUTED, fontsize=8)
        ax.set_title(f"Rolling {window}-signal Exit Accuracy", color=CYAN, fontsize=9)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=20, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Per-strategy breakdown ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Accuracy by Strategy</div>',
                unsafe_allow_html=True)

    rdf = ready_df.copy()
    rdf["strategy_label"] = rdf.apply(_extract_strategy_label, axis=1)
    if has_mfe:
        rdf["_dir"] = rdf["mfe"].notna() & (rdf["mfe"] >= mfe_thresh)

    strat_rows = []
    for strat in sorted(rdf["strategy_label"].unique()):
        sub = rdf[rdf["strategy_label"] == strat]
        sr  = sub["return_pct"].dropna()
        dir_a = (sub["_dir"].mean() * 100) if has_mfe and len(sub) > 0 else None
        t_, p_ = (scipy_stats.ttest_1samp(sr, 0) if len(sr) >= 5 else (None, None))
        strat_rows.append({
            "Strategy":        strat,
            "Signals":         len(sub),
            "Exit Accuracy %": round(sub["correct"].mean() * 100, 1) if len(sub) >= 3 else float("nan"),
            "Dir. Accuracy %": round(dir_a, 1) if dir_a is not None else float("nan"),
            "Avg Return %":    round(sr.mean(), 2) if len(sr) >= 3 else float("nan"),
            "Std Dev %":       round(sr.std(),  2) if len(sr) >= 3 else float("nan"),
            "Variance":        round(sr.var(),  3) if len(sr) >= 3 else float("nan"),
            "Sharpe":          round(sr.mean() / sr.std() * (252**0.5), 2)
                               if len(sr) >= 3 and sr.std() > 0 else float("nan"),
            "p-Value":         round(float(p_), 4) if p_ is not None else float("nan"),
            "Significant":     "✅" if p_ is not None and p_ < 0.05 else "❌",
        })

    if strat_rows:
        sdf = pd.DataFrame(strat_rows)

        def _ca(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN};font-weight:700" if v >= 60 else (
                   f"color:{AMBER}" if v >= 50 else f"color:{RED}")
        def _cr(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        def _cs(v): return f"color:{GREEN}" if "✅" in str(v) else f"color:{RED}"

        sty = sdf.style
        for col in ["Exit Accuracy %", "Dir. Accuracy %"]:
            if col in sdf.columns: sty = sty.map(_ca, subset=[col])
        for col in ["Avg Return %", "Sharpe"]:
            if col in sdf.columns: sty = sty.map(_cr, subset=[col])
        if "Significant" in sdf.columns: sty = sty.map(_cs, subset=["Significant"])
        sty = sty.format({
            "Exit Accuracy %": "{:.1f}%",
            "Dir. Accuracy %": "{:.1f}%",
            "Avg Return %":    "{:+.2f}%",
            "Std Dev %":       "{:.2f}%",
            "Variance":        "{:.3f}",
            "Sharpe":          "{:+.2f}",
            "p-Value":         "{:.4f}",
        }, na_rep="—")
        st.dataframe(sty, use_container_width=True, hide_index=True)

        # Strategy bar charts
        plot_df = sdf[sdf["Signals"] >= 3]
        if not plot_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            _style_fig(fig, list(axes))
            ax1, ax2 = axes

            acc_v = plot_df["Exit Accuracy %"].fillna(0).values
            bc1   = [GREEN if v >= 60 else (AMBER if v >= 50 else RED) for v in acc_v]
            ax1.barh(plot_df["Strategy"].values[::-1], acc_v[::-1],
                     color=bc1[::-1], edgecolor=BORDER, linewidth=0.4)
            ax1.axvline(50, color=MUTED, linewidth=0.8, linestyle="--")
            ax1.axvline(60, color=GREEN, linewidth=0.7, linestyle=":")
            ax1.set_xlim(0, 100)
            ax1.set_title("Exit Accuracy % by Strategy", color=CYAN, fontsize=9)
            ax1.set_xlabel("Accuracy %", color=MUTED, fontsize=8)
            for i, (v, n_s) in enumerate(zip(acc_v[::-1], plot_df["Signals"].values[::-1])):
                if not np.isnan(v):
                    ax1.text(v + 1, i, f"{v:.0f}% (n={n_s})", va="center", color=TEXT, fontsize=7)

            ret_v = plot_df["Avg Return %"].fillna(0).values
            rc2   = [GREEN if v > 0 else RED for v in ret_v]
            ax2.barh(plot_df["Strategy"].values[::-1], ret_v[::-1],
                     color=rc2[::-1], edgecolor=BORDER, linewidth=0.4)
            ax2.axvline(0, color=MUTED, linewidth=0.8)
            ax2.set_title("Avg Return % by Strategy", color=CYAN, fontsize=9)
            ax2.set_xlabel("Avg Return %", color=MUTED, fontsize=8)
            for i, v in enumerate(ret_v[::-1]):
                if not np.isnan(v):
                    ax2.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}%",
                             va="center", ha="left" if v >= 0 else "right",
                             color=TEXT, fontsize=7)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Per-ticker breakdown ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Accuracy by Ticker</div>',
                unsafe_allow_html=True)
    tick_rows = []
    for ticker in sorted(rdf["ticker"].unique()):
        sub = rdf[rdf["ticker"] == ticker]
        sr  = sub["return_pct"].dropna()
        dir_a = (sub["_dir"].mean() * 100) if has_mfe and "_dir" in sub.columns else None
        tick_rows.append({
            "Ticker":          ticker,
            "Signals":         len(sub),
            "BUY":             int((sub["action"] == "BUY").sum()),
            "SHORT":           int((sub["action"] == "SHORT").sum()),
            "Exit Accuracy %": round(sub["correct"].mean() * 100, 1) if len(sub) >= 3 else float("nan"),
            "Dir. Accuracy %": round(dir_a, 1) if dir_a is not None else float("nan"),
            "Avg Return %":    round(sr.mean(), 2) if len(sr) >= 3 else float("nan"),
            "Std Dev %":       round(sr.std(),  2) if len(sr) >= 3 else float("nan"),
            "Best %":          round(sr.max(),  2) if not sr.empty else float("nan"),
            "Worst %":         round(sr.min(),  2) if not sr.empty else float("nan"),
        })

    if tick_rows:
        tdf = pd.DataFrame(tick_rows).sort_values("Signals", ascending=False)
        def _ca(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN};font-weight:700" if v >= 60 else (f"color:{AMBER}" if v >= 50 else f"color:{RED}")
        def _cr(v):
            if pd.isna(v): return f"color:{MUTED}"
            return f"color:{GREEN}" if v > 0 else f"color:{RED}"
        sty2 = tdf.style
        for c in ["Exit Accuracy %", "Dir. Accuracy %"]:
            if c in tdf.columns: sty2 = sty2.map(_ca, subset=[c])
        for c in ["Avg Return %", "Best %"]:
            if c in tdf.columns: sty2 = sty2.map(_cr, subset=[c])
        sty2 = sty2.format({
            "Exit Accuracy %": "{:.1f}%", "Dir. Accuracy %": "{:.1f}%",
            "Avg Return %": "{:+.2f}%",   "Std Dev %": "{:.2f}%",
            "Best %": "{:+.2f}%",          "Worst %": "{:+.2f}%",
        }, na_rep="—")
        st.dataframe(sty2, use_container_width=True, hide_index=True)

    # ── Variance & risk profile ────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Variance & Risk Profile</div>',
                unsafe_allow_html=True)
    skew = float(scipy_stats.skew(rets))   if len(rets) >= 5 else 0.0
    kurt = float(scipy_stats.kurtosis(rets)) if len(rets) >= 5 else 0.0
    cv   = (std_ret / abs(avg_ret) * 100) if avg_ret != 0 else float("inf")

    v1, v2, v3, v4 = st.columns(4)
    for col, lbl, val, clr, tip in [
        (v1, "Std Dev",              f"{std_ret:.2f}%",
         TEXT,
         "Spread of returns — lower = more consistent"),
        (v2, "Skewness",             f"{skew:+.3f}",
         GREEN if skew > 0 else (AMBER if skew > -0.5 else RED),
         ">0 = more big wins than big losses"),
        (v3, "Kurtosis",             f"{kurt:+.3f}",
         AMBER if abs(kurt) > 1 else TEXT,
         "High = fat tails / extreme outliers common"),
        (v4, "Coeff of Variation",   f"{cv:.1f}%" if cv != float("inf") else "∞",
         GREEN if 0 < cv < 100 else AMBER,
         "Std Dev ÷ |Mean| — lower = more efficient"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box" title="{tip}">'
                f'<div class="label">{lbl}</div>'
                f'<div class="value" style="color:{clr};font-size:1.15rem;">{val}</div>'
                f'<div style="color:#4a6882;font-size:.6rem;margin-top:.2rem;">{tip}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Box plot: BUY vs SHORT spread
    if n >= 5 and "action" in rdf.columns:
        groups = [(rdf[rdf["action"] == a]["return_pct"].dropna().values, a)
                  for a in ["BUY", "SHORT"]]
        groups = [(g, l) for g, l in groups if len(g) >= 2]
        if groups:
            data_vals, data_labs = zip(*groups)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            _style_fig(fig, ax)
            bp = ax.boxplot(list(data_vals), labels=list(data_labs),
                            patch_artist=True, notch=False,
                            medianprops=dict(color=AMBER, linewidth=2),
                            whiskerprops=dict(color=MUTED),
                            capprops=dict(color=MUTED),
                            flierprops=dict(markerfacecolor=RED, marker="o",
                                            markersize=4, alpha=0.6))
            for patch, clr in zip(bp["boxes"], [GREEN, RED]):
                patch.set_facecolor(clr + "33")
                patch.set_edgecolor(clr)
            ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
            ax.set_title("Return Spread: BUY vs SHORT", color=CYAN, fontsize=9)
            ax.set_ylabel("Return %", color=MUTED, fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── MFE / MAE scatter ──────────────────────────────────────────────────
    if has_mfe:
        exc_df = rdf[rdf["mfe"].notna()].copy()
        if len(exc_df) >= 3:
            st.markdown('<div class="section-header">📐 Directional Excursion (MFE / MAE)</div>',
                        unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            _style_fig(fig, list(axes))
            ax1, ax2 = axes

            dot_c = [GREEN if c else RED for c in exc_df["correct"]]
            ax1.scatter(exc_df["mfe"], exc_df["return_pct"],
                        c=dot_c, alpha=0.75, edgecolors=BORDER, linewidth=0.5, s=55)
            ax1.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
            ax1.axvline(mfe_thresh, color=AMBER, linewidth=1, linestyle=":",
                        label=f"MFE ≥ {mfe_thresh:.1f}%")
            ax1.set_title("MFE vs Final Return %", color=CYAN, fontsize=9)
            ax1.set_xlabel("MFE % (best intraday move in signal direction)", color=MUTED, fontsize=8)
            ax1.set_ylabel("Final Return %", color=MUTED, fontsize=8)
            from matplotlib.lines import Line2D
            ax1.legend(handles=[
                Line2D([0],[0], marker="o", color="w", markerfacecolor=GREEN,
                       markersize=7, label="Correct exit"),
                Line2D([0],[0], marker="o", color="w", markerfacecolor=RED,
                       markersize=7, label="Incorrect exit"),
            ], fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)

            corr_s = exc_df[exc_df["correct"] == 1]
            incr_s = exc_df[exc_df["correct"] == 0]
            vals_mfe = [corr_s["mfe"].mean() if len(corr_s) else 0,
                        incr_s["mfe"].mean() if len(incr_s) else 0]
            vals_mae = [corr_s["mae"].mean() if len(corr_s) else 0,
                        incr_s["mae"].mean() if len(incr_s) else 0]
            x = np.arange(2)
            w = 0.35
            ax2.bar(x - w/2, vals_mfe, w, color=GREEN, edgecolor=BORDER,
                    linewidth=0.4, label="Avg MFE", alpha=0.85)
            ax2.bar(x + w/2, vals_mae, w, color=RED,   edgecolor=BORDER,
                    linewidth=0.4, label="Avg MAE", alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels(["Correct exits", "Incorrect exits"], color=TEXT, fontsize=9)
            ax2.set_title("Avg MFE vs MAE by Outcome", color=CYAN, fontsize=9)
            ax2.set_ylabel("% move", color=MUTED, fontsize=8)
            ax2.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            wrong_dir = int((incr_s["mfe"] >= mfe_thresh).sum()) if len(incr_s) else 0
            if wrong_dir > 0:
                st.warning(
                    f"⚡ **{wrong_dir} 'incorrect' exit(s) were right in direction** — "
                    f"price moved ≥{mfe_thresh:.1f}% favourably before reversing. "
                    f"Consider tighter take-profit levels."
                )

    # ── Full table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Evaluated Signals</div>',
                unsafe_allow_html=True)
    disp_cols = ["date", "entry_date", "ticker", "action", "confidence", "time_horizon",
                 "held_td", "entry_price", "exit_price", "return_pct",
                 "mfe", "mae", "directional_correct", "outcome", "exit_reason"]
    disp = ready_df[[c for c in disp_cols if c in ready_df.columns]].copy()

    def _c_a(v): return f"color:{GREEN}" if v == "BUY" else f"color:{RED}"
    def _c_o(v): return f"color:{GREEN}" if "Correct" in str(v) else f"color:{RED}"
    def _c_r(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v > 0 else f"color:{RED}"
    def _c_dc(v): return f"color:{GREEN}" if v else f"color:{MUTED}"

    sty_t = disp.style
    if "action"              in disp.columns: sty_t = sty_t.map(_c_a,  subset=["action"])
    if "outcome"             in disp.columns: sty_t = sty_t.map(_c_o,  subset=["outcome"])
    if "return_pct"          in disp.columns: sty_t = sty_t.map(_c_r,  subset=["return_pct"])
    if "mfe"                 in disp.columns: sty_t = sty_t.map(_c_r,  subset=["mfe"])
    if "directional_correct" in disp.columns: sty_t = sty_t.map(_c_dc, subset=["directional_correct"])
    sty_t = sty_t.format({
        "entry_price": "${:.2f}",
        "exit_price":  "${:.2f}",
        "return_pct":  lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
        "mfe":         lambda x: f"+{x:.2f}%" if pd.notna(x) else "—",
        "mae":         lambda x: f"-{x:.2f}%" if pd.notna(x) else "—",
    })
    st.dataframe(sty_t, use_container_width=True, hide_index=True)

    csv = ready_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV (includes MFE/MAE)",
        data=csv,
        file_name=f"strategy_signal_accuracy_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.markdown(f"""
    <div style="margin-top:1rem;padding:.8rem;border:1px solid #1e3a5f;border-radius:8px;
                color:#6b8fad;font-size:.78rem;">
        ⚠️ Entry = OPEN of next trading day after signal · Exit = CLOSE at stated horizon ·
        Directional accuracy uses MFE ≥ {mfe_thresh:.1f}% threshold (intraday High/Low) ·
        Black-swan events excluded · Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy_signals():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Apply your backtested strategy to today's market — get a real, actionable trade signal.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Uses the exact same logic as the Backtester but runs on current data to tell you what the strategy
        says to do RIGHT NOW. Includes entry price, stop loss, take profit, risk/reward ratio,
        and the exact indicator values driving the signal. Signals are auto-saved to Supabase
        so the <b>Signal Accuracy Tracker</b> below can evaluate them later.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Strategy selector ──────────────────────────────────────────────────
    STRATEGIES = {
        "📈 Moving Average Crossover": "ma",
        "🔄 RSI Mean Reversion":       "rsi",
        "🚀 Momentum":                 "momentum",
        "🤖 ML Weekly (RF)":           "ml",
    }

    strategy_label = st.radio("Strategy:", list(STRATEGIES.keys()),
                              horizontal=True, key="ss_strategy")
    strategy_key   = STRATEGIES[strategy_label]

    col_t, col_p = st.columns([3, 2])
    with col_t:
        tickers_raw = st.text_input(
            "Ticker(s) to analyse",
            placeholder={
                "ma":       "e.g. QQQ or AAPL, MSFT",
                "rsi":      "e.g. TSLA or BTC-USD, NVDA",
                "momentum": "e.g. QQQ, SPY, GLD",
                "ml":       "e.g. NVDA, QQQ, AAPL (first = target)",
            }.get(strategy_key, "e.g. QQQ"),
            key="ss_tickers"
        )
    with col_p:
        data_years = st.slider("Data history (years)", 1, 5, 3, key="ss_years")

    with st.expander("⚙️ Strategy Parameters", expanded=False):
        if strategy_key == "ma":
            p1, p2 = st.columns(2)
            with p1: fast = st.slider("Fast MA", 10, 100, 50, 5, key="ss_fast")
            with p2: slow = st.slider("Slow MA", 50, 300, 200, 10, key="ss_slow")
        elif strategy_key == "rsi":
            p1, p2, p3 = st.columns(3)
            with p1: rsi_period  = st.slider("RSI Period", 7, 21, 14, key="ss_rsi_p")
            with p2: oversold    = st.slider("Oversold",   20, 40, 30, key="ss_os")
            with p3: overbought  = st.slider("Overbought", 60, 80, 70, key="ss_ob")
        elif strategy_key == "momentum":
            p1, p2 = st.columns(2)
            with p1: lookback    = st.slider("Lookback (days)", 20, 252, 63, key="ss_lb")
            with p2: hold_period = st.slider("Hold period (days)", 5, 63, 21, key="ss_hp")
        elif strategy_key == "ml":
            n_est = st.slider("RF Estimators", 50, 200, 100, 50, key="ss_nest")

    run_btn = st.button("🎯 Generate Live Trading Signal", key="ss_run")

    if not run_btn:
        if st.session_state.get("ss_results"):
            st.caption("⬇️ Previous signals — click Generate to refresh")
            for ticker, result in st.session_state["ss_results"].items():
                _render_signal_card(ticker, result)
    else:
        if not tickers_raw.strip():
            st.error("❌ Enter at least one ticker.")
        else:
            tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

            with st.spinner("📡 Fetching current market data…"):
                closes_map = {}
                failed     = []
                for ticker in tickers:
                    closes = _fetch_data(ticker, years=data_years)
                    if closes is not None:
                        closes_map[ticker] = closes
                    else:
                        failed.append(ticker)

            for f in failed:
                st.warning(f"⚠️ Could not fetch data for {f} — skipping.")

            if not closes_map:
                st.error("❌ No valid data fetched.")
            else:
                results = {}
                target  = list(closes_map.keys())[0]

                with st.spinner(f"⚙️ Computing {strategy_label} signal…"):
                    for ticker, closes in closes_map.items():
                        if strategy_key == "ma":
                            if len(closes) < slow + 5:
                                st.warning(f"⚠️ {ticker}: not enough data for {slow}-day MA.")
                                continue
                            results[ticker] = _ma_signal(closes, fast, slow)
                        elif strategy_key == "rsi":
                            if len(closes) < rsi_period + 5:
                                continue
                            results[ticker] = _rsi_signal(closes, rsi_period, oversold, overbought)
                        elif strategy_key == "momentum":
                            if len(closes) < lookback + 5:
                                continue
                            results[ticker] = _momentum_signal(closes, lookback, hold_period)
                        elif strategy_key == "ml":
                            feature_closes = {k: v for k, v in closes_map.items() if k != target}
                            r = _ml_weekly_signal(closes_map[target], feature_closes, n_est)
                            if "error" in r:
                                st.error(f"❌ {r['error']}")
                            else:
                                results[target] = r
                            break

                if not results:
                    st.error("❌ No signals generated.")
                else:
                    st.session_state["ss_results"] = results
                    st.caption(f"Signals generated at {datetime.now().strftime('%H:%M:%S ET')} · "
                               f"Saved to Supabase for accuracy tracking below")
                    for ticker, result in results.items():
                        _render_signal_card(ticker, result)

    # ══════════════════════════════════════════════════════════════════════
    # ACCURACY TRACKER SECTION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _render_strategy_accuracy()

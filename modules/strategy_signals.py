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
    """Compute current MA crossover signal and all supporting data."""
    fast_ma  = closes.rolling(fast).mean()
    slow_ma  = closes.rolling(slow).mean()

    current_price = float(closes.iloc[-1])
    fast_now      = float(fast_ma.iloc[-1])
    slow_now      = float(slow_ma.iloc[-1])
    fast_prev     = float(fast_ma.iloc[-2])
    slow_prev     = float(slow_ma.iloc[-2])

    # Current position
    currently_above = fast_now > slow_now
    was_above       = fast_prev > slow_prev
    just_crossed    = currently_above != was_above

    if currently_above:
        signal   = "BUY"
        color    = GREEN
        rationale = f"{fast}-day MA (${fast_now:.2f}) is above the {slow}-day MA (${slow_now:.2f}) — bullish trend."
        if just_crossed:
            rationale += f" ⚡ **Golden Cross just occurred** — this is a fresh entry signal."
    else:
        signal   = "SHORT"
        color    = RED
        rationale = f"{fast}-day MA (${fast_now:.2f}) is below the {slow}-day MA (${slow_now:.2f}) — bearish trend."
        if just_crossed:
            rationale += f" ⚡ **Death Cross just occurred** — this is a fresh short signal."

    # What price would flip the signal?
    # Flip happens when fast MA crosses slow MA. Approximate by finding
    # the price that would move fast_ma to equal slow_ma over 'fast' days
    days_in_fast_ma = min(fast, len(closes))
    old_sum  = closes.iloc[-days_in_fast_ma:-1].sum()
    flip_price = slow_now * fast - old_sum
    flip_price = max(0, flip_price)

    # Distance from MAs
    dist_from_fast = (current_price - fast_now) / fast_now * 100
    dist_from_slow = (current_price - slow_now) / slow_now * 100

    # Stop loss: below slow MA (for BUY) or above slow MA (for SHORT)
    if signal == "BUY":
        stop_loss    = slow_now * 0.98   # 2% below slow MA
        take_profit  = current_price * 1.10  # 10% above entry
    else:
        stop_loss    = slow_now * 1.02   # 2% above slow MA
        take_profit  = current_price * 0.90

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

    # Determine position
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
            rationale = (f"RSI is {current_rsi:.1f} — recovering from oversold but not there yet. "
                         f"Watch for RSI to drop below {oversold} for a BUY signal.")
        else:
            rationale = (f"RSI is {current_rsi:.1f} — elevated but not yet overbought. "
                         f"Watch for RSI to exceed {overbought} for a SHORT signal.")

    # Where would a signal trigger?
    buy_trigger   = f"RSI drops below {oversold}"
    short_trigger = f"RSI rises above {overbought}"
    exit_trigger  = f"RSI returns to 45–55 neutral zone"

    # Implied price for RSI to hit oversold (rough estimate)
    recent_vol = float(closes.pct_change().tail(20).std())
    days_to_signal_buy   = max(1, int((current_rsi - oversold) / max(0.5, abs(current_rsi - prev_rsi))))
    days_to_signal_short = max(1, int((overbought - current_rsi) / max(0.5, abs(current_rsi - prev_rsi))))

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
        "days_to_buy":      days_to_signal_buy if signal != "BUY" else 0,
        "days_to_short":    days_to_signal_short if signal != "SHORT" else 0,
        "strategy":         f"RSI Mean Reversion ({oversold}/{overbought})",
    }


def _momentum_signal(closes: pd.Series, lookback: int, hold_period: int) -> dict:
    ret      = closes.pct_change()
    momentum = closes.pct_change(lookback)

    current_price    = float(closes.iloc[-1])
    current_momentum = float(momentum.iloc[-1])
    prev_momentum    = float(momentum.iloc[-2])

    # Momentum percentile vs history (how extreme is the current reading?)
    hist_mom  = momentum.dropna()
    percentile = float((hist_mom < current_momentum).mean() * 100)

    if current_momentum > 0.02:
        signal    = "BUY"
        color     = GREEN
        strength  = "strong" if current_momentum > 0.10 else "moderate"
        rationale = (f"{lookback}-day momentum is **+{current_momentum*100:.1f}%** ({strength} positive trend). "
                     f"This is in the {percentile:.0f}th percentile of historical readings — "
                     f"trend has been {'accelerating' if current_momentum > prev_momentum else 'decelerating'}.")
    elif current_momentum < -0.02:
        signal    = "SHORT"
        color     = RED
        strength  = "strong" if current_momentum < -0.10 else "moderate"
        rationale = (f"{lookback}-day momentum is **{current_momentum*100:.1f}%** ({strength} negative trend). "
                     f"This is in the {percentile:.0f}th percentile of historical readings — "
                     f"trend has been {'accelerating' if current_momentum < prev_momentum else 'decelerating'}.")
    else:
        signal    = "HOLD"
        color     = AMBER
        rationale = (f"{lookback}-day momentum is {current_momentum*100:.1f}% — "
                     f"too close to zero for a clear signal. Need >+2% for BUY or <-2% for SHORT.")

    stop_loss   = current_price * 0.95 if signal == "BUY" else current_price * 1.05
    take_profit = current_price * 1.15 if signal == "BUY" else current_price * 0.85

    return {
        "signal":            signal,
        "color":             color,
        "rationale":         rationale,
        "current_price":     current_price,
        "momentum_pct":      current_momentum * 100,
        "momentum_percentile": percentile,
        "hold_period":       hold_period,
        "stop_loss":         stop_loss,
        "take_profit":       take_profit,
        "closes":            closes,
        "momentum_series":   momentum,
        "strategy":          f"Momentum ({lookback}d lookback)",
    }


def _ml_weekly_signal(closes: pd.Series, feature_closes: dict, n_estimators: int) -> dict:
    """Run the RF weekly model on current data and return current signal."""
    from sklearn.ensemble import RandomForestClassifier

    # Build weekly data
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

    # Train on ALL available data (we are generating a forward signal, not backtesting)
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X, y)

    # Predict on LAST row (current week's features, predicting NEXT week)
    last_features = feat[feature_cols].iloc[-1].values.reshape(1, -1)
    proba = float(clf.predict_proba(last_features)[0][1])

    current_price = float(closes.iloc[-1])

    if proba > 0.60:
        signal = "BUY"
        color  = GREEN
        confidence = "HIGH" if proba > 0.70 else "MEDIUM"
        rationale = (f"Model assigns **{proba*100:.0f}% probability of UP** next week. "
                     f"Confidence: {confidence}. The model is trained on all available weekly data.")
    elif proba < 0.40:
        signal = "SHORT"
        color  = RED
        confidence = "HIGH" if proba < 0.30 else "MEDIUM"
        rationale = (f"Model assigns **{(1-proba)*100:.0f}% probability of DOWN** next week. "
                     f"Confidence: {confidence}.")
    else:
        signal = "HOLD"
        color  = AMBER
        confidence = "LOW"
        rationale = (f"Model probability is {proba*100:.0f}% — too close to 50% for a confident signal. "
                     f"No trade recommended this week.")

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(5)

    stop_loss   = current_price * 0.95 if signal == "BUY" else current_price * 1.05
    take_profit = current_price * 1.10 if signal == "BUY" else current_price * 0.90

    return {
        "signal":         signal,
        "color":          color,
        "confidence":     confidence,
        "rationale":      rationale,
        "proba_up":       proba,
        "current_price":  current_price,
        "stop_loss":      stop_loss,
        "take_profit":    take_profit,
        "top_features":   top_features,
        "closes":         closes,
        "strategy":       "ML Weekly Direction (RF)",
    }


def _render_signal_card(ticker: str, result: dict):
    """Render the full forward signal card for one ticker."""
    signal = result.get("signal", "HOLD")
    color  = result.get("color", AMBER)
    current= result.get("current_price", 0)
    sl     = result.get("stop_loss", 0)
    tp     = result.get("take_profit", 0)

    emoji  = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️"}.get(signal, "➡️")
    risk   = abs(current - sl) / current * 100 if current > 0 else 0
    reward = abs(tp - current) / current * 100 if current > 0 else 0
    rr     = reward / risk if risk > 0 else 0

    st.markdown("---")

    # ── Signal banner ──────────────────────────────────────────────────────
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

    # ── Strategy-specific details ──────────────────────────────────────────
    st.markdown("")

    # MA-specific
    if "fast_ma" in result:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.markdown(f'<div class="metric-box"><div class="label">Fast MA ({result.get("strategy","").split("/")[0].split("(")[1] if "(" in result.get("strategy","") else ""}d)</div><div class="value" style="font-size:1.1rem;">${result["fast_ma"]:.2f}</div></div>', unsafe_allow_html=True)
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

    # ── Price chart with signal overlays ───────────────────────────────────
    closes = result.get("closes")
    if closes is not None:
        display_closes = closes.tail(252)  # 1 year

        fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                  gridspec_kw={"height_ratios": [3, 1]})
        _style_fig(fig, list(axes))

        ax1, ax2 = axes

        # Price
        ax1.plot(display_closes.index, display_closes.values,
                 color=BLUE, linewidth=1.5, label="Price")

        # MA overlays
        if "fast_series" in result:
            fs = result["fast_series"].tail(252)
            ss = result["slow_series"].tail(252)
            ax1.plot(fs.index, fs.values, color=AMBER, linewidth=1, linestyle="--",
                     label=f"Fast MA", alpha=0.8)
            ax1.plot(ss.index, ss.values, color=RED, linewidth=1, linestyle="--",
                     label=f"Slow MA", alpha=0.8)

        # Signal line markers
        if signal == "BUY":
            ax1.axhline(sl, color=RED,   linewidth=0.8, linestyle=":", alpha=0.7, label=f"Stop ${sl:.2f}")
            ax1.axhline(tp, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label=f"Target ${tp:.2f}")
        elif signal == "SHORT":
            ax1.axhline(sl, color=RED,   linewidth=0.8, linestyle=":", alpha=0.7, label=f"Stop ${sl:.2f}")
            ax1.axhline(tp, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label=f"Target ${tp:.2f}")

        # Highlight current price
        ax1.axhline(float(closes.iloc[-1]), color=CYAN, linewidth=1, alpha=0.5)

        ax1.set_title(f"{ticker} — {result.get('strategy','')} · Current Signal: {signal}", color=CYAN)
        ax1.set_ylabel("Price ($)", color=MUTED)
        ax1.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # Secondary: RSI or Momentum
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
            ax2.axhline(0,   color=MUTED, linewidth=0.8)
            ax2.axhline(2,   color=GREEN, linewidth=0.6, linestyle="--", alpha=0.5)
            ax2.axhline(-2,  color=RED,   linewidth=0.6, linestyle="--", alpha=0.5)
            ax2.set_ylabel("Momentum %", color=MUTED)

        else:
            # Volume placeholder for ML
            ax2.set_visible(False)

        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Trade plan summary ─────────────────────────────────────────────────
    if signal in ("BUY", "SHORT"):
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#122338);
                    border:1px solid {color}33;border-radius:12px;padding:1rem 1.4rem;margin-top:0.5rem;">
            <div style="color:{CYAN};font-weight:700;font-size:0.88rem;margin-bottom:0.6rem;">
                📋 Trade Plan Summary — {ticker}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;font-size:0.83rem;color:#c9d8e8;">
                <div>
                    <span style="color:#6b8fad;">Direction:</span>
                    <b style="color:{color};">{signal}</b>
                </div>
                <div>
                    <span style="color:#6b8fad;">Entry near:</span>
                    <b>${current:.2f}</b>
                </div>
                <div>
                    <span style="color:#6b8fad;">Strategy:</span>
                    <b>{result.get('strategy','')}</b>
                </div>
                <div>
                    <span style="color:#6b8fad;">Stop Loss:</span>
                    <b style="color:{RED};">${sl:.2f} (-{risk:.1f}%)</b>
                </div>
                <div>
                    <span style="color:#6b8fad;">Take Profit:</span>
                    <b style="color:{GREEN};">${tp:.2f} (+{reward:.1f}%)</b>
                </div>
                <div>
                    <span style="color:#6b8fad;">Risk/Reward:</span>
                    <b style="color:{rr_color};">{rr:.1f}:1</b>
                </div>
            </div>
            <div style="margin-top:0.8rem;color:#6b8fad;font-size:0.75rem;">
                ⚠️ This is a systematic signal based on {result.get('strategy','')} applied to current market data.
                It is not financial advice. Always size positions according to your own risk tolerance.
                Paper trade first to validate the strategy before using real capital.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Log to signal history
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


def run_strategy_signals():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Apply your backtested strategy to today's market — get a real, actionable trade signal.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Uses the exact same logic as the Backtester but runs on current data to tell you what the strategy
        says to do RIGHT NOW. Includes entry price, stop loss, take profit, risk/reward ratio,
        and the exact indicator values driving the signal. Signals are auto-saved to Signal History.
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

    # ── Ticker input ───────────────────────────────────────────────────────
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

    # ── Strategy params ────────────────────────────────────────────────────
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

    run_btn = st.button("🎯 Generate Live Trading Signal", key="ss_run",
                        use_container_width=False)

    if not run_btn:
        # Show previous results
        if st.session_state.get("ss_results"):
            st.caption("⬇️ Previous signals — click Generate to refresh")
            for ticker, result in st.session_state["ss_results"].items():
                _render_signal_card(ticker, result)
        return

    if not tickers_raw.strip():
        st.error("❌ Enter at least one ticker.")
        return

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    # Fetch data for all tickers
    with st.spinner("📡 Fetching current market data…"):
        closes_map = {}
        failed     = []
        for ticker in tickers:
            closes = _fetch_data(ticker, years=data_years)
            if closes is not None:
                closes_map[ticker] = closes
            else:
                failed.append(ticker)

    if failed:
        for f in failed:
            st.warning(f"⚠️ Could not fetch data for {f} — skipping.")

    if not closes_map:
        st.error("❌ No valid data fetched. Check tickers and try again.")
        return

    results = {}
    target  = list(closes_map.keys())[0]

    with st.spinner(f"⚙️ Computing {strategy_label} signal…"):
        for ticker, closes in closes_map.items():
            if strategy_key == "ma":
                if len(closes) < slow + 5:
                    st.warning(f"⚠️ {ticker}: not enough data for {slow}-day MA. Try a longer history or shorter MA.")
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
                # First ticker = target, rest = features
                feature_closes = {k: v for k, v in closes_map.items() if k != target}
                r = _ml_weekly_signal(closes_map[target], feature_closes, n_est)
                if "error" in r:
                    st.error(f"❌ {r['error']}")
                else:
                    results[target] = r
                break  # ML handles all tickers at once

    if not results:
        st.error("❌ No signals generated. Check tickers and parameters.")
        return

    st.session_state["ss_results"] = results
    st.caption(f"Signals generated at {datetime.now().strftime('%H:%M:%S ET')} · "
               f"Signals auto-saved to Signal History")

    for ticker, result in results.items():
        _render_signal_card(ticker, result)

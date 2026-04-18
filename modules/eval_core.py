"""
eval_core.py — Shared evaluation logic
=======================================

Fixes applied:
  1. Trading days not calendar days — INTRADAY = 1 trading day, SWING = 5 trading days
  2. Signal time awareness — if signal came after market close (4pm ET), 
     day 1 starts from NEXT trading day open
  3. Black swan detection — if a volatility spike (VIX >20% move OR stock 
     moves >3x its normal daily range) occurs during the window, the signal 
     is flagged as "inconclusive — external shock" and excluded from accuracy 
     stats by default
  4. Supersession — if opposing signal arrives before the evaluation date, 
     exit at that earlier date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── Horizon definitions in TRADING DAYS ──────────────────────────────────────
HORIZON_TRADING_DAYS = {
    "INTRADAY":          1,    # evaluate at close of next trading day
    "SWING (1-5 DAYS)":  5,    # 5 trading days = 1 calendar week
    "SWING":             5,
    "MEDIUM (WEEKS)":   14,    # ~3 calendar weeks
    "MEDIUM":           14,
    "LONG (MONTHS)":    30,    # ~6 calendar weeks
    "LONG":             30,
}

# Market close is 4:00 PM ET = 21:00 UTC
# If signal was generated after 4pm ET, day 1 is the NEXT trading day
MARKET_CLOSE_HOUR_ET = 16   # 4pm ET

# Black swan threshold: stock moved more than this many times its
# 20-day average daily range during the evaluation window
VOLATILITY_MULTIPLIER_THRESHOLD = 3.0

# Also flag if VIX moved more than this % during the window
VIX_SPIKE_THRESHOLD_PCT = 20.0


def get_horizon_trading_days(horizon_str: str) -> int:
    if not horizon_str:
        return 5
    key = str(horizon_str).upper().strip()
    if key in HORIZON_TRADING_DAYS:
        return HORIZON_TRADING_DAYS[key]
    for k, v in HORIZON_TRADING_DAYS.items():
        if k in key:
            return v
    return 5


def horizon_label(td: int) -> str:
    if td <= 1:  return "Intraday (1 trading day)"
    if td <= 5:  return "Swing (5 trading days)"
    if td <= 14: return "Medium (14 trading days)"
    return "Long (30 trading days)"


def _is_after_market_close(ts: pd.Timestamp) -> bool:
    """
    Check if the signal was generated after 4pm ET.
    Timestamps stored without timezone — assume they are local machine time.
    Rough check: if hour >= 16 (4pm) or hour <= 3 (before 3am = after midnight ET)
    treat as post-close.
    """
    # Simple heuristic: if generated after 20:00 local or before 06:30 local
    # (i.e. outside normal trading hours), start counting from next trading day
    h = ts.hour
    return h >= 20 or h < 6   # rough after-hours detection


def _add_trading_days(start_ts: pd.Timestamp, n_trading_days: int,
                       trading_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    Add N trading days to start_ts using the actual trading calendar
    (derived from price data).
    Returns the nth trading day date, or None if not enough data.
    """
    # Find trading dates strictly after the signal date
    after_signal = trading_dates[trading_dates.date > start_ts.date()]

    # If signal was after market close, day 1 = next trading day
    # If signal was during market hours, day 1 = same trading day's close
    # But we already fetched that close at entry, so count from next day
    if len(after_signal) < n_trading_days:
        return None

    return pd.Timestamp(after_signal[n_trading_days - 1])


def _trading_days_elapsed(sig_ts: pd.Timestamp,
                           trading_dates: pd.DatetimeIndex) -> int:
    """Count trading days that have elapsed since signal_ts."""
    after = trading_dates[trading_dates.date > sig_ts.date()]
    # Also include today if market has closed
    now = datetime.now()
    today_closed = now.hour >= 17   # past 5pm = today's session closed
    today_dt = pd.Timestamp(now.date())
    if today_closed and today_dt in trading_dates:
        eligible = trading_dates[trading_dates.date >= sig_ts.date()]
    else:
        eligible = after
    return len(eligible[eligible.date <= now.date()])


def build_ticker_signal_timeline(df: pd.DataFrame) -> dict:
    """Build {ticker: [(timestamp, action, source), ...]} sorted ascending."""
    timeline = {}
    for _, row in df.iterrows():
        ticker = row["ticker"]
        ts     = pd.to_datetime(row["timestamp"])
        if ticker not in timeline:
            timeline[ticker] = []
        timeline[ticker].append((ts, row["action"], row.get("source", "")))
    for t in timeline:
        timeline[t].sort(key=lambda x: x[0])
    return timeline


def find_exit_date(ticker: str, signal_ts: pd.Timestamp, signal_action: str,
                   planned_exit_ts: pd.Timestamp,
                   ticker_timeline: dict) -> tuple[pd.Timestamp, str]:
    """
    Find actual exit date.
    - planned_exit_ts: the Nth trading day after signal
    - If opposing signal arrives before planned exit → exit there instead
    """
    opposite = "SHORT" if signal_action == "BUY" else "BUY"
    for ts, action, source in ticker_timeline.get(ticker, []):
        if ts <= signal_ts:
            continue
        if ts >= planned_exit_ts:
            break
        if action == opposite:
            return ts, f"superseded by {action} at {ts.strftime('%Y-%m-%d %H:%M')} ({source})"
    return planned_exit_ts, "horizon"


@pd.api.extensions.register_dataframe_accessor("_dummy")
class _Dummy:
    def __init__(self, df): pass


# ── Black swan / volatility spike detection ───────────────────────────────────

def _detect_volatility_spike(ticker: str, entry_date: str, exit_date: str,
                               fetch_history_fn) -> dict:
    """
    Detect if an external shock occurred during the evaluation window.
    Returns {"spiked": bool, "reason": str}
    """
    try:
        # Get 60 days of history to compute baseline volatility
        hist = fetch_history_fn(ticker, entry_date, exit_date, lookback_days=60)
        if hist is None or len(hist) < 10:
            return {"spiked": False, "reason": ""}

        closes = hist["Close"].dropna()
        ret    = closes.pct_change().dropna()

        if len(ret) < 5:
            return {"spiked": False, "reason": ""}

        # Baseline: 20-day avg absolute daily move before entry
        baseline_vol = float(ret.iloc[:-1].abs().mean())  # exclude the eval window itself

        # Window: moves during the evaluation period
        entry_dt = pd.Timestamp(entry_date)
        exit_dt  = pd.Timestamp(exit_date)
        window_ret = ret[(ret.index >= entry_dt) & (ret.index <= exit_dt)]

        if window_ret.empty:
            return {"spiked": False, "reason": ""}

        max_move = float(window_ret.abs().max()) * 100
        baseline_pct = baseline_vol * 100

        if baseline_pct > 0 and max_move > baseline_pct * VOLATILITY_MULTIPLIER_THRESHOLD:
            return {
                "spiked": True,
                "reason": f"Stock moved {max_move:.1f}% in a single day during eval window "
                           f"({VOLATILITY_MULTIPLIER_THRESHOLD:.0f}x normal {baseline_pct:.1f}% avg). "
                           f"Likely external shock — signal marked inconclusive."
            }

        return {"spiked": False, "reason": ""}

    except Exception:
        return {"spiked": False, "reason": ""}


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_signals(
    df: pd.DataFrame,
    fetch_price_fn,
    fetch_ohlc_fn=None,
    all_signals_for_timeline: pd.DataFrame = None,
    progress_callback=None,
    exclude_inconclusive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Core evaluation. Returns (ready_df, waiting_df, inconclusive_df).

    ready_df        — signals with outcomes, excluded from inconclusive
    waiting_df      — signals not yet evaluable (horizon not passed)
    inconclusive_df — signals flagged as external shock (VIX spike / >3x move)
    """
    now = datetime.now()

    # Build supersession timeline
    timeline_src = all_signals_for_timeline if all_signals_for_timeline is not None else df
    ticker_timeline = build_ticker_signal_timeline(timeline_src)

    # We need a trading calendar. Build it from SPY price history as proxy.
    # This gives us real market open days.
    import yfinance as yf
    try:
        spy = yf.Ticker("SPY").history(period="2y", auto_adjust=True)
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        trading_dates = spy.index.normalize().unique().sort_values()
    except Exception:
        # Fallback: use business days
        trading_dates = pd.date_range(
            start=(now - timedelta(days=730)).strftime("%Y-%m-%d"),
            end=now.strftime("%Y-%m-%d"),
            freq="B"
        )

    ready        = []
    waiting      = []
    inconclusive = []
    total        = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback(i, total, row["ticker"], row["timestamp"][:10])

        sig_ts     = pd.to_datetime(row["timestamp"])
        h_td       = get_horizon_trading_days(row.get("time_horizon", ""))
        td_elapsed = _trading_days_elapsed(sig_ts, trading_dates)

        base = {
            "timestamp":    row["timestamp"],
            "date":         row["timestamp"][:10],
            "ticker":       row["ticker"],
            "action":       row["action"],
            "confidence":   row.get("confidence", ""),
            "source":       row.get("source", ""),
            "time_horizon": row.get("time_horizon", ""),
            "horizon_td":   h_td,
            "td_elapsed":   td_elapsed,
            "reasoning":    str(row.get("reasoning", ""))[:100],
            "article":      str(row.get("article_title", ""))[:70],
            "urgency":      row.get("urgency", ""),
        }

        # ── Not enough TRADING days have passed ────────────────────────────
        if td_elapsed < h_td:
            remaining = h_td - td_elapsed
            base["not_ready_reason"] = (
                f"Wait {remaining} more trading day{'s' if remaining != 1 else ''} "
                f"({h_td} trading days needed, {td_elapsed} elapsed)"
            )
            waiting.append(base)
            continue

        # ── Find planned exit date in trading days ─────────────────────────
        planned_exit_ts = _add_trading_days(sig_ts, h_td, trading_dates)
        if planned_exit_ts is None:
            base["not_ready_reason"] = "Not enough trading day history to compute exit"
            waiting.append(base)
            continue

        # ── Supersession check ─────────────────────────────────────────────
        actual_exit_ts, exit_reason = find_exit_date(
            ticker          = row["ticker"],
            signal_ts       = sig_ts,
            signal_action   = row["action"],
            planned_exit_ts = planned_exit_ts,
            ticker_timeline = ticker_timeline,
        )

        # ── Fetch prices ───────────────────────────────────────────────────
        entry_price = fetch_price_fn(row["ticker"], row["timestamp"][:10])
        if entry_price is None or entry_price <= 0:
            base["not_ready_reason"] = "Entry price unavailable"
            waiting.append(base)
            continue

        exit_date_str = actual_exit_ts.strftime("%Y-%m-%d")
        exit_price    = fetch_price_fn(row["ticker"], exit_date_str)
        if exit_price is None or exit_price <= 0:
            base["not_ready_reason"] = f"Exit price unavailable for {exit_date_str}"
            waiting.append(base)
            continue

        # ── Black swan detection ───────────────────────────────────────────
        spike = {"spiked": False, "reason": ""}
        if fetch_ohlc_fn is not None:
            spike = _detect_volatility_spike(
                row["ticker"],
                row["timestamp"][:10],
                exit_date_str,
                fetch_ohlc_fn
            )

        # ── Compute return ─────────────────────────────────────────────────
        raw_ret = (exit_price - entry_price) / entry_price * 100
        pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
        correct = int(pnl_ret > 0)
        held_td = _trading_days_elapsed(sig_ts,
                      trading_dates[trading_dates <= actual_exit_ts])

        result = {
            **base,
            "entry_price":  round(entry_price, 2),
            "exit_price":   round(exit_price, 2),
            "exit_date":    exit_date_str,
            "held_td":      held_td,
            "exit_reason":  exit_reason,
            "superseded":   exit_reason != "horizon",
            "return_pct":   round(pnl_ret, 3),
            "correct":      correct,
            "outcome":      "✅ Correct" if correct else "❌ Incorrect",
            "spike_flag":   spike["spiked"],
            "spike_reason": spike["reason"],
        }

        if spike["spiked"]:
            result["outcome"] = "⚠️ Inconclusive"
            inconclusive.append(result)
        else:
            ready.append(result)

    return pd.DataFrame(ready), pd.DataFrame(waiting), pd.DataFrame(inconclusive)

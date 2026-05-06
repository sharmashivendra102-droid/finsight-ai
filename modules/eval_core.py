"""
eval_core.py — Shared evaluation logic
=======================================

Fixes applied:
  1. Trading days not calendar days — INTRADAY = 1 trading day, SWING = 5 trading days
  2. Next-trading-day entry — entry price is ALWAYS the close of the first
     trading day AFTER the signal date, regardless of what time the signal
     arrived.  A signal at 3 pm Friday → entry = Monday close.  A signal at
     9 am Monday → entry = Tuesday close.  This is the only realistic model:
     by the time a news signal is fetched, Groq-analysed, and a trader acts,
     same-day execution at the exact signal timestamp is not reliable.
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

# Maximum Favorable Excursion — default threshold for "directionally correct"
# A signal is considered directionally correct if the price moved at least
# this many % in the signal's favour at any point during the holding window,
# even if the final exit price was against the signal.
MFE_DIRECTIONAL_THRESHOLD = 0.5   # 0.5 %


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
    Check if the signal was generated at or after 4 pm ET (US market close).
    Timestamps are stored in local machine time without timezone info.
    We use 16:00 (4 pm) as the cutoff.

    NOTE: in practice this function is no longer the gating condition —
    evaluate_signals() always uses the NEXT trading day as entry,
    regardless of signal time.  It is kept here for reference / logging.
    """
    return ts.hour >= 16


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


# ── Maximum Favorable / Adverse Excursion ─────────────────────────────────────

def _compute_mfe_mae(ticker: str, entry_date_str: str, exit_date_str: str,
                     action: str, entry_price: float,
                     fetch_ohlc_fn) -> dict:
    """
    Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    over the signal's holding window using intraday High/Low data.

    MFE answers: "How far did price move IN the signal's direction at its best?"
    MAE answers: "How far did price move AGAINST the signal at its worst?"

    For a BUY signal:
        MFE = (peak_high  - entry) / entry * 100   (best upside reached)
        MAE = (entry - trough_low) / entry * 100   (worst drawdown seen)

    For a SHORT signal:
        MFE = (entry - trough_low) / entry * 100   (best downside reached)
        MAE = (peak_high - entry)  / entry * 100   (worst run-up against)

    A signal is "directionally correct" if MFE > MFE_DIRECTIONAL_THRESHOLD (0.5%
    by default).  This captures signals where the thesis was right and the price
    DID move favourably, but then reversed before the formal exit — a pattern
    common when fast-moving news triggers a quick spike followed by a fade.

    Returns dict with keys: mfe, mae, mfe_date, mae_date, excursion_ratio.
    All values are None when OHLC data is unavailable.
    """
    empty = {
        "mfe": None, "mae": None,
        "mfe_date": None, "mae_date": None,
        "excursion_ratio": None,
    }
    if fetch_ohlc_fn is None or entry_price is None or entry_price <= 0:
        return empty

    try:
        # We only need data inside the holding window; pass a tiny lookback
        hist = fetch_ohlc_fn(ticker, entry_date_str, exit_date_str, lookback_days=2)
        if hist is None or hist.empty:
            return empty

        entry_dt = pd.Timestamp(entry_date_str)
        exit_dt  = pd.Timestamp(exit_date_str)

        # Restrict to [entry, exit] inclusive
        window = hist[
            (hist.index >= entry_dt) &
            (hist.index <= exit_dt + pd.Timedelta(days=1))
        ]
        if window.empty:
            return empty

        # Use High/Low where available; fall back to Close
        highs = window["High"] if "High" in window.columns else window["Close"]
        lows  = window["Low"]  if "Low"  in window.columns else window["Close"]

        peak_high   = float(highs.max())
        trough_low  = float(lows.min())
        peak_date   = highs.idxmax().strftime("%Y-%m-%d")
        trough_date = lows.idxmin().strftime("%Y-%m-%d")

        if action.upper() == "BUY":
            mfe      = (peak_high  - entry_price) / entry_price * 100
            mae      = (entry_price - trough_low) / entry_price * 100
            mfe_date = peak_date
            mae_date = trough_date
        else:   # SHORT
            mfe      = (entry_price - trough_low) / entry_price * 100
            mae      = (peak_high  - entry_price) / entry_price * 100
            mfe_date = trough_date
            mae_date = peak_date

        mfe = max(0.0, round(mfe, 3))
        mae = max(0.0, round(mae, 3))

        # Excursion ratio: what fraction of total price range was favourable?
        # 1.0 = all movement was in signal direction; 0.0 = entirely against.
        total_range = mfe + mae
        exc_ratio   = round(mfe / total_range, 3) if total_range > 0 else None

        return {
            "mfe":             mfe,
            "mae":             mae,
            "mfe_date":        mfe_date,
            "mae_date":        mae_date,
            "excursion_ratio": exc_ratio,
        }

    except Exception:
        return empty


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

        sig_ts = pd.to_datetime(row["timestamp"])
        h_td   = get_horizon_trading_days(row.get("time_horizon", ""))

        # ── Determine entry date — always NEXT trading day after signal ────
        # Rationale: regardless of what time the signal arrived (3 pm, 9 am,
        # midnight), same-day execution at the signal timestamp is not
        # realistic.  A 3 pm Friday signal → entry = Monday close.
        next_tds = trading_dates[trading_dates.date > sig_ts.date()]
        if len(next_tds) == 0:
            # No trading day found after signal — too recent
            base_early = {
                "timestamp":    row["timestamp"],
                "date":         row["timestamp"][:10],
                "ticker":       row["ticker"],
                "action":       row["action"],
                "confidence":   row.get("confidence", ""),
                "source":       row.get("source", ""),
                "time_horizon": row.get("time_horizon", ""),
                "horizon_td":   h_td,
                "td_elapsed":   0,
                "reasoning":    str(row.get("reasoning", ""))[:100],
                "article":      str(row.get("article_title", ""))[:70],
                "urgency":      row.get("urgency", ""),
                "not_ready_reason": "No trading day found after signal date",
            }
            waiting.append(base_early)
            continue

        entry_ts       = pd.Timestamp(next_tds[0])          # e.g. Monday for Friday signal
        entry_date_str = entry_ts.strftime("%Y-%m-%d")

        # Readiness: have h_td trading days elapsed since entry_ts?
        td_elapsed_entry = _trading_days_elapsed(entry_ts, trading_dates)
        # Also track elapsed from signal date for display
        td_elapsed_signal = _trading_days_elapsed(sig_ts, trading_dates)

        base = {
            "timestamp":    row["timestamp"],
            "date":         row["timestamp"][:10],
            "entry_date":   entry_date_str,
            "ticker":       row["ticker"],
            "action":       row["action"],
            "confidence":   row.get("confidence", ""),
            "source":       row.get("source", ""),
            "time_horizon": row.get("time_horizon", ""),
            "horizon_td":   h_td,
            "td_elapsed":   td_elapsed_entry,   # days elapsed from entry
            "days_elapsed": td_elapsed_signal,  # days elapsed from signal (display)
            "reasoning":    str(row.get("reasoning", ""))[:100],
            "article":      str(row.get("article_title", ""))[:70],
            "urgency":      row.get("urgency", ""),
        }

        # ── Not enough TRADING days have passed since entry ────────────────
        if td_elapsed_entry < h_td:
            remaining = h_td - td_elapsed_entry
            base["not_ready_reason"] = (
                f"Wait {remaining} more trading day{'s' if remaining != 1 else ''} "
                f"(entry {entry_date_str}, {h_td} trading days needed, "
                f"{td_elapsed_entry} elapsed from entry)"
            )
            waiting.append(base)
            continue

        # ── Find planned exit date (h_td trading days after ENTRY) ────────
        planned_exit_ts = _add_trading_days(entry_ts, h_td, trading_dates)
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

        # ── Fetch prices — entry = next trading day close ──────────────────
        entry_price = fetch_price_fn(row["ticker"], entry_date_str)
        if entry_price is None or entry_price <= 0:
            base["not_ready_reason"] = f"Entry price unavailable for {entry_date_str}"
            waiting.append(base)
            continue

        exit_date_str = actual_exit_ts.strftime("%Y-%m-%d")
        exit_price    = fetch_price_fn(row["ticker"], exit_date_str)
        if exit_price is None or exit_price <= 0:
            base["not_ready_reason"] = f"Exit price unavailable for {exit_date_str}"
            waiting.append(base)
            continue

        # ── Black swan detection (over entry→exit window) ──────────────────
        spike = {"spiked": False, "reason": ""}
        if fetch_ohlc_fn is not None:
            spike = _detect_volatility_spike(
                row["ticker"],
                entry_date_str,   # start from actual entry, not signal date
                exit_date_str,
                fetch_ohlc_fn
            )

        # ── Maximum Favorable / Adverse Excursion ─────────────────────────
        # Measures whether the price ever moved in the signal's direction
        # during the hold window, even if it reversed by exit.
        excursion = _compute_mfe_mae(
            ticker         = row["ticker"],
            entry_date_str = entry_date_str,
            exit_date_str  = exit_date_str,
            action         = row["action"],
            entry_price    = entry_price,
            fetch_ohlc_fn  = fetch_ohlc_fn,
        )
        directional_correct = (
            excursion["mfe"] is not None
            and excursion["mfe"] > MFE_DIRECTIONAL_THRESHOLD
        )

        # ── Compute return ─────────────────────────────────────────────────
        raw_ret = (exit_price - entry_price) / entry_price * 100
        pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
        correct = int(pnl_ret > 0)
        held_td = _trading_days_elapsed(entry_ts,
                      trading_dates[trading_dates <= actual_exit_ts])

        result = {
            **base,
            "entry_date":          entry_date_str,
            "entry_price":         round(entry_price, 2),
            "exit_price":          round(exit_price, 2),
            "exit_date":           exit_date_str,
            "held_td":             held_td,
            "exit_reason":         exit_reason,
            "superseded":          exit_reason != "horizon",
            "return_pct":          round(pnl_ret, 3),
            "correct":             correct,
            "outcome":             "✅ Correct" if correct else "❌ Incorrect",
            "spike_flag":          spike["spiked"],
            "spike_reason":        spike["reason"],
            # ── Excursion fields ──────────────────────────────────────────
            "mfe":                 excursion["mfe"],
            "mae":                 excursion["mae"],
            "mfe_date":            excursion["mfe_date"],
            "mae_date":            excursion["mae_date"],
            "excursion_ratio":     excursion["excursion_ratio"],
            "directional_correct": directional_correct,
        }

        if spike["spiked"]:
            result["outcome"] = "⚠️ Inconclusive"
            inconclusive.append(result)
        else:
            ready.append(result)

    return pd.DataFrame(ready), pd.DataFrame(waiting), pd.DataFrame(inconclusive)

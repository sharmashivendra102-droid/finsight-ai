"""
eval_core.py — Shared evaluation logic for signal_performance and news_signal_evaluator.

Handles:
  1. Horizon matching — only evaluate when signal's own time window has passed
  2. Signal supersession — if a newer opposite signal arrives before the horizon,
     exit at that point instead (more realistic)
  3. Stale signal detection — flags signals where no follow-up check was done
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

HORIZON_MAP = {
    "INTRADAY":          1,
    "SWING (1-5 DAYS)":  5,
    "SWING":             5,
    "MEDIUM (WEEKS)":    14,
    "MEDIUM":            14,
    "LONG (MONTHS)":     30,
    "LONG":              30,
}


def get_horizon_days(horizon_str: str) -> int:
    """Convert time_horizon string to number of days."""
    if not horizon_str:
        return 5
    key = str(horizon_str).upper().strip()
    if key in HORIZON_MAP:
        return HORIZON_MAP[key]
    for k, v in HORIZON_MAP.items():
        if k in key:
            return v
    return 5


def horizon_label(days: int) -> str:
    if days <= 1:  return "Intraday (1d)"
    if days <= 5:  return "Swing (5d)"
    if days <= 14: return "Medium (14d)"
    return "Long (30d)"


def build_ticker_signal_timeline(all_signals_df: pd.DataFrame) -> dict:
    """
    Build a per-ticker timeline of all signals sorted by timestamp.
    Used to detect when a signal was superseded by a newer one.

    Returns: {ticker: [(timestamp, action, source), ...]} sorted ascending
    """
    timeline = {}
    for _, row in all_signals_df.iterrows():
        ticker = row["ticker"]
        ts     = pd.to_datetime(row["timestamp"])
        action = row["action"]
        source = row.get("source", "")
        if ticker not in timeline:
            timeline[ticker] = []
        timeline[ticker].append((ts, action, source))

    # Sort each ticker's signals by time
    for ticker in timeline:
        timeline[ticker].sort(key=lambda x: x[0])

    return timeline


def find_exit_date(
    ticker: str,
    signal_ts: pd.Timestamp,
    signal_action: str,
    horizon_days: int,
    ticker_timeline: dict,
) -> tuple[pd.Timestamp, str]:
    """
    Determine the actual exit date for a signal.

    Logic:
      - Planned exit = signal_ts + horizon_days
      - If a NEWER signal on the same ticker with OPPOSITE action arrives
        before the planned exit → use that date as exit instead
      - Returns (exit_date, exit_reason)

    exit_reason is one of:
      "horizon"     — normal, held full duration
      "superseded"  — exited early because opposing signal arrived
    """
    planned_exit = signal_ts + timedelta(days=horizon_days)
    opposite     = "SHORT" if signal_action == "BUY" else "BUY"

    signals = ticker_timeline.get(ticker, [])

    for ts, action, source in signals:
        # Only look at signals AFTER the entry and BEFORE the planned exit
        if ts <= signal_ts:
            continue
        if ts >= planned_exit:
            break
        # If an opposite signal arrived early → exit here
        if action == opposite:
            return ts, f"superseded by {action} at {ts.strftime('%Y-%m-%d')} ({source})"

    return planned_exit, "horizon"


def evaluate_signals(
    df: pd.DataFrame,
    fetch_price_fn,          # callable(ticker, date_str) -> float | None
    all_signals_for_timeline: pd.DataFrame = None,  # full signal set for supersession check
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core evaluation function.

    For each BUY/SHORT signal:
      1. Check if enough time has passed (horizon_days elapsed)
      2. Find actual exit date (horizon OR supersession by newer signal)
      3. Fetch entry and exit prices
      4. Compute return (sign-adjusted for BUY vs SHORT)
      5. Determine correct/incorrect

    Returns: (ready_df, waiting_df)
    """
    now = datetime.now()

    # Build timeline for supersession detection
    timeline_source = all_signals_for_timeline if all_signals_for_timeline is not None else df
    ticker_timeline = build_ticker_signal_timeline(timeline_source)

    ready   = []
    waiting = []
    total   = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback(i, total, row["ticker"], row["timestamp"][:10])

        sig_ts       = pd.to_datetime(row["timestamp"])
        h_days       = get_horizon_days(row.get("time_horizon", ""))
        days_elapsed = (now - sig_ts).days

        base = {
            "timestamp":    row["timestamp"],
            "date":         row["timestamp"][:10],
            "ticker":       row["ticker"],
            "action":       row["action"],
            "confidence":   row.get("confidence", ""),
            "source":       row.get("source", ""),
            "time_horizon": row.get("time_horizon", ""),
            "horizon_days": h_days,
            "days_elapsed": days_elapsed,
            "reasoning":    str(row.get("reasoning", ""))[:100],
            "article":      str(row.get("article_title", ""))[:70],
            "urgency":      row.get("urgency", ""),
        }

        # ── Not enough time has passed ─────────────────────────────────────
        if days_elapsed < h_days:
            remaining = h_days - days_elapsed
            base["not_ready_reason"] = (
                f"Wait {remaining} more day{'s' if remaining != 1 else ''} "
                f"(horizon: {h_days}d, elapsed: {days_elapsed}d)"
            )
            waiting.append(base)
            continue

        # ── Find actual exit date ──────────────────────────────────────────
        exit_date, exit_reason = find_exit_date(
            ticker         = row["ticker"],
            signal_ts      = sig_ts,
            signal_action  = row["action"],
            horizon_days   = h_days,
            ticker_timeline= ticker_timeline,
        )

        # ── Fetch entry price ──────────────────────────────────────────────
        entry_price = fetch_price_fn(row["ticker"], row["timestamp"][:10])
        if entry_price is None or entry_price <= 0:
            base["not_ready_reason"] = "Entry price unavailable from yfinance"
            waiting.append(base)
            continue

        # ── Fetch exit price ───────────────────────────────────────────────
        exit_date_str = exit_date.strftime("%Y-%m-%d")
        exit_price    = fetch_price_fn(row["ticker"], exit_date_str)
        if exit_price is None or exit_price <= 0:
            base["not_ready_reason"] = f"Exit price unavailable for {exit_date_str}"
            waiting.append(base)
            continue

        # ── Compute outcome ────────────────────────────────────────────────
        raw_ret = (exit_price - entry_price) / entry_price * 100
        pnl_ret = raw_ret if row["action"] == "BUY" else -raw_ret
        correct = int(pnl_ret > 0)

        held_days = (exit_date - sig_ts).days

        ready.append({
            **base,
            "entry_price":  round(entry_price, 2),
            "exit_price":   round(exit_price, 2),
            "exit_date":    exit_date_str,
            "held_days":    held_days,
            "exit_reason":  exit_reason,
            "superseded":   exit_reason != "horizon",
            "return_pct":   round(pnl_ret, 3),
            "correct":      correct,
            "outcome":      "✅ Correct" if correct else "❌ Incorrect",
        })

    return pd.DataFrame(ready), pd.DataFrame(waiting)

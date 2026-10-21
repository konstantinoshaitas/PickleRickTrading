"""Small deterministic stages; signals only, never cash/position accounting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import NS, PIP


@dataclass(frozen=True)
class Candidate:
    name: str
    unit: str = "pips"
    threshold: float = 20
    persistence: bool = False
    exit_offset_s: float = 120
    convergence: float | None = None  # fraction remaining; None disables, 0 is full convergence
    adverse_multiple: float = 0
    spread_multiple: float = 0


def candidates(stage):
    # These are separate stages, not a Cartesian product. No automatic winner selection.
    return {
        "baseline": [Candidate(f"pips_{n}", threshold=n) for n in (10, 20, 40)],
        "scale": [Candidate(f"bps_{n}", unit="bps", threshold=n, spread_multiple=2) for n in (1, 2)] +
                 [Candidate(f"vol_{n}", unit="vol", threshold=n, spread_multiple=2) for n in (2, 3)],
        "confirm": [Candidate("immediate"), Candidate("persistent", persistence=True)],
        "exits": [Candidate("fixed_60", exit_offset_s=60), Candidate("fixed_300", exit_offset_s=300),
                  Candidate("half_convergence", convergence=0.5, adverse_multiple=2),
                  Candidate("full_convergence", convergence=0, adverse_multiple=2)],
    }[stage]


def threshold(row, candidate):
    if candidate.unit == "pips":
        scale = PIP[row.pair]
    elif candidate.unit == "bps":
        scale = row.reference_mid / 10000
    else:
        scale = row.prewindow_volatility
    value = candidate.threshold * scale
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return max(value, candidate.spread_multiple*(row.ask-row.bid))


def direction(row, limit):
    if not np.isfinite(limit):
        return 0
    if row.target_bid-row.ask >= limit:
        return 1
    if row.bid-row.target_ask >= limit:
        return -1
    return 0


def entry_size(row, side, config):
    """Long buys base at its USD ask; short sells it at its USD bid, known at entry."""
    rate = getattr(row, "base_usd_ask" if side > 0 else "base_usd_bid", np.nan)
    if config.quantity is not None:
        return config.quantity, config.quantity*rate
    if not np.isfinite(rate) or rate <= 0:
        return np.nan, config.notional_usd
    return config.notional_usd/rate, config.notional_usd


def quote_rejection(row, side, now_ns, config):
    """Only the side we trade must be fresh; the other side may be old."""
    if row.broker_rejection:
        return row.broker_rejection
    traded = "ask" if side > 0 else "bid"
    other = "bid" if side > 0 else "ask"
    for label, limit in ((traded, config.traded_side_max_age_s),
                         (other, config.untraded_side_max_age_s)):
        if limit is None:
            continue
        recv = getattr(row, f"{label}_recv_ns", np.nan)
        if not np.isfinite(recv) or recv > now_ns:
            return f"{label}_received_after_decision"
        if now_ns-recv > limit*NS:
            return f"{label}_side_age_exceeded"
    return ""


def conversion_rejection(row, prefix, label, now_ns, config):
    """The USD rate is usable only if its used side is present and within the same freshness."""
    rate = getattr(row, f"{prefix}_{label}", np.nan)
    recv = getattr(row, f"{prefix}_{label}_recv_ns", np.nan)
    if (not np.isfinite(rate) or rate <= 0 or not np.isfinite(recv) or
            recv > now_ns or now_ns-recv > config.traded_side_max_age_s*NS):
        return "missing_fresh_base_usd_conversion" if prefix == "base_usd" else "missing_fresh_usd_conversion"
    return ""


def trade_signal(frame, candidate, config):
    frame = frame.sort_values("ts_ns").reset_index(drop=True)
    first = frame.iloc[0]
    base = {"session_id": first.session_id, "pair": first.pair, "provider": first.provider,
            "arm": first.arm, "candidate": candidate.name, "symbols": first.symbols,
            "reference_rule": getattr(first, "reference_rule", "FROZEN" if first.arm.endswith("frozen") else "INDICATIVE"),
            "reopen_ns": int(first.reopen_ns), "basis": first.basis,
            "basis_cutoff_ns": int(first.basis_cutoff_ns), "status": "no_signal",
            "rejection_reason": "threshold_or_quote_validity"}
    last_side, began, last_time = 0, None, None
    signal = None
    for row in frame.itertuples(index=False):
        if row.ts_ns >= row.reopen_ns:
            break
        side = direction(row, threshold(row, candidate)) if row.reference_rejection == "" else 0
        if side and quote_rejection(row, side, row.ts_ns, config):
            side = 0
        if side != last_side or last_time is None or row.ts_ns-last_time > config.stale_ms*1e6:
            began = row.ts_ns if side else None
        last_side, last_time = side, row.ts_ns
        if side and (not candidate.persistence or row.ts_ns-began >= config.persistence_ms*1e6):
            signal = row
            break
    if signal is None:
        return base
    side = last_side
    base.update({"signal_ns": int(signal.ts_ns), "direction": side, "rejection_reason": "",
                 "side": "buy" if side > 0 else "sell",
                 "threshold": threshold(signal, candidate), "raw_residual": signal.raw_residual,
                 "adjusted_residual": signal.adjusted_residual, "reference_mid": signal.reference_mid,
                 "reference_recv_ns": int(signal.reference_recv_ns),
                 "reference_age_ns": int(signal.reference_age_ns), "book_state": signal.book_state,
                 "reference_crossed": bool(getattr(signal, "reference_crossed", False)),
                 "signal_bid": signal.bid, "signal_ask": signal.ask,
                 "sunday_carry_anchor": bool(signal.sunday_carry_anchor)})
    earliest = signal.ts_ns + max(1, int(config.latency_ms*1e6))
    entry_side = "ask" if side > 0 else "bid"
    # Use the quote in force at signal+latency; do not wait for another update row.
    earlier = frame[(frame.ts_ns <= earliest) & (frame.ts_ns < signal.reopen_ns)]
    entry = None
    reasons = set()
    if not earlier.empty:
        row = earlier.iloc[-1]
        quantity, notional = entry_size(row, side, config)
        reason = quote_rejection(row, side, earliest, config)
        conversion = conversion_rejection(row, "base_usd", "ask" if side > 0 else "bid", earliest, config)
        if reason:
            reasons.add(reason)
        elif not np.isfinite(quantity):
            reasons.add("missing_fresh_base_usd_conversion")
        elif conversion:
            reasons.add(conversion)
        elif config.require_quote_size and getattr(row, entry_side+"_size") < quantity:
            reasons.add("insufficient_size_for_usd_notional")
        else:
            entry = row
    if entry is None:
        return {**base, "status": "unfillable_entry",
                "rejection_reason": "|".join(sorted(reasons)) or "no_quote_within_entry_wait"}
    base.update({"quantity_base": quantity, "notional_usd": notional,
                 "entry_ns": int(earliest), "entry_quote_ns": int(entry.ts_ns),
                 "entry_bid": entry.bid, "entry_ask": entry.ask,
                 "entry_age_ns": int(earliest-getattr(entry, entry_side+"_recv_ns")),
                 "entry_price": float(entry.ask if side > 0 else entry.bid)})
    # Baseline time uses actual reopening event AND its receive availability, not an invented auction.
    deadline = min(int(first.boundary_ns)-1, int(entry.ts_ns+config.max_hold_s*NS))
    fixed = max(int(first.reopen_ns+candidate.exit_offset_s*NS), int(first.reopen_recv_ns))
    target, reason = min(fixed, deadline), "fixed_reopen_offset" if fixed <= deadline else "max_hold"
    if candidate.convergence is not None or candidate.adverse_multiple:
        for row in frame[frame.ts_ns > entry.ts_ns].itertuples(index=False):
            if row.ts_ns >= target:
                break
            if row.broker_rejection or row.reference_rejection:
                continue
            remaining = row.target_bid-row.bid if side > 0 else row.ask-row.target_ask
            initial = abs(signal.adjusted_residual)
            if candidate.convergence is not None and remaining <= initial*candidate.convergence:
                target, reason = int(row.ts_ns+max(1, config.latency_ms*1e6)), "residual_convergence"
                break
            if candidate.adverse_multiple and remaining >= initial*candidate.adverse_multiple:
                target, reason = int(row.ts_ns+max(1, config.latency_ms*1e6)), "adverse_residual_stop"
                break
    # The latest quote on the closing side as of reopen+2 min is in force; no fresh-row wait.
    exits = frame[(frame.ts_ns <= min(target, deadline)) & (frame.ts_ns > entry.ts_ns)]
    exit_row = exits.iloc[-1] if not exits.empty else None
    exit_reason = "missing_exit" if exit_row is None else quote_rejection(exit_row, -side, target, config)
    if exit_row is not None and not exit_reason:
        closing = "bid" if side > 0 else "ask"
        gross_positive = (side > 0 and exit_row.bid >= entry.ask) or (side < 0 and entry.bid >= exit_row.ask)
        exit_reason = conversion_rejection(exit_row, "usd", "bid" if gross_positive else "ask",
                                           target, config)
        if not exit_reason and config.require_quote_size and getattr(exit_row, closing+"_size") < quantity:
            exit_reason = "insufficient_size_for_usd_notional"
    if exit_reason:
        return {**base, "status": "unpriced_open", "exit_reason": "missing_exit",
                "exit_target_ns": target}
    closing = "bid" if side > 0 else "ask"
    return {**base, "status": "closed", "exit_reason": reason, "exit_target_ns": target,
            "exit_ns": int(target), "exit_quote_ns": int(exit_row.ts_ns),
            "exit_bid": exit_row.bid, "exit_ask": exit_row.ask,
            "exit_price": float(exit_row.bid if side > 0 else exit_row.ask),
            "exit_age_ns": int(target-getattr(exit_row, closing+"_recv_ns")),
            "usd_bid": float(exit_row.usd_bid), "usd_ask": float(exit_row.usd_ask)}


def signals(observations, config):
    rows = []
    weeks = sorted(pd.to_datetime(observations.reopen_ns, unit="ns", utc=True).dt.strftime("%G-%V").unique())
    # Fixed calendar split, never optimize against validation/test. Entire correlated weeks stay together.
    train_end, val_end = max(1, int(len(weeks)*0.6)), max(1, int(len(weeks)*0.8))
    splits = {w: "train" if i < train_end else "validation" if i < val_end else "test"
              for i, w in enumerate(weeks)}
    for _, frame in observations.groupby(["session_id", "pair", "provider", "arm"], sort=True):
        week = pd.Timestamp(int(frame.iloc[0].reopen_ns), unit="ns", tz="UTC").strftime("%G-%V")
        for candidate in candidates(config.stage):
            rows.append({**trade_signal(frame, candidate, config), "split": splits[week], "week": week})
    result = pd.DataFrame(rows, dtype=object)
    for column in result:
        if column.endswith("_ns"):
            result[column] = pd.array(result[column], dtype="Int64")
    return result

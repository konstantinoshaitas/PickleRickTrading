"""Causal quote algebra, validity and basis-adjusted session observations."""
import numpy as np
import pandas as pd

from ..config import LEGS

# Futures price units are USD per currency unit, NOT USD per futures contract.
MULTIPLIER = {"6E": 125000, "6A": 100000, "6N": 100000}


def asof(left, right, time="ts_ns"):
    if left[time].isna().any() or right[time].isna().any():
        raise ValueError("missing as-of timestamp")
    right = right.copy()
    for column in right:
        if column.endswith("_ns") and column != time:
            right[column] = right[column].astype("Int64")
    return pd.merge_asof(left.sort_values(time), right.sort_values(time), on=time, direction="backward")


def validity(frame, now, stale_ns, quantity, allow_locked=False):
    reasons = pd.Series("", index=frame.index)
    checks = [
        (frame.bid.isna() | frame.ask.isna() | ~np.isfinite(frame.bid) | ~np.isfinite(frame.ask), "missing"),
        ((frame.bid <= 0) | (frame.ask <= 0), "nonpositive"),
        ((frame.bid > frame.ask) if allow_locked else (frame.bid >= frame.ask), "crossed_or_locked"),
        ((frame.bid_size < quantity) | (frame.ask_size < quantity) |
         frame.bid_size.isna() | frame.ask_size.isna(), "insufficient_size"),
        ((frame.indicative_flags != 0) | (frame.disabled_flags != 0), "flags"),
        ((now - frame.recv_ns > stale_ns) | (frame.recv_ns > now) | frame.recv_ns.isna(), "stale_or_future"),
    ]
    for label in ("bid", "ask"):
        column = f"{label}_recv_ns"
        if column in frame:
            checks.append(((now-frame[column] > stale_ns) | (frame[column] > now) |
                           frame[column].isna(), "stale_side"))
    for condition, name in reversed(checks):
        reasons.loc[condition.fillna(True)] = name
    return reasons


def synthetic(numerator, denominator, numerator_tenor, denominator_tenor):
    if numerator_tenor != denominator_tenor:
        raise ValueError("synthetic legs require the same contract tenor")
    bid, ask = numerator.bid / denominator.ask, numerator.ask / denominator.bid
    # Both legs must support the SAME base currency quantity (denominator leg is quote currency).
    bid_size = np.minimum(numerator.bid_size, denominator.ask_size / bid)
    ask_size = np.minimum(numerator.ask_size, denominator.bid_size / ask)
    return pd.DataFrame({"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size,
                         "recv_ns": np.minimum(numerator.recv_ns, denominator.recv_ns),
                         "event_ns": np.minimum(numerator.event_ns, denominator.event_ns),
                         "indicative_flags": 0, "disabled_flags": 0})


def future_at(quotes, times, contracts, pair, config, minimum_recv=None):
    legs = []
    for root in LEGS[pair]:
        contract = contracts[root]
        source = quotes[(quotes.venue == "cme") & (quotes.instrument_id == contract["instrument_id"])].copy()
        if minimum_recv is not None:
            source = source[source.recv_ns >= minimum_recv]
        source["source_recv_ns"] = source.recv_ns
        source["ts_ns"] = source.recv_ns
        source = source.drop(columns=["recv_ns"])
        merged = asof(pd.DataFrame({"ts_ns": times}), source).rename(columns={"source_recv_ns": "recv_ns"})
        legs.append(merged)
    if len(legs) == 2:
        result = synthetic(*legs, contracts[LEGS[pair][0]]["tenor"], contracts[LEGS[pair][1]]["tenor"])
    else:
        result = legs[0]
    result = result.reset_index(drop=True)
    result["ts_ns"] = np.asarray(times)
    result["rejection"] = validity(result, result.ts_ns, config.stale_ms*1e6, config.quantity, allow_locked=True)
    # Each direct leg must independently pass validity; reciprocal arithmetic must not mask a bad leg.
    for leg in legs:
        bad = validity(leg, result.ts_ns, config.stale_ms*1e6, config.quantity, allow_locked=True)
        result.loc[bad != "", "rejection"] = bad[bad != ""]
    result["book_state"] = np.where(result.bid == result.ask, "locked_indicative", "indicative")
    return result


def frozen_trade(trades, contracts, pair, close, stale_ns):
    prices, ages, arrivals = [], [], []
    for root in LEGS[pair]:
        data = trades[(trades.instrument_id == contracts[root]["instrument_id"]) &
                      (trades.recv_ns <= close) & (trades.event_ns <= close)].sort_values("recv_ns")
        if data.empty or close - int(data.iloc[-1].recv_ns) > stale_ns:
            raise ValueError("no sufficiently fresh last-tradable CME trade before closure")
        prices.append(float(data.iloc[-1].price))
        ages.append(close-int(data.iloc[-1].recv_ns))
        arrivals.append(int(data.iloc[-1].recv_ns))
    if any(not np.isfinite(p) or p <= 0 for p in prices):
        raise ValueError("invalid last-tradable price")
    return (prices[0] if len(prices) == 1 else prices[0]/prices[1]), max(ages), max(arrivals)


def prepare(tables, audits, config):
    observations, coverage = [], []
    quotes = tables["quotes"]
    if (quotes.recv_ns < quotes.event_ns).any():
        raise ValueError("quote receive time precedes event time")
    if quotes.duplicated(["venue", "provider", "pair", "instrument_id", "recv_ns"]).any():
        raise ValueError("ambiguous same-timestamp quote; normalize using provider sequence first")
    for session_id, audit_rows in audits.groupby("session_id"):
        for pair in config.pairs:
            roots = LEGS[pair]
            selected = audit_rows[audit_rows.root.isin(roots)]
            for provider in config.providers:
                entry = {"session_id": session_id, "pair": pair, "provider": provider}
                try:
                    if set(selected.root) != set(roots) or (selected.rejection != "").any():
                        raise ValueError("contract/session audit rejected or missing a leg")
                    contracts = {r["root"]: r for r in selected.to_dict("records")}
                    first = contracts[roots[0]]
                    for key in ("tenor", "close_ns", "preopen_ns", "reopen_ns", "boundary_ns"):
                        if len({contracts[root][key] for root in roots}) != 1:
                            raise ValueError("maturity/session mismatch between synthetic legs")
                    source = quotes[(quotes.venue == "broker") & (quotes.provider == provider) &
                                    (quotes.pair == pair)].sort_values("recv_ns").copy()
                    basis = source[(source.recv_ns >= first["basis_start_ns"]) &
                                   (source.recv_ns < first["basis_end_ns"])].reset_index(drop=True)
                    if basis.empty:
                        raise ValueError("no liquid preclosure broker basis observations")
                    future = future_at(quotes, basis.recv_ns.to_numpy(), contracts, pair, config)
                    valid = ((validity(basis, basis.recv_ns, config.stale_ms*1e6, config.quantity) == "") &
                             (future.rejection == ""))
                    differences = ((basis.bid+basis.ask)/2 - (future.bid+future.ask)/2)[valid]
                    if len(differences) < config.min_basis_samples:
                        raise ValueError("insufficient valid preclosure basis observations")
                    anchor = float(differences.median())
                    volatility = float(differences.std(ddof=1))
                    active = source[(source.recv_ns >= first["close_ns"]) &
                                    (source.recv_ns < first["boundary_ns"])].reset_index(drop=True)
                    if active.empty:
                        raise ValueError("no broker closure/reopening coverage")
                    active["ts_ns"] = active.recv_ns.astype("int64")
                    active["broker_rejection"] = validity(active, active.ts_ns, config.stale_ms*1e6,
                                                          config.quantity)
                    # Exit P&L in quote currency. Conversion uses same LP's contemporaneous USD quote.
                    ccy_pair = {"EURAUD": "AUDUSD", "EURNZD": "NZDUSD", "AUDNZD": "NZDUSD"}.get(pair)
                    active["usd_bid"], active["usd_ask"] = 1.0, 1.0
                    if ccy_pair:
                        conversion = quotes[(quotes.venue == "broker") & (quotes.provider == provider) &
                                            (quotes.pair == ccy_pair)].copy()
                        conversion["ts_ns"] = conversion.recv_ns
                        fx = asof(active[["ts_ns"]], conversion).reset_index(drop=True)
                        good = validity(fx, active.ts_ns, config.stale_ms*1e6, config.quantity) == ""
                        active["usd_bid"], active["usd_ask"] = fx.bid.where(good), fx.ask.where(good)
                    for reference in ("frozen", "preopen"):
                        frame = active.copy()
                        if reference == "frozen":
                            price, age, arrival = frozen_trade(tables["trades"], contracts, pair,
                                                              first["close_ns"], config.stale_ms*1e6)
                            frame["reference_bid"] = frame["reference_ask"] = price
                            frame["reference_age_ns"] = frame.ts_ns - first["close_ns"] + age
                            frame["reference_recv_ns"] = arrival
                            frame["book_state"] = "stale_last_trade_reference"
                            frame["reference_rejection"] = ""
                        else:
                            # Market-data availability includes a conservative feed latency allowance.
                            ref_times = frame.ts_ns.to_numpy() - int(config.latency_ms*1e6)
                            ref = future_at(quotes, ref_times, contracts, pair, config,
                                            minimum_recv=first["preopen_ns"])
                            frame["reference_bid"], frame["reference_ask"] = ref.bid, ref.ask
                            frame["reference_recv_ns"] = ref.recv_ns
                            frame["reference_age_ns"] = frame.ts_ns - ref.recv_ns
                            frame["book_state"] = ref.book_state
                            frame["reference_rejection"] = ref.rejection
                        frame["reference_mid"] = (frame.reference_bid+frame.reference_ask)/2
                        frame["basis"] = anchor
                        frame["basis_cutoff_ns"] = int(first["basis_end_ns"])
                        frame["prewindow_volatility"] = volatility
                        frame["raw_residual"] = (frame.bid+frame.ask)/2-frame.reference_mid
                        frame["adjusted_residual"] = frame.raw_residual-anchor
                        frame["target_bid"] = frame.reference_bid+anchor
                        frame["target_ask"] = frame.reference_ask+anchor
                        frame["pair"], frame["provider"], frame["session_id"] = pair, provider, session_id
                        frame["arm"] = first["kind"]+"_"+reference
                        for key in ("close_ns", "preopen_ns", "reopen_ns", "boundary_ns", "status_verified"):
                            frame[key] = first[key]
                        frame["reopen_recv_ns"] = int(first.get("reopen_recv_ns", first["reopen_ns"]))
                        frame["symbols"] = "/".join(contracts[r]["raw_symbol"] for r in roots)
                        frame["sunday_carry_anchor"] = first["kind"] == "sunday"
                        observations.append(frame)
                    coverage.append({**entry, "rejection": "", "broker_rows": len(active),
                                     "invalid_broker_rows": int((active.broker_rejection != "").sum()),
                                     "basis_samples": len(differences)})
                except ValueError as exc:
                    coverage.append({**entry, "rejection": str(exc), "broker_rows": 0})
    return (pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(),
            pd.DataFrame(coverage))

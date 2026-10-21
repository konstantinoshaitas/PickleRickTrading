"""Causal quote algebra, validity and basis-adjusted session observations."""
import numpy as np
import pandas as pd

from ..config import LEGS, NS
from ..products import MULTIPLIER, quote_in_usd  # noqa: F401 - normalization also imports MULTIPLIER


def asof(left, right, time="ts_ns"):
    if left[time].isna().any() or right[time].isna().any():
        raise ValueError("missing as-of timestamp")
    right = right.copy()
    for column in right:
        if column.endswith("_ns") and column != time:
            right[column] = right[column].astype("Int64")
    return pd.merge_asof(left.sort_values(time), right.sort_values(time), on=time, direction="backward")


def validity(frame, now, stale_ns=None, quantity=0, allow_locked=False, allow_crossed=False,
             require_size=False, require_clean_flags=False, side_max_age_ns=None,
             traded_side=None, untraded_side_max_age_ns=None):
    """Quote tests; optional rules reproduce the old strict gates when explicitly enabled."""
    reasons = pd.Series("", index=frame.index)
    checks = [
        (frame.bid.isna() | frame.ask.isna() | ~np.isfinite(frame.bid) | ~np.isfinite(frame.ask), "missing_price"),
        ((frame.bid <= 0) | (frame.ask <= 0), "nonpositive_price"),
        (((frame.bid > frame.ask) if allow_locked else (frame.bid >= frame.ask)) &
         (not allow_crossed), "crossed_book" if allow_locked else "crossed_or_locked_book"),
    ]
    if require_size:
        checks.append(((frame.bid_size <= 0) | (frame.ask_size <= 0) |
                       (frame.bid_size < quantity) | (frame.ask_size < quantity) |
                       frame.bid_size.isna() | frame.ask_size.isna(), "insufficient_size"))
    if require_clean_flags:
        checks.append(((frame.indicative_flags != 0) | (frame.disabled_flags != 0),
                       "disabled_or_indicative_flag"))
    if stale_ns is not None:
        checks.append((now - frame.recv_ns > stale_ns, "book_age_exceeded"))
    checks.extend([(frame.recv_ns > now, "received_after_decision"),
                   (frame.recv_ns.isna(), "missing_receive_time")])
    for label in ("bid", "ask"):
        column = f"{label}_recv_ns"
        if column in frame:
            if side_max_age_ns is not None and traded_side in (None, label):
                checks.append((now-frame[column] > side_max_age_ns, f"{label}_side_age_exceeded"))
            if (untraded_side_max_age_ns is not None and traded_side is not None and
                    label != traded_side):
                checks.append((now-frame[column] > untraded_side_max_age_ns,
                               f"{label}_side_age_exceeded"))
            checks.extend([(frame[column] > now, f"{label}_received_after_decision"),
                           (frame[column].isna(), f"missing_{label}_time")])
    for condition, name in reversed(checks):
        reasons.loc[condition.fillna(True)] = name
    return reasons


def broker_validity(frame, now, config, traded_side=None):
    """Default broker rule: real prices, bid below ask, no future quote, traded side <= 240 s."""
    return validity(frame, now,
                    stale_ns=None if config.broker_row_max_age_s is None else
                    config.broker_row_max_age_s*NS,
                    quantity=config.quantity or 0,
                    require_size=config.require_quote_size,
                    require_clean_flags=config.reject_flagged_quotes,
                    side_max_age_ns=config.traded_side_max_age_s*NS if traded_side else None,
                    traded_side=traded_side,
                    untraded_side_max_age_ns=None if config.untraded_side_max_age_s is None else
                    config.untraded_side_max_age_s*NS)

def reciprocal(frame):
    """Invert USD/foreign to foreign/USD or vice versa, swapping prices AND available base units."""
    result = frame.copy()
    result["bid"], result["ask"] = 1/frame.ask, 1/frame.bid
    result["bid_size"], result["ask_size"] = frame.ask_size*frame.ask, frame.bid_size*frame.bid
    if {"bid_recv_ns", "ask_recv_ns"} <= set(frame):
        result["bid_recv_ns"], result["ask_recv_ns"] = frame.ask_recv_ns, frame.bid_recv_ns
    return result


class Slices:
    """Rows of a quote/trade table grouped by key and ordered by receive time.

    Preparing a session used to re-filter the whole quote table once per window; an index built
    once lets every pass binary-search its own rows instead.
    """

    def __init__(self, frame, keys):
        self.frame, self.rows = frame, {}
        recv = frame.recv_ns.to_numpy()
        for key, rows in frame.groupby(list(keys), sort=False).indices.items():
            ordered = rows[np.argsort(recv[rows], kind="stable")]
            self.rows[key if isinstance(key, tuple) else (key,)] = recv[ordered], ordered

    def window(self, key, start=None, end=None, closed=False):
        """Rows with start <= recv_ns < end (<= end when closed), in receive order."""
        empty = np.empty(0, np.int64), np.empty(0, np.int64)
        recv, rows = self.rows.get(key, empty)
        lo = 0 if start is None else int(np.searchsorted(recv, start, "left"))
        hi = len(recv) if end is None else int(np.searchsorted(recv, end, "right" if closed else "left"))
        return self.frame.iloc[rows[lo:hi]].copy()

    def span(self, key, times):
        """Rows an as-of join over `times` can reach: their span plus one earlier row."""
        empty = np.empty(0, np.int64), np.empty(0, np.int64)
        recv, rows = self.rows.get(key, empty)
        times = np.asarray(times)
        if not len(times):
            return self.frame.iloc[rows[0:0]].copy()
        lo = max(int(np.searchsorted(recv, times.min(), "left"))-1, 0)
        hi = int(np.searchsorted(recv, times.max(), "right"))
        return self.frame.iloc[rows[lo:hi]].copy()


def indexed(table, keys):
    """A prepared index, or a one-off index for a caller that holds a plain table."""
    return table if isinstance(table, Slices) else Slices(table, keys)


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


def future_at(quotes, times, contracts, pair, config, minimum_recv=None, phase_end=None):
    quotes = indexed(quotes, ["venue", "instrument_id"])
    legs = []
    indicative = minimum_recv is not None
    mid_mode = indicative and config.indicative_crossed_mid
    stale_ns = config.indicative_max_age_s*NS if indicative else config.stale_ms*1e6
    for root in LEGS[pair]:
        contract = contracts[root]
        key = ("cme", contract["instrument_id"])
        if minimum_recv is None and phase_end is None:
            source = quotes.span(key, times)  # an as-of join never reads past its own span
        else:
            source = quotes.window(key, minimum_recv, phase_end)
            if minimum_recv is not None:
                source = source[source.event_ns >= minimum_recv]
            if phase_end is not None:
                source = source[source.event_ns < phase_end]
        source["source_recv_ns"] = source.recv_ns
        source["ts_ns"] = source.recv_ns
        source = source.drop(columns=["recv_ns"])
        merged = asof(pd.DataFrame({"ts_ns": times}), source).rename(columns={"source_recv_ns": "recv_ns"})
        legs.append(merged)
    crossed = np.logical_or.reduce([(leg.bid > leg.ask).to_numpy() for leg in legs])
    locked = np.logical_or.reduce([(leg.bid == leg.ask).to_numpy() for leg in legs])
    pricing = []
    for leg in legs:
        priced = leg.copy()
        if mid_mode:
            priced["bid"] = priced["ask"] = (leg.bid+leg.ask)/2
        pricing.append(priced)
    if len(legs) == 2:
        result = synthetic(*pricing, contracts[LEGS[pair][0]]["tenor"], contracts[LEGS[pair][1]]["tenor"])
    else:
        result = reciprocal(pricing[0]) if pair.startswith("USD") else pricing[0]
    result = result.reset_index(drop=True)
    result["ts_ns"] = np.asarray(times)
    result["rejection"] = validity(result, result.ts_ns, stale_ns, config.quantity or 0,
                                    allow_locked=True, allow_crossed=mid_mode,
                                    require_size=config.require_quote_size,
                                    require_clean_flags=config.reject_flagged_quotes,
                                    side_max_age_ns=stale_ns)
    # Validate ORIGINAL legs; mid arithmetic must not hide missing prices, flags or a phase boundary.
    for leg in legs:
        bad = validity(leg, result.ts_ns, stale_ns, config.quantity or 0,
                       allow_locked=True, allow_crossed=mid_mode,
                       require_size=config.require_quote_size,
                       require_clean_flags=config.reject_flagged_quotes,
                       side_max_age_ns=stale_ns)
        result.loc[bad != "", "rejection"] = bad[bad != ""]
    if phase_end is not None:
        result.loc[result.ts_ns >= phase_end, "rejection"] = "preopen_ended"
    result["reference_crossed"] = crossed
    result["book_state"] = np.where(crossed, "crossed_preopen_mid" if mid_mode else "crossed_book",
                                    np.where(locked, "locked_indicative", "indicative"))
    return result


def frozen_trade(trades, contracts, pair, close, stale_ns):
    trades = indexed(trades, ["instrument_id"])
    prices, ages, arrivals = [], [], []
    for root in LEGS[pair]:
        data = trades.window((contracts[root]["instrument_id"],), None, close, closed=True)
        data = data[data.event_ns <= close]
        if data.empty or close - int(data.recv_ns.iloc[-1]) > stale_ns:
            raise ValueError("FROZEN unavailable: no pre-close trade within frozen_max_age_s")
        prices.append(float(data.iloc[-1].price))
        ages.append(close-int(data.recv_ns.iloc[-1]))
        arrivals.append(int(data.recv_ns.iloc[-1]))
    if any(not np.isfinite(p) or p <= 0 for p in prices):
        raise ValueError("invalid last-tradable price")
    price = prices[0]/prices[1] if len(prices) == 2 else (1/prices[0] if pair.startswith("USD") else prices[0])
    return price, max(ages), max(arrivals)


def usd_rates(quotes, provider, currency, times, config):
    """Causal USD per currency unit; the signal checks only the side it actually uses."""
    if currency == "USD":
        return pd.DataFrame({"bid": np.ones(len(times)), "ask": np.ones(len(times)),
                             "bid_recv_ns": np.asarray(times), "ask_recv_ns": np.asarray(times)})
    pair, inverse = quote_in_usd(currency)
    quotes = indexed(quotes, ["venue", "provider", "pair"])
    source = quotes.span(("broker", provider, pair), times)
    source["ts_ns"] = source.recv_ns
    fx = asof(pd.DataFrame({"ts_ns": np.asarray(times)}), source).reset_index(drop=True)
    good = validity(fx, fx.ts_ns) == ""
    if inverse:
        fx = reciprocal(fx)
    return pd.DataFrame({"bid": fx.bid.where(good), "ask": fx.ask.where(good),
                         "bid_recv_ns": fx.bid_recv_ns, "ask_recv_ns": fx.ask_recv_ns})


def prepare(tables, audits, config):
    observations, coverage = [], []
    quotes = tables["quotes"]
    if (quotes.recv_ns < quotes.event_ns).any():
        raise ValueError("quote receive time precedes event time")
    if quotes.duplicated(["venue", "provider", "pair", "instrument_id", "recv_ns"]).any():
        raise ValueError("ambiguous same-timestamp quote; normalize using provider sequence first")
    # Index the tables once; every pass below then binary-searches its own window.
    broker = Slices(quotes, ["venue", "provider", "pair"])
    cme = Slices(quotes, ["venue", "instrument_id"])
    trades = Slices(tables["trades"], ["instrument_id"])
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
                    key = ("broker", provider, pair)
                    basis = broker.window(key, first["basis_start_ns"],
                                          first["basis_end_ns"]).reset_index(drop=True)
                    if basis.empty:
                        raise ValueError("no liquid preclosure broker basis observations")
                    future = future_at(cme, basis.recv_ns.to_numpy(), contracts, pair, config)
                    valid = ((broker_validity(basis, basis.recv_ns, config) == "") &
                             (future.rejection == ""))
                    differences = ((basis.bid+basis.ask)/2 - (future.bid+future.ask)/2)[valid]
                    if len(differences) < config.min_basis_samples:
                        raise ValueError("insufficient valid preclosure basis observations")
                    anchor = float(differences.median())
                    volatility = float(differences.std(ddof=1))
                    active = broker.window(key, first["close_ns"],
                                           first["boundary_ns"]).reset_index(drop=True)
                    if active.empty:
                        raise ValueError("no broker closure/reopening coverage")
                    active["ts_ns"] = active.recv_ns.astype("int64")
                    # Side freshness depends on trade direction and is evaluated in the signal loop.
                    active["broker_rejection"] = broker_validity(active, active.ts_ns, config)
                    # Quote currency converts exit P&L; base currency converts entry USD allocation.
                    for currency, prefix in ((pair[3:], "usd"), (pair[:3], "base_usd")):
                        fx = usd_rates(broker, provider, currency, active.ts_ns, config)
                        for label in ("bid", "ask"):
                            active[f"{prefix}_{label}"] = fx[label]
                            active[f"{prefix}_{label}_recv_ns"] = fx[f"{label}_recv_ns"]
                    frozen_rejection = ""
                    for reference in ("frozen", "preopen"):
                        frame = active.copy()
                        if reference == "frozen":
                            try:
                                price, age, arrival = frozen_trade(trades, contracts, pair,
                                                                  first["close_ns"], config.frozen_max_age_s*1e9)
                            except ValueError as exc:  # A missing last trade removes only the frozen arm.
                                frozen_rejection = str(exc)
                                continue
                            frame["reference_bid"] = frame["reference_ask"] = price
                            frame["reference_age_ns"] = frame.ts_ns - first["close_ns"] + age
                            frame["reference_recv_ns"] = arrival
                            frame["book_state"] = "frozen_last_trade"
                            frame["reference_rejection"] = ""
                            frame["reference_crossed"] = False
                        else:
                            # Market-data availability includes a conservative feed latency allowance.
                            ref_times = frame.ts_ns.to_numpy() - int(config.latency_ms*1e6)
                            ref = future_at(cme, ref_times, contracts, pair, config,
                                            minimum_recv=first["preopen_ns"], phase_end=first["reopen_ns"])
                            frame["reference_bid"], frame["reference_ask"] = ref.bid, ref.ask
                            frame["reference_recv_ns"] = ref.recv_ns
                            frame["reference_age_ns"] = frame.ts_ns - ref.recv_ns
                            frame["book_state"] = ref.book_state
                            frame["reference_crossed"] = ref.reference_crossed
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
                        frame["arm"] = first["kind"]+"_"+reference  # stable legacy IDs
                        frame["reference_rule"] = "FROZEN" if reference == "frozen" else "INDICATIVE"
                        for key in ("close_ns", "preopen_ns", "reopen_ns", "boundary_ns", "status_verified"):
                            frame[key] = first[key]
                        frame["reopen_recv_ns"] = int(first.get("reopen_recv_ns", first["reopen_ns"]))
                        frame["symbols"] = "/".join(contracts[r]["raw_symbol"] for r in roots)
                        frame["sunday_carry_anchor"] = first["kind"] == "sunday"
                        observations.append(frame)
                    coverage.append({**entry, "rejection": "", "frozen_rejection": frozen_rejection,
                                     "broker_rows": len(active),
                                     "invalid_broker_rows": int((active.broker_rejection != "").sum()),
                                     "basis_samples": len(differences)})
                except ValueError as exc:
                    coverage.append({**entry, "rejection": str(exc), "broker_rows": 0})
    return (pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(),
            pd.DataFrame(coverage))

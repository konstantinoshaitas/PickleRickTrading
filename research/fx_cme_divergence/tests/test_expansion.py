"""The new silent-money risks: inverse quotes, crossed per-leg mids, and historical rolls."""
from dataclasses import replace

import pandas as pd
import pytest

from ..config import NS, Config, ns
from ..data_access.official import calendar
from ..market_data.contracts import audit
from ..market_data.quotes import frozen_trade, future_at
from ..products import MULTIPLIER, PIP, PRODUCTS, ROOTS


@pytest.mark.parametrize("currency,root,units,tick_usd,pair,price,pip_usd", [
    ("EUR", "6E", 125000, 6.25, "EURUSD", 1.1, 12.5),
    ("GBP", "6B", 62500, 6.25, "GBPUSD", 1.3, 6.25),
    ("AUD", "6A", 100000, 5, "AUDUSD", .7, 10),
    ("NZD", "6N", 100000, 5, "NZDUSD", .6, 10),
    ("CAD", "6C", 100000, 5, "USDCAD", .72, 5.184),
    ("CHF", "6S", 125000, 6.25, "USDCHF", 1.2, 18),
    ("JPY", "6J", 12500000, 6.25, "USDJPY", .0065, 5.28125),
])
def test_root_units_tick_inverse_sides_and_broker_pip_value(currency, root, units, tick_usd, pair, price, pip_usd):
    tick = PRODUCTS[currency][2]
    assert MULTIPLIER[root] == units and tick*units == pytest.approx(tick_usd)
    quotes = pd.DataFrame([{"venue": "cme", "instrument_id": 1, "recv_ns": 10, "event_ns": 9,
                            "bid": price-tick, "ask": price+tick, "bid_size": 2*units, "ask_size": 3*units,
                            "indicative_flags": 0, "disabled_flags": 0}])
    contracts = {root: {"instrument_id": 1, "tenor": "2026-09"}}
    result = future_at(quotes, [10], contracts, pair, Config()).iloc[0]
    inverse = pair.startswith("USD")
    assert result.rejection == ""
    assert result.bid == pytest.approx(1/(price+tick) if inverse else price-tick)
    assert result.ask == pytest.approx(1/(price-tick) if inverse else price+tick)
    assert result.bid_size == pytest.approx(3*units*(price+tick) if inverse else 2*units)
    assert result.ask_size == pytest.approx(2*units*(price-tick) if inverse else 3*units)
    # Position equivalent to one CME contract, expressed in the broker pair's BASE units.
    base_quantity, usd_per_quote = (units*price, price) if inverse else (units, 1)
    assert base_quantity*PIP[pair]*usd_per_quote == pytest.approx(pip_usd)
    moved = quotes.copy()
    moved[["bid", "ask"]] *= 1.01
    assert (future_at(moved, [10], contracts, pair, Config()).bid.iloc[0] < result.bid) == inverse


def test_frozen_age_and_crossed_indicative_mid_do_not_hide_phase_or_missing_data():
    close = ns("2026-10-16T21:00:00Z")+123
    trades = pd.DataFrame({"instrument_id": [1], "recv_ns": [close-3*NS+7],
                           "event_ns": [close-3*NS], "price": [1.1]})
    contracts = {"6E": {"instrument_id": 1, "tenor": "2026-09"},
                 "6A": {"instrument_id": 2, "tenor": "2026-09"}}
    with pytest.raises(ValueError, match="frozen_max_age_s"):
        frozen_trade(trades, contracts, "EURUSD", close, 2*NS)
    assert frozen_trade(trades, contracts, "EURUSD", close, 5*NS) == (1.1, 3*NS-7, close-3*NS+7)
    quotes = pd.DataFrame([
        {"venue": "cme", "instrument_id": iid, "recv_ns": 10*NS, "event_ns": 10*NS-1,
         "bid": bid, "ask": ask, "bid_size": 1000000, "ask_size": 1000000,
         "indicative_flags": 0, "disabled_flags": 0}
        for iid, bid, ask in [(1, 1.2, 1.0), (2, .6, .8)]])
    config = Config(indicative_max_age_s=3600)
    off = future_at(quotes, [20*NS], contracts, "EURAUD", config, minimum_recv=5*NS)
    assert off.rejection.iloc[0] != ""  # age relaxation alone cannot un-cross a leg
    on = future_at(quotes, [20*NS, 30*NS], contracts, "EURAUD", replace(config, indicative_crossed_mid=True),
                   minimum_recv=5*NS, phase_end=30*NS)
    assert on.loc[0, "bid"] == on.loc[0, "ask"] == pytest.approx(1.1/.7)
    assert on.loc[0, "rejection"] == "" and on.loc[0, "reference_crossed"]
    assert on.loc[1, "rejection"] == "preopen_ended"
    quotes.loc[0, "bid"] = float("nan")
    assert future_at(quotes, [20*NS], contracts, "EURAUD", replace(config, indicative_crossed_mid=True),
                     minimum_recv=5*NS).rejection.iloc[0] == "missing_price"


def test_official_holiday_exclusions_and_roll_flag_on_every_root(tmp_path):
    config = Config(start="2026-04-23", end="2026-10-20")
    calendar(config, tmp_path)
    cal = pd.read_csv(tmp_path/"calendar.csv")
    assert set(cal.root) == set(ROOTS)
    # The official page really includes a Saturday close: keep it, require status verification.
    assert cal[cal.session_id == "sunday-2026-06-21"].close_ns.eq(ns("2026-06-20T22:00:00Z")).all()
    assert {"2026-06-20", "2026-07-04", "2026-05-25", "2026-09-07"} <= set(
        pd.read_csv(tmp_path/"calendar_exclusions.csv").date)
    cal = cal[cal.session_id.isin(["weekday-2026-09-10", "weekday-2026-09-14"])]
    known = ns("2026-09-01T00:00:00Z")
    definitions, volumes, statuses = [], [], []
    for i, root in enumerate(ROOTS):
        for j, expiry in enumerate([f"2026-09-{'15' if root == '6C' else '14'}", "2026-12-14"]):
            definitions.append({"root": root, "raw_symbol": f"{root}_{j}", "instrument_id": i*2+j,
                                "known_ns": known, "expiry_ns": ns(expiry+"T14:16:00Z"),
                                "last_trade_ns": ns(expiry+"T14:16:00Z"), "tenor": expiry[:7]})
            volumes.append({"root": root, "instrument_id": i*2+j, "available_ns": known+1,
                            "session_end_ns": known, "volume": 1000 if j == 0 else 100})
    for r in cal.itertuples():
        for phase in ("close", "preopen", "reopen"):
            t = getattr(r, phase+"_ns")
            statuses.append({"root": r.root, "session_id": r.session_id, "phase": phase,
                             "event_ns": t, "recv_ns": t+1})
    result = audit({"sessions": cal, "status": pd.DataFrame(statuses),
                    "definitions": pd.DataFrame(definitions), "volumes": pd.DataFrame(volumes)}, config)
    assert result.rejection.eq("").all() and result.groupby("root").roll.sum().eq(1).all()
    assert (result.volume_available_ns <= result.selection_cutoff_ns).all()


def test_terms_use_cad_one_day_rule_and_reject_unverified_expiry(tmp_path, monkeypatch):
    import json
    from importlib import import_module

    from ..config import TIR_COMMIT
    from ..data_access.official import contract_terms
    from ..data_access.storage import digest

    path = tmp_path/"definitions.bin"
    path.write_bytes(b"synthetic definitions")
    data = pd.DataFrame({"instrument_class": ["F"]*3, "raw_symbol": ["6CU6", "6EU6", "6EU7"],
                         "expiration": [ns("2026-09-15T14:16:00Z"), ns("2026-09-14T14:16:00Z"), 1],
                         "maturity_year": [2026, 2026, 2027], "maturity_month": [9]*3})
    monkeypatch.setattr(import_module("tir_data.databento.convert"), "load_frame", lambda _: data)
    downloads = tmp_path/"downloads.json"
    downloads.write_text(json.dumps([{"path": str(path), "sha256": digest(path), "cost_usd": 0,
                                      "tir_commit": TIR_COMMIT, "request": {"schema": "definition"}}]))
    assert contract_terms(downloads, tmp_path) == (2, 1)
    terms = pd.read_csv(tmp_path/"terms.csv").set_index("root")
    assert terms.loc["6C", "last_trade_ns"]-terms.loc["6E", "last_trade_ns"] == 86400*NS

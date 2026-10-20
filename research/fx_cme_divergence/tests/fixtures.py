"""Deterministic synthetic data generator. No real prices, quotes, identifiers or calendars."""
from pathlib import Path

import pandas as pd

from ..config import NS, PIP, ns
from ..data_access.storage import TABLES, seal
from ..market_data.contracts import chicago_ns
from ..products import PRODUCTS, broker_pairs


def make_bundle(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(config.start, config.end, inclusive="left", freq="D")
    # Four weeks/month, each with one Sunday and one weekday. Tiny synthetic event clock.
    days = [d for d in days if d.dayofweek in (0, 6) and d.day <= 21]
    prices = {"EUR": 1.1, "GBP": 1.3, "AUD": .7, "NZD": .6, "CAD": .72, "CHF": 1.2, "JPY": .0065}
    roots = {p[0]: (1001+i, prices[c]) for i, (c, p) in enumerate(PRODUCTS.items()) if p}
    known = ns(config.start+"T00:00:00Z") - 20*86400*NS
    expiry = ns(config.end+"T00:00:00Z") + 100*86400*NS
    definitions = [{"root": root, "raw_symbol": root+"_SYNTHETIC", "instrument_id": iid,
                    "known_ns": known, "expiry_ns": expiry, "last_trade_ns": expiry,
                    "tenor": "SYNTHETIC_SHARED"} for root, (iid, _) in roots.items()]
    sessions, status, volumes, quotes, trades = [], [], [], [], []
    for number, day in enumerate(days):
        day_text = str(day.date())
        kind = "sunday" if day.dayofweek == 6 else "weekday"
        reopening = chicago_ns(day_text, "17:00")
        closing = reopening-20*NS if kind == "weekday" else chicago_ns(str((day-pd.Timedelta(days=2)).date()), "16:00")
        preopen = reopening-10*NS
        sid = f"synthetic-{day_text}"
        basis_times = [closing-i*NS for i in range(10, 0, -1)]
        event_times = list(range(preopen, reopening+NS, NS)) + [reopening+60*NS,
                      reopening+120*NS, reopening+121*NS, reopening+122*NS, reopening+300*NS]
        for root, (iid, price) in roots.items():
            sessions.append({"session_id": sid, "root": root, "kind": kind, "basis_start_ns": closing-10*NS,
                             "basis_end_ns": closing, "close_ns": closing, "preopen_ns": preopen,
                             "reopen_ns": reopening, "boundary_ns": reopening+600*NS,
                             "calendar_source": "synthetic://calendar", "calendar_version": "synthetic-v1"})
            for phase, stamp in (("close", closing), ("preopen", preopen), ("reopen", reopening)):
                status.append({"session_id": sid, "root": root, "phase": phase,
                               "event_ns": stamp, "recv_ns": stamp+1})
            volumes.append({"root": root, "instrument_id": iid, "session_end_ns": closing-86400*NS,
                            "available_ns": closing-86399*NS, "volume": 1000})
            trades.append({"instrument_id": iid, "recv_ns": closing-NS, "event_ns": closing-NS-1,
                           "price": price})
            for stamp in basis_times+event_times:
                quotes.append({"venue": "cme", "provider": "CME", "pair": "", "instrument_id": iid,
                               "recv_ns": stamp, "event_ns": stamp-1, "bid": price-0.00001,
                               "ask": price+0.00001, "bid_size": 1000000, "ask_size": 1000000,
                               "bid_recv_ns": stamp, "ask_recv_ns": stamp,
                               "indicative_flags": 0, "disabled_flags": 0})
        for pair in broker_pairs(config.pairs):
            fair = prices.get(pair[:3], 1)/prices.get(pair[3:], 1)
            for provider in config.providers:
                for stamp in basis_times+event_times:
                    displacement = 80*PIP[pair]*(-1 if number % 2 == 0 else 1) if preopen <= stamp < reopening else 0
                    mid = fair+2*PIP[pair]+displacement
                    quotes.append({"venue": "broker", "provider": provider, "pair": pair, "instrument_id": 0,
                                   "recv_ns": stamp, "event_ns": stamp, "bid": mid-0.00005,
                                   "ask": mid+0.00005, "bid_size": 1000000, "ask_size": 1000000,
                                   "bid_recv_ns": stamp, "ask_recv_ns": stamp,
                                   "indicative_flags": 0, "disabled_flags": 0})
    tables = {name: pd.DataFrame(value) for name, value in zip(TABLES,
              (definitions, volumes, sessions, status, quotes, trades))}
    if not sessions:
        raise ValueError("demo range needs a Sunday or Monday in the first 21 days of a month")
    for name, frame in tables.items():
        frame.to_csv(directory/f"{name}.csv", index=False)
    seal(directory, [f"{name}.csv" for name in TABLES], {"synthetic": True, "source": "deterministic fixture v1"})
    return tables

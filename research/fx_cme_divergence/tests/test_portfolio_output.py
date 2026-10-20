"""USD 100k sizing and cash P&L checked by hand, independent of the signal fixture."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ..backtests.portfolio import account
from ..config import NS, Config, ns
from ..reporting.trades import cumulative_pnl, display_ledger, statistics, summaries
from ..signals.rules import entry_size


@pytest.mark.parametrize("pair,side,entry,exit_price,base_bid,base_ask,usd_bid,usd_ask,quantity,pips,gross", [
    ("EURUSD", 1, 1.25, 1.251, 1.2499, 1.25, 1, 1, 80000, 10, 80),
    ("EURUSD", -1, 1.25, 1.249, 1.25, 1.2501, 1, 1, 80000, 10, 80),
    ("USDJPY", 1, 150, 150.15, 1, 1, 1/150.15, 1/150.16, 100000, 15, 15000/150.15),
    ("USDJPY", -1, 150, 149.85, 1, 1, 1/149.85, 1/149.86, 100000, 15, 15000/149.85),
    ("EURAUD", 1, 1.6, 1.602, 1.2499, 1.25, .8, .8001, 80000, 20, 128),
    ("EURAUD", -1, 1.6, 1.598, 1.25, 1.2501, .8, .8001, 80000, 20, 128),
])
def test_long_short_jpy_and_cross_size_prices_and_usd_pnl(pair, side, entry, exit_price,
                                                         base_bid, base_ask, usd_bid, usd_ask,
                                                         quantity, pips, gross):
    config = Config(commission_bps=1, financing_quote=0)
    quote = SimpleNamespace(base_usd_bid=base_bid, base_usd_ask=base_ask)
    size, notional = entry_size(quote, side, config)
    assert size == pytest.approx(quantity) and notional == 100000
    start = ns("2026-10-18T21:50:00Z")+123
    ledger = pd.DataFrame([{"pair": pair, "status": "closed", "direction": side,
                            "quantity_base": size, "notional_usd": notional,
                            "entry_price": entry, "exit_price": exit_price,
                            "entry_ns": start, "exit_ns": start+720*NS, "signal_ns": start-NS,
                            "usd_bid": usd_bid, "usd_ask": usd_ask}])
    row = account(ledger, config).iloc[0]
    assert row.engine_entry_price == entry and row.engine_exit_price == exit_price
    assert row.gross_pnl_usd == pytest.approx(gross)
    fees_quote = quantity*(entry+exit_price)/10000
    assert row.scenario_pnl_usd == pytest.approx(gross-fees_quote*usd_bid)
    assert row.costs_usd == pytest.approx(fees_quote*usd_bid)
    assert row.pips == pytest.approx(pips) and row.hold_s == 720
    assert row.cost_status == "net_under_assumptions"
    shown = display_ledger(account(ledger, Config()))
    assert shown.loc[0, "cost_status"] == "gross_provisional_unknown_costs"
    assert shown.loc[0, "entry_utc"].startswith("2026-10-18T21:50:00")
    assert shown.loc[0, "entry_chicago"].startswith("2026-10-18T16:50:00")
    assert shown.loc[0, "entry_ns"] == start


def test_no_size_when_base_conversion_missing_and_legacy_quantity_is_explicit():
    quote = SimpleNamespace(base_usd_bid=np.nan, base_usd_ask=np.nan)
    assert np.isnan(entry_size(quote, 1, Config())[0])
    assert entry_size(quote, 1, Config(quantity=1000))[0] == 1000


def test_drawdown_nets_simultaneous_exits_and_never_fills_missing_pnl_with_zero():
    frame = pd.DataFrame({"status": ["closed"]*3, "exit_ns": [1, 1, 2],
                          "scenario_pnl_usd": [-100, 200, -50], "gross_pnl_usd": [-100, 200, -50],
                          "pips": [-10, 20, -5]})
    assert cumulative_pnl(frame).tolist() == [100, 50]
    stats = statistics(frame)
    assert stats["net_pnl_usd"] == 50 and stats["max_drawdown_usd"] == 50
    assert stats["win_rate_priced"] == pytest.approx(1/3)
    frame.loc[0, "scenario_pnl_usd"] = np.nan
    stats = statistics(frame)
    assert stats["unpriced_usd"] == 1 and np.isnan(stats["net_pnl_usd"])
    assert cumulative_pnl(frame).empty and np.isnan(stats["max_drawdown_usd"])


def test_portfolio_combines_pairs_not_parameter_or_provider_alternatives():
    t = ns("2026-10-18T22:02:00Z")
    rows = []
    for candidate in ("pips_10", "pips_20"):
        for provider in ("Broadway", "ICMLD"):
            for pair in ("EURUSD", "USDJPY"):
                rows.append({"candidate": candidate, "provider": provider, "pair": pair,
                             "arm": "sunday_frozen", "status": "closed", "rejection_reason": "",
                             "exit_ns": t, "reopen_ns": t-120*NS, "scenario_pnl_usd": 5,
                             "gross_pnl_usd": 6, "pips": 2})
    result = summaries(pd.DataFrame(rows))
    assert len(result["portfolio"]) == 4
    assert result["portfolio"].closed.eq(2).all() and result["portfolio"].net_pnl_usd.eq(10).all()
    assert len(result["by_pair"]) == 8 and len(result["by_week"]) == len(result["by_month"]) == 4

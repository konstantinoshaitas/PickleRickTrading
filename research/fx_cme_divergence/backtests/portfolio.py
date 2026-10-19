"""Stock vectorbt orders only. One independent quote-currency account per roundtrip."""
import numpy as np
import pandas as pd
import vectorbt as vbt


def roundtrip(entry_price, exit_price, direction, quantity, cash, commission_bps=0,
              slippage_bps=0, financing_quote=0):
    index = pd.DatetimeIndex(["2000-01-01T00:00:00Z", "2000-01-01T00:00:01Z"])
    prices = pd.Series([entry_price, exit_price], index=index)
    sizes = pd.Series([direction*quantity, -direction*quantity], index=index)
    portfolio = vbt.Portfolio.from_orders(
        close=prices, price=prices, size=sizes, size_type="amount", direction="both",
        init_cash=cash, cash_sharing=False, fees=commission_bps/10000,
        fixed_fees=pd.Series([0.0, financing_quote*quantity], index=index),
        slippage=slippage_bps/10000, allow_partial=False, raise_reject=True, lock_cash=True,
        freq="1s")
    if portfolio.orders.count() != 2 or portfolio.trades.closed.count() != 1:
        raise ValueError("engine did not fill exactly one complete roundtrip")
    trade = portfolio.trades.closed.records.iloc[0]
    return {"pnl": float(trade.pnl), "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price),
            "fees": float(trade.entry_fees+trade.exit_fees)}


def account(ledger, config):
    ledger = ledger.copy()
    for i, row in ledger.iterrows():
        ledger.loc[i, "quantity_base"] = config.quantity
        ledger.loc[i, "latency_ms"] = config.latency_ms
        ledger.loc[i, "slippage_bps"] = config.slippage_bps
        ledger.loc[i, "commission_bps"] = config.commission_bps if config.commission_bps is not None else np.nan
        ledger.loc[i, "financing_quote_per_unit"] = (
            config.financing_quote if config.financing_quote is not None else np.nan)
        ledger.loc[i, "cost_status"] = ("net_under_assumptions" if config.commission_bps is not None and
                                       config.financing_quote is not None else "gross_provisional_unknown_costs")
        if row.status != "closed":
            continue
        gross = roundtrip(row.entry_price, row.exit_price, int(row.direction), config.quantity, config.cash)
        # A second stock-engine run is a sensitivity, NOT a second accounting implementation.
        costs = roundtrip(row.entry_price, row.exit_price, int(row.direction), config.quantity, config.cash,
                          config.commission_bps or 0, config.slippage_bps, config.financing_quote or 0)
        ledger.loc[i, "gross_pnl_quote"] = gross["pnl"]
        ledger.loc[i, "scenario_pnl_quote"] = costs["pnl"]
        ledger.loc[i, "fees_quote"] = costs["fees"]
        ledger.loc[i, "engine_entry_price"] = costs["entry_price"]
        ledger.loc[i, "engine_exit_price"] = costs["exit_price"]
        # Positive quote-currency cash is sold at bid; a liability is bought back at ask.
        rate = row.usd_bid if costs["pnl"] >= 0 else row.usd_ask
        ledger.loc[i, "usd_conversion"] = rate
        ledger.loc[i, "scenario_pnl_usd"] = costs["pnl"]*rate
        if not np.isfinite(rate):
            ledger.loc[i, "cost_status"] = "unpriced_usd_conversion"
    return ledger

"""Trade display and realized all-pairs P&L, separated by candidate/provider/reference arm."""
import json

import numpy as np
import pandas as pd

TRADE_COLUMNS = ["candidate", "pair", "provider", "reference_rule", "arm", "session_id", "status",
                 "signal_utc", "entry_utc", "exit_utc", "signal_chicago", "entry_chicago", "exit_chicago",
                 "side", "entry_price", "exit_price", "engine_entry_price", "engine_exit_price",
                 "quantity_base", "notional_usd", "pips",
                 "gross_pnl_usd", "costs_usd", "scenario_pnl_usd", "cost_status", "exit_reason", "hold_s"]


def display_ledger(ledger):
    result = ledger.copy()
    for phase in ("signal", "entry", "exit"):
        column = phase+"_ns"
        times = pd.to_datetime(pd.array(result.get(column, [None]*len(result)), dtype="Int64"),
                               unit="ns", utc=True)
        result[phase+"_utc"] = times.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        result[phase+"_chicago"] = times.tz_convert("America/Chicago").strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return result


def cumulative_pnl(frame):
    """Realized USD cash only. Net simultaneous exits before computing drawdown."""
    closed = frame[frame.status == "closed"]
    if closed.empty:
        return pd.Series(dtype=float)
    values = pd.to_numeric(closed.scenario_pnl_usd, errors="coerce")
    if values.isna().any():
        return pd.Series(dtype=float)  # never turn an unpriced trade into zero
    return pd.Series(values.to_numpy(), index=closed.exit_ns.astype("int64")).groupby(level=0).sum().sort_index().cumsum()


def statistics(frame):
    closed = frame[frame.status == "closed"]
    pnl = pd.to_numeric(closed.get("scenario_pnl_usd", pd.Series(dtype=float)), errors="coerce")
    gross = pd.to_numeric(closed.get("gross_pnl_usd", pd.Series(dtype=float)), errors="coerce")
    pips = pd.to_numeric(closed.get("pips", pd.Series(dtype=float)), errors="coerce")
    priced = pnl.dropna()
    curve = cumulative_pnl(frame)
    complete = len(priced) == len(closed)
    cost_status = ("net_under_assumptions" if "cost_status" in frame and
                   frame.cost_status.eq("net_under_assumptions").all() else "provisional_unknown_costs")
    return {"cost_status": cost_status, "sessions": len(frame), "signals": int((frame.status != "no_signal").sum()),
            "closed": len(closed), "unfillable": int((frame.status == "unfillable_entry").sum()),
            "unpriced_open": int((frame.status == "unpriced_open").sum()),
            "unpriced_usd": len(closed)-len(priced),
            "gross_pnl_usd": gross.sum() if complete else np.nan,
            "net_pnl_usd": priced.sum() if complete else np.nan,
            "win_rate_priced": float((priced > 0).mean()) if len(priced) else np.nan,
            "average_pips": pips.mean(), "average_trade_usd": priced.mean(), "worst_trade_usd": priced.min(),
            "max_drawdown_usd": float((curve.cummax().clip(lower=0)-curve).max()) if len(curve) else
                                (0.0 if complete else np.nan)}


def summaries(ledger):
    """No pooling providers, reference rules or parameter candidates into an invented portfolio."""
    frame = ledger.copy()
    exits = pd.to_datetime(pd.array(frame.get("exit_ns", [None]*len(frame)), dtype="Int64"), unit="ns", utc=True)
    reopens = pd.to_datetime(pd.array(frame.reopen_ns, dtype="Int64"), unit="ns", utc=True)
    times = exits.where(exits.notna(), reopens)  # no-entry rows need a calendar bucket for rejection counts
    frame["exit_week"] = times.strftime("%G-%V")
    frame["exit_month"] = times.strftime("%Y-%m")
    cohorts = ["candidate", "provider", "arm"]
    results = {}
    for name, extra in (("portfolio", []), ("by_pair", ["pair"]),
                        ("by_week", ["exit_week"]), ("by_month", ["exit_month"])):
        keys = cohorts+extra
        results[name] = pd.DataFrame([{**dict(zip(keys, values)), **statistics(g)}
                                     for values, g in frame.groupby(keys, dropna=False, sort=True)])
    keys = cohorts+["pair", "status", "rejection_reason"]
    frame["rejection_reason"] = frame.get("rejection_reason", "")
    results["rejections"] = frame[frame.status != "closed"].groupby(keys, dropna=False).size().reset_index(name="count")
    return results


def table(name, frame, columns=None):
    """Small TOON table; JSON quoting on each cell prevents delimiter/newline ambiguity."""
    columns = [c for c in (columns or list(frame)) if c in frame]
    print(f"{name}[{len(frame)}]{{{','.join(columns)}}}:")
    for row in frame[columns].itertuples(index=False, name=None):
        values = [None if pd.isna(v) else round(float(v), 8) if isinstance(v, (float, np.floating))
                  else v.item() if isinstance(v, np.generic) else v for v in row]
        print("  "+",".join(json.dumps(v, allow_nan=False) for v in values))


def print_results(ledger):
    trades = display_ledger(ledger[ledger.status.isin(["closed", "unpriced_open"])])
    table("trades", trades, TRADE_COLUMNS)
    for name, frame in summaries(ledger).items():
        table(name, frame)

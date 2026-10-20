"""Static local research summaries. Never pool arms/providers into a fictitious equity feed."""
from html import escape
from pathlib import Path

import pandas as pd

from ..data_access.storage import verify
from .trades import TRADE_COLUMNS, cumulative_pnl, display_ledger, summaries


def report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    manifest = verify(directory, ["ledger.csv", "config.json"])
    columns = pd.read_csv(directory/"ledger.csv", nrows=0).columns
    ledger = pd.read_csv(directory/"ledger.csv", keep_default_na=False,
                         dtype={c: str for c in columns if c.endswith("_ns")})
    for column in ledger:
        if column.endswith("_ns"):
            ledger[column] = pd.array(ledger[column].replace("", None), dtype="Int64")
    label = "SYNTHETIC — NOT EMPIRICAL" if manifest["provenance"].get("synthetic") else "PILOT, SMALL SAMPLE"
    keys = ["pair", "provider", "arm", "candidate", "split"]
    summary = ledger.groupby(keys, dropna=False).agg(
        sessions=("status", "size"), closed=("status", lambda x: (x == "closed").sum()),
        unpriced_open=("status", lambda x: (x == "unpriced_open").sum()),
        unfillable=("status", lambda x: (x == "unfillable_entry").sum())).reset_index()
    if "scenario_pnl_usd" in ledger:
        ledger["scenario_pnl_usd"] = pd.to_numeric(ledger.scenario_pnl_usd, errors="coerce")
        totals = ledger.groupby(keys, dropna=False).scenario_pnl_usd.sum(min_count=1).rename("scenario_pnl_usd")
        summary = summary.merge(totals, on=keys, how="left")
    summary.to_csv(directory/"summary.csv", index=False)
    portfolios = summaries(ledger)
    for name, frame in portfolios.items():
        frame.to_csv(directory/f"{name}.csv", index=False)
    trades = display_ledger(ledger[ledger.status.isin(["closed", "unpriced_open"])])
    trades.to_csv(directory/"trades.csv", index=False)
    figures = directory/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("event_pnl.png", "cumulative_pnl.png"):
        (figures/name).unlink(missing_ok=True)
    if "scenario_pnl_usd" in ledger and ledger.scenario_pnl_usd.notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        # Only baseline first candidate, one pair/provider, arms separately: diagnostic, not investable equity.
        first = ledger.iloc[0]
        sample = ledger[(ledger.pair == first.pair) & (ledger.provider == first.provider) &
                        (ledger.candidate == first.candidate)].sort_values("reopen_ns")
        for arm, events in sample.groupby("arm"):
            curve = cumulative_pnl(events)
            ax.plot(range(len(curve)), curve.to_numpy(), label=arm)
        ax.set(title=f"{label}: independent-event cumulative P&L (not portfolio equity)",
               xlabel="Session", ylabel="Scenario USD P&L")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures/"event_pnl.png", dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for cohort, events in ledger.groupby(["candidate", "provider", "arm"]):
            curve = cumulative_pnl(events)
            if len(curve):
                ax.plot(pd.to_datetime(curve.index, unit="ns", utc=True), curve.to_numpy(), label=" / ".join(cohort))
        ax.set(title=f"{label}: all-pairs realized cumulative P&L", xlabel="Exit time UTC", ylabel="USD")
        ax.legend(fontsize="x-small")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figures/"cumulative_pnl.png", dpi=120)
        plt.close(fig)
    text = ("Unknown fees/financing remain provisional. Missing exits are open and unpriced, not zero-profit. "
            "A few Sundays and weekday closures cannot establish robust alpha. "
            "Crosses and providers are correlated. No configurations were fitted on validation/test. "
            "FROZEN is the last CME trade before close; INDICATIVE is the pre-open reference. "
            "CME catching up at reopen is not broker convergence. "
            "Each USD-notional trade is independent: no shared capital, netting or pair priority. "
            "Portfolio totals combine all pairs, but never mix candidates, providers or reference arms. "
            "Drawdown is realized cash by exit time, not mark-to-market equity. Unknown USD values are not zeros.")
    images = "".join(f"<img src='figures/{name}' alt='Realized P&amp;L diagnostic'>"
                     for name in ("cumulative_pnl.png", "event_pnl.png") if (figures/name).exists())
    (directory/"report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>FX/CME pilot</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1200px}table{border-collapse:collapse}"
        "td,th{padding:.4rem;border:1px solid #bbb}main{overflow:auto}</style>"
        f"<h1>{escape(label)}</h1><p>{escape(text)}</p>"
        f"{images or '<p>No priced closed trades to chart.</p>'}"
        f"<main><h2>Trades</h2>{trades[[c for c in TRADE_COLUMNS if c in trades]].to_html(index=False)}"
        + "".join(f"<h2>{escape(name)}</h2>{frame.to_html(index=False)}" for name, frame in portfolios.items())
        + f"<h2>By chronological split</h2>{summary.to_html(index=False)}</main></html>")
    return len(summary)

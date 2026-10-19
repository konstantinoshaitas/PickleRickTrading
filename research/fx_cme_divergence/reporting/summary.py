"""Static local research summaries. Never pool arms/providers into a fictitious equity feed."""
from html import escape
from pathlib import Path

import pandas as pd

from ..data_access.storage import verify


def report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    manifest = verify(directory, ["ledger.csv", "config.json"])
    ledger = pd.read_csv(directory/"ledger.csv", keep_default_na=False)
    label = "SYNTHETIC — NOT EMPIRICAL" if manifest["provenance"].get("synthetic") else "THREE-MONTH PILOT"
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
    figures = directory.parent/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    if "scenario_pnl_usd" in ledger and ledger.scenario_pnl_usd.notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        # Only baseline first candidate, one pair/provider, arms separately: diagnostic, not investable equity.
        first = ledger.iloc[0]
        sample = ledger[(ledger.pair == first.pair) & (ledger.provider == first.provider) &
                        (ledger.candidate == first.candidate)].sort_values("reopen_ns")
        for arm, events in sample.groupby("arm"):
            ax.plot(range(len(events)), events.scenario_pnl_usd.fillna(0).cumsum(), label=arm)
        ax.set(title=f"{label}: independent-event cumulative P&L (not portfolio equity)",
               xlabel="Session", ylabel="Scenario USD P&L")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures/"event_pnl.png", dpi=120)
        plt.close(fig)
    text = ("Unknown fees/financing remain provisional. Missing exits are open and unpriced, not zero-profit. "
            "Approximately 13 Sundays in a full three-month sample cannot establish robust alpha. "
            "Crosses and providers are correlated. No configurations were fitted on validation/test. "
            "Frozen CME is stale, not ground truth; CME catching up at reopen is not broker convergence.")
    (directory/"report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>FX/CME pilot</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1200px}table{border-collapse:collapse}"
        "td,th{padding:.4rem;border:1px solid #bbb}main{overflow:auto}</style>"
        f"<h1>{escape(label)}</h1><p>{escape(text)}</p><main>{summary.to_html(index=False)}</main></html>")
    return len(summary)

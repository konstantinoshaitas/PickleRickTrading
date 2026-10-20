"""Public CME input builders. No network, credentials, or inferred holiday hours."""
import json
import re
from pathlib import Path

import pandas as pd

from ..config import LEGS, NS, ns
from ..market_data.contracts import CALENDAR_URL, chicago_ns
from ..products import PRODUCTS, ROOTS, SOURCES

INPUTS = Path(__file__).resolve().parents[1]/"configs"
RETRIEVED = "2026-10-20"


def calendar(config, output):
    """Expand the seven verified, identical product schedules; fail outside their sourced window."""
    if config.start < "2026-04-23" or config.end > "2026-10-20":
        raise ValueError("bundled official calendar covers 2026-04-23..2026-10-20 only; supply a new sourced calendar")
    source = pd.read_csv(INPUTS/"cme_sessions_2026.csv")
    roots = sorted({r for pair in config.pairs for r in LEGS[pair]})
    rows = []
    for row in source.itertuples(index=False):
        close, preopen, reopen = map(ns, (row.close_utc, row.preopen_utc, row.reopen_utc))
        if not ns(config.start+"T00:00:00Z") <= reopen < ns(config.end+"T00:00:00Z"):
            continue
        for root in roots:
            rows.append({"session_id": row.session_id, "root": root, "kind": row.kind,
                         "basis_start_ns": close-31*60*NS, "basis_end_ns": close-60*NS,
                         "close_ns": close, "preopen_ns": preopen, "reopen_ns": reopen, "boundary_ns": reopen+600*NS,
                         "calendar_source": CALENDAR_URL,
                         "calendar_version": f"CME-product-schedules-retrieved-{RETRIEVED}"})
    if not rows:
        raise ValueError("no complete sourced sessions in requested range")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"calendar.csv", index=False)
    exclusions = pd.read_csv(INPUTS/"cme_exclusions_2026.csv")
    exclusions = exclusions[(exclusions.date >= config.start) & (exclusions.date < config.end)]
    exclusions.to_csv(output/"calendar_exclusions.csv", index=False)
    return len(rows)


def contract_terms(downloads, output):
    """Read ONLY receipted downloaded definitions, not a live/current contract shortcut.

    Check definition expiry against each root's CME termination rule. Business-day
    exceptions that disagree are explicitly excluded, never silently assigned a date.
    """
    from tir_data.databento.convert import load_frame

    from ..config import TIR_COMMIT
    from .storage import digest

    frames = []
    for item in json.loads(Path(downloads).read_text()):
        if item["request"]["schema"] != "definition":
            continue
        if (item.get("tir_commit") != TIR_COMMIT or item.get("cost_usd") != 0 or
                digest(item["path"]) != item["sha256"]):
            raise ValueError("definition receipt must be pinned, zero-cost and hash verified")
        frames.append(load_frame(item["path"]))
    if not frames:
        raise ValueError("download manifest has no definitions")
    definitions = pd.concat(frames, ignore_index=True)
    definitions = definitions[(definitions.instrument_class == "F") &
                              definitions.raw_symbol.str.fullmatch(r"6[EBANCSJ][FGHJKMNQUVXZ]\d{1,2}")]
    rows, excluded = [], []
    for symbol, group in definitions.groupby("raw_symbol"):
        root = re.match(r"6[EBANCSJ]", symbol).group()
        if root not in ROOTS:
            continue
        if any(group[c].nunique() != 1 for c in ("expiration", "maturity_year", "maturity_month")):
            raise ValueError(f"ambiguous historical terms for {symbol}")
        year, month = int(group.maturity_year.iloc[0]), int(group.maturity_month.iloc[0])
        wednesdays = pd.date_range(f"{year}-{month:02d}-01", periods=31)
        third = [d for d in wednesdays if d.month == month and d.dayofweek == 2][2]
        days = PRODUCTS[ROOTS[root]][3]
        day = third-pd.offsets.BDay(days)
        rule = chicago_ns(str(day.date()), "09:16")
        expiry = int(group.expiration.iloc[0])
        if expiry != rule:
            excluded.append({"raw_symbol": symbol, "definition_expiry_ns": expiry, "rule_expiry_ns": rule,
                             "reason": "CME business-day exception needs explicit verification"})
            continue
        rows.append({"root": root, "raw_symbol": symbol, "last_trade_ns": expiry, "tenor": f"{year}-{month:02d}",
                     "calendar_source": SOURCES[root], "retrieved": RETRIEVED})
    if not rows:
        raise ValueError("no verified outright contract terms")
    output = Path(output)
    pd.DataFrame(rows).sort_values(["root", "last_trade_ns"]).to_csv(output/"terms.csv", index=False)
    pd.DataFrame(excluded, columns=["raw_symbol", "definition_expiry_ns", "rule_expiry_ns", "reason"]).to_csv(
        output/"terms_exclusions.csv", index=False)
    return len(rows), len(excluded)

"""Thin installed tir_data adapters. No network calls happen on import."""
import json
import math
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from ..config import NS, TIR_COMMIT, ns
from .storage import digest, write_json

SCHEMAS = ("definition", "status", "mbp-1", "trades")


def session_windows(calendar, start, end, pad_ns=NS):
    """Merged [basis_start, boundary) quote windows for official-calendar sessions inside [start, end)."""
    import pandas as pd
    sessions = pd.read_csv(calendar)
    sessions = sessions[(sessions.reopen_ns >= ns(start)) & (sessions.reopen_ns < ns(end))]
    spans = sorted({(int(a)-pad_ns, int(b)+pad_ns) for a, b in zip(sessions.basis_start_ns, sessions.boundary_ns)})
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged or len(merged) > 400:
        raise ValueError("calendar yields no sessions or more than 400 quote windows")
    return merged


def requests(start, end, symbols, stype="parent", windows=None):
    """Definitions/status/trades cover the whole range; mbp-1 covers the declared session windows only."""
    from tir_data.databento import Request
    a, b = ns(start), ns(end)
    if a >= b or b-a > 100*86400*NS:
        raise ValueError("request requires 0 < range <= 100 days")
    if stype not in {"parent", "raw_symbol"}:
        raise ValueError("only parent or explicit raw_symbol requests are supported")
    if not symbols or len(symbols) > 12 or any(not s.startswith(("6E", "6A", "6N")) for s in symbols):
        raise ValueError("request needs at most 12 6E/6A/6N symbols")
    batch = []
    for schema in SCHEMAS:
        spans = windows if (windows and schema == "mbp-1") else [[a, b]]
        batch += [Request("GLBX.MDP3", schema, symbols, stype, lo, hi) for lo, hi in spans]
    return batch


def client(env_file=None, cache=None):
    from importlib.metadata import distribution

    from tir_data.config import load_settings
    from tir_data.databento import DatabentoClient
    direct = json.loads(distribution("tir-tools").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != TIR_COMMIT:
        raise ValueError("install the pinned tir-tools Git dependency from pyproject.toml")
    return DatabentoClient.from_settings(load_settings(env_file=env_file), cache)


def estimate_all(provider, batch, audit_path):
    estimates = []
    for request in batch:
        try:
            estimate = provider.estimate(request)
            cost = float(estimate.cost_usd)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("unknown cost")
            estimates.append({"request": request.describe(), "cost_usd": cost,
                              "billable_bytes": int(estimate.billable_bytes)})
        except Exception:  # noqa: BLE001 - never surface provider text (may echo credentials)
            write_json(audit_path, {"status": "blocked", "reason": "incomplete_cost_metadata",
                                   "estimates": estimates, "failed_request": request.describe()})
            raise ValueError("cost metadata unavailable; nothing downloaded") from None
    write_json(audit_path, {"status": "estimated", "estimates": estimates,
                           "total_cost_usd": sum(e["cost_usd"] for e in estimates)})
    return estimates


def check_cached_provenance(provider, batch):
    cache = getattr(provider, "cache", None)
    if cache is None:
        return
    for request in batch:
        path = cache.path_for(request)
        if path.exists():
            receipt = path.with_name(path.name+".research.json")
            if not receipt.exists():
                raise ValueError("existing raw cache lacks research provenance; use a separate empty cache root")
            record = json.loads(receipt.read_text())
            if (record.get("request") != request.describe() or record.get("sha256") != digest(path) or
                    record.get("cost_usd") != 0 or record.get("tir_commit") != TIR_COMMIT):
                raise ValueError("raw cache provenance mismatch")


def fetch_zero(provider, batch, audit_path, max_cost=0, max_bytes=5_000_000_000):
    if max_cost != 0:
        raise ValueError("authority is USD 0; positive caps/credit consumption are forbidden")
    if {r.schema for r in batch} != set(SCHEMAS) or len(batch) > 1600:
        raise ValueError("all four schemas (definitions/status/book/trades) must be estimated together")
    estimates = estimate_all(provider, batch, audit_path)
    if any(e["cost_usd"] != 0 for e in estimates):
        raise ValueError("positive estimated usage; no downloads or credit consumption permitted")
    if sum(e["billable_bytes"] for e in estimates) > max_bytes:
        raise ValueError("zero-cost but larger than --max-bytes; narrow windows (--calendar), nothing downloaded")
    check_cached_provenance(provider, batch)
    # Retain tir_data's second estimate+zero-cap gate. Our preflight includes cached requests too.
    results = provider.fetch(batch, max_cost=0.0)
    records = []
    for result in results:
        if result.status == "invalid" or not result.validation.ok:
            raise ValueError("invalid provider cache; inspect before reusing")
        records.append({"request": result.request.describe(), "path": str(result.path),
                        "sha256": digest(result.path), "status": result.status,
                        "cost_usd": 0.0, "tir_commit": TIR_COMMIT,
                        "fetched_at": datetime.now(UTC).isoformat()})
        write_json(result.path.with_name(result.path.name+".research.json"), records[-1])
    write_json(Path(audit_path).with_name("downloads.json"), records)
    return records


def clickhouse(creds_file):
    from tir_data.clickhouse import ClickHouseClient
    from tir_data.config import load_settings
    # Explicit opt-in only; no ambient read of another project's credentials.
    with Path(creds_file).expanduser().open("rb") as stream:
        prod = tomllib.load(stream)["prod"]
    url = urlsplit(prod["url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("ClickHouse configuration requires a credential-free https URL")
    settings = replace(load_settings(environ={}), clickhouse_host=url.hostname,
                       clickhouse_port=url.port or 8443, clickhouse_username=prod["username"],
                       clickhouse_password=prod["password"])
    return ClickHouseClient.from_settings(settings)


def broker_export(connection, start, end, pairs, providers, output, row_limit=200000, windows=None):
    """One-hour chunks with partition pruning. Refuse saturated chunks, never truncate silently.

    `windows` ([[lo_ns, hi_ns], ...], e.g. from session_windows) restricts the export to the session quote
    windows instead of every hour of [start, end)."""
    from .storage import seal
    a, b = ns(start), ns(end)
    if not (0 < b-a <= 100*86400*NS) or not (1 <= row_limit <= 1000000):
        raise ValueError("bounded dates and row limit 1..1000000 required")
    mapping = connection.query_df(
        "SELECT symbol_id, symbol_name FROM ICMarkets_Risk.symbols "
        "WHERE has({pairs:Array(String)}, symbol_name)", {"pairs": list(pairs)},
        settings={"max_execution_time": 15, "max_rows_to_read": 1000000, "max_result_rows": 1000})
    if set(mapping.symbol_name) != set(pairs) or mapping.symbol_name.duplicated().any():
        raise ValueError("broker symbol mapping missing or ambiguous")
    tables = connection.query_df(
        "SELECT name FROM system.tables WHERE database = 'ticks_and_markouts' "
        "AND has({tables:Array(String)}, name)", {"tables": [f"quotes_{p}" for p in providers]},
        settings={"max_execution_time": 15})
    if set(tables.name) != {f"quotes_{p}" for p in providers}:
        raise ValueError("provider tables not discovered")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files, empty = [], 0
    settings = {"max_execution_time": 30, "max_threads": 2, "max_rows_to_read": 10000000,
                "max_result_rows": row_limit, "result_overflow_mode": "throw"}
    spans = [(max(a, lo), min(b, hi)) for lo, hi in (windows or [(a, b)]) if max(a, lo) < min(b, hi)]
    chunks = [(hour, max(wa, hour*3600*NS), min(wb, (hour+1)*3600*NS)) for wa, wb in spans
              for hour in range(wa // (3600*NS), (wb-1)//(3600*NS)+1)]
    if not chunks or len({hour for hour, _, _ in chunks}) != len(chunks):
        raise ValueError("broker windows are empty or share an hour chunk")
    for lp in providers:
        for hour, lo, hi in chunks:
            # tir loader is bounded/readonly, but omits size; use its table/parameters with enriched SQL.
            from tir_data.sources.lp_quotes import quote_parameters
            params = quote_parameters(lp, mapping.symbol_id, lo, hi)
            from importlib.resources import files as resources
            sql = resources("tir_data.sources").joinpath("sql/lp_quotes_top_of_book.sql").read_text()
            sql = sql.replace("    disabled_flags\n", "    disabled_flags, size, quote_flags, time_sent\n")
            frame = connection.query_df(sql + f" LIMIT {row_limit + 1}", params, settings=settings)
            if len(frame) >= row_limit:
                raise ValueError("broker chunk at row cap; narrow request, do not use partial export")
            if frame.empty:  # closed market (weekend, maintenance): no columns come back, nothing to keep
                empty += 1
                continue
            name = f"{lp}_{hour}.csv"
            frame.merge(mapping, left_on="core_symbol", right_on="symbol_id").to_csv(output/name, index=False)
            files.append(name)
    seal(output, files, {"source": "read-only ClickHouse LP top of book", "synthetic": False,
                         "start_ns": a, "end_ns": b, "pairs": list(pairs), "providers": list(providers),
                         "empty_chunks": empty})
    return len(files)

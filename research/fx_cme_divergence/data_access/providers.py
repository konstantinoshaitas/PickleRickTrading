"""Thin installed tir_data adapters. No network calls happen on import."""
import json
import math
import re
import tomllib
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlsplit

from ..config import NS, TIR_COMMIT, ns
from ..products import ROOTS, broker_pairs
from .storage import digest, write_json

SCHEMAS = ("definition", "status", "mbp-1", "trades")


def session_windows(calendar, start, end, pad_ns=NS, split_closed=False):
    """Merge session windows; broker exports omit the closed interval (including weekends).

    Keep Databento's existing contiguous windows so raw requests/receipts remain unchanged.
    """
    import pandas as pd
    sessions = pd.read_csv(calendar)
    sessions = sessions[(sessions.reopen_ns >= ns(start)) & (sessions.reopen_ns < ns(end))]
    intervals = [(sessions.basis_start_ns, sessions.boundary_ns)]
    if split_closed:
        intervals = [(sessions.basis_start_ns, sessions.basis_end_ns), (sessions.preopen_ns, sessions.boundary_ns)]
    spans = sorted({(int(a)-pad_ns, int(b)+pad_ns) for starts, ends in intervals for a, b in zip(starts, ends)})
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged or len(merged) > 800:
        raise ValueError("calendar yields no sessions or more than 800 quote windows")
    return merged


def requests(start, end, symbols, stype="parent", windows=None):
    """At most 366 days; every provider request is <=90 days, without changing schemas/contracts."""
    from tir_data.databento import Request
    a, b = ns(start), ns(end)
    if a >= b or b-a > 366*86400*NS:
        raise ValueError("request requires 0 < range <= 366 days")
    if stype not in {"parent", "raw_symbol"}:
        raise ValueError("only parent or explicit raw_symbol requests are supported")
    pattern = r"(6[EBANCSJ])" + (r"\.FUT" if stype == "parent" else r"[FGHJKMNQUVXZ]\d{1,2}")
    if (not symbols or len(symbols) > 12 or len(set(symbols)) != len(symbols) or
            any(not re.fullmatch(pattern, s) or s[:2] not in ROOTS for s in symbols)):
        raise ValueError("request needs at most 12 unique supported CME outright/parent symbols")
    if windows is not None and (not windows or any(not a <= lo < hi <= b for lo, hi in windows)):
        raise ValueError("quote windows must be nonempty and contained in request bounds")
    batch = []
    step = 90*86400*NS
    for schema in SCHEMAS:
        spans = windows if (windows and schema == "mbp-1") else [[a, b]]
        for lo, hi in spans:
            batch += [Request("GLBX.MDP3", schema, symbols, stype, t, min(t+step, hi))
                      for t in range(lo, hi, step)]
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
    from tir_data.clickhouse.client import make_connection
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
    return ClickHouseClient(make_connection(settings, send_receive_timeout=1800))


def broker_plan(hour_counts):
    """One stream up to 20M rows / 31 days; otherwise at most three row-balanced hour slices."""
    if hour_counts.empty:
        return [([], 0)]
    hours = hour_counts.sort_values("epoch_hour")
    cumulative = hours.rows.cumsum().tolist()
    total = int(cumulative[-1])
    span = int(hours.epoch_hour.iloc[-1])-int(hours.epoch_hour.iloc[0])+1
    parts = min(3, len(hours), max(1, math.ceil(total/20_000_000), math.ceil(span/(31*24))))
    cuts = [0]
    for i in range(1, parts):
        cuts.append(max(cuts[-1]+1, min(len(hours)-(parts-i), bisect_left(cumulative, total*i/parts)+1)))
    cuts.append(len(hours))
    return [(hours.epoch_hour.iloc[a:b].astype(int).tolist(), int(hours.rows.iloc[a:b].sum()))
            for a, b in pairwise(cuts)]


def stream_broker(connection, sql, params, settings, path, expected, connection_factory=None):
    """Write only bounded Arrow batches; never publish a truncated Parquet part."""
    import pyarrow.parquet as pq

    temporary = path.with_suffix(".parquet.tmp")
    rows = 0
    try:
        with ExitStack() as stack:
            if connection_factory is not None:
                connection = connection_factory()
                stack.callback(connection.connection.close)
            stream = stack.enter_context(connection.connection.query_arrow_stream(sql, params, settings=settings,
                                                                                  use_strings=True))
            writer = None
            for batch in stream:
                if writer is None:
                    writer = stack.enter_context(pq.ParquetWriter(temporary, batch.schema, compression="zstd"))
                writer.write_batch(batch)
                rows += batch.num_rows
        if rows != expected:
            raise ValueError(f"broker row count mismatch: expected {expected}, streamed {rows}; export refused")
        if writer is not None:
            temporary.replace(path)
        return rows
    finally:
        temporary.unlink(missing_ok=True)


def broker_export(connection, start, end, pairs, providers, output, windows=None, connection_factory=None, on_plan=None):
    """One count-by-hour query per provider, then 1–3 streamed Parquet parts according to size."""
    from .storage import seal
    a, b = ns(start), ns(end)
    if not 0 < b-a <= 366*86400*NS:
        raise ValueError("bounded dates required")
    if not providers or len(set(providers)) != len(providers) or not set(providers) <= {"Broadway", "ICMLD", "Finalto"}:
        raise ValueError("choose unique supported providers")
    spans = sorted({(max(a, lo), min(b, hi)) for lo, hi in (windows or [(a, b)]) if max(a, lo) < min(b, hi)})
    if not spans or len(spans) > 800:
        raise ValueError("broker windows are empty or exceed 800 ranges")
    pairs = broker_pairs(pairs)
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
    (output/"manifest.json").unlink(missing_ok=True)  # A failed rerun must not leave a usable partial bundle.
    files, counts = [], {}
    settings = {"readonly": 1, "max_execution_time": 1800, "max_threads": 2,
                "max_rows_to_read": 0, "max_result_rows": 0, "result_overflow_mode": "throw",
                "max_block_size": 65536}
    hours = sorted({hour for lo, hi in spans for hour in range(lo//(3600*NS), (hi-1)//(3600*NS)+1)})
    params = {"symbol_ids": [int(i) for i in mapping.symbol_id], "names": list(mapping.symbol_name), "hours": hours}
    ranges = []
    for i, (lo, hi) in enumerate(spans):
        params.update({f"lo{i}": lo, f"hi{i}": hi})
        ranges.append(f"(time_received >= {{lo{i}:UInt64}} AND time_received < {{hi{i}:UInt64}})")
    source = (" FROM ticks_and_markouts.{table:Identifier} "
              "PREWHERE core_symbol IN {symbol_ids:Array(UInt32)} AND epoch_hour IN {hours:Array(UInt32)} "
              "WHERE depth = 0 AND side IN (0, 1) AND (" + " OR ".join(ranges) + ")")
    for lp in providers:
        params["table"] = f"quotes_{lp}"
        hours = connection.query_df("SELECT epoch_hour, count() AS rows" + source + " GROUP BY epoch_hour",
                                    params, settings=settings)
        plan = broker_plan(hours)
        expected = sum(rows for _, rows in plan)
        if on_plan:
            on_plan(provider=lp, parts=len(plan), expected_rows=expected)
        if len(plan) > 1 and connection_factory is None:
            raise ValueError("parallel broker export requires a separate connection factory")
        sql = ("SELECT core_symbol, time_received, side, toFloat64(price) AS price, indicative_flags, "
               "disabled_flags, size, quote_flags, time_sent, "
               "transform(core_symbol, {symbol_ids:Array(UInt32)}, {names:Array(String)}, '') AS symbol_name" + source)
        names = [f"{lp}_{i}.parquet" for i in range(len(plan))]
        with ThreadPoolExecutor(max_workers=len(plan)) as pool:
            jobs = [pool.submit(stream_broker, connection, sql, {**params, "hours": hours}, settings,
                                output/name, rows, connection_factory if len(plan) > 1 else None)
                    for name, (hours, rows) in zip(names, plan)]
            results = [job.result() for job in jobs]
        if sum(results) != expected:
            raise ValueError("broker total row count mismatch; export refused")
        counts[lp] = expected
        files.extend(name for name, rows in zip(names, results) if rows)
    seal(output, files, {"source": "read-only ClickHouse LP top of book", "synthetic": False,
                         "start_ns": a, "end_ns": b, "pairs": list(pairs), "providers": list(providers),
                         "rows": counts, "windows": spans})
    return sum(counts.values())

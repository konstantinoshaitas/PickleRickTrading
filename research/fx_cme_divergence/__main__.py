"""Thin script CLI. Offline commands never read credentials or construct a provider."""
import argparse
import json
import sys
from pathlib import Path

from .config import PAIRS, Config, load_config
from .data_access.storage import read_bundle, seal, verify, write_json

ROOT = Path(__file__).resolve().parent
PROG = "python -m research.fx_cme_divergence"


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message + "\n" + self.format_help())


def parser():
    p = Parser(prog=PROG, description="Auditable FX/CME pilot; downloads require cost exactly USD 0.",
               allow_abbrev=False)
    p.add_argument("--version", action="version", version="0.1.0")
    subs = p.add_subparsers(dest="command", parser_class=Parser)
    descriptions = {
        "demo": "Generate deterministic synthetic inputs, never empirical data",
        "estimate": "Free metadata estimate for all four CME schemas; no data download",
        "fetch": "Download only after complete fresh USD 0 estimates (no credits)",
        "broker-fetch": "Bounded read-only LP export using explicit ch-axi credentials",
        "normalize": "Offline DBN/LP normalization with official contract/session inputs",
        "audit": "Audit historical contracts, prior volume and product sessions first",
        "prepare": "Prepare causal references, basis and quote validity after contract audit",
        "backtest": "Run a bounded candidate stage through stock vectorbt orders",
        "report": "Write static tables/chart/HTML locally from a hashed result bundle",
    }
    for command, desc in descriptions.items():
        sub = subs.add_parser(command, help=desc, description=desc, allow_abbrev=False,
                              epilog=f"Example: {PROG} {command} --help\n"
                                     f"Workflow: {PROG} demo; then audit, prepare, backtest, report.")
        sub.add_argument("--workspace", type=Path, default=ROOT,
                         help="data/ and outputs/ root (default: installed experiment directory)")
        sub.add_argument("--config", type=Path, help="strict settings YAML; no credentials; flags override")
        sub.add_argument("--start", help="UTC inclusive YYYY-MM-DD (default: 3 completed months ago)")
        sub.add_argument("--end", help="UTC exclusive YYYY-MM-DD (default: current month's first day)")
        sub.add_argument("--pairs", help="comma-separated supported pairs (default: all six)")
        sub.add_argument("--providers", help="comma-separated Broadway,ICMLD,Finalto (default: first two)")
        if command in {"estimate", "fetch"}:
            sub.add_argument("--request-start", required=True, help="inclusive timezone-aware request timestamp")
            sub.add_argument("--request-end", required=True, help="exclusive request timestamp; <=100 days")
            sub.add_argument("--symbols", default="6E.FUT,6A.FUT,6N.FUT", help="bounded parent/raw CME symbols")
            sub.add_argument("--stype", choices=("parent", "raw_symbol"), default="parent")
            sub.add_argument("--calendar", type=Path, help="official-calendar sessions CSV; limits mbp-1 to its "
                             "basis->boundary windows (all windows are estimated; definitions/status/trades stay full range)")
            sub.add_argument("--env-file", type=Path, help="explicit ~/.config/tir-data/databento.env; never auto-read")
            sub.add_argument("--max-cost", type=float, default=0, help="must be 0; no paid/credit authority")
            sub.add_argument("--max-bytes", type=int, default=5_000_000_000,
                             help="fetch refuses larger zero-cost totals (default 5 GB billable)")
            sub.add_argument("--cache-root", type=Path, help="default data/raw/tir (outside TIR_Tools repository)")
        if command == "broker-fetch":
            sub.add_argument("--creds-file", type=Path, required=True,
                             help="explicit ~/.config/ch-axi/creds.toml; [prod] reused securely")
            sub.add_argument("--row-limit", type=int, default=200000, help="per hourly chunk; saturation refuses")
            sub.add_argument("--calendar", type=Path, help="official-calendar sessions CSV; exports only its "
                             "basis->boundary windows instead of every hour of [start, end); --start must reach the "
                             "Friday basis of a first-Sunday session")
        if command == "normalize":
            for name in ("downloads", "broker-chunks", "calendar", "contract-terms", "broker-size-units"):
                sub.add_argument("--"+name, type=Path, required=True)
        if command in {"audit", "prepare"}:
            sub.add_argument("--inputs", type=Path, help="default data/inputs/bundle; hashed normalized CSVs")
            sub.add_argument("--allow-calendar-fallback", action="store_true", default=None,
                             help="explicitly allow sourced calendar if status absent; marked in audit")
        if command in {"prepare", "backtest"}:
            sub.add_argument("--stage", choices=("baseline", "scale", "confirm", "exits"), help="one small stage")
        if command == "backtest":
            sub.add_argument("--prepared", type=Path, help="default data/processed/prepared")
            sub.add_argument("--commission-bps", type=float, help="per side; unknown default => provisional")
            sub.add_argument("--slippage-bps", type=float, help="adverse per side, default 0")
            sub.add_argument("--financing-quote", type=float, help="roundtrip per base unit; unknown => provisional")
        if command == "report":
            sub.add_argument("--results", type=Path, help="default outputs/results/<stage>")
            sub.add_argument("--stage", choices=("baseline", "scale", "confirm", "exits"), help="result stage")
    return p


def settings(args):
    overrides = {k: getattr(args, k, None) for k in ("start", "end", "stage", "commission_bps",
                 "slippage_bps", "financing_quote", "allow_calendar_fallback")}
    for key in ("pairs", "providers"):
        if getattr(args, key, None) is not None:
            overrides[key] = tuple(getattr(args, key).split(","))
    return load_config(args.config, **overrides)


def output(**values):
    # Flat TOON scalar map; always quote strings to prevent delimiter/newline ambiguity.
    for key, value in values.items():
        print(f"{key}: {json.dumps(value, allow_nan=False, default=str)}")


def run(args, config):
    ws = args.workspace.expanduser().resolve()
    for folder in ("data/inputs", "data/raw", "data/processed", "outputs/audits", "outputs/results", "outputs/figures"):
        (ws/folder).mkdir(parents=True, exist_ok=True)
    command = args.command
    if command == "demo":
        from .tests.fixtures import make_bundle
        path = ws/"data/inputs/bundle"
        make_bundle(path, config)
        output(status="synthetic_only", inputs=path, pairs=len(PAIRS))
    elif command in {"estimate", "fetch"}:
        from .data_access.providers import client, estimate_all, fetch_zero, requests, session_windows
        if args.max_cost != 0:
            raise UsageError("--max-cost must be 0; no credit consumption permitted")
        windows = session_windows(args.calendar, args.request_start, args.request_end) if args.calendar else None
        batch = requests(args.request_start, args.request_end, args.symbols.split(","), args.stype, windows)
        path = ws/"outputs/audits/cost.json"
        try:
            provider = client(args.env_file, args.cache_root or ws/"data/raw/tir")
            if command == "estimate":
                estimates = estimate_all(provider, batch, path)
                total = sum(e["cost_usd"] for e in estimates)
                output(status="zero_cost" if total == 0 else "blocked_positive_cost", total_cost_usd=total,
                       billable_bytes=sum(e["billable_bytes"] for e in estimates), requests=len(batch), audit=path)
            else:
                results = fetch_zero(provider, batch, path, max_bytes=args.max_bytes)
                output(status="fetched_or_valid_cache", requests=len(results), audit=path)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - provider errors can echo credentials
            raise ValueError("data access failed; verify pinned install, explicit credential file and entitlement; no secrets logged") from None
    elif command == "broker-fetch":
        from .data_access.providers import broker_export, clickhouse, session_windows
        if not 1 <= args.row_limit <= 1000000:
            raise UsageError("--row-limit must be 1..1000000")
        start, end = config.start+"T00:00:00Z", config.end+"T00:00:00Z"
        windows = session_windows(args.calendar, start, end) if args.calendar else None
        try:
            count = broker_export(clickhouse(args.creds_file), start, end, config.pairs, config.providers,
                                  ws/"data/raw/broker", args.row_limit, windows)
        except Exception:  # noqa: BLE001 - provider errors can echo credentials
            raise ValueError("bounded broker export failed or hit limits; narrow dates/check explicit credentials") from None
        output(status="exported", chunks=count, path=ws/"data/raw/broker")
    elif command == "normalize":
        from .data_access.normalize import normalize
        normalize(args.downloads, args.broker_chunks, args.calendar, args.contract_terms,
                  args.broker_size_units, ws/"data/inputs/bundle")
        output(status="normalized", inputs=ws/"data/inputs/bundle")
    elif command in {"audit", "prepare"}:
        from .market_data.contracts import audit
        inputs = args.inputs or ws/"data/inputs/bundle"
        tables, manifest = read_bundle(inputs)
        synthetic = manifest["provenance"].get("synthetic", False)
        audits = audit(tables, config, synthetic)
        if audits.empty:
            raise ValueError("no sessions in requested date range")
        audit_path = ws/"outputs/audits/contracts.csv"
        audits.to_csv(audit_path, index=False)
        if command == "audit":
            output(status="synthetic_audit" if synthetic else "audit", sessions=len(audits),
                   rejected=int((audits.rejection != "").sum()), path=audit_path)
        else:
            from .market_data.quotes import prepare
            observations, coverage = prepare(tables, audits, config)
            coverage.to_csv(ws/"outputs/audits/coverage.csv", index=False)
            if observations.empty:
                raise ValueError("no valid prepared sessions; inspect contract/coverage audits")
            path = ws/"data/processed/prepared"
            path.mkdir(parents=True, exist_ok=True)
            observations.to_parquet(path/"observations.parquet", index=False)
            coverage.to_csv(path/"coverage.csv", index=False)
            audits.to_csv(path/"contracts.csv", index=False)
            write_json(path/"config.json", config.dump())
            seal(path, ["observations.parquet", "config.json", "coverage.csv", "contracts.csv"],
                 {"synthetic": synthetic, "source": "causal prepared normalized inputs",
                  "input_manifest": manifest})
            output(status="prepared_synthetic" if synthetic else "prepared", rows=len(observations), path=path)
    elif command == "backtest":
        import pandas as pd

        from .backtests.portfolio import account
        from .signals.rules import candidates, signals
        path = args.prepared or ws/"data/processed/prepared"
        manifest = verify(path, ["observations.parquet", "config.json"])
        original = Config(**json.loads((path/"config.json").read_text()))
        # Prepared market-data/causality settings cannot change after preparation.
        permitted = {"stage", "commission_bps", "slippage_bps", "financing_quote"}
        for key, value in original.dump().items():
            if key not in permitted and value != config.dump()[key]:
                raise ValueError(f"prepared config differs at {key}; reuse original config or prepare again")
        observations = pd.read_parquet(path/"observations.parquet")
        output(candidate_count=len(candidates(config.stage)), stage=config.stage)
        ledger = account(signals(observations, config), config)
        results = ws/f"outputs/results/{config.stage}"
        results.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(results/"ledger.csv", index=False)
        write_json(results/"config.json", config.dump())
        seal(results, ["ledger.csv", "config.json"], {"synthetic": manifest["provenance"].get("synthetic", False),
             "source": "stock vectorbt 0.28.1 independent quote-currency accounts", "prepared_manifest": manifest})
        output(status="synthetic_only" if manifest["provenance"].get("synthetic") else "pilot_only",
               rows=len(ledger), closed=int((ledger.status == "closed").sum()), results=results)
    elif command == "report":
        from .reporting.summary import report
        path = args.results or ws/f"outputs/results/{config.stage}"
        output(status="reported", groups=report(path), path=path/"report.html")


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command is None:
            output(bin=Path(__file__).resolve(), description="Offline-first six-pair FX/CME pilot",
                   data_status="no data loaded", cost_authority_usd=0,
                   help=f"{PROG} demo --help; {PROG} estimate --help")
            return 0
        config = settings(args)
        run(args, config)
        return 0
    except UsageError as exc:
        output(error=str(exc), help=f"{PROG} --help")
        return 2
    except (ValueError, KeyError, OSError, TypeError) as exc:
        # Provider paths above deliberately suppress raw third-party exceptions/credentials.
        output(error=str(exc), help="Check input schemas, date bounds and local audit files; use command --help")
        return 1


if __name__ == "__main__":
    sys.exit(main())

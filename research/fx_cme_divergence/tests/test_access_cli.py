"""Zero-cost gate, cache provenance, manifests, CLI strictness. Providers are mocks; no network or secrets."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from .. import __main__ as cli
from ..config import Config, load_config
from ..data_access import providers
from ..data_access.storage import digest, read_bundle, seal, verify
from .fixtures import make_bundle

START, END = "2026-07-06T20:00:00Z", "2026-07-06T23:00:00Z"


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def path_for(self, request):
        return self.root / f"{request.schema}.dbn.zst"


class Provider:
    def __init__(self, costs, tmp_path, cache=None):
        self.costs, self.fetched, self.cache = costs, [], cache
        self.tmp = tmp_path

    def estimate(self, request):
        cost = self.costs[request.schema]
        if isinstance(cost, Exception):
            raise cost
        return SimpleNamespace(cost_usd=cost, billable_bytes=10)

    def fetch(self, batch, max_cost=0.0):
        self.fetched.append(max_cost)
        results = []
        for request in batch:
            path = self.tmp / f"{request.schema}.bin"
            path.write_bytes(request.schema.encode())
            results.append(SimpleNamespace(request=request, path=path, status="fetched",
                                           validation=SimpleNamespace(ok=True)))
        return results


def batch():
    return providers.requests(START, END, ["6E.FUT", "6A.FUT"])


def zero():
    return dict.fromkeys(providers.SCHEMAS, 0.0)


def test_all_four_schemas_are_always_estimated():
    assert [r.schema for r in batch()] == ["definition", "status", "mbp-1", "trades"]


def test_zero_estimate_fetches_with_zero_cap_and_records_provenance(tmp_path):
    provider = Provider(zero(), tmp_path)
    records = providers.fetch_zero(provider, batch(), tmp_path / "audit" / "cost.json")
    assert provider.fetched == [0.0] and len(records) == 4
    assert all(r["cost_usd"] == 0 and r["sha256"] and r["tir_commit"] for r in records)
    text = (tmp_path / "audit" / "downloads.json").read_text()
    assert "DATABENTO_API_KEY" not in text and "db-" not in text


@pytest.mark.parametrize("bad", [0.01, float("nan"), -1.0, RuntimeError("secret db-key rejected")])
def test_positive_unknown_or_failed_cost_never_downloads(tmp_path, bad):
    costs = zero()
    costs["status"] = bad  # a non-main schema must also gate the whole fetch
    provider = Provider(costs, tmp_path)
    with pytest.raises(ValueError) as error:
        providers.fetch_zero(provider, batch(), tmp_path / "cost.json")
    assert provider.fetched == [] and "db-key" not in str(error.value)
    assert "db-key" not in (tmp_path / "cost.json").read_text()


def test_positive_cap_and_partial_batches_are_rejected(tmp_path):
    provider = Provider(zero(), tmp_path)
    with pytest.raises(ValueError, match="USD 0"):
        providers.fetch_zero(provider, batch(), tmp_path / "c.json", max_cost=0.01)
    with pytest.raises(ValueError, match="four schemas"):
        providers.fetch_zero(provider, batch()[:3], tmp_path / "c.json")
    assert provider.fetched == []


def test_request_bounds_rejected_before_api():
    with pytest.raises(ValueError):
        providers.requests("2025-01-01T00:00:00Z", "2026-06-01T00:00:00Z", ["6E.FUT"])
    with pytest.raises(ValueError):
        providers.requests(END, START, ["6E.FUT"])
    with pytest.raises(ValueError):
        providers.requests(START, END, ["ES.FUT"])
    with pytest.raises(ValueError):
        providers.requests("2026-07-06T20:00:00", END, ["6E.FUT"])  # naive timestamps are ambiguous


def test_cache_requires_matching_receipt(tmp_path):
    cache = Cache(tmp_path)
    provider = Provider(zero(), tmp_path, cache)
    requests = batch()
    cache.path_for(requests[0]).write_bytes(b"cached")
    with pytest.raises(ValueError, match="lacks research provenance"):
        providers.fetch_zero(provider, requests, tmp_path / "cost.json")
    assert provider.fetched == []
    path = cache.path_for(requests[0])
    receipt = path.with_name(path.name + ".research.json")
    receipt.write_text(json.dumps({"request": requests[0].describe(), "sha256": "0" * 64, "cost_usd": 0,
                                   "tir_commit": providers.TIR_COMMIT}))
    with pytest.raises(ValueError, match="mismatch"):
        providers.fetch_zero(provider, requests, tmp_path / "cost.json")
    receipt.write_text(json.dumps({"request": requests[0].describe(), "sha256": digest(path), "cost_usd": 0,
                                   "tir_commit": providers.TIR_COMMIT}))
    providers.fetch_zero(provider, requests, tmp_path / "cost.json")
    assert provider.fetched == [0.0]


def test_manifest_detects_tampering_and_missing_source(tmp_path):
    config = Config(start="2026-07-01", end="2026-08-01", pairs=("EURUSD",), providers=("Broadway",))
    make_bundle(tmp_path, config)
    assert read_bundle(tmp_path)[1]["provenance"]["synthetic"] is True
    (tmp_path / "quotes.csv").write_text((tmp_path / "quotes.csv").read_text() + "x\n")
    with pytest.raises(ValueError, match="mismatch"):
        read_bundle(tmp_path)
    with pytest.raises(ValueError, match="source provenance"):
        seal(tmp_path, ["quotes.csv"], {"synthetic": False})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "db-" not in json.dumps(manifest) and manifest["tir_commit"] == providers.TIR_COMMIT
    with pytest.raises(ValueError, match="missing required files"):
        verify(tmp_path, ["absent.csv"])


def test_config_is_strict(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("api_key: nope\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)
    for kwargs in ({"start": "2026-07-01", "end": "2026-06-01"}, {"start": "2025-01-01", "end": "2026-07-01"},
                   {"pairs": ("USDEUR",)}, {"stale_ms": 0}, {"commission_bps": -1},
                   {"start": "2026-07-01T12:00:00", "end": "2026-08-01"}):
        with pytest.raises(ValueError):
            Config(**kwargs)
    assert load_config(None, start="2026-07-01", end="2026-08-01", stage=None).stage == "baseline"
    sample = Path(cli.ROOT / "configs/pilot.yml")
    assert len(load_config(sample).pairs) == 28 and "password" not in sample.read_text()


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_rejects_unknown_flags_and_bad_ranges_before_apis(capsys, tmp_path):
    code, out = run(capsys, "audit", "--workspace", str(tmp_path), "--nope")
    assert code == 2 and out.startswith("error:") and "--nope" in out
    code, out = run(capsys, "audit", "--workspace", str(tmp_path), "--start", "2026-08-01",
                    "--end", "2026-07-01")
    assert code == 1 and "start < end" in out
    code, out = run(capsys, "estimate", "--workspace", str(tmp_path), "--request-start", START,
                    "--request-end", END, "--max-cost", "1")
    assert code == 2 and "must be 0" in out
    code, out = run(capsys, "fetch", "--workspace", str(tmp_path))
    assert code == 2 and "--request-start" in out


def test_cli_home_and_help(capsys):
    code, out = run(capsys, "")
    assert code == 2  # empty command is not a valid command
    code, out = run(capsys)
    assert code == 0 and "cost_authority_usd: 0" in out
    with pytest.raises(SystemExit):
        cli.main(["estimate", "--help"])
    help_text = capsys.readouterr().out
    assert "--max-cost" in help_text and "must be 0" in help_text and "Example" in help_text


def test_demo_cli_smoke_is_labelled_synthetic_and_reproducible(capsys, tmp_path):
    flags = ["--workspace", str(tmp_path), "--start", "2026-07-01", "--end", "2026-08-01",
             "--pairs", "EURUSD,EURAUD", "--providers", "Broadway"]
    outputs = {}
    for command in ("demo", "audit", "prepare", "backtest", "report"):
        code, out = run(capsys, command, *flags)
        assert code == 0, out
        outputs[command] = out
    assert "synthetic" in outputs["demo"] and "prepared_synthetic" in outputs["prepare"]
    assert "candidate_count: 3" in outputs["backtest"] and "synthetic_only" in outputs["backtest"]
    results = tmp_path / "outputs/results/baseline"
    assert "SYNTHETIC" in (results / "report.html").read_text()
    first = (results / "ledger.csv").read_text()
    for command in ("backtest",):
        assert run(capsys, command, *flags)[0] == 0
    assert (results / "ledger.csv").read_text() == first
    resolved = json.loads((results / "config.json").read_text())
    assert resolved["start"] == "2026-07-01" and "key" not in json.dumps(resolved).lower()


def test_backtest_refuses_changed_prepared_settings(capsys, tmp_path):
    flags = ["--workspace", str(tmp_path), "--start", "2026-07-01", "--end", "2026-08-01",
             "--pairs", "EURUSD", "--providers", "Broadway"]
    for command in ("demo", "prepare"):
        assert run(capsys, command, *flags)[0] == 0
    code, out = run(capsys, "backtest", *flags[:-2], "--providers", "ICMLD")
    assert code == 1 and "prepared config differs" in out


def test_generated_directories_are_git_ignored():
    root = Path(cli.ROOT)
    for folder in ("data/inputs/x.csv", "data/raw/x.dbn", "data/processed/x.parquet",
                   "outputs/audits/x.csv", "outputs/results/x.csv", "outputs/figures/x.png"):
        result = subprocess.run(["git", "check-ignore", "-q", str(root / folder)], cwd=root, check=False)
        assert result.returncode == 0, folder
    assert not list(root.rglob("*.ipynb"))


def test_session_windows_cover_every_estimated_request(tmp_path):
    config = Config(start="2026-07-01", end="2026-08-01", pairs=("EURUSD",), providers=("Broadway",))
    make_bundle(tmp_path, config)
    windows = providers.session_windows(tmp_path / "sessions.csv", "2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z")
    batch = providers.requests("2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z", ["6E.FUT"], windows=windows)
    quotes = [r for r in batch if r.schema == "mbp-1"]
    assert len(quotes) == len(windows) > 1 and len(batch) == len(windows) + 3
    assert all(r.end - r.start < 3 * 86400 * 10**9 for r in quotes)
    provider = Provider(zero(), tmp_path)
    with pytest.raises(ValueError, match="max-bytes"):
        providers.fetch_zero(provider, batch, tmp_path / "c.json", max_bytes=1)
    assert provider.fetched == []


def test_session_windows_reject_empty_calendar(tmp_path):
    config = Config(start="2026-07-01", end="2026-08-01", pairs=("EURUSD",), providers=("Broadway",))
    make_bundle(tmp_path, config)
    with pytest.raises(ValueError, match="no sessions"):
        providers.session_windows(tmp_path / "sessions.csv", "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z")


class Connection:
    """Read-only ClickHouse stand-in: records the hour bounds of every quote query."""

    def __init__(self, rows=0, empty_after=None):
        self.rows, self.empty_after, self.spans = rows, empty_after, []

    def query_df(self, sql, params=None, settings=None):
        if "ICMarkets_Risk.symbols" in sql:
            return pd.DataFrame({"symbol_id": [7], "symbol_name": ["EURUSD"]})
        if "system.tables" in sql:
            return pd.DataFrame({"name": ["quotes_Broadway"]})
        self.spans.append((params["start_ns"], params["end_ns"]))
        if self.empty_after is not None and params["start_ns"] >= self.empty_after:
            return pd.DataFrame()  # ClickHouse returns no columns at all when nothing matches
        return pd.DataFrame({"core_symbol": [7] * self.rows, "price": [1.0] * self.rows})


def test_broker_export_windows_limit_hourly_chunks(tmp_path):
    hour = 3600 * 10**9
    base = 1_790_000_000 // 3600 * 3600 * 10**9
    stamp = lambda n: str(pd.Timestamp(n, unit="ns", tz="UTC"))
    connection = Connection(rows=2, empty_after=base + 6 * hour)
    windows = [(base + 5 * hour + hour // 2, base + 7 * hour), (base + 40 * hour, base + 40 * hour + hour // 4)]
    count = providers.broker_export(connection, stamp(base), stamp(base + 48 * hour), ("EURUSD",), ("Broadway",),
                                    tmp_path, windows=windows)
    assert count == 1 and len(connection.spans) == 3  # 3 of 48 hours queried; 2 returned nothing and are skipped
    assert verify(tmp_path)["provenance"]["empty_chunks"] == 2
    assert connection.spans == [(base + 5 * hour + hour // 2, base + 6 * hour), (base + 6 * hour, base + 7 * hour),
                                (base + 40 * hour, base + 40 * hour + hour // 4)]
    with pytest.raises(ValueError, match="empty"):
        providers.broker_export(connection, stamp(base), stamp(base + 4 * hour), ("EURUSD",), ("Broadway",),
                                tmp_path, windows=windows)
    with pytest.raises(ValueError, match="row cap"):
        providers.broker_export(Connection(rows=5), stamp(base), stamp(base + hour), ("EURUSD",), ("Broadway",),
                                tmp_path, row_limit=5)

"""Offline normalization of synthetic DBN-shaped frames (tir_data's reader is mocked)."""
import json
from decimal import Decimal

import pandas as pd
import pytest

from ..config import NS, TIR_COMMIT, ns
from ..data_access import normalize as norm
from ..data_access.storage import digest, read_bundle, seal
from ..market_data.contracts import chicago_ns

REOPEN = chicago_ns("2026-07-06", "17:00")
CLOSE = chicago_ns("2026-07-06", "16:00")


def frames():
    iid = {"6EU6": 11, "6EZ6": 12}
    definition = pd.DataFrame({"instrument_id": [11, 12], "raw_symbol": ["6EU6", "6EZ6"],
                               "ts_recv": [CLOSE-9*86400*NS]*2,
                               "expiration": [REOPEN+20*86400*NS, REOPEN+110*86400*NS]})
    status = pd.DataFrame({"instrument_id": [11]*3 + [12]*3,
                           "action": [9, 1, 7]*2, "ts_event": [CLOSE, REOPEN-600*NS, REOPEN]*2,
                           "ts_recv": [CLOSE+1, REOPEN-600*NS+1, REOPEN+1]*2})
    undefined = Decimal("9223372036.854775807")
    mbp = pd.DataFrame({"instrument_id": [11, 11, 12], "ts_recv": [CLOSE-5*NS, CLOSE-4*NS, CLOSE-4*NS],
                        "ts_event": [CLOSE-5*NS-1, CLOSE-4*NS-1, CLOSE-4*NS-1],
                        "bid_px_00": [Decimal("1.1000"), undefined, Decimal("1.1100")],
                        "ask_px_00": [Decimal("1.1002"), Decimal("1.1003"), Decimal("1.1102")],
                        "bid_sz_00": [3, 3, 3], "ask_sz_00": [4, 4, 4], "sequence": [100, 101, 101]})
    trades = pd.DataFrame({"instrument_id": [11, 11], "ts_recv": [CLOSE-6*NS, CLOSE-3*NS],
                           "ts_event": [CLOSE-6*NS-1, CLOSE-3*NS-1], "price": [Decimal("1.1001")]*2,
                           "size": [5, 7]})
    data = {"definition": definition, "status": status, "mbp-1": mbp, "trades": trades}
    for frame in data.values():  # Databento's reader yields uint64 times; mixing with int64 would become float64
        for column in {"ts_recv", "ts_event", "expiration"} & set(frame):
            frame[column] = frame[column].astype("uint64")
    return data, iid


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data, _ = frames()
    schemas = list(data)
    downloads = []
    for schema in schemas:
        path = tmp_path / f"{schema}.bin"
        path.write_bytes(schema.encode())
        downloads.append({"path": str(path), "sha256": digest(path), "cost_usd": 0.0, "tir_commit": TIR_COMMIT,
                          "request": {"schema": schema, "start": CLOSE-86400*NS*3, "end": REOPEN+86400*NS}})
    by_name = {f"{s}.bin": data[s] for s in schemas}
    monkeypatch.setattr(norm, "load_frame", lambda path: by_name[path.name].copy())
    (tmp_path / "downloads.json").write_text(json.dumps(downloads))
    broker = tmp_path / "broker"
    broker.mkdir()
    pd.DataFrame({"symbol_name": ["EURUSD"]*2, "side": [0, 1], "time_received": [CLOSE-5*NS, CLOSE-5*NS+1],
                  "price": [1.0999, 1.1003], "size": [1, 1], "indicative_flags": [0, 0],
                  "disabled_flags": [0, 0]}).to_csv(broker / "Broadway_0.csv", index=False)
    seal(broker, ["Broadway_0.csv"], {"source": "mock read-only export", "synthetic": False})
    pd.DataFrame([{"session_id": "s1", "root": "6E", "kind": "weekday", "basis_start_ns": CLOSE-600*NS,
                   "basis_end_ns": CLOSE, "close_ns": CLOSE, "preopen_ns": REOPEN-600*NS, "reopen_ns": REOPEN,
                   "boundary_ns": REOPEN+600*NS, "calendar_source": "https://www.cmegroup.com/trading-hours.html",
                   "calendar_version": "mock-2026"}]).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame({"root": ["6E"]*2, "raw_symbol": ["6EU6", "6EZ6"], "last_trade_ns": [REOPEN+19*86400*NS,
                  REOPEN+109*86400*NS], "tenor": ["2026-09", "2026-12"],
                  "calendar_source": ["https://www.cmegroup.com/"]*2}).to_csv(tmp_path / "terms.csv", index=False)
    (tmp_path / "sizes.json").write_text(json.dumps({"EURUSD": 100000}))
    return tmp_path


def run(inputs, **override):
    paths = {"downloads_path": inputs / "downloads.json", "broker_dir": inputs / "broker",
             "calendar_path": inputs / "calendar.csv", "terms_path": inputs / "terms.csv",
             "sizes_path": inputs / "sizes.json", "output": inputs / "bundle"}
    return norm.normalize(**{**paths, **override})


def test_normalize_builds_sealed_point_in_time_bundle(inputs):
    tables = run(inputs)
    quotes = tables["quotes"]
    cme = quotes[quotes.venue == "cme"]
    assert len(cme) == 3
    # Futures size is currency units (125000 per contract), and DBN's undefined bid is NaN, never a price.
    assert cme.bid_size.iloc[0] == 3 * 125000 and cme.bid.isna().sum() == 1
    assert (tables["volumes"].available_ns >= tables["volumes"].session_end_ns + NS).all()
    assert set(tables["status"].phase) == {"close", "preopen", "reopen"} and len(tables["status"]) == 3
    loaded, _ = read_bundle(inputs / "bundle")  # hashes, required files and exact-integer timestamp columns
    assert loaded["quotes"].bid.dtype == "float64" and loaded["quotes"].bid.isna().any()  # blank is NaN, not ""
    assert (CLOSE - 5*NS) in set(loaded["quotes"].recv_ns) and loaded["quotes"].recv_ns.dtype == "int64"
    broker = quotes[quotes.venue == "broker"]
    assert broker.bid_size.iloc[0] == 1 * 100000 and broker.ask.isna().iloc[0]


def test_same_packet_book_events_keep_the_final_state(inputs, monkeypatch):
    def with_packet(sequence):
        data = frames()[0]
        extra = data["mbp-1"].iloc[[0]].assign(bid_px_00=Decimal("1.1005"), ask_px_00=Decimal("1.1007"),
                                               sequence=sequence)
        data["mbp-1"] = pd.concat([data["mbp-1"], extra], ignore_index=True)
        monkeypatch.setattr(norm, "load_frame", lambda path: data[path.stem].copy())
    with_packet(100)  # same packet: both events share ts_recv and sequence, the later one is the book state
    cme = run(inputs)["quotes"].query("venue == 'cme' and instrument_id == 11")
    assert len(cme) == 2 and cme.sort_values("recv_ns").bid.iloc[0] == 1.1005
    with_packet(105)  # another packet at the same ts_recv cannot be ordered
    with pytest.raises(ValueError, match="several packets"):
        run(inputs)


def test_normalize_rejects_tampered_or_costly_downloads(inputs):
    downloads = json.loads((inputs / "downloads.json").read_text())
    downloads[0]["cost_usd"] = 0.01
    (inputs / "downloads.json").write_text(json.dumps(downloads))
    with pytest.raises(ValueError, match="zero-cost"):
        run(inputs)
    downloads[0]["cost_usd"] = 0.0
    (inputs / "downloads.json").write_text(json.dumps(downloads))
    (inputs / "definition.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(inputs)


def test_normalize_requires_official_contract_terms(inputs):
    terms = pd.read_csv(inputs / "terms.csv")
    terms["calendar_source"] = "https://example.invalid/schedule"
    terms.to_csv(inputs / "terms.csv", index=False)
    with pytest.raises(ValueError, match="official CME"):
        run(inputs)


def test_ambiguous_status_leaves_session_unverified(inputs, monkeypatch):
    data, _ = frames()
    data["status"].loc[3:, "ts_event"] += 30 * NS  # the two instruments disagree on every transition
    monkeypatch.setattr(norm, "load_frame",
                        lambda path: (data["status"] if path.name == "status.bin" else frames()[0][path.stem]).copy())
    assert run(inputs)["status"].empty


def test_reopen_timestamps_are_chicago_dst_aware():
    assert REOPEN == ns("2026-07-06T22:00:00Z") and CLOSE == ns("2026-07-06T21:00:00Z")

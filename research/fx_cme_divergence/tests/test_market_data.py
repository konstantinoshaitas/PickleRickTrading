import pandas as pd
import pytest

from ..config import NS, PIP, Config, completed_months, ns
from ..data_access.normalize import broker_snapshots
from ..data_access.storage import read_bundle
from ..market_data.contracts import audit, chicago_ns, select_contract, session_times
from ..market_data.quotes import asof, prepare, synthetic, validity
from .fixtures import make_bundle


@pytest.fixture
def config():
    return Config(start="2026-07-01", end="2026-08-01", pairs=("EURUSD",), providers=("Broadway",))


@pytest.fixture
def bundle(tmp_path, config):
    make_bundle(tmp_path, config)
    return read_bundle(tmp_path)[0]


def test_completed_months_and_pips():
    assert completed_months("2026-10-19") == ("2026-07-01", "2026-10-01")
    assert completed_months("2026-01-02") == ("2025-10-01", "2026-01-01")
    assert set(PIP.values()) == {0.0001}


def test_chicago_dst_and_nonexistent_time():
    assert chicago_ns("2026-03-06", "17:00") == ns("2026-03-06T23:00:00Z")
    assert chicago_ns("2026-03-09", "17:00") == ns("2026-03-09T22:00:00Z")
    assert chicago_ns("2026-11-02", "17:00") == ns("2026-11-02T23:00:00Z")
    with pytest.raises(Exception):  # noqa: B017 - pandas/pytz nonexistent-time type
        chicago_ns("2026-03-08", "02:30")


def test_contract_prior_volume_not_nearest_expiry(bundle, config):
    cutoff = int(bundle["sessions"].iloc[0].close_ns)
    definitions = bundle["definitions"].copy()
    second = definitions[definitions.root == "6E"].copy()
    second["instrument_id"], second["raw_symbol"], second["expiry_ns"] = 999, "6E_NEXT", cutoff+200*86400*NS
    definitions = pd.concat([definitions, second], ignore_index=True)
    volumes = bundle["volumes"].copy()
    past = {"root": "6E", "instrument_id": 999, "session_end_ns": cutoff-86400*NS,
            "available_ns": cutoff-86399*NS, "volume": 9999}
    future = {**past, "instrument_id": 1001, "available_ns": cutoff+1, "volume": 999999}
    volumes = pd.concat([volumes, pd.DataFrame([past, future])], ignore_index=True)
    # Restrict repeated synthetic sessions to prior cutoff; unavailable future record must not win.
    selected = select_contract(definitions, volumes, "6E", cutoff)
    assert selected["instrument_id"] == 999
    assert selected["volume_available_ns"] <= cutoff
    second["known_ns"] = cutoff+1
    definitions = pd.concat([bundle["definitions"], second], ignore_index=True)
    assert select_contract(definitions, volumes, "6E", cutoff)["instrument_id"] == 1001


def test_expired_contract_and_roll_buffer(bundle):
    cutoff = int(bundle["sessions"].iloc[0].close_ns)
    definitions = bundle["definitions"].copy()
    definitions["last_trade_ns"] = cutoff+NS
    with pytest.raises(ValueError, match="active definitions"):
        select_contract(definitions, bundle["volumes"], "6E", cutoff)


def test_holiday_calendar_no_invented_sessions_and_status(config, bundle):
    # A holiday is represented by absent/modified official rows, never a generic weekday generator.
    one = bundle["sessions"].iloc[:1].copy()
    row = next(one.itertuples(index=False))
    with pytest.raises(ValueError, match="status absent"):
        session_times(row, bundle["status"].iloc[:0], False, True)
    fallback = session_times(row, bundle["status"].iloc[:0], True, True)
    assert fallback["status_verified"] is False
    verified = session_times(row, bundle["status"], False, True)
    assert verified["status_verified"] is True
    bundle["sessions"] = one
    assert len(audit(bundle, config, True)) == 1


def test_asof_never_future_and_preserves_ns():
    t = ns("2026-07-06T22:00:00Z") + 123
    left = pd.DataFrame({"ts_ns": [t-1, t, t+1]})
    right = pd.DataFrame({"ts_ns": [t], "recv_ns": [t], "value": [2]})
    result = asof(left, right)
    assert pd.isna(result.iloc[0].value)
    assert result.loc[1, "recv_ns"] == t
    assert result.loc[2, "recv_ns"] == t


def leg(bid, ask, size=1000):
    return pd.DataFrame({"bid": [bid], "ask": [ask], "bid_size": [size], "ask_size": [size],
                         "recv_ns": [10], "event_ns": [9], "indicative_flags": [0], "disabled_flags": [0]})


def test_cross_sides_size_and_tenor():
    cross = synthetic(leg(1.1, 1.2), leg(.7, .8), "Z6", "Z6")
    assert cross.bid.iloc[0] == pytest.approx(1.1/.8)
    assert cross.ask.iloc[0] == pytest.approx(1.2/.7)
    assert cross.bid_size.iloc[0] == pytest.approx(1000/(1.1/.8))
    with pytest.raises(ValueError, match="tenor"):
        synthetic(leg(1.1, 1.2), leg(.7, .8), "Z6", "H7")


@pytest.mark.parametrize("column,value,expected", [
    ("indicative_flags", 1, "flags"), ("disabled_flags", 1, "flags"),
    ("bid", 3, "crossed_or_locked"), ("bid_size", 0, "insufficient_size"),
    ("recv_ns", 20, "stale_or_future"), ("recv_ns", 1, "stale_or_future")])
def test_quote_rejections(column, value, expected):
    frame = leg(1, 2)
    frame[column] = value
    assert validity(frame, 10, 5, 100).iloc[0] == expected


def test_per_side_staleness_survives_pairing():
    rows = pd.DataFrame({"symbol_name": ["EURUSD"]*3, "side": [0, 1, 1],
                         "time_received": [100, 101, 1000], "price": [1.0, 1.01, 1.02],
                         "size": [1]*3, "indicative_flags": [0]*3, "disabled_flags": [0]*3})
    result = broker_snapshots(rows, "Broadway", {"EURUSD": 100000})
    assert result.iloc[-1].bid_recv_ns == 100
    assert validity(result, result.recv_ns, 50, 1000).iloc[-1] == "stale_side"
    assert pd.isna(result.iloc[0].ask)  # Never backfill a future ask.


def test_same_timestamp_broker_updates_are_invalid_not_guessed():
    rows = pd.DataFrame({"symbol_name": ["EURUSD"]*5, "side": [0, 1, 1, 1, 0],
                         "time_received": [100, 100, 100, 200, 300], "price": [1.0, 1.01, 1.02, 1.03, 1.0],
                         "size": [1]*5, "indicative_flags": [0]*5, "disabled_flags": [0]*5})
    result = broker_snapshots(rows, "Broadway", {"EURUSD": 100000})
    assert result.recv_ns.tolist() == [100, 200, 300]
    assert result.indicative_flags.tolist() == [1, 0, 0]  # 100: two asks, flagged; 200: the next clean ask
    assert result.iloc[0].ask in (1.01, 1.02)  # one of the observed prices, never an average


def test_prepare_and_basis_no_future_outcomes(bundle, config):
    audits = audit(bundle, config, True)
    observed, coverage = prepare(bundle, audits, config)
    assert not observed.empty
    assert (coverage.rejection == "").all()
    assert set(observed.arm) == {"sunday_frozen", "sunday_preopen", "weekday_frozen", "weekday_preopen"}
    original = observed.basis.copy()
    future = bundle["quotes"].venue == "broker"
    # Change ONLY post-reopen rows; preclosure basis must remain byte-for-byte identical.
    for row in bundle["sessions"].drop_duplicates("session_id").itertuples(index=False):
        mask = future & (bundle["quotes"].recv_ns == row.reopen_ns+120*NS)
        bundle["quotes"].loc[mask, ["bid", "ask"]] += 1
    changed, _ = prepare(bundle, audits, config)
    assert original.equals(changed.basis)


def test_missing_last_trade_removes_only_the_frozen_arm(bundle, config):
    audits = audit(bundle, config, True)
    bundle["trades"] = bundle["trades"].iloc[0:0]
    observed, coverage = prepare(bundle, audits, config)
    assert set(observed.arm) == {"sunday_preopen", "weekday_preopen"}
    assert (coverage.rejection == "").all() and coverage.frozen_rejection.str.contains("last-tradable").all()


def test_rejected_audit_preserves_exact_ns(bundle, config):
    bad = bundle["sessions"].iloc[-1].session_id
    bundle["status"] = bundle["status"][bundle["status"].session_id != bad]
    audits = audit(bundle, config, True)
    good = audits[audits.rejection == ""].iloc[0]
    expected = bundle["sessions"].iloc[0].close_ns
    assert good.close_ns == expected
    assert isinstance(good.close_ns, int)

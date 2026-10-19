"""Signal causality and stock-vectorbt accounting checks on synthetic data only."""
import pandas as pd
import pytest

from ..backtests.portfolio import account, roundtrip
from ..config import NS, Config
from ..data_access.storage import read_bundle
from ..market_data.contracts import audit
from ..market_data.quotes import prepare
from ..signals.rules import Candidate, candidates, signals, trade_signal
from .fixtures import make_bundle


@pytest.fixture
def config():
    return Config(start="2026-07-01", end="2026-08-01", pairs=("EURUSD", "EURAUD"), providers=("Broadway",))


@pytest.fixture
def observations(tmp_path, config):
    make_bundle(tmp_path, config)
    tables = read_bundle(tmp_path)[0]
    observed, _ = prepare(tables, audit(tables, config, True), config)
    return observed


def one_session(observations, arm="weekday_frozen", pair="EURUSD"):
    frames = observations[(observations.arm == arm) & (observations.pair == pair)]
    return frames[frames.session_id == frames.session_id.iloc[0]].reset_index(drop=True)


def test_vectorbt_long_short_costs_hand_calculated():
    assert roundtrip(1.1000, 1.1010, 1, 1000, 1e5)["pnl"] == pytest.approx(1.0)
    assert roundtrip(1.1010, 1.1000, -1, 1000, 1e5)["pnl"] == pytest.approx(1.0)
    costs = roundtrip(1.1000, 1.1010, 1, 1000, 1e5, commission_bps=1, financing_quote=0.0002)
    assert costs["fees"] == pytest.approx(1.1*1000e-4 + 1.101*1000e-4 + 0.2)
    assert costs["pnl"] == pytest.approx(1.0 - costs["fees"])
    slipped = roundtrip(1.1000, 1.1010, 1, 1000, 1e5, slippage_bps=1)
    assert slipped["pnl"] == pytest.approx((1.101*0.9999 - 1.1*1.0001)*1000)


def test_unknown_costs_stay_provisional_and_accounts_are_independent(observations, config):
    ledger = account(signals(observations, config), config)
    closed = ledger[ledger.status == "closed"]
    assert (ledger.cost_status == "gross_provisional_unknown_costs").all()
    # Every row is its own isolated one-roundtrip account, so repeated identical trades give identical P&L.
    same = closed.groupby(["pair", "arm", "candidate", "direction", "entry_price", "exit_price"]).gross_pnl_quote
    assert (same.nunique() == 1).all()
    known = Config(**{**config.dump(), "commission_bps": 0.5, "financing_quote": 0.0})
    priced = account(signals(observations, known), known)
    assert (priced.cost_status == "net_under_assumptions").all()


def test_baseline_causality_fills_and_exit(observations, config):
    ledger = signals(observations, config)
    assert set(ledger.candidate) == {"pips_10", "pips_20", "pips_40"}
    closed = ledger[ledger.status == "closed"]
    assert not closed.empty
    latency = int(config.latency_ms*1e6)
    assert (closed.signal_ns < closed.reopen_ns).all()
    assert (closed.entry_ns >= closed.signal_ns + latency).all()
    assert (closed.exit_ns >= closed.reopen_ns + 120*NS).all()
    longs, shorts = closed[closed.direction == 1], closed[closed.direction == -1]
    assert (longs.entry_price == longs.entry_ask).all() and (longs.exit_price == longs.exit_bid).all()
    assert (shorts.entry_price == shorts.entry_bid).all() and (shorts.exit_price == shorts.exit_ask).all()
    # Only reference/basis known strictly before the entry may be used.
    assert (closed.basis_cutoff_ns <= closed.signal_ns).all()


def test_future_rows_cannot_change_signal(observations, config):
    frame = one_session(observations)
    candidate = Candidate("pips_10", threshold=10)
    before = trade_signal(frame, candidate, config)
    altered = frame.copy()
    after_reopen = altered.ts_ns >= altered.reopen_ns
    altered.loc[after_reopen, ["bid", "ask", "raw_residual", "adjusted_residual"]] += 5
    after = trade_signal(altered, candidate, config)
    for key in ("signal_ns", "direction", "entry_ns", "entry_price", "adjusted_residual"):
        assert before[key] == after[key]


def test_persistence_requires_sustained_crossing(observations, config):
    frame = one_session(observations).copy()
    start = frame[frame.ts_ns >= frame.preopen_ns].index[0]
    # A one-tick displacement, then the broker returns to the adjusted reference.
    calm = frame.loc[start:, ["bid", "ask"]].mean(axis=1)
    frame["adjusted_residual"] = 0.0
    inside = frame.ts_ns < frame.reopen_ns
    frame.loc[inside, "bid"] = frame.loc[inside, "target_bid"]
    frame.loc[inside, "ask"] = frame.loc[inside, "target_ask"] + 0.00001
    frame.loc[start, ["bid", "ask"]] = [frame.target_ask.iloc[start] + 0.0050,
                                       frame.target_ask.iloc[start] + 0.0051]
    assert calm.notna().all()
    immediate = trade_signal(frame, Candidate("immediate", threshold=10), config)
    persistent = trade_signal(frame, Candidate("persistent", threshold=10, persistence=True), config)
    assert immediate["direction"] == -1
    assert persistent["status"] == "no_signal"


def test_missing_exit_is_reported_not_hidden(observations, config):
    frame = one_session(observations).copy()
    frame.loc[frame.ts_ns >= frame.reopen_ns + 100*NS, "broker_rejection"] = "stale_side"
    result = trade_signal(frame, Candidate("pips_10", threshold=10), config)
    assert result["status"] == "unpriced_open" and result["exit_reason"] == "missing_exit"


def test_invalid_entry_quotes_are_unfillable(observations, config):
    frame = one_session(observations).copy()
    frame["broker_rejection"] = "flags"
    assert trade_signal(frame, Candidate("pips_10", threshold=10), config)["status"] == "no_signal"


def test_staged_candidate_counts_are_small_and_deterministic():
    counts = {stage: len(candidates(stage)) for stage in ("baseline", "scale", "confirm", "exits")}
    assert counts == {"baseline": 3, "scale": 4, "confirm": 2, "exits": 4}
    assert candidates("exits") == candidates("exits")


def test_exit_stages_run_and_reproduce(observations, config):
    staged = Config(**{**config.dump(), "stage": "exits"})
    first, second = signals(observations, staged), signals(observations, staged)
    pd.testing.assert_frame_equal(first, second)
    assert {"half_convergence", "full_convergence"} <= set(first.candidate)
    assert set(first.split) <= {"train", "validation", "test"}


def test_splits_keep_whole_weeks_together(observations, config):
    ledger = signals(observations, config)
    assert (ledger.groupby("week").split.nunique() == 1).all()

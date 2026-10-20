"""Point-in-time front-contract selection and explicitly sourced session boundaries."""
import pandas as pd

from ..config import LEGS, NS, ns

CALENDAR_URL = "https://www.cmegroup.com/trading-hours.html"


def chicago_ns(day, clock):
    """Reject ambiguous/nonexistent local times; do not guess a DST fold."""
    return pd.Timestamp(f"{day} {clock}").tz_localize(
        "America/Chicago", ambiguous="raise", nonexistent="raise").tz_convert("UTC").value


def select_contract(definitions, volumes, root, cutoff, buffer_days=2):
    required = {"root", "raw_symbol", "instrument_id", "known_ns", "expiry_ns", "last_trade_ns", "tenor"}
    if not required <= set(definitions):
        raise ValueError("definitions missing historical identity/expiry/knowledge fields")
    known = definitions[(definitions.root == root) & (definitions.known_ns <= cutoff)]
    known = known.sort_values("known_ns").drop_duplicates("instrument_id", keep="last")
    active = known[(known.last_trade_ns > cutoff + buffer_days*86400*NS) & (known.expiry_ns > cutoff)]
    # Volumes must be COMPLETE prior sessions, with publication time as well as session end before selection.
    prior = volumes[(volumes.root == root) & (volumes.available_ns <= cutoff) &
                    (volumes.session_end_ns < cutoff)]
    if prior.empty or active.empty:
        raise ValueError("no point-in-time active definitions/prior complete volume")
    prior = prior[prior.session_end_ns == prior.session_end_ns.max()]
    if prior.instrument_id.duplicated().any():
        raise ValueError("ambiguous prior-volume observation")
    ranked = active.merge(prior, on=["root", "instrument_id"], suffixes=("", "_volume"))
    ranked = ranked[ranked.volume > 0].sort_values(["volume", "expiry_ns", "raw_symbol"],
                                                ascending=[False, True, True])
    if ranked.empty:
        raise ValueError("no prior-volume leader with known unexpired definition")
    row = ranked.iloc[0]
    return {"root": root, "raw_symbol": row.raw_symbol, "instrument_id": int(row.instrument_id),
            "expiry_ns": int(row.expiry_ns), "last_trade_ns": int(row.last_trade_ns),
            "definition_known_ns": int(row.known_ns), "tenor": row.tenor,
            "selection_cutoff_ns": int(cutoff), "volume_available_ns": int(row.available_ns),
            "volume_session_end_ns": int(row.session_end_ns), "prior_volume": float(row.volume)}


def session_times(row, status, allow_fallback, synthetic=False):
    if row.kind not in {"sunday", "weekday"}:
        raise ValueError("session kind must be sunday or weekday")
    if not row.calendar_version or not (row.calendar_source.startswith("https://www.cmegroup.com/") or
                                      (synthetic and row.calendar_source == "synthetic://calendar")):
        raise ValueError("session requires official CME calendar source/version")
    times = {name: int(getattr(row, name)) for name in
             ("basis_start_ns", "basis_end_ns", "close_ns", "preopen_ns", "reopen_ns", "boundary_ns")}
    if not (times["basis_start_ns"] < times["basis_end_ns"] <= times["close_ns"] <=
            times["preopen_ns"] < times["reopen_ns"] < times["boundary_ns"]):
        raise ValueError("invalid session/calendar ordering")
    local = pd.Timestamp(times["reopen_ns"], unit="ns", tz="UTC").tz_convert("America/Chicago")
    if (local.dayofweek == 6) != (row.kind == "sunday"):
        raise ValueError("session kind disagrees with Chicago reopening day")
    events = status[(status.session_id == row.session_id) & (status.root == row.root)]
    verified = False
    if not events.empty:
        if "unexpected_transition" in set(events.phase):
            raise ValueError("CME closed, paused or traded during preopen; resting reference cannot be verified")
        if set(events.phase) != {"close", "preopen", "reopen"} or events.phase.duplicated().any():
            raise ValueError("incomplete or ambiguous status transitions")
        if (events.recv_ns < events.event_ns).any():
            raise ValueError("status receive timestamp precedes event")
        for phase in ("close", "preopen", "reopen"):
            event = events[events.phase == phase].iloc[0]
            times[f"{phase}_ns"] = int(event.event_ns)
            times[f"{phase}_recv_ns"] = int(event.recv_ns)
        if not (times["basis_end_ns"] <= times["close_ns"] <= times["preopen_ns"] <
                times["reopen_ns"] < times["boundary_ns"]):
            raise ValueError("status events violate session order")
        verified = True
    if not verified and not allow_fallback:
        raise ValueError("status absent; explicit calendar fallback required")
    times["status_verified"] = verified
    times["calendar_source"] = row.calendar_source
    times["calendar_version"] = row.calendar_version
    return times


def audit(tables, config, synthetic=False):
    rows, previous = [], {}
    sessions = tables["sessions"].sort_values(["close_ns", "root"])
    start, end = ns(config.start + "T00:00:00Z"), ns(config.end + "T00:00:00Z")
    sessions = sessions[(sessions.reopen_ns >= start) & (sessions.reopen_ns < end)]
    if sessions.duplicated(["session_id", "root"]).any():
        raise ValueError("duplicate session/product")
    roots = {root for pair in config.pairs for root in LEGS[pair]}
    for row in sessions.itertuples(index=False):
        if row.root not in roots:
            continue
        base = {"session_id": row.session_id, "root": row.root, "kind": row.kind,
                "calendar_source": row.calendar_source, "calendar_version": row.calendar_version}
        try:
            times = session_times(row, tables["status"], config.allow_calendar_fallback, synthetic)
            selected = select_contract(tables["definitions"], tables["volumes"], row.root,
                                       times["close_ns"], config.roll_buffer_days)
            key = row.root
            roll = key in previous and previous[key] != selected["instrument_id"]
            previous[key] = selected["instrument_id"]
            rows.append({**base, **times, **selected, "roll": roll, "rejection": ""})
        except ValueError as exc:
            rows.append({**base, "rejection": str(exc)})
    # Mixed accepted/rejected rows must not cast nanosecond integers through float64.
    return pd.DataFrame(rows, dtype=object)

"""Offline DBN/ClickHouse normalization. Official session/contract terms are explicit inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tir_data.databento.convert import load_frame

from ..config import NS, TIR_COMMIT
from ..market_data.quotes import MULTIPLIER
from .storage import TABLES, digest, seal, verify


def broker_snapshots(frame, provider, contract_sizes):
    pieces = []
    for pair, rows in frame.groupby("symbol_name"):
        if pair not in contract_sizes or contract_sizes[pair] <= 0:
            raise ValueError("explicit broker size-unit multiplier required for each pair")
        # Several prices for one side at one timestamp carry no provider sequence, so their order is unknowable:
        # keep one row but flag it, which makes the quote invalid until the next unambiguous update.
        ambiguous = rows.duplicated(["side", "time_received"], keep=False)
        rows = rows.assign(indicative_flags=rows.indicative_flags.where(~ambiguous, 1))
        rows = rows.drop_duplicates(["side", "time_received"], keep="first")
        times = pd.DataFrame({"recv_ns": sorted(rows.time_received.unique())})
        sides = []
        for side, label in ((0, "bid"), (1, "ask")):
            quotes = rows[rows.side == side].copy().sort_values("time_received")
            renamed = pd.DataFrame({"recv_ns": quotes.time_received, label: quotes.price.astype(float),
                                    f"{label}_size": quotes["size"].astype(float)*contract_sizes[pair],
                                    f"{label}_recv_ns": quotes.time_received.astype("Int64"),
                                    f"{label}_flags": ((quotes.indicative_flags != 0) |
                                                      (quotes.disabled_flags != 0)).astype(int)})
            sides.append(pd.merge_asof(times, renamed, on="recv_ns", direction="backward"))
        snap = sides[0].merge(sides[1], on="recv_ns", validate="one_to_one")
        snap["venue"], snap["provider"], snap["pair"], snap["instrument_id"] = "broker", provider, pair, 0
        snap["event_ns"] = snap.recv_ns  # No comparable broker exchange event time is available.
        snap["indicative_flags"] = snap.bid_flags.fillna(1).astype(int) | snap.ask_flags.fillna(1).astype(int)
        snap["disabled_flags"] = 0
        # Missing side times are explicit invalid snapshots, not zero-age valid quotes.
        for label in ("bid", "ask"):
            snap[f"{label}_recv_ns"] = snap[f"{label}_recv_ns"].fillna(0).astype("int64")
        pieces.append(snap)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def exact_ns(frame, columns=("recv_ns", "event_ns", "known_ns", "expiry_ns")):
    """DBN timestamps are uint64; concatenated with int64 broker times pandas would silently go to float64."""
    frame = frame.copy()
    for column in set(columns) & set(frame):
        frame[column] = frame[column].astype("int64")
    return frame


def normalize(downloads_path, broker_dir, calendar_path, terms_path, sizes_path, output):
    downloads = json.loads(Path(downloads_path).read_text())
    frames = {}
    for item in downloads:
        if item.get("tir_commit") != TIR_COMMIT or item.get("cost_usd") != 0:
            raise ValueError("download provenance is not pinned/zero-cost")
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("download hash mismatch")
        frames.setdefault(item["request"]["schema"], []).append(load_frame(path))
    if set(frames) != {"definition", "status", "mbp-1", "trades"}:
        raise ValueError("all downloaded schemas required")
    frames = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    sessions = pd.read_csv(calendar_path, keep_default_na=False)
    terms = pd.read_csv(terms_path, keep_default_na=False)
    if not terms.calendar_source.str.startswith("https://www.cmegroup.com/").all():
        raise ValueError("last-trade/tenor terms require official CME source URLs")
    definitions = frames["definition"]
    # Databento raw_symbol, expiration, ts_recv are authoritative definition fields.
    definitions = definitions.merge(terms, on="raw_symbol", validate="many_to_one")
    if definitions.empty:
        raise ValueError("no downloaded definitions match official contract terms")
    definitions = definitions.rename(columns={"ts_recv": "known_ns", "expiration": "expiry_ns"})
    definitions = exact_ns(definitions)
    definitions = definitions[["root", "raw_symbol", "instrument_id", "known_ns", "expiry_ns",
                               "last_trade_ns", "tenor"]].drop_duplicates()
    ids = definitions.sort_values("known_ns").drop_duplicates("instrument_id").set_index("instrument_id")
    book = frames["mbp-1"]
    book = book[book.instrument_id.isin(ids.index)].copy()
    book["root"] = book.instrument_id.map(ids.root)
    book = book.rename(columns={"ts_recv": "recv_ns", "ts_event": "event_ns", "bid_px_00": "bid",
                                "ask_px_00": "ask", "bid_sz_00": "bid_size", "ask_sz_00": "ask_size"})
    book = exact_ns(book)
    for key in ("bid", "ask"):
        book[key] = book[key].map(lambda v: float(v) if v is not None else np.nan)
        book.loc[book[key].abs() > 1e6, key] = np.nan  # DBN undefined-price sentinel, never a real FX price.
        book[f"{key}_size"] = book[f"{key}_size"].astype(float)*book.root.map(MULTIPLIER)
        book[f"{key}_recv_ns"] = book.recv_ns
    book["venue"], book["provider"], book["pair"] = "cme", "CME", ""
    book["indicative_flags"], book["disabled_flags"] = 0, 0
    # One MDP3 packet carries several book events sharing ts_recv and sequence; the book known at ts_recv is the
    # state after the LAST of them in feed order. Different packets at one ts_recv are genuinely ambiguous.
    if (book.groupby(["instrument_id", "recv_ns"]).sequence.nunique() > 1).any():
        raise ValueError("CME timestamp spans several packets; sequence-aware input required")
    book = book.drop_duplicates(["instrument_id", "recv_ns"], keep="last")
    trades = frames["trades"].rename(columns={"ts_recv": "recv_ns", "ts_event": "event_ns"})
    trades = exact_ns(trades[trades.instrument_id.isin(ids.index)].copy())
    trades["price"] = trades.price.astype(float)
    volumes = []
    # Complete Chicago 17:00 -> next weekday 16:00 sessions only, never current-day volume.
    trade_days = pd.to_datetime(trades.event_ns, unit="ns", utc=True).dt.tz_convert("America/Chicago")
    trades["local_end_day"] = (trade_days + pd.Timedelta(hours=7)).dt.date
    from ..market_data.contracts import chicago_ns
    starts = [int(r["request"]["start"]) for r in downloads if r["request"]["schema"] == "trades"]
    ends = [int(r["request"]["end"]) for r in downloads if r["request"]["schema"] == "trades"]
    for (iid, day), group in trades.groupby(["instrument_id", "local_end_day"]):
        opening_day = pd.Timestamp(day)-pd.Timedelta(days=1)
        opening, ending = chicago_ns(str(opening_day.date()), "17:00"), chicago_ns(str(day), "16:00")
        if not any(a <= opening and b >= ending+NS for a, b in zip(starts, ends)):
            continue
        group = group[(group.event_ns >= opening) & (group.event_ns < ending)]
        volumes.append({"root": ids.loc[iid].root, "instrument_id": iid, "session_end_ns": ending,
                        "available_ns": max(ending+NS, int(group.recv_ns.max())) if len(group) else ending+NS,
                        "volume": float(group["size"].sum())})
    statuses = exact_ns(frames["status"], ("ts_event", "ts_recv"))
    import databento as db
    observed = []
    for row in sessions.itertuples(index=False):
        root_ids = ids[ids.root == row.root].index
        events = statuses[statuses.instrument_id.isin(root_ids)]
        for phase, actions in (("close", (int(db.StatusAction.CLOSE), int(db.StatusAction.PAUSE))),
                               ("preopen", (int(db.StatusAction.PRE_OPEN),)),
                               ("reopen", (int(db.StatusAction.TRADING),))):
            target = int(getattr(row, f"{phase}_ns"))
            matches = events[(events.action.astype(int).isin(actions)) &
                             ((events.ts_event-target).abs() <= 600*NS)]
            if matches.empty:
                continue
            # Instruments can all emit the same product transition. Different times are ambiguous.
            transitions = matches[["ts_event", "ts_recv"]].drop_duplicates()
            if transitions.ts_event.nunique() != 1:
                continue  # Ambiguous: leave unverified so audit rejects it unless fallback is explicit.
            observed.append({"session_id": row.session_id, "root": row.root, "phase": phase,
                             "event_ns": int(transitions.ts_event.iloc[0]),
                             "recv_ns": int(transitions.ts_recv.max())})
    broker_manifest = verify(broker_dir)
    chunks = []
    for name in broker_manifest["files"]:
        provider = name.split("_")[0]
        chunks.append(broker_snapshots(pd.read_csv(Path(broker_dir)/name), provider,
                                      json.loads(Path(sizes_path).read_text())))
    if not chunks:
        raise ValueError("no broker data chunks")
    columns = ["venue", "provider", "pair", "instrument_id", "recv_ns", "event_ns", "bid", "ask",
               "bid_size", "ask_size", "bid_recv_ns", "ask_recv_ns", "indicative_flags", "disabled_flags"]
    broker = pd.concat(chunks, ignore_index=True)
    # Chunks have no forward state across boundaries; missing sides remain invalid until both arrive.
    tables = {"definitions": definitions, "sessions": sessions,
              "volumes": pd.DataFrame(volumes, columns=["root", "instrument_id", "session_end_ns", "available_ns", "volume"]),
              "status": pd.DataFrame(observed, columns=["session_id", "root", "phase", "event_ns", "recv_ns"]),
              "quotes": pd.concat([book[columns], broker[columns]], ignore_index=True),
              "trades": trades[["instrument_id", "recv_ns", "event_ns", "price"]]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        for column in table:
            if column.endswith("_ns") and len(table) and not pd.api.types.is_integer_dtype(table[column]):
                raise ValueError(f"{name}.{column} lost integer nanosecond precision")
        table.to_csv(output/f"{name}.csv", index=False)
    seal(output, [f"{t}.csv" for t in TABLES], {
        "synthetic": False, "source": "zero-cost Databento and authorized bounded ClickHouse",
        "downloads_hash": digest(downloads_path), "broker_manifest_hash": digest(Path(broker_dir)/"manifest.json"),
        "calendar_hash": digest(calendar_path), "terms_hash": digest(terms_path), "sizes_hash": digest(sizes_path)})
    return tables

"""Fetch official CME FX session schedules and merge the requested 2026 window."""
import argparse
import calendar
import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOTS = ("6A", "6B", "6C", "6E", "6J", "6N", "6S")
PRODUCT_IDS = "37,42,48,58,69,78,86"
SOURCE = "https://www.cmegroup.com/trading-hours.html"
API = "https://www.cmegroup.com/services/trading-hours-by-product"
CHICAGO = ZoneInfo("America/Chicago")
START, END = date(2026, 1, 1), date(2026, 4, 23)
SESSIONS = Path(__file__).resolve().parents[1] / "configs/cme_sessions_2026.csv"
EXCLUSIONS = SESSIONS.with_name("cme_exclusions_2026.csv")


def month_schedules(year, month, json_dir=None):
    if json_dir:
        data = json.loads((json_dir/f"{year}-{month:02d}.json").read_text())
    else:
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
        url = (f"{API}?id={PRODUCT_IDS}&pageNumber=1&pageSize=500&sortAsc=true"
               f"&fromEventDate={first}&toEventDate={last}&isProtected")
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/131 Safari/537.36",
                                       "Accept": "application/json,text/plain,*/*"})
        try:
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"CME request blocked ({error.code}); provide that month's Chrome-fetched JSON") from error
    products = data["products"]
    if isinstance(products[0], str):
        if set(products) != set(ROOTS):
            raise ValueError(f"CME response did not contain exactly {ROOTS}")
        return [(item["eventDate"], item["events"]) for item in data["schedules"]]
    if {item["globex"] for item in products} != set(ROOTS):
        raise ValueError(f"CME response did not contain exactly {ROOTS}")
    reference = [(item["eventDate"], item["events"]) for item in products[0]["tradingHours"]["schedules"]]
    for product in products[1:]:
        actual = [(item["eventDate"], item["events"]) for item in product["tradingHours"]["schedules"]]
        if actual != reference:
            raise ValueError(f"CME schedules differ for {product['globex']}")
    return reference


def iso_utc(day, time_ct):
    return datetime.fromisoformat(f"{day}T{time_ct}").replace(tzinfo=CHICAGO).astimezone(UTC).isoformat()


def build(schedules, retrieved):
    rows, exclusions = [], []
    last_close = preopen = None
    for day, events in sorted(schedules):
        for event in sorted(events, key=lambda item: item["eventTime"]):
            kind = event["marketEventType"]
            stamp = iso_utc(day, event["eventTime"])
            if kind == "closed":
                last_close, preopen = stamp, None
            elif kind == "preopen":
                preopen = stamp
            elif kind == "open":
                if last_close and preopen:
                    if START.isoformat() <= day < END.isoformat():
                        session_kind = "sunday" if date.fromisoformat(day).weekday() == 6 else "weekday"
                        rows.append({"session_id": f"{session_kind}-{day}", "kind": session_kind,
                                     "close_utc": last_close, "preopen_utc": preopen, "reopen_utc": stamp})
                elif START.isoformat() <= day < END.isoformat():
                    exclusions.append({"date": day, "time_ct": event["eventTime"],
                                       "reason": "open without a preceding sourced close and preopen",
                                       "source": SOURCE, "retrieved": retrieved})
                last_close = preopen = None
            else:
                raise ValueError(f"unknown CME event type: {kind}")
    return rows, exclusions


def merge(path, rows, key, replace_start, replace_end):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames)
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        existing = [row for row in reader if not replace_start <= row[key][:10] < replace_end]
    for row in existing:
        if "calendar_source" in fields and not row.get("calendar_source"):
            row["calendar_source"] = SOURCE
        if "calendar_retrieved" in fields and not row.get("calendar_retrieved"):
            row["calendar_retrieved"] = "2026-10-20"
    added = rows
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(existing + added, key=lambda row: row[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, help="monthly JSON fetched via visible Chrome if Python is blocked")
    args = parser.parse_args()
    retrieved = datetime.now(UTC).date().isoformat()
    monthly = []
    for year, month in ((2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4)):
        monthly.extend(month_schedules(year, month, args.json_dir))
    rows, exclusions = build(monthly, retrieved)
    merge(SESSIONS, rows, "reopen_utc", START.isoformat(), END.isoformat())
    merge(EXCLUSIONS, exclusions, "date", START.isoformat(), END.isoformat())
    print(f"sessions={len(rows)} exclusions={len(exclusions)} source={SOURCE} retrieved={retrieved}")


if __name__ == "__main__":
    main()

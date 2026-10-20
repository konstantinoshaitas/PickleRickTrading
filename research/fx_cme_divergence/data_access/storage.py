"""Small, hashed local bundles; no credentials or raw provider errors in manifests."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from ..config import TIR_COMMIT

TABLES = ("definitions", "volumes", "sessions", "status", "quotes", "trades")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(directory, names, provenance):
    directory = Path(directory)
    if provenance.get("synthetic") is not True and not provenance.get("source"):
        raise ValueError("real inputs require source provenance")
    write_json(directory / "manifest.json", {
        "version": 1, "tir_commit": TIR_COMMIT, "provenance": provenance,
        "files": {name: digest(directory / name) for name in names},
    })


def verify(directory, required=()):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != 1 or manifest.get("tir_commit") != TIR_COMMIT:
        raise ValueError("unsupported manifest version/dependency provenance")
    files = manifest.get("files", {})
    if not set(required) <= set(files):
        raise ValueError("bundle manifest is missing required files")
    for name, expected in files.items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError("bundle hash/path mismatch")
    provenance = manifest.get("provenance", {})
    if provenance.get("synthetic") is not True and not provenance.get("source"):
        raise ValueError("missing source provenance")
    return manifest


def read_bundle(directory):
    manifest = verify(directory, [f"{t}.parquet" for t in TABLES])
    tables = {t: pd.read_parquet(Path(directory) / f"{t}.parquet") for t in TABLES}
    # Never let timestamps pass through float64 (nanoseconds exceed its exact integer range).
    for table in tables.values():
        for column in table:
            if column.endswith("_ns"):
                if not pd.api.types.is_integer_dtype(table[column]) and len(table):
                    raise ValueError(f"{column} must contain exact integer UTC nanoseconds")
                if not len(table):
                    table[column] = table[column].astype("int64")
    return tables, manifest

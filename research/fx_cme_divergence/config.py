"""Resolved, credential-free experiment settings. Dates are UTC [start, end)."""
import math
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from .products import LEGS, PIP, SUPPORTED_PAIRS  # noqa: F401 - existing import surface

TIR_COMMIT = "d77ea849b450b5a2ab8c940694d6a82affe5e1b9"
NS = 1_000_000_000


def ns(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("timestamps require a timezone")
    return stamp.tz_convert("UTC").value


def completed_months(today=None):
    end = pd.Timestamp(today or date.today()).normalize().replace(day=1)
    return str((end - pd.DateOffset(months=3)).date()), str(end.date())


@dataclass(frozen=True)
class Config:
    start: str = ""
    end: str = ""
    pairs: tuple = SUPPORTED_PAIRS
    providers: tuple = ("Broadway",)
    stage: str = "baseline"
    latency_ms: float = 100.0
    stale_ms: float = 2000.0  # executable broker quotes; independent of CME references
    frozen_max_age_s: float = 2.0  # age at close, NOT elapsed age through the closure
    indicative_max_age_s: float = 2.0  # unchanged resting book since pre-open
    indicative_crossed_mid: bool = False  # combine per-leg mids, including crossed books
    persistence_ms: float = 1000.0
    notional_usd: float = 100000.0  # fixed allocation per independent trade, no pair priority
    quantity: float | None = None  # legacy fixed base units, only for reproducing the old pilot
    cash: float = 100000.0
    commission_bps: float | None = None
    slippage_bps: float = 0.0
    financing_quote: float | None = None  # per base unit, roundtrip; 0 must be explicit
    exit_wait_s: float = 30.0
    max_hold_s: float = 7200.0
    roll_buffer_days: float = 2.0
    min_basis_samples: int = 5
    allow_calendar_fallback: bool = False

    def __post_init__(self):
        start, end = completed_months()
        object.__setattr__(self, "start", self.start or start)
        object.__setattr__(self, "end", self.end or end)
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "providers", tuple(self.providers))
        a, b = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if a.tzinfo is not None or b.tzinfo is not None or a != a.normalize() or b != b.normalize():
            raise ValueError("dates must be midnight YYYY-MM-DD, interpreted as UTC")
        if a >= b or (b-a).days > 366:
            raise ValueError("require start < end and a bounded range of at most 366 days")
        if not self.pairs or not set(self.pairs) <= set(SUPPORTED_PAIRS) or len(set(self.pairs)) != len(self.pairs):
            raise ValueError("choose unique supported pairs")
        if not self.providers or not set(self.providers) <= {"Broadway", "ICMLD", "Finalto"}:
            raise ValueError("choose Broadway, ICMLD or Finalto")
        if self.stage not in {"baseline", "scale", "confirm", "exits"}:
            raise ValueError("stage must be baseline, scale, confirm or exits")
        for key in ("latency_ms", "slippage_bps", "roll_buffer_days", "commission_bps", "financing_quote", "quantity"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("stale_ms", "frozen_max_age_s", "indicative_max_age_s", "persistence_ms",
                    "notional_usd", "cash", "exit_wait_s", "max_hold_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.quantity == 0:
            raise ValueError("quantity must be positive or null (USD-notional sizing)")
        if self.min_basis_samples < 2 or not isinstance(self.min_basis_samples, int):
            raise ValueError("min_basis_samples must be an integer >= 2")
        for key in ("allow_calendar_fallback", "indicative_crossed_mid"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")

    def dump(self):
        return asdict(self)


def load_config(path=None, **overrides):
    payload = yaml.safe_load(Path(path).read_text()) if path else {}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError("config must be a mapping")
    unknown = set(payload) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    payload.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**payload)

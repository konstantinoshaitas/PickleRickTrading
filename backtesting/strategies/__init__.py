"""Strategy registry with only the Triple EMA implementation for now."""

from .triple_ema import TripleEMAStrategy

StrategyFactory = {
    "triple_ema": TripleEMAStrategy,
}

__all__ = ["StrategyFactory", "TripleEMAStrategy"]

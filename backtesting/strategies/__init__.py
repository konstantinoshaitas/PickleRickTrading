"""Strategy registry with Triple EMA, Triple MACD, and Ensemble implementations."""

from .ensemble import EnsembleStrategy
from .triple_ema import TripleEMAStrategy
from .triple_macd import TripleMACDStrategy

StrategyFactory = {
    "triple_ema": TripleEMAStrategy,
    "triple_macd": TripleMACDStrategy,
    "ensemble": EnsembleStrategy,
}

__all__ = [
    "StrategyFactory",
    "TripleEMAStrategy",
    "TripleMACDStrategy",
    "EnsembleStrategy",
]

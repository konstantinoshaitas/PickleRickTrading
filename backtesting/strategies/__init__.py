"""Strategy registry with Triple EMA, Triple MACD, and Ensemble implementations."""

from .ensemble import EnsembleStrategy
from .ensemble_unconstrained import EnsembleUnconstrainedStrategy
from .rsi_filter_portfolio import RSIFilterPortfolioStrategy
from .triple_ema import TripleEMAStrategy
from .triple_ema_unconstrained import TripleEMAUnconstrainedStrategy
from .triple_macd import TripleMACDStrategy

StrategyFactory = {
    "triple_ema": TripleEMAStrategy,
    "triple_ema_unconstrained": TripleEMAUnconstrainedStrategy,
    "triple_macd": TripleMACDStrategy,
    "ensemble": EnsembleStrategy,
    "ensemble_unconstrained": EnsembleUnconstrainedStrategy,
}

__all__ = [
    "StrategyFactory",
    "TripleEMAStrategy",
    "TripleEMAUnconstrainedStrategy",
    "TripleMACDStrategy",
    "EnsembleStrategy",
    "EnsembleUnconstrainedStrategy",
    "RSIFilterPortfolioStrategy",
]

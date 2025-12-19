"""Strategy registry with Triple EMA, Triple MACD, and Ensemble implementations."""

from .ensemble import EnsembleStrategy
from .ensemble_unconstrained import EnsembleUnconstrainedStrategy
from .rsi_filter_portfolio import RSIFilterPortfolioStrategy
from .triple_ema import TripleEMAStrategy
from .triple_ema_unconstrained import TripleEMAUnconstrainedStrategy
from .macd import MACDStrategy

StrategyFactory = {
    "triple_ema": TripleEMAStrategy,
    "triple_ema_unconstrained": TripleEMAUnconstrainedStrategy,
    "macd": MACDStrategy,
    "ensemble": EnsembleStrategy,
    "ensemble_unconstrained": EnsembleUnconstrainedStrategy,
}

__all__ = [
    "StrategyFactory",
    "TripleEMAStrategy",
    "TripleEMAUnconstrainedStrategy",
    "MACDStrategy",
    "EnsembleStrategy",
    "EnsembleUnconstrainedStrategy",
    "RSIFilterPortfolioStrategy",
]

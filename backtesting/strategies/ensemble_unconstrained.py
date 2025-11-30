"""Ensemble strategy using unconstrained Triple EMA and standard Triple MACD.

Combines the unconstrained EMA (no parameter ordering) with MACD using OR logic.
This allows testing EMA parameters in any order while keeping MACD constrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from .triple_ema_unconstrained import TripleEMAUnconstrainedStrategy
from .triple_macd import TripleMACDStrategy


@dataclass
class EnsembleUnconstrainedStrategy:
    """Ensemble strategy with unconstrained EMA parameters.
    
    Combines signals using OR logic:
    - Buy if Unconstrained EMA signals buy OR MACD signals buy
    - Sell if Unconstrained EMA signals sell OR MACD signals sell
    
    The EMA parameters have no ordering constraint (any combination allowed).
    MACD still requires fastperiod < slowperiod for proper calculation.
    
    Parameters:
        ema_fast: First EMA period (not necessarily shortest)
        ema_mid: Second EMA period  
        ema_slow: Third EMA period (not necessarily longest)
        fastperiod: Fast EMA period for MACD (must be < slowperiod)
        slowperiod: Slow EMA period for MACD
        signalperiod: Signal line EMA period for MACD
    """
    ema_fast: int
    ema_mid: int
    ema_slow: int
    fastperiod: int
    slowperiod: int
    signalperiod: int
    
    def generate_signals(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate ensemble signals by combining unconstrained EMA and MACD with OR logic."""
        # Generate signals from unconstrained EMA strategy
        ema_strategy = TripleEMAUnconstrainedStrategy(
            ema_fast=self.ema_fast,
            ema_mid=self.ema_mid,
            ema_slow=self.ema_slow
        )
        ema_entries, ema_exits = ema_strategy.generate_signals(close)
        
        # Generate signals from standard MACD strategy
        macd_strategy = TripleMACDStrategy(
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        macd_entries, macd_exits = macd_strategy.generate_signals(close)
        
        # Combine signals using OR logic
        ensemble_entries = ema_entries | macd_entries
        ensemble_exits = ema_exits | macd_exits
        
        return ensemble_entries, ensemble_exits
    
    def params(self):
        return {
            "ema_fast": self.ema_fast,
            "ema_mid": self.ema_mid,
            "ema_slow": self.ema_slow,
            "fastperiod": self.fastperiod,
            "slowperiod": self.slowperiod,
            "signalperiod": self.signalperiod,
        }


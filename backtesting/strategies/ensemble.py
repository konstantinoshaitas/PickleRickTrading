"""Ensemble strategy combining Triple EMA and Triple MACD using OR logic.

This implements the ensemble approach described in NOTEBOOK 06 - THE CASINO,
where multiple technical indicators are combined to increase robustness
and reduce overfitting. Signals from both strategies are combined using OR logic,
meaning a trade is taken if EITHER indicator signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from .triple_ema import TripleEMAStrategy
from .triple_macd import TripleMACDStrategy


@dataclass
class EnsembleStrategy:
    """Ensemble strategy combining Triple EMA and Triple MACD.
    
    Combines signals using OR logic:
    - Buy if EMA signals buy OR MACD signals buy
    - Sell if EMA signals sell OR MACD signals sell
    
    This allows the indicators to correct each other's false or delayed signals,
    increasing robustness as described in the ensemble theory.
    
    Parameters:
        ema_fast: Fast EMA period for Triple EMA strategy
        ema_mid: Medium EMA period for Triple EMA strategy
        ema_slow: Slow EMA period for Triple EMA strategy
        fastperiod: Fast EMA period for MACD (same as triple_macd)
        slowperiod: Slow EMA period for MACD (same as triple_macd)
        signalperiod: Signal line EMA period for MACD (same as triple_macd)
    """
    ema_fast: int
    ema_mid: int
    ema_slow: int
    fastperiod: int
    slowperiod: int
    signalperiod: int
    
    def generate_signals(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate ensemble signals by combining EMA and MACD with OR logic.
        
        The OR logic means:
        - If EITHER EMA or MACD signals a buy, we enter
        - If EITHER EMA or MACD signals a sell, we exit
        
        This allows the indicators to:
        1. Correct each other's false signals
        2. Compensate for delayed signals (one may react earlier)
        3. Increase robustness through diversification
        """
        # Generate signals from both strategies
        ema_strategy = TripleEMAStrategy(
            ema_fast=self.ema_fast,
            ema_mid=self.ema_mid,
            ema_slow=self.ema_slow
        )
        ema_entries, ema_exits = ema_strategy.generate_signals(close)
        
        macd_strategy = TripleMACDStrategy(
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        macd_entries, macd_exits = macd_strategy.generate_signals(close)
        
        # Combine signals using OR logic
        # Buy if EMA signals buy OR MACD signals buy
        ensemble_entries = ema_entries | macd_entries
        
        # Sell if EMA signals sell OR MACD signals sell
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


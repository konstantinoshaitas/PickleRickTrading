"""Triple EMA strategy without parameter ordering constraints.

Unlike TripleEMAStrategy, this version allows ANY parameter combination
without requiring ema_fast < ema_mid < ema_slow. This enables testing
all possible EMA crossover combinations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import vectorbt as vbt

# Suppress pandas FutureWarning for fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')


@dataclass
class TripleEMAUnconstrainedStrategy:
    """Triple EMA crossover strategy without parameter ordering constraints.
    
    Parameters can be in any order (no requirement for ema_fast < ema_mid < ema_slow).
    This allows testing all possible EMA combinations to find potentially
    unconventional but profitable setups.
    
    Signal Logic:
        - Entry: ANY EMA crosses above ANY other EMA
        - Exit: ANY EMA crosses below ANY other EMA
    
    Parameters:
        ema_fast: First EMA period (not necessarily the shortest)
        ema_mid: Second EMA period
        ema_slow: Third EMA period (not necessarily the longest)
    """
    ema_fast: int
    ema_mid: int
    ema_slow: int
    
    def generate_signals(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals.
        
        Uses the same crossover logic as TripleEMAStrategy but without
        any parameter ordering constraints.
        """
        # Compute EMAs
        ema1_ma = vbt.MA.run(close, self.ema_fast, ewm=True).ma
        ema2_ma = vbt.MA.run(close, self.ema_mid, ewm=True).ma
        ema3_ma = vbt.MA.run(close, self.ema_slow, ewm=True).ma
        
        # Triple EMA crossover signals (same logic, no constraints)
        e1 = ema1_ma.vbt.crossed_above(ema2_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        e2 = ema1_ma.vbt.crossed_above(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        e3 = ema2_ma.vbt.crossed_above(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        entries_raw = e1 | e2 | e3
        
        x1 = ema1_ma.vbt.crossed_below(ema2_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        x2 = ema1_ma.vbt.crossed_below(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        x3 = ema2_ma.vbt.crossed_below(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        exits_raw = x1 | x2 | x3
        
        # Shift to avoid lookahead bias
        entries = entries_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        exits = exits_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        
        return entries, exits
    
    def params(self):
        return {
            "ema_fast": self.ema_fast,
            "ema_mid": self.ema_mid,
            "ema_slow": self.ema_slow,
        }


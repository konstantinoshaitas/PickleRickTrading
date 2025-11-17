"""Regime-aware Triple EMA strategy implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

# Suppress pandas FutureWarning for fillna downcasting (we handle dtype conversion explicitly)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')


@dataclass
class TripleEMAStrategy:
    ema_fast: int
    ema_mid: int
    ema_slow: int
    
    def generate_signals(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals matching notebook logic exactly.
        
        Entries: ANY EMA crosses above ANY other EMA
        Exits: ANY EMA crosses below ANY other EMA
        
        Note: Uses .vbt.crossed_above() pattern like notebook validation section
        for consistency across train/val splits.
        """
        # Extract .ma series like notebook validation (line 2441)
        ema1_ma = vbt.MA.run(close, self.ema_fast, ewm=True).ma
        ema2_ma = vbt.MA.run(close, self.ema_mid, ewm=True).ma
        ema3_ma = vbt.MA.run(close, self.ema_slow, ewm=True).ma
        
        # Triple EMA crossover signals (matches notebook validation exactly - line 2445)
        e1 = ema1_ma.vbt.crossed_above(ema2_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        e2 = ema1_ma.vbt.crossed_above(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        e3 = ema2_ma.vbt.crossed_above(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        entries_raw = e1 | e2 | e3
        
        x1 = ema1_ma.vbt.crossed_below(ema2_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        x2 = ema1_ma.vbt.crossed_below(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        x3 = ema2_ma.vbt.crossed_below(ema3_ma).reindex(close.index).fillna(False).infer_objects(copy=False).astype(bool)
        exits_raw = x1 | x2 | x3
        
        # Shift and fix lookahead bias (matches notebook - line 2457)
        entries = entries_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        exits = exits_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        
        return entries, exits
    
    def params(self):
        return {
            "ema_fast": self.ema_fast,
            "ema_mid": self.ema_mid,
            "ema_slow": self.ema_slow,
        }

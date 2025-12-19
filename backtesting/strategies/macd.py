"""MACD strategy implementation for trend following.

Uses MACD line crossing signal line for entry/exit signals.
MACD consists of three components: MACD line, Signal line, and Histogram.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import vectorbt as vbt

# Suppress pandas FutureWarning for fillna downcasting (we handle dtype conversion explicitly)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')


@dataclass
class MACDStrategy:
    """MACD strategy: Buy when MACD line crosses above Signal line, sell when MACD line crosses below Signal line.
    
    MACD is calculated using vectorbt's MACD indicator:
    - MACD Line = Fast EMA - Slow EMA (of price)
    - Signal Line = EMA of the MACD Line
    
    Parameters:
        fastperiod: Period for fast EMA used in MACD line calculation (default 12)
        slowperiod: Period for slow EMA used in MACD line calculation (default 26)
        signalperiod: Period for EMA of MACD line (the signal line) (default 9)
    """
    fastperiod: int
    slowperiod: int
    signalperiod: int
    
    def generate_signals(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals using MACD crossover.
        
        Entries: MACD line crosses above Signal line (bullish momentum)
        Exits: MACD line crosses below Signal line (bearish momentum)
        
        This is a trend-following strategy that captures momentum shifts.
        """
        # Calculate MACD components using vectorbt (matches grid search)
        macd = vbt.MACD.run(
            close,
            fast_window=self.fastperiod,
            slow_window=self.slowperiod,
            signal_window=self.signalperiod
        )
        
        macd_line = macd.macd
        signal_line = macd.signal
        
        # Generate crossover signals
        # Entry: MACD crosses above Signal (bullish momentum)
        entries_raw = macd_line.vbt.crossed_above(signal_line).reindex(close.index)
        
        # Exit: MACD crosses below Signal (bearish momentum)
        exits_raw = macd_line.vbt.crossed_below(signal_line).reindex(close.index)
        
        # Shift and fix lookahead bias (matches notebook pattern)
        # Fill NaN values and convert to bool (using infer_objects to avoid FutureWarning)
        entries = entries_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        exits = exits_raw.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        
        return entries, exits
    
    def params(self):
        return {
            "fastperiod": self.fastperiod,
            "slowperiod": self.slowperiod,
            "signalperiod": self.signalperiod,
        }


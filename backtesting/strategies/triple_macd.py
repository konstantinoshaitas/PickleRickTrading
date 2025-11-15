"""Triple MACD strategy implementation for trend following.

Uses MACD line crossing signal line for entry/exit signals.
The 'triple' refers to the three components: MACD line, Signal line, and Histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import talib
import vectorbt as vbt


@dataclass
class TripleMACDStrategy:
    """MACD strategy: Buy when MACD line crosses above Signal line, sell when MACD line crosses below Signal line.
    
    MACD is calculated using EMAs internally:
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
        # Calculate MACD components using TA-Lib
        macd_line, signal_line, histogram = talib.MACD(
            close.values,
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod
        )
        
        # Convert to pandas Series with proper index
        macd_series = pd.Series(macd_line, index=close.index)
        signal_series = pd.Series(signal_line, index=close.index)
        
        # Generate crossover signals
        # Entry: MACD crosses above Signal (bullish momentum)
        entries_raw = macd_series.vbt.crossed_above(signal_series).reindex(close.index).fillna(False)
        
        # Exit: MACD crosses below Signal (bearish momentum)
        exits_raw = macd_series.vbt.crossed_below(signal_series).reindex(close.index).fillna(False)
        
        # Shift and fix lookahead bias (matches notebook pattern)
        entries = entries_raw.shift(1).fillna(False)
        exits = exits_raw.shift(1).fillna(False)
        
        return entries, exits
    
    def params(self):
        return {
            "fastperiod": self.fastperiod,
            "slowperiod": self.slowperiod,
            "signalperiod": self.signalperiod,
        }


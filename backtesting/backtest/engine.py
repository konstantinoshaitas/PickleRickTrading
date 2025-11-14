"""Concise vectorbt wrapper used by the CLI and notebook."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import vectorbt as vbt

from ..config import BacktestConfig


class BacktestEngine:
    """Tiny helper that turns strategy signals into a vectorbt portfolio."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run(self, close: pd.Series, signals: Tuple[pd.Series, pd.Series]) -> vbt.Portfolio:
        entries, exits = signals
        return vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            slippage=self.config.slippage,
            freq=self.config.freq,
        )

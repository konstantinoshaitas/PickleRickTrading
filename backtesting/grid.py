"""Tiny brute-force grid search helper."""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd

from .backtest import BacktestEngine
from .metrics import compute_metrics


class GridSearch:
    def __init__(self, engine: BacktestEngine, strategy_cls):
        self.engine = engine
        self.strategy_cls = strategy_cls
        self.results: List[Dict] = []
    
    def run(self, close: pd.Series, grid: Dict[str, List[int]], base_params: Dict[str, int]):
        keys = list(grid.keys())
        # Calculate total combinations for progress tracking
        all_combos = list(itertools.product(*[grid[k] for k in keys]))
        total_combos = len(all_combos)
        
        for i, combo in enumerate(all_combos, 1):
            params = dict(base_params)
            params.update(dict(zip(keys, combo)))
            
            # Strategy-agnostic parameter validation
            # Skip invalid parameter combinations
            if not self._validate_params(params):
                continue
            
            strat = self.strategy_cls(**params)
            entries, exits = strat.generate_signals(close)
            portfolio = self.engine.run(close, (entries, exits))
            metrics = compute_metrics(portfolio, close, self.engine.config.freq)
            metrics.update(params)
            self.results.append(metrics)
            
            # Print progress every 10% or every 50 iterations, whichever is more frequent
            progress_interval = max(1, min(50, total_combos // 10))
            if i % progress_interval == 0 or i == total_combos:
                progress_pct = (i / total_combos) * 100
                print(f"Progress: {i}/{total_combos} ({progress_pct:.1f}%) - {len(self.results)} valid results")
        
        return self.results
    
    def _validate_params(self, params: Dict[str, int]) -> bool:
        """Validate parameter combinations based on strategy type.
        
        Returns True if parameters are valid, False otherwise.
        """
        # Triple EMA validation: ema_fast < ema_mid < ema_slow
        if "ema_fast" in params and "ema_mid" in params and "ema_slow" in params:
            if not (params["ema_fast"] < params["ema_mid"] < params["ema_slow"]):
                return False
        
        # MACD validation: fastperiod < slowperiod
        if "fastperiod" in params and "slowperiod" in params:
            if not (params["fastperiod"] < params["slowperiod"]):
                return False
        
        # Ensemble validation: both EMA and MACD constraints must be satisfied
        # (handled by the checks above)
        
        return True
    
    def best(self, metric: str):
        if not self.results:
            raise ValueError("Run grid search first.")
        df = pd.DataFrame(self.results)
        return df.sort_values(metric, ascending=False).iloc[0]

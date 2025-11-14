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
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(base_params)
            params.update(dict(zip(keys, combo)))
            if not (params["ema_fast"] < params["ema_mid"] < params["ema_slow"]):
                continue
            strat = self.strategy_cls(**params)
            entries, exits = strat.generate_signals(close)
            portfolio = self.engine.run(close, (entries, exits))
            metrics = compute_metrics(portfolio, close, self.engine.config.freq)
            metrics.update(params)
            self.results.append(metrics)
        return self.results
    
    def best(self, metric: str):
        if not self.results:
            raise ValueError("Run grid search first.")
        df = pd.DataFrame(self.results)
        return df.sort_values(metric, ascending=False).iloc[0]

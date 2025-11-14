"""Minimal backtesting toolkit built for the Triple EMA notebook workflow."""

from .config import BacktestConfig, DataConfig, StrategyConfig, WorkflowConfig, load_config
from .data import DataFetcher, split_train_val
from .metrics import buy_and_hold, compute_metrics
from .pipeline import (
    load_prices,
    run_grid_search,
    run_single_backtest,
    save_grid_results,
)
from .strategies import StrategyFactory, TripleEMAStrategy

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "StrategyConfig",
    "WorkflowConfig",
    "DataFetcher",
    "split_train_val",
    "compute_metrics",
    "buy_and_hold",
    "StrategyFactory",
    "TripleEMAStrategy",
    "load_prices",
    "run_single_backtest",
    "run_grid_search",
    "save_grid_results",
    "load_config",
]

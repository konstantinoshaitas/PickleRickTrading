"""Minimal backtesting toolkit built"""

from .config import BacktestConfig, DataConfig, StrategyConfig, WorkflowConfig, load_config
from .data import DataFetcher, split_train_val
from .metrics import buy_and_hold, compute_metrics
from .pipeline import (
    load_prices,
    run_grid_search,
    run_single_backtest,
    save_grid_results,
)
from .strategies import (
    EnsembleStrategy,
    StrategyFactory,
    TripleEMAStrategy,
    TripleMACDStrategy,
)
from .visualization import (
    plot_cumulative_equity,
    plot_drawdowns,
    plot_equity_curves,
    plot_rolling_sharpe,
    plot_signals,
    plot_trade_returns,
)

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
    "TripleMACDStrategy",
    "EnsembleStrategy",
    "load_prices",
    "run_single_backtest",
    "run_grid_search",
    "save_grid_results",
    "load_config",
    "plot_rolling_sharpe",
    "plot_drawdowns",
    "plot_signals",
    "plot_equity_curves",
    "plot_trade_returns",
    "plot_cumulative_equity",
]

"""Minimal backtesting toolkit built"""

from .config import (
    BacktestConfig,
    DataConfig,
    OptimizationConfig,
    RegistryEntry,
    StrategyConfig,
    WorkflowConfig,
    get_asset_cache_path,
    get_asset_results_dir,
    get_registry_entry,
    load_config,
    load_registry,
    load_template,
    save_registry,
    update_registry_entry,
)
from .data import DataFetcher, split_train_val, split_train_val_test, get_split_info
from .grid import GridSearch, VectorizedGridSearch, VectorizedPortfolioGridSearch
from .metrics import buy_and_hold, compute_batch_metrics, compute_metrics
from .optimizer import AssetOptimizer, OptimizationResult, optimize_asset
from .pipeline import (
    load_prices,
    run_grid_search,
    run_portfolio_grid_search,
    run_single_backtest,
    save_grid_results,
)
from .strategies import (
    EnsembleStrategy,
    EnsembleUnconstrainedStrategy,
    StrategyFactory,
    TripleEMAStrategy,
    TripleEMAUnconstrainedStrategy,
    MACDStrategy,
)
from .visualization import (
    plot_cumulative_equity,
    plot_drawdowns,
    plot_equity_curves,
    plot_full_sample_equity,
    plot_rolling_sharpe,
    plot_signals,
    plot_trade_returns,
)

__all__ = [
    # Config
    "BacktestConfig",
    "DataConfig",
    "OptimizationConfig",
    "RegistryEntry",
    "StrategyConfig",
    "WorkflowConfig",
    "load_config",
    "load_template",
    # Registry
    "load_registry",
    "save_registry",
    "get_registry_entry",
    "update_registry_entry",
    "get_asset_cache_path",
    "get_asset_results_dir",
    # Data
    "DataFetcher",
    "split_train_val",
    "split_train_val_test",
    "get_split_info",
    # Metrics
    "compute_metrics",
    "compute_batch_metrics",
    "buy_and_hold",
    # Grid Search
    "GridSearch",
    "VectorizedGridSearch",
    "VectorizedPortfolioGridSearch",
    # Optimizer
    "AssetOptimizer",
    "OptimizationResult",
    "optimize_asset",
    # Strategies
    "StrategyFactory",
    "TripleEMAStrategy",
    "TripleEMAUnconstrainedStrategy",
    "MACDStrategy",
    "EnsembleStrategy",
    "EnsembleUnconstrainedStrategy",
    # Pipeline
    "load_prices",
    "run_single_backtest",
    "run_grid_search",
    "run_portfolio_grid_search",
    "save_grid_results",
    # Visualization
    "plot_rolling_sharpe",
    "plot_drawdowns",
    "plot_signals",
    "plot_equity_curves",
    "plot_full_sample_equity",
    "plot_trade_returns",
    "plot_cumulative_equity",
]

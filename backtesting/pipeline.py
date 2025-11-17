"""Reusable building blocks for CLI subcommands and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .backtest import BacktestEngine
from .config import WorkflowConfig
from .data import DataFetcher, split_train_val
from .grid import GridSearch
from .metrics import buy_and_hold, compute_metrics
from .strategies import StrategyFactory


def load_prices(cfg: WorkflowConfig, force_download: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """Fetch prices based on config, returning close series and full OHLCV frame."""
    fetcher = DataFetcher(
        ticker=cfg.data.ticker,
        start=cfg.data.start,
        end=cfg.data.end,
        interval=cfg.data.interval,
        data_source=cfg.data.data_source,
        asset_type=cfg.data.asset_type,
        local_csv=cfg.data.local_csv,
        cache_csv=cfg.data.cache_csv,
    )
    ohlcv = fetcher.load(force_download=force_download)
    close = fetcher.close()
    return close, ohlcv


def run_single_backtest(cfg: WorkflowConfig, close: pd.Series, return_portfolios: bool = False) -> Dict[str, Any]:
    """Run the configured strategy on train/validation splits.
    
    Args:
        cfg: Workflow configuration
        close: Price series
        return_portfolios: If True, include portfolio objects and signals in output for plotting
        
    Returns:
        Dictionary with metrics and optionally portfolios/signals for visualization
    """
    train_close, val_close = split_train_val(close, cfg.backtest.train_ratio)
    strategy_cls = StrategyFactory[cfg.strategy.name]
    strategy = strategy_cls(**cfg.strategy.params)
    engine = BacktestEngine(cfg.backtest)
    
    train_entries, train_exits = strategy.generate_signals(train_close)
    train_portfolio = engine.run(train_close, (train_entries, train_exits))
    train_metrics = compute_metrics(train_portfolio, train_close, cfg.backtest.freq)
    
    outputs = {
        "train": train_metrics,
        "train_window": (train_close.index[0], train_close.index[-1]),
    }
    
    if return_portfolios:
        outputs["train_portfolio"] = train_portfolio
        outputs["train_entries"] = train_entries
        outputs["train_exits"] = train_exits
        outputs["train_close"] = train_close
    
    if len(val_close) > 0:
        val_entries, val_exits = strategy.generate_signals(val_close)
        val_portfolio = engine.run(val_close, (val_entries, val_exits))
        outputs["validation"] = compute_metrics(val_portfolio, val_close, cfg.backtest.freq)
        outputs["benchmark"] = buy_and_hold(val_close, cfg.backtest)
        outputs["validation_window"] = (val_close.index[0], val_close.index[-1])
        
        if return_portfolios:
            outputs["val_portfolio"] = val_portfolio
            outputs["val_entries"] = val_entries
            outputs["val_exits"] = val_exits
            outputs["val_close"] = val_close

    return outputs


def run_grid_search(cfg: WorkflowConfig, close: pd.Series):
    """Execute brute-force grid search on the training slice."""
    if not cfg.strategy.grid:
        raise ValueError("No grid defined in config.")
    train_close, _ = split_train_val(close, cfg.backtest.train_ratio)
    engine = BacktestEngine(cfg.backtest)
    strategy_cls = StrategyFactory[cfg.strategy.name]
    
    search = GridSearch(engine, strategy_cls)
    search.run(train_close, cfg.strategy.grid, cfg.strategy.params)
    return search


def save_grid_results(search: GridSearch, path: Path, sort_by: str = "sharpe_ratio", ascending: bool = False):
    """Persist grid search results to CSV, sorted by specified metric.
    
    Args:
        search: GridSearch instance with results
        path: Output file path
        sort_by: Metric to sort by (default: "sharpe_ratio")
        ascending: Sort order (default: False = descending)
    
    Note:
        Metrics that cannot be calculated (e.g., deflated_sharpe_ratio if method unavailable,
        winning_streak/losing_streak if no trades) will be np.nan, which pandas writes as
        empty strings in CSV. This is expected behavior.
    """
    if not search.results:
        raise ValueError("No grid results to save.")
    df = pd.DataFrame(search.results)
    
    # Sort by specified metric if it exists
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    else:
        # Fallback to first available metric if sort_by not found
        metric_cols = [c for c in df.columns if c not in ['ema_fast', 'ema_mid', 'ema_slow', 
                                                          'fastperiod', 'slowperiod', 'signalperiod']]
        if metric_cols:
            df = df.sort_values(metric_cols[0], ascending=ascending)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    # Note: np.nan values will be written as empty strings in CSV (standard pandas behavior)
    df.to_csv(path, index=False)
    return path

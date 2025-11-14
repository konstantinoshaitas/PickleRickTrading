"""Reusable building blocks for CLI subcommands and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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
        local_csv=cfg.data.local_csv,
        cache_csv=cfg.data.cache_csv,
    )
    ohlcv = fetcher.load(force_download=force_download)
    close = fetcher.close()
    return close, ohlcv


def run_single_backtest(cfg: WorkflowConfig, close: pd.Series) -> Dict[str, Dict]:
    """Run the configured strategy on train/validation splits."""
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
    if len(val_close) > 0:
        val_entries, val_exits = strategy.generate_signals(val_close)
        val_portfolio = engine.run(val_close, (val_entries, val_exits))
        outputs["validation"] = compute_metrics(val_portfolio, val_close, cfg.backtest.freq)
        outputs["benchmark"] = buy_and_hold(val_close, cfg.backtest)
        outputs["validation_window"] = (val_close.index[0], val_close.index[-1])

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


def save_grid_results(search: GridSearch, path: Path):
    """Persist grid search results to CSV."""
    if not search.results:
        raise ValueError("No grid results to save.")
    df = pd.DataFrame(search.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

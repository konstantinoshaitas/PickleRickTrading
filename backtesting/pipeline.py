"""Reusable building blocks for CLI subcommands and notebooks."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .backtest import BacktestEngine, PortfolioBuilder
from .config import (
    WorkflowConfig, 
    get_asset_cache_path,
    calculate_warmup_from_grid,
    calculate_warmup_from_params,
)
from .data import DataFetcher, load_multi_asset_prices, split_train_val
from .grid import GridSearch, VectorizedGridSearch, VectorizedPortfolioGridSearch
from .metrics import buy_and_hold, compute_metrics
from .strategies import StrategyFactory
from .strategies.rsi_filter_portfolio import RSIFilterPortfolioStrategy


def load_prices(
    cfg: WorkflowConfig, 
    force_download: bool = False,
    warmup_bars: Optional[int] = None,
    allow_partial_warmup: bool = False,
    min_backtest_bars: int = 252,
) -> Tuple[pd.Series, pd.DataFrame, "DataFetcher"]:
    """Fetch prices based on config, returning close series and full OHLCV frame.
    
    Auto-resolves cache path to new asset-centric structure (assets/{TICKER}/cache.csv).
    If cache_csv is not set in config, uses the new path (creating it if needed).
    Falls back to config value for backward compatibility if explicitly set.
    
    Args:
        cfg: Workflow configuration
        force_download: Force fresh download (ignore cache)
        warmup_bars: Override warmup bars (None = use cfg.data.warmup_bars or auto-calculate)
        allow_partial_warmup: Allow proceeding with insufficient warmup
        min_backtest_bars: Minimum bars required for backtesting (default 252)
        
    Returns:
        Tuple of (close_series, ohlcv_dataframe, fetcher)
        - close_series: Close prices (including warmup period for indicator calculation)
        - ohlcv_dataframe: Full OHLCV data (including warmup)
        - fetcher: DataFetcher instance (use fetcher.close_trimmed() for backtest evaluation)
    """
    # Auto-resolve cache path to new asset-centric structure
    cache_csv = cfg.data.cache_csv
    
    # If cache_csv not set or doesn't exist, try new asset-centric path
    if not cache_csv or not Path(cache_csv).exists():
        new_cache_path = get_asset_cache_path(cfg.data.ticker)
        if new_cache_path.exists():
            cache_csv = str(new_cache_path)
        elif cfg.data.cache_csv:
            # Keep original if specified (for backward compatibility)
            cache_csv = cfg.data.cache_csv
        else:
            # Use new path even if it doesn't exist yet (so it can be created)
            cache_csv = str(new_cache_path)
    
    # Determine warmup bars
    if warmup_bars is None:
        warmup_bars = cfg.data.warmup_bars
        
        # Auto-calculate from grid if warmup_bars is 0 and grid exists
        if warmup_bars == 0 and cfg.strategy.grid:
            warmup_bars = calculate_warmup_from_grid(cfg.strategy.grid, cfg.strategy.name)
            if warmup_bars > 0:
                print(f"Auto-calculated warmup: {warmup_bars} bars from grid")
        elif warmup_bars == 0 and cfg.strategy.params:
            # Calculate from params for single backtest
            warmup_bars = calculate_warmup_from_params(cfg.strategy.params, cfg.strategy.name)
    
    fetcher = DataFetcher(
        ticker=cfg.data.ticker,
        start=cfg.data.start,
        end=cfg.data.end,
        interval=cfg.data.interval,
        data_source=cfg.data.data_source,
        asset_type=cfg.data.asset_type,
        local_csv=cfg.data.local_csv,
        cache_csv=cache_csv,
        warmup_bars=warmup_bars,
    )
    ohlcv = fetcher.load(
        force_download=force_download,
        validate_warmup=(warmup_bars > 0),
        allow_partial_warmup=allow_partial_warmup,
        min_backtest_bars=min_backtest_bars,
    )
    close = fetcher.close()
    return close, ohlcv, fetcher


def load_prices_legacy(cfg: WorkflowConfig, force_download: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """Legacy load_prices for backward compatibility (no warmup).
    
    Deprecated: Use load_prices() instead for new code.
    """
    close, ohlcv, _ = load_prices(cfg, force_download=force_download, warmup_bars=0)
    return close, ohlcv


def run_single_backtest(
    cfg: WorkflowConfig, 
    close: pd.Series, 
    return_portfolios: bool = False,
    close_full: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Run the configured strategy on train/validation splits.
    
    Args:
        cfg: Workflow configuration
        close: Price series for backtest evaluation (trimmed to original start, no warmup)
        return_portfolios: If True, include portfolio objects and signals in output for plotting
        close_full: Full price series including warmup period for indicator calculation.
                   If None, uses close for both indicator calculation and backtest.
        
    Returns:
        Dictionary with metrics and optionally portfolios/signals for visualization
        
    Note:
        When using warmup, pass close_full containing the warmup period for indicator 
        calculation and close trimmed to original start for backtest evaluation.
    """
    # Use full series for indicator calculation if provided
    indicator_close = close_full if close_full is not None else close
    
    train_close, val_close = split_train_val(close, cfg.backtest.train_ratio)
    strategy_cls = StrategyFactory[cfg.strategy.name]
    strategy = strategy_cls(**cfg.strategy.params)
    engine = BacktestEngine(cfg.backtest)
    
    # Calculate indicators on full data (including warmup) then trim
    if close_full is not None:
        # Get the train/val split points from trimmed data
        train_start = train_close.index[0]
        train_end = train_close.index[-1]
        
        # Generate signals on full data
        full_entries, full_exits = strategy.generate_signals(indicator_close)
        
        # Trim signals to match train close
        train_entries = full_entries.loc[train_start:train_end]
        train_exits = full_exits.loc[train_start:train_end]
    else:
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
        if close_full is not None:
            val_start = val_close.index[0]
            val_end = val_close.index[-1]
            val_entries = full_entries.loc[val_start:val_end]
            val_exits = full_exits.loc[val_start:val_end]
        else:
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


def run_grid_search(
    cfg: WorkflowConfig, 
    close: pd.Series, 
    n_jobs: Optional[int] = None,
    use_vectorized: bool = True,
    batch_size: int = 5000,
    min_trades_per_year: float = 0.5,
    close_full: Optional[pd.Series] = None,
    warmup_bars: int = 0,
):
    """Execute grid search on the training slice.
    
    Args:
        cfg: Workflow configuration
        close: Price series (trimmed to original start - used for backtest)
        n_jobs: Number of CPU cores to use (default: all - 1)
        use_vectorized: Use vectorized grid search for supported strategies (default: True)
        batch_size: Batch size for vectorized search (default: 5000)
        min_trades_per_year: Minimum trades per year filter for vectorized search (default: 2.0)
        close_full: Full price series including warmup period for indicator calculation.
                   If provided with warmup_bars > 0, indicators are calculated on this 
                   and then trimmed to match close.
        warmup_bars: Number of warmup bars in close_full (used for trimming)
        
    Returns:
        GridSearch or VectorizedGridSearch object with results
        
    Note:
        Vectorized search is ~10-20x faster but only supports:
        - triple_ema
        - triple_ema_unconstrained
        - macd
        - ensemble
        - ensemble_unconstrained
        
        For other strategies, falls back to multiprocessing-based GridSearch.
        
        When close_full and warmup_bars are provided, indicators are calculated
        on the full series and then trimmed to the original start date for backtesting.
    """
    if not cfg.strategy.grid:
        raise ValueError("No grid defined in config.")
    
    # Use full close for indicator calculation if provided
    indicator_close = close_full if close_full is not None else close
    
    train_close, _ = split_train_val(close, cfg.backtest.train_ratio)
    
    # Also split indicator close if using warmup
    if close_full is not None and warmup_bars > 0:
        train_indicator_close, _ = split_train_val(indicator_close, cfg.backtest.train_ratio)
    else:
        train_indicator_close = train_close
    
    engine = BacktestEngine(cfg.backtest)
    
    # Check if vectorized search is available for this strategy
    vectorized_strategies = VectorizedGridSearch.SUPPORTED_STRATEGIES
    
    if use_vectorized and cfg.strategy.name in vectorized_strategies:
        print(f"Using VectorizedGridSearch for '{cfg.strategy.name}' strategy...")
        search = VectorizedGridSearch(
            engine=engine,
            strategy_name=cfg.strategy.name,
            batch_size=batch_size,
            n_jobs=n_jobs,
            min_trades_per_year=min_trades_per_year,
            warmup_bars=warmup_bars,
        )
        # Pass both indicator close (for signal calc) and train close (for backtest)
        search.run(
            train_indicator_close, 
            cfg.strategy.grid, 
            cfg.strategy.params,
            backtest_close=train_close if warmup_bars > 0 else None,
        )
    else:
        if use_vectorized and cfg.strategy.name not in vectorized_strategies:
            print(f"Note: Strategy '{cfg.strategy.name}' not supported for vectorization. Using GridSearch.")
        strategy_cls = StrategyFactory[cfg.strategy.name]
        search = GridSearch(engine, strategy_cls, n_jobs=n_jobs, warmup_bars=warmup_bars)
        search.run(
            train_indicator_close, 
            cfg.strategy.grid, 
            cfg.strategy.params,
            backtest_close=train_close if warmup_bars > 0 else None,
        )
    
    return search


def save_grid_results(search_results: List[Dict], path: Path, sort_by: str = "sharpe_ratio", ascending: bool = False):
    """Persist grid search results to Parquet format, sorted by specified metric.
    
    Args:
        search_results: List of result dictionaries (from GridSearch.results or portfolio grid)
        path: Output file path (will use .parquet extension if not specified)
        sort_by: Metric to sort by (default: "sharpe_ratio")
        ascending: Sort order (default: False = descending)
    
    Note:
        Parquet format preserves data types and is much more efficient for large datasets.
        Metrics that cannot be calculated will be np.nan (preserved in Parquet).
    """
    # Handle both GridSearch object and raw list of dicts
    if hasattr(search_results, 'results'):
        results_data = search_results.results
    else:
        results_data = search_results

    if not results_data:
        raise ValueError("No grid results to save.")
    
    df = pd.DataFrame(results_data)
    
    # Sort by specified metric if it exists
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    else:
        # Fallback to first available metric if sort_by not found
        metric_cols = [c for c in df.columns if c not in ['ema_fast', 'ema_mid', 'ema_slow', 
                                                          'fastperiod', 'slowperiod', 'signalperiod',
                                                          'rsi_period', 'rsi_threshold', 'top_k']]
        if metric_cols:
            df = df.sort_values(metric_cols[0], ascending=ascending)
    
    # Ensure .parquet extension
    if path.suffix.lower() != '.parquet':
        path = path.with_suffix('.parquet')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet with snappy compression (good balance of speed and size)
    print(f"Writing {len(df):,} rows to {path}...")
    df.to_parquet(
        path,
        compression='snappy',
        index=False,
        engine='pyarrow'
    )
    
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    print(f"Saved {file_size:.1f} MB to {path}")
    
    return path


def run_portfolio_backtest(
    cfg: WorkflowConfig,
    return_portfolios: bool = False,
) -> Dict[str, Any]:
    """Run multi-asset portfolio backtest with RSI filtering and top-K selection.
    
    Args:
        cfg: Workflow configuration (must have portfolio config)
        return_portfolios: If True, include portfolio objects and weights in output
        
    Returns:
        Dictionary with metrics and optionally portfolios/weights for visualization
        
    Raises:
        ValueError: If portfolio config is not provided
    """
    if cfg.portfolio is None:
        raise ValueError(
            "Portfolio config is required for portfolio backtest. "
            "Set cfg.portfolio with tickers, RSI parameters, etc."
        )
    
    # Build per-asset cache_csv overrides from portfolio.assets if any are specified
    per_asset_cache_csv = {}
    if cfg.portfolio.assets:
        for ticker, asset_cfg in cfg.portfolio.assets.items():
            if asset_cfg.cache_csv:
                per_asset_cache_csv[ticker] = asset_cfg.cache_csv
    
    # Load multi-asset prices
    prices = load_multi_asset_prices(
        tickers=cfg.portfolio.tickers,
        config=cfg.data,
        force_download=False,
        per_asset_cache_csv=per_asset_cache_csv if per_asset_cache_csv else None,
    )
    
    # Ensure safe haven ticker is in prices
    if cfg.portfolio.safe_haven_ticker not in prices.columns:
        raise ValueError(
            f"Safe haven ticker '{cfg.portfolio.safe_haven_ticker}' not found in prices. "
            f"Available columns: {list(prices.columns)}. "
            f"Make sure it's included in portfolio.tickers."
        )
    
    # Split into train/validation if needed
    if cfg.backtest.train_ratio < 1.0:
        split_idx = int(len(prices) * cfg.backtest.train_ratio)
        train_prices = prices.iloc[:split_idx].copy()
        val_prices = prices.iloc[split_idx:].copy()
    else:
        train_prices = prices.copy()
        val_prices = pd.DataFrame()
    
    # Generate signals for each asset using per-asset strategies or default
    train_entries_dict = {}
    train_exits_dict = {}
    val_entries_dict = {}
    val_exits_dict = {}

    # Instantiate default strategy once if needed
    default_strategy_cls = StrategyFactory[cfg.strategy.name]
    default_strategy = default_strategy_cls(**cfg.strategy.params)
    
    # Safe haven ticker - we don't generate signals for it (only used for allocation)
    safe_haven = cfg.portfolio.safe_haven_ticker
    
    for asset in cfg.portfolio.tickers:
        # Skip safe haven ticker - no signals needed (it's only for allocation)
        if asset == safe_haven:
            continue
            
        # Determine strategy for this asset
        if asset in cfg.portfolio.assets:
            asset_cfg = cfg.portfolio.assets[asset]
            strat_cls = StrategyFactory[asset_cfg.strategy]
            strategy = strat_cls(**asset_cfg.params)
        else:
            strategy = default_strategy
        
        # Generate signals for training set
        if asset in train_prices.columns:
            asset_close = train_prices[asset]
            entries, exits = strategy.generate_signals(asset_close)
            train_entries_dict[asset] = entries
            train_exits_dict[asset] = exits
            
        # Generate signals for validation set
        if len(val_prices) > 0 and asset in val_prices.columns:
            asset_close = val_prices[asset]
            entries, exits = strategy.generate_signals(asset_close)
            val_entries_dict[asset] = entries
            val_exits_dict[asset] = exits
    
    # Convert to DataFrames
    train_entries = pd.DataFrame(train_entries_dict, index=train_prices.index)
    train_exits = pd.DataFrame(train_exits_dict, index=train_prices.index)
    
    # Create RSI filter portfolio strategy
    rsi_strategy = RSIFilterPortfolioStrategy(
        rsi_period=cfg.portfolio.rsi_period,
        rsi_threshold=cfg.portfolio.rsi_threshold,
        top_k=cfg.portfolio.top_k,
        safe_haven_ticker=cfg.portfolio.safe_haven_ticker,
    )
    
    # Generate weight allocation matrix
    train_weights = rsi_strategy.generate_weights(
        prices=train_prices,
        ensemble_entries=train_entries,
        ensemble_exits=train_exits,
    )
    
    # Build portfolio
    portfolio_builder = PortfolioBuilder(cfg.backtest)
    train_portfolio = portfolio_builder.build_from_weights(
        prices=train_prices,
        weights=train_weights,
    )
    
    # Compute metrics
    portfolio_value = train_portfolio.value()
    train_metrics = compute_metrics(train_portfolio, portfolio_value, cfg.backtest.freq)
    
    outputs = {
        "train": train_metrics,
        "train_window": (train_prices.index[0], train_prices.index[-1]),
    }
    
    if return_portfolios:
        outputs["train_portfolio"] = train_portfolio
        outputs["train_weights"] = train_weights
        outputs["train_prices"] = train_prices
        outputs["train_entries"] = train_entries
        outputs["train_exits"] = train_exits
    
    # Validation set if available
    if len(val_prices) > 0:
        val_entries = pd.DataFrame(val_entries_dict, index=val_prices.index)
        val_exits = pd.DataFrame(val_exits_dict, index=val_prices.index)
        
        # Generate weights for validation
        val_weights = rsi_strategy.generate_weights(
            prices=val_prices,
            ensemble_entries=val_entries,
            ensemble_exits=val_exits,
        )
        
        # Build validation portfolio
        val_portfolio = portfolio_builder.build_from_weights(
            prices=val_prices,
            weights=val_weights,
        )
        
        # Compute validation metrics
        val_portfolio_value = val_portfolio.value()
        outputs["validation"] = compute_metrics(
            val_portfolio, val_portfolio_value, cfg.backtest.freq
        )
        outputs["validation_window"] = (val_prices.index[0], val_prices.index[-1])
        
        # Benchmark: buy and hold on equal-weighted portfolio
        equal_weights = pd.DataFrame(
            1.0 / len(cfg.portfolio.tickers),
            index=val_prices.index,
            columns=val_prices.columns,
        )
        benchmark_portfolio = portfolio_builder.build_from_weights(
            prices=val_prices,
            weights=equal_weights,
        )
        benchmark_value = benchmark_portfolio.value()
        outputs["benchmark"] = compute_metrics(
            benchmark_portfolio, benchmark_value, cfg.backtest.freq
        )
        
        if return_portfolios:
            outputs["val_portfolio"] = val_portfolio
            outputs["val_weights"] = val_weights
            outputs["val_prices"] = val_prices
            outputs["val_entries"] = val_entries
            outputs["val_exits"] = val_exits
    
    return outputs


def run_portfolio_grid_search(cfg: WorkflowConfig, use_vectorized: bool = True) -> List[Dict]:
    """Run grid search for portfolio parameters (RSI, Top-K, etc.).
    
    Args:
        cfg: Workflow configuration with portfolio grid defined
        use_vectorized: Use vectorized grid search for faster execution (default: True)
        
    Returns:
        List of metric dictionaries for each parameter combination
    """
    if not cfg.portfolio or not cfg.portfolio.grid:
        raise ValueError("No portfolio grid defined in configuration.")
    
    # Build per-asset cache_csv overrides from portfolio.assets if any are specified
    per_asset_cache_csv = {}
    if cfg.portfolio.assets:
        for ticker, asset_cfg in cfg.portfolio.assets.items():
            if asset_cfg.cache_csv:
                per_asset_cache_csv[ticker] = asset_cfg.cache_csv
    
    # Load multi-asset prices once
    prices = load_multi_asset_prices(
        tickers=cfg.portfolio.tickers,
        config=cfg.data,
        force_download=False,
        per_asset_cache_csv=per_asset_cache_csv if per_asset_cache_csv else None,
    )
    
    # Split into train set (we optimize on training data only)
    if cfg.backtest.train_ratio < 1.0:
        split_idx = int(len(prices) * cfg.backtest.train_ratio)
        train_prices = prices.iloc[:split_idx].copy()
    else:
        train_prices = prices.copy()
    
    # Generate ensemble signals for each asset once (they don't change with portfolio params)
    safe_haven = cfg.portfolio.safe_haven_ticker
    default_strategy_cls = StrategyFactory[cfg.strategy.name]
    default_strategy = default_strategy_cls(**cfg.strategy.params)
    
    train_entries_dict = {}
    train_exits_dict = {}
    
    for asset in cfg.portfolio.tickers:
        if asset == safe_haven:
            continue
            
        # Determine strategy for this asset
        if asset in cfg.portfolio.assets:
            asset_cfg = cfg.portfolio.assets[asset]
            strat_cls = StrategyFactory[asset_cfg.strategy]
            strategy = strat_cls(**asset_cfg.params)
        else:
            strategy = default_strategy
        
        if asset in train_prices.columns:
            asset_close = train_prices[asset]
            entries, exits = strategy.generate_signals(asset_close)
            train_entries_dict[asset] = entries
            train_exits_dict[asset] = exits
    
    train_entries = pd.DataFrame(train_entries_dict, index=train_prices.index)
    train_exits = pd.DataFrame(train_exits_dict, index=train_prices.index)
    
    if use_vectorized:
        print("Using VectorizedPortfolioGridSearch...")
        search = VectorizedPortfolioGridSearch(config=cfg.backtest, batch_size=500)
        results = search.run(
            prices=train_prices,
            ensemble_entries=train_entries,
            ensemble_exits=train_exits,
            grid=cfg.portfolio.grid,
            safe_haven_ticker=safe_haven,
        )
        return results
    
    # Fallback: sequential grid search (original implementation)
    keys = list(cfg.portfolio.grid.keys())
    combos = list(itertools.product(*[cfg.portfolio.grid[k] for k in keys]))
    
    results = []
    total = len(combos)
    
    print(f"Running portfolio grid search: {total} combinations (sequential)")
    print("-" * 50)
    
    for i, combo in enumerate(combos, 1):
        # Create a deep copy of config to avoid modifying the original
        run_cfg = copy.deepcopy(cfg)
        param_dict = dict(zip(keys, combo))
        
        # Update portfolio parameters in the copy
        for k, v in param_dict.items():
            if hasattr(run_cfg.portfolio, k):
                setattr(run_cfg.portfolio, k, v)
            else:
                print(f"Warning: Unknown portfolio parameter '{k}' in grid. Ignoring.")
        
        try:
            # Run backtest (we focus on training metrics for optimization)
            metrics = run_portfolio_backtest(run_cfg, return_portfolios=False)
            result = metrics["train"]
            # Add parameters to result for analysis
            result.update(param_dict)
            results.append(result)
        except Exception as e:
            print(f"Failed combo {param_dict}: {e}")
            
        # Progress update
        if i % 10 == 0 or i == total:
            print(f"Progress: {i}/{total} ({(i/total)*100:.1f}%)")
            
    return results

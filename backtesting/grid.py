"""Optimized brute-force grid search with multiprocessing, NumPy pre-filtering, and generators."""

from __future__ import annotations

import itertools
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestEngine
from .config import BacktestConfig
from .metrics import compute_metrics
from .strategies import StrategyFactory


def _validate_params_static(params: Dict[str, int]) -> bool:
    """Static validation function for parameter combinations.
    
    Can be called from both main process and worker processes.
    
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


def _is_valid_combo(combo: Tuple, keys: List[str]) -> bool:
    """Check if a parameter combination is valid using static validation."""
    params_dict = dict(zip(keys, combo))
    return _validate_params_static(params_dict)


def _run_single_backtest_worker(args: Tuple) -> Optional[Dict]:
    """Worker function for multiprocessing - runs a single backtest.
    
    Args:
        args: Tuple containing:
            - combo: Parameter combination tuple
            - keys: Parameter names list
            - base_params: Base parameters dict
            - close_values: Close price values (numpy array)
            - close_index: Close price index (DatetimeIndex values)
            - engine_config: BacktestConfig dict
            - strategy_cls_name: Strategy class name string
            - freq: Frequency string
    
    Returns:
        Metrics dict if valid and successful, None if invalid/failed
    """
    (combo, keys, base_params, close_values, close_index, 
     engine_config, strategy_cls_name, freq) = args
    
    # Reconstruct objects (needed for multiprocessing pickling)
    close = pd.Series(close_values, index=pd.DatetimeIndex(close_index))
    engine = BacktestEngine(BacktestConfig(**engine_config))
    strategy_cls = StrategyFactory[strategy_cls_name]
    
    # Build params
    params = dict(base_params)
    params.update(dict(zip(keys, combo)))
    
    # Validate
    if not _validate_params_static(params):
        return None
    
    # Run backtest
    try:
        strat = strategy_cls(**params)
        entries, exits = strat.generate_signals(close)
        portfolio = engine.run(close, (entries, exits))
        metrics = compute_metrics(portfolio, close, freq)
        metrics.update(params)
        return metrics
    except Exception:
        # Skip failed backtests
        return None


class GridSearch:
    def __init__(self, engine: BacktestEngine, strategy_cls, n_jobs: Optional[int] = None):
        self.engine = engine
        self.strategy_cls = strategy_cls
        self.results: List[Dict] = []
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)  # Leave 1 core free
    
    def run(
        self, 
        close: pd.Series, 
        grid: Dict[str, List[int]], 
        base_params: Dict[str, int],
        use_multiprocessing: bool = True
    ):
        """Run optimized grid search with optional multiprocessing and pre-filtering.
        
        Args:
            close: Price series
            grid: Parameter grid dictionary
            base_params: Base parameters
            use_multiprocessing: Enable multiprocessing (default: True)
        """
        keys = list(grid.keys())
        
        # Calculate total combinations for progress tracking
        total_possible = int(np.prod([len(grid[k]) for k in keys]))
        
        # Use generator to avoid materializing all combinations in memory
        all_combos = itertools.product(*[grid[k] for k in keys])
        
        # Pre-filter invalid combinations using NumPy
        print("Pre-filtering combinations...")
        valid_combos = [combo for combo in all_combos if _is_valid_combo(combo, keys)]
        total_valid = len(valid_combos)
        print(f"Found {total_valid:,} valid combinations (from {total_possible:,} total)")
        
        # Route to multiprocessing or sequential based on flag and size
        if use_multiprocessing and total_valid > 100:
            self._run_multiprocessing(valid_combos, keys, base_params, close, total_valid)
        else:
            self._run_sequential(valid_combos, keys, base_params, close, total_valid)
        
        return self.results
    
    def _run_multiprocessing(
        self, 
        valid_combos: List[Tuple], 
        keys: List[str], 
        base_params: Dict[str, int],
        close: pd.Series, 
        total_valid: int
    ):
        """Run grid search using multiprocessing."""
        print(f"Using {self.n_jobs} processes for parallel execution...")
        
        # Prepare picklable arguments
        close_values = close.values
        close_index = close.index.values
        engine_config = {
            'init_cash': self.engine.config.init_cash,
            'fees': self.engine.config.fees,
            'slippage': self.engine.config.slippage,
            'freq': self.engine.config.freq,
        }
        # Find strategy key in StrategyFactory (reverse lookup)
        strategy_cls_name = next(
            (key for key, cls in StrategyFactory.items() if cls == self.strategy_cls),
            self.strategy_cls.__name__
        )
        freq = self.engine.config.freq
        
        # Create args list for workers
        args_list = [
            (
                combo, keys, base_params,
                close_values, close_index,
                engine_config, strategy_cls_name, freq
            )
            for combo in valid_combos
        ]
        
        # Process in chunks for progress tracking
        chunk_size = max(100, len(args_list) // (self.n_jobs * 10))
        processed = 0
        
        with mp.Pool(processes=self.n_jobs) as pool:
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                results_chunk = pool.map(_run_single_backtest_worker, chunk)
                
                # Filter out None results (invalid/failed)
                valid_results = [r for r in results_chunk if r is not None]
                self.results.extend(valid_results)
                
                processed += len(chunk)
                progress_pct = (processed / total_valid) * 100
                print(f"Progress: {processed}/{total_valid} ({progress_pct:.1f}%) - {len(self.results)} valid results")
    
    def _run_sequential(
        self, 
        valid_combos: List[Tuple], 
        keys: List[str], 
        base_params: Dict[str, int],
        close: pd.Series, 
        total_valid: int
    ):
        """Run grid search sequentially (fallback)."""
        for i, combo in enumerate(valid_combos, 1):
            params = dict(base_params)
            params.update(dict(zip(keys, combo)))
            
            if not self._validate_params(params):
                continue
            
            strat = self.strategy_cls(**params)
            entries, exits = strat.generate_signals(close)
            portfolio = self.engine.run(close, (entries, exits))
            metrics = compute_metrics(portfolio, close, self.engine.config.freq)
            metrics.update(params)
            self.results.append(metrics)
            
            # Progress tracking
            progress_interval = max(1, total_valid // 10)
            if i % progress_interval == 0 or i == total_valid:
                progress_pct = (i / total_valid) * 100
                print(f"Progress: {i}/{total_valid} ({progress_pct:.1f}%) - {len(self.results)} valid results")
    
    def _validate_params(self, params: Dict[str, int]) -> bool:
        """Validate parameter combinations based on strategy type.
        
        Returns True if parameters are valid, False otherwise.
        """
        return _validate_params_static(params)
    
    def best(self, metric: str):
        if not self.results:
            raise ValueError("Run grid search first.")
        df = pd.DataFrame(self.results)
        return df.sort_values(metric, ascending=False).iloc[0]

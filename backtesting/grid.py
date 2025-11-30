"""Optimized brute-force grid search with multiprocessing, NumPy pre-filtering, and generators."""

from __future__ import annotations

import gc
import itertools
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from .backtest import BacktestEngine
from .config import BacktestConfig
from .metrics import compute_batch_metrics, compute_metrics
from .strategies import StrategyFactory


def _validate_params_static(params: Dict[str, int], strategy_name: str = "") -> bool:
    """Static validation function for parameter combinations.
    
    Can be called from both main process and worker processes.
    
    Args:
        params: Parameter dictionary
        strategy_name: Strategy name (used for unconstrained strategies)
    
    Returns True if parameters are valid, False otherwise.
    """
    # Unconstrained strategies skip EMA ordering validation
    is_unconstrained = "unconstrained" in strategy_name.lower()
    
    # Triple EMA validation: ema_fast < ema_mid < ema_slow (skip for unconstrained)
    if not is_unconstrained:
        if "ema_fast" in params and "ema_mid" in params and "ema_slow" in params:
            if not (params["ema_fast"] < params["ema_mid"] < params["ema_slow"]):
                return False
    
    # MACD validation: fastperiod < slowperiod (always required for MACD to work)
    if "fastperiod" in params and "slowperiod" in params:
        if not (params["fastperiod"] < params["slowperiod"]):
            return False
    
    return True


def _is_valid_combo(combo: Tuple, keys: List[str], strategy_name: str = "") -> bool:
    """Check if a parameter combination is valid using static validation."""
    params_dict = dict(zip(keys, combo))
    return _validate_params_static(params_dict, strategy_name)


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
        
        # Get strategy name from StrategyFactory (reverse lookup)
        self.strategy_name = next(
            (key for key, cls in StrategyFactory.items() if cls == strategy_cls),
            strategy_cls.__name__
        )
    
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
        
        # Pre-filter invalid combinations (unconstrained strategies skip EMA ordering)
        print("Pre-filtering combinations...")
        valid_combos = [combo for combo in all_combos if _is_valid_combo(combo, keys, self.strategy_name)]
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


def _run_vectorized_batch_worker(args: Tuple) -> Optional[pd.DataFrame]:
    """Worker function for multiprocessing - runs a vectorized batch backtest.
    
    Args:
        args: Tuple containing:
            - batch_params: List of parameter tuples for this batch
            - keys: Parameter names list
            - close_values: Close price values (numpy array)
            - close_index: Close price index (DatetimeIndex values)
            - engine_config: BacktestConfig dict
            - strategy_name: Strategy name string
            - freq: Frequency string
            - min_trades_per_year: Minimum trades per year filter
    
    Returns:
        DataFrame with metrics for valid parameter combinations, or None if failed
    """
    (batch_params, keys, close_values, close_index, 
     engine_config, strategy_name, freq, min_trades_per_year) = args
    
    try:
        # Reconstruct close Series
        close = pd.Series(close_values, index=pd.DatetimeIndex(close_index))
        batch_size = len(batch_params)
        
        # Unzip parameters
        param_lists = list(zip(*batch_params))
        param_dict = dict(zip(keys, param_lists))
        
        # Standardize column names
        common_cols = pd.Index(range(batch_size), name='combo_id')
        
        # Compute indicators based on strategy type
        # Note: unconstrained strategies use the same signal logic, just without param ordering
        if strategy_name in ("triple_ema", "triple_ema_unconstrained"):
            entries, exits = _compute_ema_signals_batch(
                close, 
                list(param_dict['ema_fast']),
                list(param_dict['ema_mid']),
                list(param_dict['ema_slow']),
                common_cols
            )
        elif strategy_name == "triple_macd":
            entries, exits = _compute_macd_signals_batch(
                close,
                list(param_dict['fastperiod']),
                list(param_dict['slowperiod']),
                list(param_dict['signalperiod']),
                common_cols
            )
        elif strategy_name in ("ensemble", "ensemble_unconstrained"):
            entries, exits = _compute_ensemble_signals_batch(
                close,
                list(param_dict['ema_fast']),
                list(param_dict['ema_mid']),
                list(param_dict['ema_slow']),
                list(param_dict['fastperiod']),
                list(param_dict['slowperiod']),
                list(param_dict['signalperiod']),
                common_cols
            )
        else:
            return None
        
        # Run batch backtest
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=engine_config['init_cash'],
            fees=engine_config['fees'],
            slippage=engine_config['slippage'],
            freq=freq
        )
        
        # Build parameter DataFrame
        param_df = pd.DataFrame({k: list(v) for k, v in param_dict.items()})
        
        # Compute batch metrics
        result_df = compute_batch_metrics(pf, close, freq, param_df)
        
        # Filter by minimum trades per year
        if min_trades_per_year > 0:
            result_df = result_df[result_df['trades_per_year'] >= min_trades_per_year]
        
        return result_df
        
    except Exception as e:
        print(f"Batch worker error: {e}")
        return None


def _compute_ema_signals_batch(
    close: pd.Series,
    ema_fast_list: List[int],
    ema_mid_list: List[int],
    ema_slow_list: List[int],
    common_cols: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Triple EMA signals for a batch of parameters."""
    # Vectorized EMA computation
    ema1 = vbt.MA.run(close, ema_fast_list, ewm=True)
    ema2 = vbt.MA.run(close, ema_mid_list, ewm=True)
    ema3 = vbt.MA.run(close, ema_slow_list, ewm=True)
    
    # Standardize column names
    df_ema1 = ema1.ma.copy()
    df_ema1.columns = common_cols
    df_ema2 = ema2.ma.copy()
    df_ema2.columns = common_cols
    df_ema3 = ema3.ma.copy()
    df_ema3.columns = common_cols
    
    # Crossover signals (OR logic)
    c1 = df_ema1.vbt.crossed_above(df_ema2)
    c2 = df_ema1.vbt.crossed_above(df_ema3)
    c3 = df_ema2.vbt.crossed_above(df_ema3)
    entries_raw = c1 | c2 | c3
    
    d1 = df_ema1.vbt.crossed_below(df_ema2)
    d2 = df_ema1.vbt.crossed_below(df_ema3)
    d3 = df_ema2.vbt.crossed_below(df_ema3)
    exits_raw = d1 | d2 | d3
    
    # Shift to avoid lookahead bias
    entries = entries_raw.shift(1).fillna(False).astype(bool)
    exits = exits_raw.shift(1).fillna(False).astype(bool)
    
    return entries, exits


def _compute_macd_signals_batch(
    close: pd.Series,
    fast_list: List[int],
    slow_list: List[int],
    signal_list: List[int],
    common_cols: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Triple MACD signals for a batch of parameters."""
    # Vectorized MACD computation
    macd = vbt.MACD.run(
        close,
        fast_window=fast_list,
        slow_window=slow_list,
        signal_window=signal_list
    )
    
    # Standardize column names
    macd_line = macd.macd.copy()
    macd_line.columns = common_cols
    signal_line = macd.signal.copy()
    signal_line.columns = common_cols
    
    # Crossover signals
    entries_raw = macd_line.vbt.crossed_above(signal_line)
    exits_raw = macd_line.vbt.crossed_below(signal_line)
    
    # Shift to avoid lookahead bias
    entries = entries_raw.shift(1).fillna(False).astype(bool)
    exits = exits_raw.shift(1).fillna(False).astype(bool)
    
    return entries, exits


def _compute_ensemble_signals_batch(
    close: pd.Series,
    ema_fast_list: List[int],
    ema_mid_list: List[int],
    ema_slow_list: List[int],
    fast_list: List[int],
    slow_list: List[int],
    signal_list: List[int],
    common_cols: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Ensemble (EMA + MACD with OR logic) signals for a batch."""
    # Get EMA signals
    ema_entries, ema_exits = _compute_ema_signals_batch(
        close, ema_fast_list, ema_mid_list, ema_slow_list, common_cols
    )
    
    # Get MACD signals
    macd_entries, macd_exits = _compute_macd_signals_batch(
        close, fast_list, slow_list, signal_list, common_cols
    )
    
    # Combine with OR logic
    entries = ema_entries | macd_entries
    exits = ema_exits | macd_exits
    
    return entries, exits


class VectorizedGridSearch:
    """Vectorized grid search using vectorbt's batch processing capabilities.
    
    This class uses vectorbt's parameter broadcasting to compute thousands of
    indicator combinations simultaneously, then runs batch backtests.
    
    Supports: triple_ema, triple_ema_unconstrained, triple_macd, ensemble, ensemble_unconstrained.
    
    Args:
        engine: BacktestEngine instance with configuration
        strategy_name: Strategy name (see SUPPORTED_STRATEGIES)
        batch_size: Number of parameter combinations per batch (default: 5000)
        n_jobs: Number of CPU cores for multiprocessing (default: all - 1)
        min_trades_per_year: Filter out strategies with fewer trades (default: 0.5)
    """
    
    SUPPORTED_STRATEGIES = [
        "triple_ema", 
        "triple_ema_unconstrained",
        "triple_macd", 
        "ensemble",
        "ensemble_unconstrained",
    ]
    
    def __init__(
        self, 
        engine: BacktestEngine, 
        strategy_name: str,
        batch_size: int = 5000,
        n_jobs: Optional[int] = None,
        min_trades_per_year: float = 0.5
    ):
        if strategy_name not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Strategy '{strategy_name}' not supported for vectorization. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )
        
        self.engine = engine
        self.strategy_name = strategy_name
        self.batch_size = batch_size
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)
        self.min_trades_per_year = min_trades_per_year
        self.results: List[Dict] = []
    
    def run(
        self,
        close: pd.Series,
        grid: Dict[str, List[int]],
        base_params: Dict[str, int],
        use_multiprocessing: bool = True
    ) -> List[Dict]:
        """Run vectorized grid search.
        
        Args:
            close: Price series
            grid: Parameter grid dictionary
            base_params: Base parameters (merged with grid params)
            use_multiprocessing: Use multiprocessing for batch distribution
            
        Returns:
            List of metric dictionaries for each valid parameter combination
        """
        keys = list(grid.keys())
        
        # Calculate total combinations
        total_possible = int(np.prod([len(grid[k]) for k in keys]))
        
        # Generate all combinations
        all_combos = list(itertools.product(*[grid[k] for k in keys]))
        
        # Pre-filter invalid combinations (unconstrained strategies skip EMA ordering)
        print("Pre-filtering combinations...")
        valid_combos = [combo for combo in all_combos if _is_valid_combo(combo, keys, self.strategy_name)]
        total_valid = len(valid_combos)
        print(f"Found {total_valid:,} valid combinations (from {total_possible:,} total)")
        
        if total_valid == 0:
            print("No valid combinations to test.")
            return []
        
        # Split into batches
        num_batches = (total_valid + self.batch_size - 1) // self.batch_size
        batches = [
            valid_combos[i:i + self.batch_size] 
            for i in range(0, total_valid, self.batch_size)
        ]
        
        print(f"Processing {num_batches} batches of up to {self.batch_size} combinations...")
        print(f"Strategy: {self.strategy_name} | Vectorization: ENABLED")
        
        # Prepare engine config for serialization
        engine_config = {
            'init_cash': self.engine.config.init_cash,
            'fees': self.engine.config.fees,
            'slippage': self.engine.config.slippage,
            'freq': self.engine.config.freq,
        }
        
        if use_multiprocessing and num_batches > 1 and self.n_jobs > 1:
            self._run_multiprocessing(
                batches, keys, close, engine_config, num_batches
            )
        else:
            self._run_sequential(
                batches, keys, close, engine_config, num_batches
            )
        
        print(f"\nCompleted: {len(self.results):,} valid strategies found")
        return self.results
    
    def _run_sequential(
        self,
        batches: List[List[Tuple]],
        keys: List[str],
        close: pd.Series,
        engine_config: Dict,
        num_batches: int
    ):
        """Process batches sequentially with vectorization."""
        for batch_idx, batch_params in enumerate(batches):
            try:
                batch_size = len(batch_params)
                print(f"Batch {batch_idx + 1}/{num_batches} ({batch_size} combinations)...")
                
                # Unzip parameters
                param_lists = list(zip(*batch_params))
                param_dict = dict(zip(keys, param_lists))
                
                # Standardize column names
                common_cols = pd.Index(range(batch_size), name='combo_id')
                
                # Compute signals based on strategy
                # Note: unconstrained strategies use the same signal logic, just without param ordering
                if self.strategy_name in ("triple_ema", "triple_ema_unconstrained"):
                    entries, exits = _compute_ema_signals_batch(
                        close,
                        list(param_dict['ema_fast']),
                        list(param_dict['ema_mid']),
                        list(param_dict['ema_slow']),
                        common_cols
                    )
                elif self.strategy_name == "triple_macd":
                    entries, exits = _compute_macd_signals_batch(
                        close,
                        list(param_dict['fastperiod']),
                        list(param_dict['slowperiod']),
                        list(param_dict['signalperiod']),
                        common_cols
                    )
                elif self.strategy_name in ("ensemble", "ensemble_unconstrained"):
                    entries, exits = _compute_ensemble_signals_batch(
                        close,
                        list(param_dict['ema_fast']),
                        list(param_dict['ema_mid']),
                        list(param_dict['ema_slow']),
                        list(param_dict['fastperiod']),
                        list(param_dict['slowperiod']),
                        list(param_dict['signalperiod']),
                        common_cols
                    )
                
                # Run batch backtest
                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=entries,
                    exits=exits,
                    init_cash=engine_config['init_cash'],
                    fees=engine_config['fees'],
                    slippage=engine_config['slippage'],
                    freq=engine_config['freq']
                )
                
                # Build parameter DataFrame
                param_df = pd.DataFrame({k: list(v) for k, v in param_dict.items()})
                
                # Compute batch metrics
                result_df = compute_batch_metrics(pf, close, engine_config['freq'], param_df)
                
                # Filter by minimum trades per year
                if self.min_trades_per_year > 0:
                    result_df = result_df[result_df['trades_per_year'] >= self.min_trades_per_year]
                
                # Convert to dict records and extend results
                self.results.extend(result_df.to_dict('records'))
                
                print(f"  → {len(result_df)} valid strategies (total: {len(self.results)})")
                
                # Memory cleanup
                del entries, exits, pf, param_df, result_df
                gc.collect()
                
            except Exception as e:
                print(f"Batch {batch_idx + 1} failed: {e}")
                continue
    
    def _run_multiprocessing(
        self,
        batches: List[List[Tuple]],
        keys: List[str],
        close: pd.Series,
        engine_config: Dict,
        num_batches: int
    ):
        """Distribute batches across multiple CPU cores."""
        print(f"Using {self.n_jobs} processes for parallel batch execution...")
        
        # Prepare serializable arguments for each batch
        close_values = close.values
        close_index = close.index.values
        
        args_list = [
            (
                batch_params, keys, close_values, close_index,
                engine_config, self.strategy_name, engine_config['freq'],
                self.min_trades_per_year
            )
            for batch_params in batches
        ]
        
        completed = 0
        with mp.Pool(processes=self.n_jobs) as pool:
            for result_df in pool.imap_unordered(_run_vectorized_batch_worker, args_list):
                completed += 1
                if result_df is not None and len(result_df) > 0:
                    self.results.extend(result_df.to_dict('records'))
                print(f"Progress: {completed}/{num_batches} batches | {len(self.results)} valid strategies")
    
    def best(self, metric: str = "sharpe_ratio") -> pd.Series:
        """Get the best parameter combination by specified metric."""
        if not self.results:
            raise ValueError("Run grid search first.")
        df = pd.DataFrame(self.results)
        return df.sort_values(metric, ascending=False).iloc[0]

"""
3-Phase Asset Optimization Pipeline

This module implements a robust parameter selection pipeline that automates
the process of finding stable, generalizable trading strategy parameters.

Pipeline Overview:
    Phase 1 (TRAIN): Wide grid search → Keep top N% by Sharpe
    Phase 2 (VAL): Validation filter → Remove overfit parameters (transfer ratio)
    Phase 3 (TEST): Sensitivity analysis → Select most stable parameters

Key Concepts:
    - Transfer Ratio: val_sharpe / train_sharpe (detects overfitting)
    - Stability Score: How robust parameters are to small perturbations
    - Composite Score: test_sharpe * consistency * stability

Usage:
    from backtesting.optimizer import AssetOptimizer
    
    optimizer = AssetOptimizer(ticker="GOOG", strategy="ensemble_unconstrained")
    results = optimizer.run()
    
    # Save best to registry
    optimizer.save_to_registry()
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from .backtest import BacktestEngine
from .config import (
    ASSETS_DIR,
    REGISTRY_FILE,
    TEMPLATES_DIR,
    BacktestConfig,
    OptimizationConfig,
    RegistryEntry,
    WorkflowConfig,
    get_asset_cache_path,
    get_asset_results_dir,
    load_config,
    load_registry,
    load_template,
    save_registry,
    update_registry_entry,
    calculate_warmup_from_grid,
    extend_start_date,
    InsufficientWarmupDataError,
    InsufficientBacktestDataError,
)
from .data import DataFetcher, split_train_val_test, get_split_info
from .grid import (
    VectorizedGridSearch,
    _compute_ema_signals_batch,
    _compute_macd_signals_batch,
    _compute_ensemble_signals_batch,
)
from .metrics import compute_batch_metrics, compute_metrics
from .strategies import StrategyFactory


@dataclass
class OptimizationCandidate:
    """A candidate parameter set with metrics across all splits."""
    params: Dict[str, Any]
    train_sharpe: float
    val_sharpe: Optional[float] = None
    test_sharpe: Optional[float] = None
    transfer_ratio: Optional[float] = None
    stability_score: Optional[float] = None
    consistency: Optional[float] = None
    composite_score: Optional[float] = None
    neighbor_sharpes: List[float] = field(default_factory=list)
    
    # Additional metrics for analysis
    train_return: Optional[float] = None
    val_return: Optional[float] = None
    test_return: Optional[float] = None
    train_max_dd: Optional[float] = None
    val_max_dd: Optional[float] = None
    test_max_dd: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "params": self.params,
            "train_sharpe": self.train_sharpe,
            "val_sharpe": self.val_sharpe,
            "test_sharpe": self.test_sharpe,
            "transfer_ratio": self.transfer_ratio,
            "stability_score": self.stability_score,
            "consistency": self.consistency,
            "composite_score": self.composite_score,
            "neighbor_sharpes": self.neighbor_sharpes,
            "train_return": self.train_return,
            "val_return": self.val_return,
            "test_return": self.test_return,
            "train_max_dd": self.train_max_dd,
            "val_max_dd": self.val_max_dd,
            "test_max_dd": self.test_max_dd,
        }


@dataclass
class OptimizationResult:
    """Result of the full optimization pipeline."""
    ticker: str
    strategy: str
    candidates: List[OptimizationCandidate]
    split_info: Dict[str, Any]
    phase1_count: int  # Candidates after phase 1
    phase2_count: int  # Candidates after phase 2
    phase3_count: int  # Final candidates after phase 3
    best_candidate: Optional[OptimizationCandidate] = None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Optimization Results for {self.ticker}",
            "=" * 50,
            f"Strategy: {self.strategy}",
            f"Phase 1 (Grid Search): {self.phase1_count} candidates",
            f"Phase 2 (Validation): {self.phase2_count} candidates",
            f"Phase 3 (Sensitivity): {self.phase3_count} candidates",
            "",
        ]
        
        if self.best_candidate:
            bc = self.best_candidate
            lines.extend([
                "Best Candidate:",
                f"  Params: {bc.params}",
                f"  Train Sharpe: {bc.train_sharpe:.4f}" if bc.train_sharpe else "",
                f"  Val Sharpe: {bc.val_sharpe:.4f}" if bc.val_sharpe else "",
                f"  Test Sharpe: {bc.test_sharpe:.4f}" if bc.test_sharpe else "",
                f"  Transfer Ratio: {bc.transfer_ratio:.4f}" if bc.transfer_ratio else "",
                f"  Stability Score: {bc.stability_score:.4f}" if bc.stability_score else "",
                f"  Composite Score: {bc.composite_score:.4f}" if bc.composite_score else "",
            ])
        
        return "\n".join(filter(None, lines))


def _run_phase2_validation_worker(args: Tuple) -> Optional[Dict]:
    """Worker function for Phase 2 multiprocessing - validates a single candidate.
    
    Args:
        args: Tuple containing:
            - candidate_dict: Candidate as dictionary (params, train_sharpe, etc.)
            - val_values: Validation close price values (numpy array)
            - val_index: Validation close price index (DatetimeIndex values)
            - engine_config: BacktestConfig dict
            - strategy_name: Strategy name string
            - freq: Frequency string
            - transfer_threshold: Transfer ratio threshold
    
    Returns:
        Updated candidate dict if passes, None if fails
    """
    (candidate_dict, val_values, val_index, engine_config, 
     strategy_name, freq, transfer_threshold) = args
    
    try:
        # Reconstruct objects
        val_close = pd.Series(val_values, index=pd.DatetimeIndex(val_index))
        engine = BacktestEngine(BacktestConfig(**engine_config))
        strategy_cls = StrategyFactory[strategy_name]
        
        # Run backtest on validation
        strategy = strategy_cls(**candidate_dict['params'])
        entries, exits = strategy.generate_signals(val_close)
        portfolio = engine.run(val_close, (entries, exits))
        metrics = compute_metrics(portfolio, val_close, freq)
        
        val_sharpe = metrics.get('sharpe_ratio', 0)
        train_sharpe = candidate_dict.get('train_sharpe', 0)
        
        # Compute transfer ratio
        if train_sharpe > 0:
            transfer_ratio = val_sharpe / train_sharpe
        else:
            transfer_ratio = 0
        
        # Update candidate dict
        candidate_dict['val_sharpe'] = val_sharpe
        candidate_dict['val_return'] = metrics.get('total_return', 0)
        candidate_dict['val_max_dd'] = metrics.get('max_drawdown', 0)
        candidate_dict['transfer_ratio'] = transfer_ratio
        
        # Return if passes threshold, None otherwise
        if transfer_ratio >= transfer_threshold:
            return candidate_dict
        else:
            return None
            
    except Exception:
        return None


def _run_phase3_sensitivity_worker(args: Tuple) -> Optional[Dict]:
    """Worker function for Phase 3 multiprocessing - analyzes a single candidate.
    
    Args:
        args: Tuple containing:
            - candidate_dict: Candidate as dictionary
            - test_values: Test close price values (numpy array)
            - test_index: Test close price index (DatetimeIndex values)
            - engine_config: BacktestConfig dict
            - strategy_name: Strategy name string
            - freq: Frequency string
            - sensitivity_step: Step size for neighbor generation
    
    Returns:
        Updated candidate dict with test metrics and stability scores
    """
    (candidate_dict, test_values, test_index, engine_config,
     strategy_name, freq, sensitivity_step) = args
    
    try:
        # Reconstruct objects
        test_close = pd.Series(test_values, index=pd.DatetimeIndex(test_index))
        engine = BacktestEngine(BacktestConfig(**engine_config))
        strategy_cls = StrategyFactory[strategy_name]
        
        params = candidate_dict['params']
        
        # Run on TEST data
        strategy = strategy_cls(**params)
        entries, exits = strategy.generate_signals(test_close)
        portfolio = engine.run(test_close, (entries, exits))
        metrics = compute_metrics(portfolio, test_close, freq)
        
        test_sharpe = metrics.get('sharpe_ratio', 0)
        candidate_dict['test_sharpe'] = test_sharpe
        candidate_dict['test_return'] = metrics.get('total_return', 0)
        candidate_dict['test_max_dd'] = metrics.get('max_drawdown', 0)
        
        # Generate neighbors and test sensitivity
        neighbors = _generate_neighbors_static(params, sensitivity_step)
        
        neighbor_sharpes = []
        for neighbor_params in neighbors:
            try:
                n_strategy = strategy_cls(**neighbor_params)
                n_entries, n_exits = n_strategy.generate_signals(test_close)
                n_portfolio = engine.run(test_close, (n_entries, n_exits))
                n_metrics = compute_metrics(n_portfolio, test_close, freq)
                neighbor_sharpes.append(n_metrics.get('sharpe_ratio', 0))
            except Exception:
                continue
        
        candidate_dict['neighbor_sharpes'] = neighbor_sharpes
        
        # Compute stability score
        if neighbor_sharpes and test_sharpe and test_sharpe > 0:
            min_neighbor = min(neighbor_sharpes) if neighbor_sharpes else 0
            stability_score = min_neighbor / test_sharpe
        else:
            stability_score = 0
        
        candidate_dict['stability_score'] = stability_score
        
        # Compute consistency (min/max across all periods)
        train_sharpe = candidate_dict.get('train_sharpe', 0)
        val_sharpe = candidate_dict.get('val_sharpe', 0)
        sharpes = [s for s in [train_sharpe, val_sharpe, test_sharpe] if s]
        if sharpes and max(sharpes) > 0:
            consistency = min(sharpes) / max(sharpes)
        else:
            consistency = 0
        
        candidate_dict['consistency'] = consistency
        
        # Compute composite score
        composite_score = test_sharpe * max(consistency, 0.1) * max(stability_score, 0.1) if test_sharpe > 0 else 0
        candidate_dict['composite_score'] = composite_score
        
        return candidate_dict
        
    except Exception:
        return None


def _generate_neighbors_static(params: Dict[str, Any], step: int = 2) -> List[Dict[str, Any]]:
    """Static version of neighbor generation for multiprocessing."""
    neighbors = []
    
    for key in params:
        original = params[key]
        
        # Skip non-numeric params
        if not isinstance(original, (int, float)):
            continue
        
        # Generate perturbations
        for delta in [-step, -1, 1, step]:
            new_val = original + delta
            
            # Ensure positive values (periods must be > 0)
            if new_val <= 0:
                continue
            
            neighbor = dict(params)
            neighbor[key] = int(new_val) if isinstance(original, int) else new_val
            neighbors.append(neighbor)
    
    return neighbors


class AssetOptimizer:
    """
    3-Phase optimization pipeline for single asset parameter selection.
    
    This class implements a robust workflow for finding trading strategy
    parameters that generalize well and are stable to perturbations.
    
    Args:
        ticker: Asset ticker symbol (e.g., "GOOG", "BTC")
        strategy: Strategy name (e.g., "ensemble_unconstrained")
        template: Template config name (default: "wide_ensemble_grid.yml")
        config: Optional WorkflowConfig override
        opt_config: Optional OptimizationConfig override
        verbose: Print progress messages (default: True)
    
    Example:
        >>> optimizer = AssetOptimizer("GOOG", "ensemble_unconstrained")
        >>> result = optimizer.run()
        >>> print(result.summary())
        >>> optimizer.save_to_registry()
    """
    
    def __init__(
        self,
        ticker: str,
        strategy: str = "ensemble_unconstrained",
        template: str = "wide_ensemble_grid.yml",
        config: Optional[WorkflowConfig] = None,
        opt_config: Optional[OptimizationConfig] = None,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ):
        self.ticker = ticker.upper()
        self.strategy = strategy
        self.verbose = verbose
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)  # Leave 1 core free
        
        # Load or build configuration
        if config:
            self.config = config
        else:
            try:
                self.config = load_template(template)
            except FileNotFoundError:
                self.config = WorkflowConfig()
        
        # Override strategy name
        self.config.strategy.name = strategy
        
        # Set up optimization config
        if opt_config:
            self.opt_config = opt_config
        elif self.config.optimization:
            self.opt_config = self.config.optimization
        else:
            self.opt_config = OptimizationConfig()
        
        # Set up data config for this ticker
        self.config.data.ticker = ticker
        cache_path = get_asset_cache_path(ticker)
        if cache_path.exists():
            self.config.data.cache_csv = str(cache_path)
        
        # State
        self.close: Optional[pd.Series] = None  # Trimmed to original start (for backtest)
        self.close_full: Optional[pd.Series] = None  # Full data including warmup (for indicators)
        self.train: Optional[pd.Series] = None
        self.val: Optional[pd.Series] = None
        self.test: Optional[pd.Series] = None
        self.train_full: Optional[pd.Series] = None  # Train data + warmup for indicator calc
        self.val_full: Optional[pd.Series] = None
        self.test_full: Optional[pd.Series] = None
        self.split_info: Optional[Dict] = None
        self.result: Optional[OptimizationResult] = None
        self.warmup_bars: int = 0
    
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_data(
        self, 
        force_download: bool = False,
        allow_partial_warmup: bool = False,
        min_backtest_bars: int = 252,
    ) -> pd.Series:
        """Load price data for the ticker with warmup support.
        
        Args:
            force_download: Force re-download even if cache exists
            allow_partial_warmup: Allow proceeding with insufficient warmup data
            min_backtest_bars: Minimum bars required for backtesting
            
        Returns:
            Close price series (trimmed to original start)
        """
        self.log(f"Loading data for {self.ticker}...")
        
        # Auto-calculate warmup from grid
        if self.config.strategy.grid:
            self.warmup_bars = calculate_warmup_from_grid(
                self.config.strategy.grid, 
                self.strategy
            )
            if self.warmup_bars > 0:
                self.log(f"  Auto-calculated warmup: {self.warmup_bars} bars from grid")
        
        fetcher = DataFetcher(
            ticker=self.config.data.ticker,
            start=self.config.data.start,
            end=self.config.data.end,
            interval=self.config.data.interval,
            data_source=self.config.data.data_source,
            asset_type=self.config.data.asset_type,
            cache_csv=self.config.data.cache_csv,
            warmup_bars=self.warmup_bars,
        )
        
        fetcher.load(
            force_download=force_download,
            validate_warmup=(self.warmup_bars > 0),
            allow_partial_warmup=allow_partial_warmup,
            min_backtest_bars=min_backtest_bars,
        )
        
        # Store both full data (with warmup) and trimmed data
        self.close_full = fetcher.close()  # Includes warmup for indicator calculation
        self.close = fetcher.close_trimmed()  # Trimmed to original start for backtest
        
        self.log(f"  Loaded {len(self.close_full)} bars (including {self.warmup_bars} warmup)")
        self.log(f"  Backtest range: {self.close.index[0].date()} to {self.close.index[-1].date()}")
        self.log(f"  Backtest bars: {len(self.close)}")
        
        return self.close
    
    def split_data(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split data into train/val/test sets.
        
        Also creates corresponding full series (with warmup) for indicator calculation.
        
        Returns:
            Tuple of (train, val, test) series
        """
        if self.close is None:
            self.load_data()
        
        # Split trimmed data for backtest evaluation
        self.train, self.val, self.test = split_train_val_test(
            self.close,
            train_ratio=self.opt_config.train_ratio,
            val_ratio=self.opt_config.val_ratio,
            test_ratio=self.opt_config.test_ratio,
        )
        
        # Create full series for indicator calculation (includes warmup)
        if self.close_full is not None and self.warmup_bars > 0:
            # For train, we need warmup + train data
            train_end_idx = self.train.index[-1]
            self.train_full = self.close_full[self.close_full.index <= train_end_idx].copy()
            
            # For val, we need enough lookback from val start
            # Use the full data up to val end for proper indicator calculation
            val_end_idx = self.val.index[-1]
            self.val_full = self.close_full[self.close_full.index <= val_end_idx].copy()
            
            # For test, use all data up to test end
            test_end_idx = self.test.index[-1]
            self.test_full = self.close_full[self.close_full.index <= test_end_idx].copy()
        else:
            self.train_full = self.train
            self.val_full = self.val
            self.test_full = self.test
        
        self.split_info = get_split_info(
            self.close,
            train_ratio=self.opt_config.train_ratio,
            val_ratio=self.opt_config.val_ratio,
            test_ratio=self.opt_config.test_ratio,
        )
        
        self.log(f"\nData Split:")
        self.log(f"  TRAIN: {len(self.train)} bars ({self.split_info['train']['start'].date()} to {self.split_info['train']['end'].date()})")
        self.log(f"  VAL:   {len(self.val)} bars ({self.split_info['val']['start'].date()} to {self.split_info['val']['end'].date()})")
        self.log(f"  TEST:  {len(self.test)} bars ({self.split_info['test']['start'].date()} to {self.split_info['test']['end'].date()})")
        if self.warmup_bars > 0:
            self.log(f"  (Using {self.warmup_bars} warmup bars for indicator calculation)")
        
        return self.train, self.val, self.test
    
    def run_phase1_grid_search(self) -> List[OptimizationCandidate]:
        """
        Phase 1: Wide grid search on TRAIN data.
        
        Runs vectorized grid search and keeps top N% by Sharpe ratio.
        Uses warmup data for indicator calculation if available.
        
        Returns:
            List of top candidates from grid search
        """
        if self.train is None:
            self.split_data()
        
        self.log(f"\n{'='*60}")
        self.log("PHASE 1: Wide Grid Search on TRAIN")
        self.log(f"{'='*60}")
        
        # Set up engine
        engine = BacktestEngine(self.config.backtest)
        
        # Run vectorized grid search
        search = VectorizedGridSearch(
            engine=engine,
            strategy_name=self.strategy,
            batch_size=5000,
            min_trades_per_year=self.opt_config.min_trades_per_year,
            warmup_bars=self.warmup_bars,
        )
        
        if not self.config.strategy.grid:
            raise ValueError("No grid defined in config. Set strategy.grid in template.")
        
        # Use train_full for indicator calculation, train for backtest evaluation
        search.run(
            close=self.train_full if self.warmup_bars > 0 else self.train,
            grid=self.config.strategy.grid,
            base_params=self.config.strategy.params,
            backtest_close=self.train if self.warmup_bars > 0 else None,
        )
        
        if not search.results:
            self.log("  No valid results from grid search!")
            return []
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(search.results)
        self.log(f"  Grid search returned {len(df)} valid strategies")
        
        # Keep top N% by Sharpe
        top_n = max(1, int(len(df) * self.opt_config.top_percent_phase1))
        df_top = df.nlargest(top_n, 'sharpe_ratio')
        
        self.log(f"  Keeping top {self.opt_config.top_percent_phase1*100:.1f}% = {len(df_top)} candidates")
        
        # Convert to OptimizationCandidate objects
        param_cols = self._get_param_columns(df)
        candidates = []
        
        for _, row in df_top.iterrows():
            params = {col: int(row[col]) for col in param_cols}
            candidate = OptimizationCandidate(
                params=params,
                train_sharpe=float(row['sharpe_ratio']),
                train_return=float(row.get('total_return', 0)),
                train_max_dd=float(row.get('max_drawdown', 0)),
            )
            candidates.append(candidate)
        
        return candidates
    
    def run_phase2_validation(
        self, 
        candidates: List[OptimizationCandidate]
    ) -> List[OptimizationCandidate]:
        """
        Phase 2: Validation filter - remove overfit parameters.
        
        Runs each candidate on VALIDATION data and filters by transfer ratio.
        Uses multiprocessing for parallel execution when beneficial.
        
        Args:
            candidates: Candidates from Phase 1
            
        Returns:
            Filtered list of candidates that pass transfer ratio threshold
        """
        if self.val is None:
            self.split_data()
        
        self.log(f"\n{'='*60}")
        self.log("PHASE 2: Validation Filter")
        self.log(f"{'='*60}")
        self.log(f"  Testing {len(candidates)} candidates on VALIDATION data...")
        
        # Check if strategy supports vectorization
        supports_vectorization = self.strategy in VectorizedGridSearch.SUPPORTED_STRATEGIES
        
        # Use vectorization if supported and we have enough candidates
        if supports_vectorization and len(candidates) > 10:
            self.log(f"  Using vectorized batch processing...")
            passed = self._run_phase2_vectorized(candidates)
        # Use multiprocessing if we have enough candidates and cores
        elif len(candidates) > 50 and self.n_jobs > 1:
            self.log(f"  Using {self.n_jobs} processes for parallel execution...")
            passed = self._run_phase2_multiprocessing(candidates)
        else:
            passed = self._run_phase2_sequential(candidates)
        
        self.log(f"  {len(passed)} candidates passed (transfer ratio >= {self.opt_config.transfer_ratio_threshold})")
        
        return passed
    
    def _run_phase2_sequential(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 2 sequentially (fallback for small candidate sets)."""
        engine = BacktestEngine(self.config.backtest)
        strategy_cls = StrategyFactory[self.strategy]
        
        passed = []
        
        # Use val_full for indicator calculation if warmup is enabled
        indicator_series = self.val_full if self.warmup_bars > 0 else self.val
        
        for i, candidate in enumerate(candidates):
            # Run backtest on validation
            try:
                strategy = strategy_cls(**candidate.params)
                # Generate signals on full data (includes warmup)
                entries, exits = strategy.generate_signals(indicator_series)
                
                # Trim signals to match val series if using warmup
                if self.warmup_bars > 0:
                    val_start = self.val.index[0]
                    val_end = self.val.index[-1]
                    entries = entries.loc[val_start:val_end]
                    exits = exits.loc[val_start:val_end]
                
                portfolio = engine.run(self.val, (entries, exits))
                metrics = compute_metrics(portfolio, self.val, self.config.backtest.freq)
                
                val_sharpe = metrics.get('sharpe_ratio', 0)
                candidate.val_sharpe = val_sharpe
                candidate.val_return = metrics.get('total_return', 0)
                candidate.val_max_dd = metrics.get('max_drawdown', 0)
                
                # Compute transfer ratio
                if candidate.train_sharpe > 0:
                    candidate.transfer_ratio = val_sharpe / candidate.train_sharpe
                else:
                    candidate.transfer_ratio = 0
                
                # Filter by transfer ratio
                if candidate.transfer_ratio >= self.opt_config.transfer_ratio_threshold:
                    passed.append(candidate)
                    
            except Exception as e:
                if self.verbose:
                    self.log(f"    Candidate {i} failed: {e}")
                continue
            
            # Progress update
            if (i + 1) % 100 == 0:
                self.log(f"    Processed {i+1}/{len(candidates)} ({len(passed)} passed)")
        
        return passed
    
    def _run_phase2_vectorized(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 2 using vectorized batch processing."""
        # Extract parameters from candidates
        param_dicts = [candidate.params for candidate in candidates]
        
        # Get parameter keys (assuming all candidates have same keys)
        if not param_dicts:
            return []
        keys = list(param_dicts[0].keys())
        
        # Convert to parameter lists for batch processing
        param_lists = {key: [params[key] for params in param_dicts] for key in keys}
        
        # Create column index
        n_candidates = len(candidates)
        common_cols = pd.Index(range(n_candidates), name='candidate_id')
        
        # Use val_full for indicator calculation if warmup is enabled
        indicator_series = self.val_full if self.warmup_bars > 0 else self.val
        
        # Compute signals based on strategy type (on full data with warmup)
        if self.strategy in ("triple_ema", "triple_ema_unconstrained"):
            entries, exits = _compute_ema_signals_batch(
                indicator_series,
                param_lists['ema_fast'],
                param_lists['ema_mid'],
                param_lists['ema_slow'],
                common_cols
            )
        elif self.strategy == "macd":
            entries, exits = _compute_macd_signals_batch(
                indicator_series,
                param_lists['fastperiod'],
                param_lists['slowperiod'],
                param_lists['signalperiod'],
                common_cols
            )
        elif self.strategy in ("ensemble", "ensemble_unconstrained"):
            entries, exits = _compute_ensemble_signals_batch(
                indicator_series,
                param_lists['ema_fast'],
                param_lists['ema_mid'],
                param_lists['ema_slow'],
                param_lists['fastperiod'],
                param_lists['slowperiod'],
                param_lists['signalperiod'],
                common_cols
            )
        else:
            # Fallback to sequential if strategy not supported
            return self._run_phase2_sequential(candidates)
        
        # Trim signals to match val series if using warmup
        if self.warmup_bars > 0:
            val_start = self.val.index[0]
            val_end = self.val.index[-1]
            entries = entries.loc[val_start:val_end]
            exits = exits.loc[val_start:val_end]
        
        # Run batch backtest on val (trimmed)
        pf = vbt.Portfolio.from_signals(
            close=self.val,
            entries=entries,
            exits=exits,
            init_cash=self.config.backtest.init_cash,
            fees=self.config.backtest.fees,
            slippage=self.config.backtest.slippage,
            freq=self.config.backtest.freq
        )
        
        # Build parameter DataFrame
        param_df = pd.DataFrame(param_lists)
        
        # Compute batch metrics
        result_df = compute_batch_metrics(pf, self.val, self.config.backtest.freq, param_df)
        
        # Map results back to candidates and compute transfer ratios
        passed = []
        for i, candidate in enumerate(candidates):
            if i < len(result_df):
                row = result_df.iloc[i]
                val_sharpe = float(row.get('sharpe_ratio', 0))
                candidate.val_sharpe = val_sharpe
                candidate.val_return = float(row.get('total_return', 0))
                candidate.val_max_dd = float(row.get('max_drawdown', 0))
                
                # Compute transfer ratio
                if candidate.train_sharpe > 0:
                    candidate.transfer_ratio = val_sharpe / candidate.train_sharpe
                else:
                    candidate.transfer_ratio = 0
                
                # Filter by transfer ratio
                if candidate.transfer_ratio >= self.opt_config.transfer_ratio_threshold:
                    passed.append(candidate)
        
        return passed
    
    def _run_phase2_multiprocessing(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 2 using multiprocessing."""
        # Prepare serializable arguments
        val_values = self.val.values
        val_index = self.val.index.values
        engine_config = {
            'init_cash': self.config.backtest.init_cash,
            'fees': self.config.backtest.fees,
            'slippage': self.config.backtest.slippage,
            'freq': self.config.backtest.freq,
        }
        
        # Convert candidates to dictionaries for serialization
        candidate_dicts = [candidate.to_dict() for candidate in candidates]
        
        # Create args list for workers
        args_list = [
            (
                candidate_dict, val_values, val_index,
                engine_config, self.strategy, self.config.backtest.freq,
                self.opt_config.transfer_ratio_threshold
            )
            for candidate_dict in candidate_dicts
        ]
        
        passed = []
        processed = 0
        
        # Process in chunks for progress tracking
        chunk_size = max(50, len(args_list) // (self.n_jobs * 4))
        
        with mp.Pool(processes=self.n_jobs) as pool:
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                results_chunk = pool.map(_run_phase2_validation_worker, chunk)
                
                # Filter out None results and convert back to candidates
                for result_dict in results_chunk:
                    if result_dict is not None:
                        candidate = OptimizationCandidate(
                            params=result_dict['params'],
                            train_sharpe=result_dict.get('train_sharpe', 0),
                            val_sharpe=result_dict.get('val_sharpe'),
                            transfer_ratio=result_dict.get('transfer_ratio'),
                            train_return=result_dict.get('train_return'),
                            val_return=result_dict.get('val_return'),
                            val_max_dd=result_dict.get('val_max_dd'),
                        )
                        passed.append(candidate)
                
                processed += len(chunk)
                if processed % 100 == 0 or processed == len(args_list):
                    self.log(f"    Processed {processed}/{len(candidates)} ({len(passed)} passed)")
        
        return passed
    
    def run_phase3_sensitivity(
        self, 
        candidates: List[OptimizationCandidate],
        max_candidates: int = 100,
    ) -> List[OptimizationCandidate]:
        """
        Phase 3: Sensitivity analysis on TEST data.
        
        For each candidate, perturb parameters and check stability.
        Compute final composite scores.
        Uses multiprocessing for parallel execution when beneficial.
        
        Args:
            candidates: Candidates from Phase 2
            max_candidates: Maximum candidates to analyze (default 100)
            
        Returns:
            Final ranked list of candidates with composite scores
        """
        if self.test is None:
            self.split_data()
        
        self.log(f"\n{'='*60}")
        self.log("PHASE 3: Sensitivity Analysis on TEST")
        self.log(f"{'='*60}")
        
        # Limit candidates for performance
        candidates = candidates[:max_candidates]
        self.log(f"  Analyzing {len(candidates)} candidates...")
        
        # Check if strategy supports vectorization
        supports_vectorization = self.strategy in VectorizedGridSearch.SUPPORTED_STRATEGIES
        
        # Use vectorization if supported and we have enough candidates
        if supports_vectorization and len(candidates) > 5:
            self.log(f"  Using vectorized batch processing...")
            analyzed = self._run_phase3_vectorized(candidates)
        # Use multiprocessing if we have enough candidates and cores
        elif len(candidates) > 10 and self.n_jobs > 1:
            self.log(f"  Using {self.n_jobs} processes for parallel execution...")
            analyzed = self._run_phase3_multiprocessing(candidates)
        else:
            analyzed = self._run_phase3_sequential(candidates)
        
        # Sort by composite score
        analyzed.sort(key=lambda c: c.composite_score or 0, reverse=True)
        
        # Keep top N
        final = analyzed[:self.opt_config.final_candidates]
        
        self.log(f"  Top {len(final)} candidates selected by composite score")
        
        return final
    
    def _run_phase3_vectorized(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 3 using vectorized batch processing (candidates + neighbors)."""
        # Collect all parameter sets: candidates + their neighbors
        all_param_sets = []
        candidate_indices = []  # Maps each param set to its candidate index
        neighbor_indices = []   # Maps each param set to neighbor index (-1 for original candidate)
        
        for i, candidate in enumerate(candidates):
            # Add original candidate
            all_param_sets.append(candidate.params)
            candidate_indices.append(i)
            neighbor_indices.append(-1)  # -1 means original candidate
            
            # Add neighbors
            neighbors = self._generate_neighbors(
                candidate.params,
                step=self.opt_config.sensitivity_step
            )
            for neighbor in neighbors:
                all_param_sets.append(neighbor)
                candidate_indices.append(i)
                neighbor_indices.append(len(neighbors) - 1)  # Track which neighbor
        
        if not all_param_sets:
            return candidates
        
        # Get parameter keys
        keys = list(all_param_sets[0].keys())
        
        # Convert to parameter lists for batch processing
        param_lists = {key: [params[key] for params in all_param_sets] for key in keys}
        
        # Create column index
        n_total = len(all_param_sets)
        common_cols = pd.Index(range(n_total), name='combo_id')
        
        # Use test_full for indicator calculation if warmup is enabled
        indicator_series = self.test_full if self.warmup_bars > 0 else self.test
        
        # Compute signals based on strategy type (on full data with warmup)
        if self.strategy in ("triple_ema", "triple_ema_unconstrained"):
            entries, exits = _compute_ema_signals_batch(
                indicator_series,
                param_lists['ema_fast'],
                param_lists['ema_mid'],
                param_lists['ema_slow'],
                common_cols
            )
        elif self.strategy == "macd":
            entries, exits = _compute_macd_signals_batch(
                indicator_series,
                param_lists['fastperiod'],
                param_lists['slowperiod'],
                param_lists['signalperiod'],
                common_cols
            )
        elif self.strategy in ("ensemble", "ensemble_unconstrained"):
            entries, exits = _compute_ensemble_signals_batch(
                indicator_series,
                param_lists['ema_fast'],
                param_lists['ema_mid'],
                param_lists['ema_slow'],
                param_lists['fastperiod'],
                param_lists['slowperiod'],
                param_lists['signalperiod'],
                common_cols
            )
        else:
            # Fallback to sequential if strategy not supported
            return self._run_phase3_sequential(candidates)
        
        # Trim signals to match test series if using warmup
        if self.warmup_bars > 0:
            test_start = self.test.index[0]
            test_end = self.test.index[-1]
            entries = entries.loc[test_start:test_end]
            exits = exits.loc[test_start:test_end]
        
        # Run batch backtest on test (trimmed)
        pf = vbt.Portfolio.from_signals(
            close=self.test,
            entries=entries,
            exits=exits,
            init_cash=self.config.backtest.init_cash,
            fees=self.config.backtest.fees,
            slippage=self.config.backtest.slippage,
            freq=self.config.backtest.freq
        )
        
        # Build parameter DataFrame
        param_df = pd.DataFrame(param_lists)
        
        # Compute batch metrics
        result_df = compute_batch_metrics(pf, self.test, self.config.backtest.freq, param_df)
        
        # Map results back to candidates
        for i, candidate in enumerate(candidates):
            # Find original candidate result (neighbor_index == -1)
            candidate_idx = None
            for j, (cand_idx, neigh_idx) in enumerate(zip(candidate_indices, neighbor_indices)):
                if cand_idx == i and neigh_idx == -1:
                    candidate_idx = j
                    break
            
            if candidate_idx is not None and candidate_idx < len(result_df):
                row = result_df.iloc[candidate_idx]
                candidate.test_sharpe = float(row.get('sharpe_ratio', 0))
                candidate.test_return = float(row.get('total_return', 0))
                candidate.test_max_dd = float(row.get('max_drawdown', 0))
                
                # Collect neighbor sharpes
                neighbor_sharpes = []
                for j, (cand_idx, neigh_idx) in enumerate(zip(candidate_indices, neighbor_indices)):
                    if cand_idx == i and neigh_idx != -1 and j < len(result_df):
                        neighbor_row = result_df.iloc[j]
                        neighbor_sharpes.append(float(neighbor_row.get('sharpe_ratio', 0)))
                
                candidate.neighbor_sharpes = neighbor_sharpes
                
                # Compute stability score
                if neighbor_sharpes and candidate.test_sharpe and candidate.test_sharpe > 0:
                    min_neighbor = min(neighbor_sharpes) if neighbor_sharpes else 0
                    candidate.stability_score = min_neighbor / candidate.test_sharpe
                else:
                    candidate.stability_score = 0
                
                # Compute consistency (min/max across all periods)
                sharpes = [s for s in [candidate.train_sharpe, candidate.val_sharpe, candidate.test_sharpe] if s]
                if sharpes and max(sharpes) > 0:
                    candidate.consistency = min(sharpes) / max(sharpes)
                else:
                    candidate.consistency = 0
                
                # Compute composite score
                candidate.composite_score = self._compute_composite_score(candidate)
            else:
                # Fallback if candidate not found
                candidate.composite_score = 0
        
        return candidates
    
    def _run_phase3_sequential(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 3 sequentially (fallback for small candidate sets)."""
        engine = BacktestEngine(self.config.backtest)
        strategy_cls = StrategyFactory[self.strategy]
        
        # Use test_full for indicator calculation if warmup is enabled
        indicator_series = self.test_full if self.warmup_bars > 0 else self.test
        
        for i, candidate in enumerate(candidates):
            try:
                # Run on TEST data
                strategy = strategy_cls(**candidate.params)
                # Generate signals on full data (includes warmup)
                entries, exits = strategy.generate_signals(indicator_series)
                
                # Trim signals to match test series if using warmup
                if self.warmup_bars > 0:
                    test_start = self.test.index[0]
                    test_end = self.test.index[-1]
                    entries = entries.loc[test_start:test_end]
                    exits = exits.loc[test_start:test_end]
                
                portfolio = engine.run(self.test, (entries, exits))
                metrics = compute_metrics(portfolio, self.test, self.config.backtest.freq)
                
                candidate.test_sharpe = metrics.get('sharpe_ratio', 0)
                candidate.test_return = metrics.get('total_return', 0)
                candidate.test_max_dd = metrics.get('max_drawdown', 0)
                
                # Generate neighbors and test sensitivity
                neighbors = self._generate_neighbors(
                    candidate.params, 
                    step=self.opt_config.sensitivity_step
                )
                
                neighbor_sharpes = []
                for neighbor_params in neighbors:
                    try:
                        n_strategy = strategy_cls(**neighbor_params)
                        n_entries, n_exits = n_strategy.generate_signals(indicator_series)
                        
                        # Trim neighbor signals too
                        if self.warmup_bars > 0:
                            n_entries = n_entries.loc[test_start:test_end]
                            n_exits = n_exits.loc[test_start:test_end]
                        
                        n_portfolio = engine.run(self.test, (n_entries, n_exits))
                        n_metrics = compute_metrics(n_portfolio, self.test, self.config.backtest.freq)
                        neighbor_sharpes.append(n_metrics.get('sharpe_ratio', 0))
                    except Exception:
                        continue
                
                candidate.neighbor_sharpes = neighbor_sharpes
                
                # Compute stability score
                if neighbor_sharpes and candidate.test_sharpe and candidate.test_sharpe > 0:
                    min_neighbor = min(neighbor_sharpes) if neighbor_sharpes else 0
                    candidate.stability_score = min_neighbor / candidate.test_sharpe
                else:
                    candidate.stability_score = 0
                
                # Compute consistency (min/max across all periods)
                sharpes = [s for s in [candidate.train_sharpe, candidate.val_sharpe, candidate.test_sharpe] if s]
                if sharpes and max(sharpes) > 0:
                    candidate.consistency = min(sharpes) / max(sharpes)
                else:
                    candidate.consistency = 0
                
                # Compute composite score
                candidate.composite_score = self._compute_composite_score(candidate)
                
            except Exception as e:
                if self.verbose:
                    self.log(f"    Candidate {i} failed: {e}")
                candidate.composite_score = 0
                continue
            
            # Progress update
            if (i + 1) % 20 == 0:
                self.log(f"    Processed {i+1}/{len(candidates)}")
        
        return candidates
    
    def _run_phase3_multiprocessing(self, candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
        """Run Phase 3 using multiprocessing."""
        # Prepare serializable arguments
        test_values = self.test.values
        test_index = self.test.index.values
        engine_config = {
            'init_cash': self.config.backtest.init_cash,
            'fees': self.config.backtest.fees,
            'slippage': self.config.backtest.slippage,
            'freq': self.config.backtest.freq,
        }
        
        # Convert candidates to dictionaries for serialization
        candidate_dicts = [candidate.to_dict() for candidate in candidates]
        
        # Create args list for workers
        args_list = [
            (
                candidate_dict, test_values, test_index,
                engine_config, self.strategy, self.config.backtest.freq,
                self.opt_config.sensitivity_step
            )
            for candidate_dict in candidate_dicts
        ]
        
        analyzed = []
        processed = 0
        
        # Process in chunks for progress tracking
        chunk_size = max(10, len(args_list) // (self.n_jobs * 2))
        
        with mp.Pool(processes=self.n_jobs) as pool:
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                results_chunk = pool.map(_run_phase3_sensitivity_worker, chunk)
                
                # Convert results back to candidates
                for result_dict in results_chunk:
                    if result_dict is not None:
                        candidate = OptimizationCandidate(
                            params=result_dict['params'],
                            train_sharpe=result_dict.get('train_sharpe', 0),
                            val_sharpe=result_dict.get('val_sharpe'),
                            test_sharpe=result_dict.get('test_sharpe'),
                            transfer_ratio=result_dict.get('transfer_ratio'),
                            stability_score=result_dict.get('stability_score'),
                            consistency=result_dict.get('consistency'),
                            composite_score=result_dict.get('composite_score'),
                            neighbor_sharpes=result_dict.get('neighbor_sharpes', []),
                            train_return=result_dict.get('train_return'),
                            val_return=result_dict.get('val_return'),
                            test_return=result_dict.get('test_return'),
                            train_max_dd=result_dict.get('train_max_dd'),
                            val_max_dd=result_dict.get('val_max_dd'),
                            test_max_dd=result_dict.get('test_max_dd'),
                        )
                        analyzed.append(candidate)
                
                processed += len(chunk)
                if processed % 20 == 0 or processed == len(args_list):
                    self.log(f"    Processed {processed}/{len(candidates)}")
        
        return analyzed
    
    def run(self, force_download: bool = False) -> OptimizationResult:
        """
        Run the full 3-phase optimization pipeline.
        
        Args:
            force_download: Force re-download of price data
            
        Returns:
            OptimizationResult with final candidates
        """
        self.log(f"\n{'#'*60}")
        self.log(f"OPTIMIZING: {self.ticker}")
        self.log(f"Strategy: {self.strategy}")
        self.log(f"{'#'*60}")
        
        # Validate and print grid parameter ranges
        if self.config.strategy.grid:
            self.log(f"\n{'='*60}")
            self.log("PARAMETER RANGES (VALIDATION)")
            self.log(f"{'='*60}")
            for param_name, param_values in self.config.strategy.grid.items():
                if isinstance(param_values, list) and len(param_values) > 0:
                    min_val = min(param_values)
                    max_val = max(param_values)
                    step = param_values[1] - param_values[0] if len(param_values) > 1 else 1
                    count = len(param_values)
                    self.log(f"  {param_name:15s}: {min_val:5d} to {max_val:5d} (step {step:3d}) = {count:4d} values")
                    self.log(f"    Values: {param_values[:5]}{'...' if len(param_values) > 5 else ''} {param_values[-1:] if len(param_values) > 5 else ''}")
                else:
                    self.log(f"  {param_name:15s}: INVALID - {type(param_values).__name__} = {param_values}")
            self.log(f"{'='*60}\n")
        else:
            self.log("WARNING: No grid defined in config!")
        
        # Load and split data
        self.load_data(force_download=force_download)
        self.split_data()
        
        # Phase 1: Grid search
        phase1_candidates = self.run_phase1_grid_search()
        phase1_count = len(phase1_candidates)
        
        if not phase1_candidates:
            self.log("\nOptimization failed: No candidates from Phase 1")
            self.result = OptimizationResult(
                ticker=self.ticker,
                strategy=self.strategy,
                candidates=[],
                split_info=self.split_info or {},
                phase1_count=0,
                phase2_count=0,
                phase3_count=0,
            )
            return self.result
        
        # Phase 2: Validation filter
        phase2_candidates = self.run_phase2_validation(phase1_candidates)
        phase2_count = len(phase2_candidates)
        
        if not phase2_candidates:
            self.log("\nOptimization failed: No candidates passed validation")
            self.result = OptimizationResult(
                ticker=self.ticker,
                strategy=self.strategy,
                candidates=[],
                split_info=self.split_info or {},
                phase1_count=phase1_count,
                phase2_count=0,
                phase3_count=0,
            )
            return self.result
        
        # Phase 3: Sensitivity analysis
        final_candidates = self.run_phase3_sensitivity(phase2_candidates)
        phase3_count = len(final_candidates)
        
        # Build result
        self.result = OptimizationResult(
            ticker=self.ticker,
            strategy=self.strategy,
            candidates=final_candidates,
            split_info=self.split_info or {},
            phase1_count=phase1_count,
            phase2_count=phase2_count,
            phase3_count=phase3_count,
            best_candidate=final_candidates[0] if final_candidates else None,
        )
        
        # Print summary
        self.log(f"\n{self.result.summary()}")
        
        return self.result
    
    def save_to_registry(self, path: Optional[Path] = None) -> None:
        """
        Save the best candidate to the asset registry.
        
        Args:
            path: Optional path to registry file (default: PROJECT_ROOT/registry.yml)
        """
        if not self.result or not self.result.best_candidate:
            raise ValueError("No optimization result to save. Run optimize() first.")
        
        bc = self.result.best_candidate
        
        entry = RegistryEntry(
            strategy=self.strategy,
            params=bc.params,
            last_optimized=datetime.now().strftime("%Y-%m-%d"),
            train_sharpe=bc.train_sharpe,
            val_sharpe=bc.val_sharpe,
            test_sharpe=bc.test_sharpe,
            stability_score=bc.stability_score,
            data_source=self.config.data.data_source,
            start_date=self.config.data.start,
        )
        
        update_registry_entry(self.ticker, entry, path)
        self.log(f"\nSaved {self.ticker} to registry")
    
    def _get_param_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract parameter column names from results DataFrame."""
        # Known metric columns to exclude
        metric_cols = {
            'total_return', 'annualized_return', 'max_drawdown', 'volatility',
            'annualized_volatility', 'sharpe_ratio', 'sortino_ratio', 
            'information_ratio', 'tail_ratio', 'deflated_sharpe_ratio', 
            'ulcer_index', 'calmar_ratio', 'total_trades', 'win_rate_pct', 
            'profit_factor', 'expectancy', 'avg_win_amount', 'avg_loss_amount',
            'payoff_ratio', 'largest_win', 'largest_loss', 'winning_streak',
            'losing_streak', 'gain_to_pain_ratio', 'recovery_factor', 
            'net_profit', 'sqn', 'omega_ratio', 'serenity_index',
            'max_drawdown_dollars', 'trades_per_year', 'n_trades', 'win_rate',
        }
        
        return [col for col in df.columns if col.lower() not in metric_cols]
    
    def _generate_neighbors(
        self, 
        params: Dict[str, Any], 
        step: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter neighbors for sensitivity analysis.
        
        For each parameter, try values +/- step from the original.
        
        Args:
            params: Original parameter dictionary
            step: Step size for perturbation
            
        Returns:
            List of neighbor parameter dictionaries
        """
        neighbors = []
        
        for key in params:
            original = params[key]
            
            # Skip non-numeric params
            if not isinstance(original, (int, float)):
                continue
            
            # Generate perturbations
            for delta in [-step, -1, 1, step]:
                new_val = original + delta
                
                # Ensure positive values (periods must be > 0)
                if new_val <= 0:
                    continue
                
                neighbor = dict(params)
                neighbor[key] = int(new_val) if isinstance(original, int) else new_val
                neighbors.append(neighbor)
        
        return neighbors
    
    def _compute_composite_score(self, candidate: OptimizationCandidate) -> float:
        """
        Compute composite score for final ranking.
        
        Formula: test_sharpe * consistency * stability
        
        This rewards parameters that:
        - Perform well on test data
        - Are consistent across train/val/test
        - Are stable to perturbations
        """
        test_sharpe = candidate.test_sharpe or 0
        consistency = candidate.consistency or 0
        stability = candidate.stability_score or 0
        
        # Guard against negative or zero values
        if test_sharpe <= 0:
            return 0
        
        # Composite: multiply all three factors
        # All factors are in [0, 1+] range, so product is reasonable
        return test_sharpe * max(consistency, 0.1) * max(stability, 0.1)


def optimize_asset(
    ticker: str,
    strategy: str = "ensemble_unconstrained",
    template: str = "wide_ensemble_grid.yml",
    save: bool = False,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> OptimizationResult:
    """
    Convenience function to run full optimization pipeline.
    
    Args:
        ticker: Asset ticker symbol
        strategy: Strategy name
        template: Template config name
        save: Save best result to registry
        verbose: Print progress
        n_jobs: Number of parallel processes (default: CPU count - 1)
        
    Returns:
        OptimizationResult with final candidates
        
    Example:
        >>> result = optimize_asset("GOOG", save=True)
        >>> print(result.best_candidate.params)
    """
    optimizer = AssetOptimizer(
        ticker=ticker,
        strategy=strategy,
        template=template,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    
    result = optimizer.run()
    
    if save and result.best_candidate:
        optimizer.save_to_registry()
    
    return result


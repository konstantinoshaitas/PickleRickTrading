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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
)
from .data import DataFetcher, split_train_val_test, get_split_info
from .grid import VectorizedGridSearch
from .metrics import compute_metrics
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
    ):
        self.ticker = ticker.upper()
        self.strategy = strategy
        self.verbose = verbose
        
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
        self.close: Optional[pd.Series] = None
        self.train: Optional[pd.Series] = None
        self.val: Optional[pd.Series] = None
        self.test: Optional[pd.Series] = None
        self.split_info: Optional[Dict] = None
        self.result: Optional[OptimizationResult] = None
    
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_data(self, force_download: bool = False) -> pd.Series:
        """Load price data for the ticker.
        
        Args:
            force_download: Force re-download even if cache exists
            
        Returns:
            Close price series
        """
        self.log(f"Loading data for {self.ticker}...")
        
        fetcher = DataFetcher(
            ticker=self.config.data.ticker,
            start=self.config.data.start,
            end=self.config.data.end,
            interval=self.config.data.interval,
            data_source=self.config.data.data_source,
            asset_type=self.config.data.asset_type,
            cache_csv=self.config.data.cache_csv,
        )
        
        fetcher.load(force_download=force_download)
        self.close = fetcher.close()
        
        self.log(f"  Loaded {len(self.close)} bars")
        self.log(f"  Date range: {self.close.index[0].date()} to {self.close.index[-1].date()}")
        
        return self.close
    
    def split_data(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split data into train/val/test sets.
        
        Returns:
            Tuple of (train, val, test) series
        """
        if self.close is None:
            self.load_data()
        
        self.train, self.val, self.test = split_train_val_test(
            self.close,
            train_ratio=self.opt_config.train_ratio,
            val_ratio=self.opt_config.val_ratio,
            test_ratio=self.opt_config.test_ratio,
        )
        
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
        
        return self.train, self.val, self.test
    
    def run_phase1_grid_search(self) -> List[OptimizationCandidate]:
        """
        Phase 1: Wide grid search on TRAIN data.
        
        Runs vectorized grid search and keeps top N% by Sharpe ratio.
        
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
        )
        
        if not self.config.strategy.grid:
            raise ValueError("No grid defined in config. Set strategy.grid in template.")
        
        search.run(
            close=self.train,
            grid=self.config.strategy.grid,
            base_params=self.config.strategy.params,
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
        
        engine = BacktestEngine(self.config.backtest)
        strategy_cls = StrategyFactory[self.strategy]
        
        passed = []
        
        for i, candidate in enumerate(candidates):
            # Run backtest on validation
            try:
                strategy = strategy_cls(**candidate.params)
                entries, exits = strategy.generate_signals(self.val)
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
                self.log(f"    Candidate {i} failed: {e}")
                continue
            
            # Progress update
            if (i + 1) % 100 == 0:
                self.log(f"    Processed {i+1}/{len(candidates)} ({len(passed)} passed)")
        
        self.log(f"  {len(passed)} candidates passed (transfer ratio >= {self.opt_config.transfer_ratio_threshold})")
        
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
        
        engine = BacktestEngine(self.config.backtest)
        strategy_cls = StrategyFactory[self.strategy]
        
        for i, candidate in enumerate(candidates):
            try:
                # Run on TEST data
                strategy = strategy_cls(**candidate.params)
                entries, exits = strategy.generate_signals(self.test)
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
                        n_entries, n_exits = n_strategy.generate_signals(self.test)
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
                self.log(f"    Candidate {i} failed: {e}")
                candidate.composite_score = 0
                continue
            
            # Progress update
            if (i + 1) % 20 == 0:
                self.log(f"    Processed {i+1}/{len(candidates)}")
        
        # Sort by composite score
        candidates.sort(key=lambda c: c.composite_score or 0, reverse=True)
        
        # Keep top N
        final = candidates[:self.opt_config.final_candidates]
        
        self.log(f"  Top {len(final)} candidates selected by composite score")
        
        return final
    
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
) -> OptimizationResult:
    """
    Convenience function to run full optimization pipeline.
    
    Args:
        ticker: Asset ticker symbol
        strategy: Strategy name
        template: Template config name
        save: Save best result to registry
        verbose: Print progress
        
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
    )
    
    result = optimizer.run()
    
    if save and result.best_candidate:
        optimizer.save_to_registry()
    
    return result


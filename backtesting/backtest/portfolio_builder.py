"""Portfolio construction from weight allocation matrix using vectorbt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig


@dataclass
class PortfolioWrapper:
    """Wrapper to hold init_cash for compatibility with metrics.py."""
    init_cash: float


@dataclass
class MultiAssetPortfolioResult:
    """Result container for multi-asset portfolio backtest.
    
    This wraps the portfolio value series and provides methods
    compatible with vectorbt Portfolio for metrics computation.
    """
    portfolio_value: pd.Series
    _returns: pd.Series
    weights: pd.DataFrame
    prices: pd.DataFrame
    init_cash: float
    fees: float
    slippage: float
    freq: str
    
    def __post_init__(self):
        """Initialize wrapper after dataclass init."""
        self.wrapper = PortfolioWrapper(init_cash=self.init_cash)
    
    def value(self) -> pd.Series:
        """Return portfolio value series."""
        return self.portfolio_value
    
    def returns(self) -> pd.Series:
        """Return portfolio returns series (as method for vbt compatibility)."""
        return self._returns
    
    def total_return(self) -> float:
        """Calculate total return."""
        if len(self.portfolio_value) < 2:
            return 0.0
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1.0
    
    def annualized_return(self, freq: str = "D") -> float:
        """Calculate annualized return."""
        total_ret = self.total_return()
        n_periods = len(self.portfolio_value)
        
        # Determine periods per year based on frequency
        if freq.upper() in ("D", "1D"):
            periods_per_year = 252
        elif freq.upper() in ("W", "1W"):
            periods_per_year = 52
        elif freq.upper() in ("M", "1M"):
            periods_per_year = 12
        else:
            periods_per_year = 252
        
        if n_periods <= 1:
            return 0.0
        
        years = n_periods / periods_per_year
        if years <= 0:
            return 0.0
        
        return (1 + total_ret) ** (1 / years) - 1
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_value) < 2:
            return 0.0
        
        peak = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - peak) / peak
        return float(drawdown.min())
    
    def annualized_volatility(self, freq: str = "D") -> float:
        """Calculate annualized volatility."""
        if len(self._returns) < 2:
            return 0.0
        
        # Determine periods per year based on frequency
        if freq.upper() in ("D", "1D"):
            periods_per_year = 252
        elif freq.upper() in ("W", "1W"):
            periods_per_year = 52
        elif freq.upper() in ("M", "1M"):
            periods_per_year = 12
        else:
            periods_per_year = 252
        
        return float(self._returns.std() * np.sqrt(periods_per_year))
    
    def sharpe_ratio(self, freq: str = "D", risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        ann_return = self.annualized_return(freq)
        ann_vol = self.annualized_volatility(freq)
        
        if ann_vol == 0 or np.isnan(ann_vol):
            return np.nan
        
        return (ann_return - risk_free_rate) / ann_vol
    
    def sortino_ratio(self, freq: str = "D", risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(self._returns) < 2:
            return np.nan
        
        # Determine periods per year based on frequency
        if freq.upper() in ("D", "1D"):
            periods_per_year = 252
        elif freq.upper() in ("W", "1W"):
            periods_per_year = 52
        elif freq.upper() in ("M", "1M"):
            periods_per_year = 12
        else:
            periods_per_year = 252
        
        ann_return = self.annualized_return(freq)
        
        # Downside deviation (only negative returns)
        negative_returns = self._returns[self._returns < 0]
        if len(negative_returns) == 0:
            return np.inf
        
        downside_std = float(negative_returns.std() * np.sqrt(periods_per_year))
        
        if downside_std == 0 or np.isnan(downside_std):
            return np.nan
        
        return (ann_return - risk_free_rate) / downside_std
    
    def drawdown(self) -> pd.Series:
        """Return drawdown series."""
        peak = self.portfolio_value.cummax()
        return (self.portfolio_value - peak) / peak
    
    def information_ratio(self, freq: str = "D") -> float:
        """Information ratio - not applicable without benchmark, return NaN."""
        return np.nan
    
    def tail_ratio(self, freq: str = "D") -> float:
        """Tail ratio - ratio of 95th percentile to 5th percentile of returns."""
        if len(self._returns) < 20:
            return np.nan
        
        percentile_95 = np.percentile(self._returns, 95)
        percentile_5 = np.percentile(self._returns, 5)
        
        if percentile_5 == 0:
            return np.nan
        
        return abs(percentile_95 / percentile_5)
    
    def deflated_sharpe_ratio(self, freq: str = "D") -> float:
        """Deflated Sharpe ratio - not easily computable, return NaN."""
        return np.nan
    
    @property
    def trades(self) -> "MultiAssetTrades":
        """Return trades object for compatibility with metrics."""
        return MultiAssetTrades(self.weights, self.prices, self._returns)


@dataclass
class MultiAssetTrades:
    """Minimal trades interface for multi-asset portfolio metrics."""
    weights: pd.DataFrame
    prices: pd.DataFrame
    portfolio_returns: pd.Series
    
    def __len__(self) -> int:
        """Count number of rebalancing events (weight changes)."""
        # Count significant weight changes as "trades"
        weight_changes = self.weights.diff().abs()
        # A trade occurs when any asset's weight changes by more than 1%
        trade_mask = (weight_changes > 0.01).any(axis=1)
        return int(trade_mask.sum())
    
    @property
    def returns(self) -> pd.Series:
        """Return portfolio returns as trade returns proxy."""
        # For multi-asset portfolios, use daily returns as trade returns
        return self.portfolio_returns
    
    def count(self) -> int:
        """Return trade count."""
        return len(self)
    
    def win_rate(self) -> float:
        """Calculate win rate based on positive return days."""
        if len(self.portfolio_returns) == 0:
            return 0.0
        positive = (self.portfolio_returns > 0).sum()
        return positive / len(self.portfolio_returns)
    
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gains = self.portfolio_returns[self.portfolio_returns > 0].sum()
        losses = abs(self.portfolio_returns[self.portfolio_returns < 0].sum())
        if losses == 0:
            return np.inf if gains > 0 else np.nan
        return gains / losses
    
    def winning_streak(self) -> int:
        """Calculate maximum winning streak."""
        if len(self.portfolio_returns) == 0:
            return 0
        
        wins = (self.portfolio_returns > 0).astype(int)
        # Find consecutive wins
        streaks = wins.groupby((wins != wins.shift()).cumsum()).sum()
        win_streaks = streaks[streaks > 0]
        return int(win_streaks.max()) if len(win_streaks) > 0 else 0
    
    def losing_streak(self) -> int:
        """Calculate maximum losing streak."""
        if len(self.portfolio_returns) == 0:
            return 0
        
        losses = (self.portfolio_returns < 0).astype(int)
        # Find consecutive losses
        streaks = losses.groupby((losses != losses.shift()).cumsum()).sum()
        loss_streaks = streaks[streaks > 0]
        return int(loss_streaks.max()) if len(loss_streaks) > 0 else 0


class PortfolioBuilder:
    """Builds portfolio from weight allocation matrix.
    
    This builder calculates portfolio returns from weights and asset returns,
    then provides a result object compatible with metrics computation.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize portfolio builder with backtest configuration.
        
        Args:
            config: BacktestConfig with fees, slippage, init_cash, etc.
        """
        self.config = config or BacktestConfig()
    
    def build_from_weights(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> MultiAssetPortfolioResult:
        """Build portfolio from weight allocation matrix.
        
        This method calculates portfolio returns as weighted sum of asset returns,
        accounting for transaction costs when weights change.
        
        Args:
            prices: DataFrame with asset prices (columns = assets, index = dates)
            weights: DataFrame with weight allocations (columns = assets, index = dates)
                    Each row should sum to <= 1.0 (remainder is cash)
                    
        Returns:
            MultiAssetPortfolioResult with portfolio value, returns, etc.
        """
        # Validate inputs
        if not prices.index.equals(weights.index):
            raise ValueError("prices and weights must have the same index")
        
        # Align columns (weights may have fewer columns than prices)
        common_cols = list(set(prices.columns) & set(weights.columns))
        if len(common_cols) == 0:
            raise ValueError("No common columns between prices and weights")
        
        prices = prices[common_cols].copy()
        weights = weights[common_cols].copy()
        
        # Calculate asset returns
        asset_returns = prices.pct_change().fillna(0.0)
        
        # Calculate weight changes for transaction cost estimation
        weight_changes = weights.diff().abs().fillna(0.0)
        turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 to avoid double counting
        
        # Transaction costs: fees + slippage on turnover
        transaction_cost_rate = self.config.fees + self.config.slippage
        transaction_costs = turnover * transaction_cost_rate
        
        # Calculate portfolio returns as weighted sum of asset returns
        # Use previous day's weights (we hold these positions, earn these returns)
        prev_weights = weights.shift(1).fillna(0.0)
        
        # Weighted portfolio returns (before costs)
        portfolio_returns_gross = (asset_returns * prev_weights).sum(axis=1)
        
        # Net returns after transaction costs
        portfolio_returns = portfolio_returns_gross - transaction_costs
        
        # Calculate portfolio value series
        portfolio_value = (1 + portfolio_returns).cumprod() * self.config.init_cash
        
        # Ensure first value is init_cash
        portfolio_value.iloc[0] = self.config.init_cash
        
        return MultiAssetPortfolioResult(
            portfolio_value=portfolio_value,
            _returns=portfolio_returns,
            weights=weights,
            prices=prices,
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            slippage=self.config.slippage,
            freq=self.config.freq,
        )

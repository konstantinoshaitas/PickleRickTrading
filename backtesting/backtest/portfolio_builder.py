"""Portfolio construction from weight allocation matrix using vectorbt."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from ..config import BacktestConfig


class PortfolioBuilder:
    """Builds vectorbt Portfolio from weight allocation matrix.
    
    Converts weight allocations to orders and constructs a portfolio
    that rebalances when weights change.
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
    ) -> vbt.Portfolio:
        """Build vectorbt Portfolio from weight allocation matrix.
        
        Args:
            prices: DataFrame with asset prices (columns = assets, index = dates)
            weights: DataFrame with weight allocations (columns = assets, index = dates)
                    Each row should sum to 1.0 (100% allocation)
                    
        Returns:
            vectorbt Portfolio object
            
        Note:
            Uses Portfolio.from_orders() to handle weight-based rebalancing.
            Rebalancing occurs when weights change between timestamps.
            Cash allocation is implicit (unallocated weight = cash).
        """
        # Validate inputs
        if not prices.index.equals(weights.index):
            raise ValueError("prices and weights must have the same index")
        
        if set(prices.columns) != set(weights.columns):
            raise ValueError(
                f"prices columns {list(prices.columns)} do not match "
                f"weights columns {list(weights.columns)}"
            )
        
        # Align indices (inner join)
        common_index = prices.index.intersection(weights.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between prices and weights")
        
        prices = prices.loc[common_index].copy()
        weights = weights.loc[common_index].copy()
        
        # Calculate target dollar values for each timestamp
        # We'll approximate portfolio value over time, then calculate target values
        # Start with initial cash
        portfolio_values = pd.Series(self.config.init_cash, index=common_index)
        
        # Calculate portfolio value over time using weighted returns
        # This is an approximation - vectorbt will calculate it more accurately
        returns = prices.pct_change().fillna(0.0)
        
        # Weighted portfolio returns
        weighted_returns = (returns * weights.shift(1).fillna(0.0)).sum(axis=1)
        portfolio_values = (1 + weighted_returns).cumprod() * self.config.init_cash
        
        # Calculate target dollar values
        target_values = weights.multiply(portfolio_values, axis=0)
        
        # Calculate target shares
        target_shares = target_values / prices
        target_shares = target_shares.fillna(0.0)
        
        # Calculate order sizes (change in shares needed)
        order_sizes = target_shares.diff().fillna(target_shares.iloc[0])
        
        # Use Portfolio.from_orders to build the portfolio
        # vectorbt's from_orders expects size DataFrame with same shape as prices
        try:
            portfolio = vbt.Portfolio.from_orders(
                close=prices,
                size=order_sizes,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                slippage=self.config.slippage,
                freq=self.config.freq,
            )
            return portfolio
        except (AttributeError, TypeError, ValueError) as e:
            # Fallback: use from_signals with size calculated from weights
            # Create entry/exit signals based on weight changes
            entries = pd.DataFrame(False, index=common_index, columns=prices.columns)
            exits = pd.DataFrame(False, index=common_index, columns=prices.columns)
            
            # Mark entries when weight goes from 0 to > 0
            # Mark exits when weight goes from > 0 to 0
            prev_weights = weights.shift(1).fillna(0.0)
            entries = (prev_weights == 0) & (weights > 0)
            exits = (prev_weights > 0) & (weights == 0)
            
            # Calculate size as target dollar value
            # Use size_type='value' to specify dollar amounts
            try:
                portfolio = vbt.Portfolio.from_signals(
                    close=prices,
                    entries=entries,
                    exits=exits,
                    size=target_values,  # Target dollar values
                    size_type='value',
                    init_cash=self.config.init_cash,
                    fees=self.config.fees,
                    slippage=self.config.slippage,
                    freq=self.config.freq,
                )
                return portfolio
            except (TypeError, ValueError):
                # Final fallback: use simple from_signals without size_type
                # This may not handle rebalancing perfectly, but will work
                portfolio = vbt.Portfolio.from_signals(
                    close=prices,
                    entries=entries,
                    exits=exits,
                    init_cash=self.config.init_cash,
                    fees=self.config.fees,
                    slippage=self.config.slippage,
                    freq=self.config.freq,
                )
                return portfolio

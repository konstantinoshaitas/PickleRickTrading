"""RSI momentum filter portfolio strategy.

Generates weight allocation matrix based on RSI ranking, top-K selection,
and ensemble entry/exit signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class RSIFilterPortfolioStrategy:
    """Portfolio strategy using RSI momentum filtering and top-K selection.
    
    Parameters:
        rsi_period: Period for RSI calculation (default 14)
        rsi_threshold: Minimum RSI value to qualify for selection (default 50.0)
        top_k: Number of top assets to select (default 5)
        safe_haven_ticker: Ticker for safe haven asset (default "GLD")
    """
    rsi_period: int = 14
    rsi_threshold: float = 50.0
    top_k: int = 5
    safe_haven_ticker: str = "GLD"
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        ensemble_entries: pd.DataFrame,
        ensemble_exits: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate weight allocation matrix based on RSI ranking and ensemble signals.
        
        Entry Rules (ALL must be true):
        - Ensemble entry signal = True
        - RSI > threshold
        - Asset is in top K by RSI ranking
        
        Exit Rules (ANY triggers exit):
        - Ensemble exit signal = True, OR
        - RSI < threshold, OR
        - Asset falls out of top K
        
        Args:
            prices: DataFrame with columns = assets, index = dates
            ensemble_entries: DataFrame with ensemble entry signals (columns = assets, index = dates)
            ensemble_exits: DataFrame with ensemble exit signals (columns = assets, index = dates)
            
        Returns:
            DataFrame with weight allocations (columns = assets + safe_haven_ticker, index = dates)
            Weights sum to 1.0 for each row (100% allocation)
        """
        # Ensure safe haven ticker is in prices DataFrame
        if self.safe_haven_ticker not in prices.columns:
            raise ValueError(
                f"Safe haven ticker '{self.safe_haven_ticker}' not found in prices DataFrame. "
                f"Available columns: {list(prices.columns)}"
            )
        
        # Ensure ensemble signals have same columns as prices (excluding safe haven)
        asset_columns = [col for col in prices.columns if col != self.safe_haven_ticker]
        
        if set(ensemble_entries.columns) != set(asset_columns):
            raise ValueError(
                f"Ensemble entries columns {list(ensemble_entries.columns)} "
                f"do not match asset columns {asset_columns}"
            )
        
        if set(ensemble_exits.columns) != set(asset_columns):
            raise ValueError(
                f"Ensemble exits columns {list(ensemble_exits.columns)} "
                f"do not match asset columns {asset_columns}"
            )
        
        # Align indices (inner join to ensure all data available)
        common_index = prices.index.intersection(
            ensemble_entries.index.intersection(ensemble_exits.index)
        )
        prices = prices.loc[common_index]
        ensemble_entries = ensemble_entries.loc[common_index]
        ensemble_exits = ensemble_exits.loc[common_index]
        
        # Initialize weights DataFrame with zeros
        # Include all assets plus safe haven
        weights = pd.DataFrame(
            0.0,
            index=prices.index,
            columns=prices.columns,
        )
        
        # Calculate RSI for each asset (excluding safe haven for ranking)
        rsi_df = pd.DataFrame(
            index=prices.index,
            columns=asset_columns,
        )
        
        for asset in asset_columns:
            rsi_series = vbt.RSI.run(prices[asset], window=self.rsi_period).rsi
            rsi_df[asset] = rsi_series
        
        # Process each timestamp
        for date in prices.index:
            # Get RSI values for this date
            rsi_values = rsi_df.loc[date]
            
            # Skip if any RSI values are NaN (insufficient history)
            if rsi_values.isna().any():
                # Allocate to safe haven if RSI not available
                weights.loc[date, self.safe_haven_ticker] = 0.5
                # Remaining 0.5 stays as cash (implicit)
                continue
            
            # Rank assets by RSI (descending: highest RSI = rank 1)
            # Use method='dense' to handle ties properly
            rsi_ranks = rsi_values.rank(method='dense', ascending=False)
            
            # Filter: RSI > threshold
            above_threshold = rsi_values > self.rsi_threshold
            
            # Select top K assets (by rank, among those above threshold)
            # Assets with rank <= top_k and above threshold
            top_k_mask = (rsi_ranks <= self.top_k) & above_threshold
            top_k_assets = rsi_values[top_k_mask].index.tolist()
            
            # Get ensemble signals for this date
            entries = ensemble_entries.loc[date]
            exits = ensemble_exits.loc[date]
            
            # Determine qualifying assets (entry rules: ALL must be true)
            # 1. In top K
            # 2. RSI > threshold (already filtered above)
            # 3. Ensemble entry signal = True
            qualifying_assets = []
            
            for asset in top_k_assets:
                # Check entry rules: ALL must be true
                if entries[asset]:  # Ensemble entry signal
                    qualifying_assets.append(asset)
            
            # For each asset, check if it should be held
            # An asset should be held if it qualifies AND doesn't have exit signal
            held_assets = []
            
            for asset in qualifying_assets:
                # Exit rules (ANY triggers exit):
                # 1. Ensemble exit signal = True, OR
                # 2. RSI < threshold (shouldn't happen if in top_k, but check), OR
                # 3. Asset falls out of top K (shouldn't happen if in top_k_assets, but check)
                should_exit = (
                    exits[asset] or  # Ensemble exit signal
                    rsi_values[asset] < self.rsi_threshold or  # RSI < threshold
                    asset not in top_k_assets  # Fell out of top K
                )
                
                if not should_exit:
                    held_assets.append(asset)
            
            # Allocate weights
            num_held = len(held_assets)
            
            if num_held == 0:
                # No qualifying assets → 100% safe haven (50% GLD + 50% Cash implicit)
                weights.loc[date, self.safe_haven_ticker] = 0.5
                # Remaining 0.5 stays as cash (implicit)
            else:
                # Allocate 1/K to each held asset
                weight_per_asset = 1.0 / self.top_k
                
                for asset in held_assets:
                    weights.loc[date, asset] = weight_per_asset
                
                # Remaining weight → safe haven (50% of remainder → GLD, 50% → cash)
                allocated_weight = num_held * weight_per_asset
                remaining_weight = 1.0 - allocated_weight
                
                if remaining_weight > 0:
                    # 50% of remainder to GLD, 50% stays as cash (implicit)
                    weights.loc[date, self.safe_haven_ticker] = remaining_weight * 0.5
                    # Remaining 0.5 * remaining_weight stays as cash
        
        # Normalize weights to ensure they sum to 1.0 (safety check)
        row_sums = weights.sum(axis=1)
        # Only normalize if sum is not already 1.0 (within tolerance)
        tolerance = 1e-6
        needs_normalization = (row_sums - 1.0).abs() > tolerance
        if needs_normalization.any():
            # Normalize rows that don't sum to 1.0
            weights.loc[needs_normalization] = weights.loc[needs_normalization].div(
                row_sums[needs_normalization], axis=0
            )
        
        return weights
    
    def params(self):
        """Return strategy parameters as dictionary."""
        return {
            "rsi_period": self.rsi_period,
            "rsi_threshold": self.rsi_threshold,
            "top_k": self.top_k,
            "safe_haven_ticker": self.safe_haven_ticker,
        }


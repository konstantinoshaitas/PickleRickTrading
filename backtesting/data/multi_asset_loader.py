"""Multi-asset data loading utilities.

Loads multiple assets into a single DataFrame with aligned dates using inner join.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .fetcher import DataFetcher
from ..config import DataConfig, get_asset_cache_path


def load_multi_asset_prices(
    tickers: List[str],
    config: DataConfig,
    force_download: bool = False,
    per_asset_cache_csv: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load close prices for multiple assets into a single DataFrame.
    
    Args:
        tickers: List of asset tickers to load (e.g., ['BTC-USD', 'ETH-USD', 'AAPL'])
        config: DataConfig with data source, date range, interval, etc.
        force_download: If True, force download even if cached data exists
        per_asset_cache_csv: Optional dict mapping ticker -> cache_csv path for overrides
        
    Returns:
        DataFrame with columns = tickers, index = dates (DatetimeIndex)
        Only includes dates where ALL assets have data (inner join)
        
    Example:
        >>> config = DataConfig(start='2020-01-01', interval='1d')
        >>> prices = load_multi_asset_prices(['BTC-USD', 'ETH-USD'], config)
        >>> prices.head()
                        BTC-USD    ETH-USD
        2020-01-01      7191.0     129.50
        2020-01-02      7200.0     130.00
        ...
    """
    if not tickers:
        raise ValueError("tickers list cannot be empty")
    
    # Load close prices for each ticker
    close_series_list = []
    for ticker in tickers:
        # Determine cache_csv path for this ticker
        # Priority: 1) per-asset override, 2) auto-construct from config, 3) use config as-is
        ticker_cache_csv = None
        
        if per_asset_cache_csv and ticker in per_asset_cache_csv:
            # Use per-asset override if provided
            ticker_cache_csv = per_asset_cache_csv[ticker]
        elif config.cache_csv:
            # Use new asset-centric path: assets/{TICKER}/cache.csv
            # Check if it exists, otherwise fall back to old path for backward compatibility
            new_path = get_asset_cache_path(ticker)
            if new_path.exists():
                ticker_cache_csv = str(new_path)
            else:
                # Fallback to old path for backward compatibility
                ticker_cache_csv = f"data/cache/{ticker}_{config.interval}.csv"
        else:
            # Try new asset-centric path first
            new_path = get_asset_cache_path(ticker)
            if new_path.exists():
                ticker_cache_csv = str(new_path)
            else:
                # Fallback: no cache
                ticker_cache_csv = None
        
        fetcher = DataFetcher(
            ticker=ticker,
            start=config.start,
            end=config.end,
            interval=config.interval,
            data_source=config.data_source,
            asset_type=config.asset_type,
            local_csv=config.local_csv,
            cache_csv=ticker_cache_csv,
        )
        ohlcv = fetcher.load(force_download=force_download)
        close = fetcher.close()
        close.name = ticker  # Name the series with the ticker
        close_series_list.append(close)
    
    # Concatenate all series into a DataFrame with inner join
    # This ensures we only keep dates where ALL assets have data
    prices_df = pd.concat(close_series_list, axis=1, join='inner')
    
    # Sort by date to ensure chronological order
    prices_df = prices_df.sort_index()
    
    # Remove any rows with NaN values (shouldn't happen with inner join, but safety check)
    prices_df = prices_df.dropna()
    
    if prices_df.empty:
        raise ValueError(
            f"No overlapping dates found for tickers {tickers}. "
            "Check date ranges and data availability."
        )
    
    return prices_df


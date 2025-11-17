"""Simple yfinance/local CSV loader plus train/val split helper."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DataFetcher:
    def __init__(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        data_source: str = "yfinance",
        asset_type: Optional[str] = None,
        local_csv: Optional[str] = None,
        cache_csv: Optional[str] = None,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.data_source = data_source.lower()
        self.asset_type = asset_type.lower() if asset_type else None
        self.local_csv = Path(local_csv) if local_csv else None
        self.cache_csv = Path(cache_csv) if cache_csv else None
        self.data: Optional[pd.DataFrame] = None
    
    def load(self, force_download: bool = False) -> pd.DataFrame:
        if not force_download:
            if self.local_csv and self.local_csv.exists():
                self.data = self._read_csv(self.local_csv)
            elif self.cache_csv and self.cache_csv.exists():
                self.data = self._read_csv(self.cache_csv)
        if self.data is None or self.data.empty:
            self.data = self._download_with_retry()
            if self.cache_csv and not self.data.empty:
                self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
                self.data.to_csv(self.cache_csv)
        self._normalize_frame()
        self._apply_date_window()
        if self.data.empty:
            raise ValueError(
                f"No data found for ticker '{self.ticker}' with parameters "
                f"(start={self.start}, end={self.end}, interval={self.interval}, "
                f"source={self.data_source}). "
                "This could be due to: network issues, API problems, "
                "or invalid ticker/date range."
            )
        return self.data
    
    def _detect_asset_type(self) -> str:
        """Detect asset type from ticker format or use explicit setting.
        
        Returns:
            "crypto" or "stock"
        """
        if self.asset_type:
            if self.asset_type in ("crypto", "stock"):
                return self.asset_type
            else:
                raise ValueError(
                    f"Invalid asset_type '{self.asset_type}'. "
                    "Must be 'crypto' or 'stock'."
                )
        
        # Auto-detect: tickers with format XXX-YYY (e.g., BTC-USD) are crypto
        if "-" in self.ticker:
            parts = self.ticker.split("-")
            if len(parts) == 2:
                # Common crypto ticker format: BTC-USD, ETH-USD, etc.
                return "crypto"
        
        # Default to stock for ambiguous cases
        return "stock"
    
    def _download_with_retry(self, max_retries: int = 3) -> pd.DataFrame:
        """Download data with retry logic based on selected data source."""
        if self.data_source == "alphavantage":
            return self._download_alphavantage(max_retries)
        else:  # default to yfinance
            return self._download_yfinance(max_retries)
    
    def _download_yfinance(self, max_retries: int = 3) -> pd.DataFrame:
        """Download data from yfinance with retry logic."""
        for attempt in range(max_retries):
            try:
                # Try method 1: Using Ticker.history() directly (skip .info to avoid rate limit)
                # This avoids the 429 error from accessing ticker.info
                ticker_obj = yf.Ticker(self.ticker)
                hist = ticker_obj.history(
                    start=self.start,
                    end=self.end,
                    interval=self.interval,
                    auto_adjust=True,
                )
                if hist is not None and not hist.empty:
                    return hist
                
                # Try method 2: Standard download (fallback)
                data = yf.download(
                    self.ticker,
                    start=self.start,
                    end=self.end,
                    interval=self.interval,
                    progress=False,
                    show_errors=False,
                )
                
                # Check if download was successful
                if data is not None and not data.empty:
                    return data
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limiting (429 or "too many requests")
                if "429" in error_str or "too many requests" in error_str:
                    wait_time = 60 * (attempt + 1)  # Wait 1min, 2min, 3min for rate limits
                    print(f"⚠️  Rate limited by Yahoo Finance API. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    print("   (This happens when making too many requests. Consider using cached data.)")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s for other errors
                    print(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} download attempts failed. Last error: {e}")
        
        # Return empty DataFrame if all attempts failed
        return pd.DataFrame()
    
    def _download_alphavantage(self, max_retries: int = 3) -> pd.DataFrame:
        """Download data from Alpha Vantage API (routes to crypto or stock endpoint)."""
        asset_type = self._detect_asset_type()
        
        if asset_type == "crypto":
            return self._download_alphavantage_crypto(max_retries)
        else:
            return self._download_alphavantage_stock(max_retries)
    
    def _download_alphavantage_stock(self, max_retries: int = 3) -> pd.DataFrame:
        """Download stock data from Alpha Vantage API."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY not found in environment variables. "
                "Please set it in your .env file or export it as an environment variable."
            )
        
        # Alpha Vantage only supports daily intervals for TIME_SERIES_DAILY_ADJUSTED
        if self.interval != "1d":
            raise ValueError(
                f"Alpha Vantage TIME_SERIES_DAILY_ADJUSTED only supports daily (1d) intervals. "
                f"Requested interval: {self.interval}"
            )
        
        base_url = "https://www.alphavantage.co/query"
        
        for attempt in range(max_retries):
            try:
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": self.ticker,
                    "apikey": api_key,
                    "outputsize": "full",  # Get full historical data
                    "datatype": "json",
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
                if "Note" in data:
                    raise ValueError(f"Alpha Vantage API rate limit: {data['Note']}")
                if "Information" in data:
                    raise ValueError(f"Alpha Vantage API info: {data['Information']}")
                
                # Extract time series data
                if "Time Series (Daily)" not in data:
                    raise ValueError(f"Unexpected Alpha Vantage response format. Keys: {list(data.keys())}")
                
                time_series = data["Time Series (Daily)"]
                if not time_series:
                    raise ValueError("No time series data returned from Alpha Vantage")
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                
                # Map Alpha Vantage columns to standard format
                # Alpha Vantage uses: 1. open, 2. high, 3. low, 4. close, 5. adjusted close, 6. volume, 7. dividend amount, 8. split coefficient
                df = df.rename(columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",  # Raw close (we'll use adjusted)
                    "5. adjusted close": "Adj Close",
                    "6. volume": "Volume",
                })
                
                # Use adjusted close as Close (matching yfinance auto_adjust=True behavior)
                df["Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
                
                # Convert other columns to numeric
                for col in ["Open", "High", "Low", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Select only the columns we need (matching yfinance format)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                
                # Sort by date (ascending)
                df = df.sort_index()
                
                # Filter by date range if specified
                if self.start:
                    start_ts = pd.to_datetime(self.start)
                    df = df[df.index >= start_ts]
                if self.end:
                    end_ts = pd.to_datetime(self.end)
                    df = df[df.index <= end_ts]
                
                if df.empty:
                    raise ValueError("No data in specified date range")
                
                return df
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"Alpha Vantage download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} Alpha Vantage download attempts failed. Last error: {e}")
            except Exception as e:
                # For non-network errors, don't retry
                print(f"Alpha Vantage download failed: {e}")
                break
        
        # Return empty DataFrame if all attempts failed
        return pd.DataFrame()
    
    def _download_alphavantage_crypto(self, max_retries: int = 3) -> pd.DataFrame:
        """Download cryptocurrency data from Alpha Vantage API using DIGITAL_CURRENCY_DAILY."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY not found in environment variables. "
                "Please set it in your .env file or export it as an environment variable."
            )
        
        # Alpha Vantage only supports daily intervals for DIGITAL_CURRENCY_DAILY
        if self.interval != "1d":
            raise ValueError(
                f"Alpha Vantage DIGITAL_CURRENCY_DAILY only supports daily (1d) intervals. "
                f"Requested interval: {self.interval}"
            )
        
        # Parse ticker format: BTC-USD -> symbol="BTC", market="USD"
        if "-" not in self.ticker:
            raise ValueError(
                f"Crypto ticker must be in format 'SYMBOL-MARKET' (e.g., 'BTC-USD'). "
                f"Got: {self.ticker}"
            )
        
        parts = self.ticker.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Crypto ticker must be in format 'SYMBOL-MARKET' (e.g., 'BTC-USD'). "
                f"Got: {self.ticker}"
            )
        
        symbol = parts[0].upper()
        market = parts[1].upper()
        
        base_url = "https://www.alphavantage.co/query"
        
        for attempt in range(max_retries):
            try:
                params = {
                    "function": "DIGITAL_CURRENCY_DAILY",
                    "symbol": symbol,
                    "market": market,
                    "apikey": api_key,
                    "outputsize": "full",  # Get full historical data
                    "datatype": "json",
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
                if "Note" in data:
                    raise ValueError(f"Alpha Vantage API rate limit: {data['Note']}")
                if "Information" in data:
                    raise ValueError(f"Alpha Vantage API info: {data['Information']}")
                
                # Extract time series data
                time_series_key = "Time Series (Digital Currency Daily)"
                if time_series_key not in data:
                    raise ValueError(f"Unexpected Alpha Vantage response format. Keys: {list(data.keys())}")
                
                time_series = data[time_series_key]
                if not time_series:
                    raise ValueError("No time series data returned from Alpha Vantage")
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                
                # Debug: Print date range before filtering
                if len(df) > 0:
                    print(f"DEBUG: Alpha Vantage returned {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
                
                # Map Alpha Vantage crypto columns to standard format
                # Alpha Vantage crypto returns: "1. open", "2. high", "3. low", "4. close", "5. volume"
                # (Same format as stock endpoint, but no adjusted close - crypto doesn't have splits/dividends)
                # Try to match exact column names first (most common case)
                column_mapping = {}
                
                # Check for exact column names (most common format)
                if "1. open" in df.columns:
                    column_mapping["1. open"] = "Open"
                if "2. high" in df.columns:
                    column_mapping["2. high"] = "High"
                if "3. low" in df.columns:
                    column_mapping["3. low"] = "Low"
                if "4. close" in df.columns:
                    column_mapping["4. close"] = "Close"
                if "5. volume" in df.columns:
                    column_mapping["5. volume"] = "Volume"
                
                # If exact names not found, try market-specific format (e.g., "1a. open (USD)")
                if not all(key in column_mapping for key in ["1. open", "2. high", "3. low", "4. close"]):
                    market_suffix = f" ({market})"
                    market_suffix_lower = market_suffix.lower()
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if f"open{market_suffix_lower}" in col_lower and "1. open" not in column_mapping:
                            column_mapping[col] = "Open"
                        elif f"high{market_suffix_lower}" in col_lower and "2. high" not in column_mapping:
                            column_mapping[col] = "High"
                        elif f"low{market_suffix_lower}" in col_lower and "3. low" not in column_mapping:
                            column_mapping[col] = "Low"
                        elif f"close{market_suffix_lower}" in col_lower and "4. close" not in column_mapping:
                            column_mapping[col] = "Close"
                
                # Final fallback: search for columns containing keywords
                if "Open" not in column_mapping.values():
                    for col in df.columns:
                        if "open" in col.lower() and "high" not in col.lower() and "low" not in col.lower() and "close" not in col.lower():
                            column_mapping[col] = "Open"
                            break
                if "High" not in column_mapping.values():
                    for col in df.columns:
                        if "high" in col.lower() and "open" not in col.lower() and "low" not in col.lower() and "close" not in col.lower():
                            column_mapping[col] = "High"
                            break
                if "Low" not in column_mapping.values():
                    for col in df.columns:
                        if "low" in col.lower() and "open" not in col.lower() and "high" not in col.lower() and "close" not in col.lower():
                            column_mapping[col] = "Low"
                            break
                if "Close" not in column_mapping.values():
                    for col in df.columns:
                        if "close" in col.lower() and "open" not in col.lower() and "high" not in col.lower() and "low" not in col.lower():
                            column_mapping[col] = "Close"
                            break
                if "Volume" not in column_mapping.values():
                    for col in df.columns:
                        if "volume" in col.lower() and "market cap" not in col.lower():
                            column_mapping[col] = "Volume"
                            break
                
                # Verify we found all required columns
                required = ["Open", "High", "Low", "Close", "Volume"]
                if not all(col in column_mapping.values() for col in required):
                    missing = [col for col in required if col not in column_mapping.values()]
                    raise ValueError(
                        f"Could not find all required OHLC columns. Missing: {missing}. "
                        f"Available columns: {list(df.columns)}"
                    )
                
                # Rename columns to standard format
                df = df.rename(columns=column_mapping)
                
                # Convert to numeric
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Select only the columns we need (matching yfinance format)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                
                # Sort by date (ascending)
                df = df.sort_index()
                
                # Filter by date range if specified
                if self.start:
                    start_ts = pd.to_datetime(self.start)
                    df = df[df.index >= start_ts]
                if self.end:
                    end_ts = pd.to_datetime(self.end)
                    df = df[df.index <= end_ts]
                
                if df.empty:
                    raise ValueError("No data in specified date range")
                
                return df
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"Alpha Vantage crypto download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} Alpha Vantage crypto download attempts failed. Last error: {e}")
            except Exception as e:
                # For non-network errors, don't retry
                print(f"Alpha Vantage crypto download failed: {e}")
                break
        
        # Return empty DataFrame if all attempts failed
        return pd.DataFrame()
    
    def _read_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)
    
    def _normalize_frame(self) -> None:
        if self.data is None:
            return
        raw_index = pd.Index(self.data.index)
        if not isinstance(raw_index, pd.DatetimeIndex):
            str_index = raw_index.astype(str)
            non_data_mask = ~str_index.str.contains(r"\d")
            if non_data_mask.any():
                self.data = self.data.loc[~non_data_mask]
                raw_index = pd.Index(self.data.index)
                str_index = raw_index.astype(str)
            idx = pd.to_datetime(str_index, errors="coerce", cache=False)
        else:
            idx = raw_index
        valid_mask = ~idx.isna()
        if not valid_mask.all():
            self.data = self.data.loc[valid_mask]
            idx = idx[valid_mask]
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        self.data.index = idx
        self.data.index.name = "Date"
        self.data.sort_index(inplace=True)
        if isinstance(self.data.columns, pd.MultiIndex):
            # flatten multi-index columns (yfinance default when group_by='ticker')
            if len(self.data.columns.levels[-1]) == 1:
                self.data.columns = self.data.columns.get_level_values(0)
            else:
                cols = []
                for col in self.data.columns.to_flat_index():
                    parts = [str(part) for part in col if part and part != "nan"]
                    cols.append("_".join(parts))
                self.data.columns = cols
    
    def _apply_date_window(self) -> None:
        if self.data is None or not isinstance(self.data.index, pd.DatetimeIndex):
            return
        start_ts = self._parse_date(self.start)
        end_ts = self._parse_date(self.end)
        if start_ts is not None:
            self.data = self.data.loc[self.data.index >= start_ts]
        if end_ts is not None:
            self.data = self.data.loc[self.data.index <= end_ts]
    
    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
        if not value:
            return None
        try:
            ts = pd.to_datetime(value)
        except (TypeError, ValueError):
            return None
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts
    
    def close(self) -> pd.Series:
        if self.data is None:
            raise ValueError("Call load() first.")
        close = self.data["Close"]
        if isinstance(close, pd.DataFrame):
            # Select the first column if yfinance returned a multi-index frame
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce")
        close = close.dropna()
        close.name = "close"
        return close.squeeze()


def split_train_val(series: pd.Series, train_ratio: float) -> Tuple[pd.Series, pd.Series]:
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx].copy()
    val = series.iloc[split_idx:].copy()
    return train, val

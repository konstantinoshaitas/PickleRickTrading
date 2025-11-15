"""Simple yfinance/local CSV loader plus train/val split helper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf


class DataFetcher:
    def __init__(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        local_csv: Optional[str] = None,
        cache_csv: Optional[str] = None,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
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
                f"(start={self.start}, end={self.end}, interval={self.interval}). "
                "This could be due to: network issues, yfinance API problems, "
                "or invalid ticker/date range."
            )
        return self.data
    
    def _download_with_retry(self, max_retries: int = 3) -> pd.DataFrame:
        """Download data with retry logic to handle yfinance API issues."""
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

"""Simple yfinance/local CSV loader plus train/val split helper."""

from __future__ import annotations

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
            self.data = yf.download(
                self.ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                group_by="column",
                progress=False,
            )
            if self.cache_csv:
                self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
                self.data.to_csv(self.cache_csv)
        self._normalize_frame()
        self._apply_date_window()
        if self.data.empty:
            raise ValueError("No data found for given parameters.")
        return self.data
    
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

"""Lightweight configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    ticker: str = "BTC-USD"
    start: str = "2018-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    data_source: str = "yfinance"  # Options: "yfinance" or "alphavantage"
    asset_type: Optional[str] = None  # Options: "crypto", "stock", or None (auto-detect)
    local_csv: Optional[str] = None
    cache_csv: Optional[str] = "data/cache.csv"


@dataclass
class StrategyConfig:
    name: str = "triple_ema"
    params: Dict[str, int] = field(
        default_factory=lambda: {"ema_fast": 21, "ema_mid": 55, "ema_slow": 200}
    )
    grid: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "ema_fast": list[int](range(5, 35, 5)),
            "ema_mid": list[int](range(30, 100, 10)),
            "ema_slow": list[int](range(100, 260, 20)),
        }
    )


@dataclass
class BacktestConfig:
    init_cash: float = 100_000
    fees: float = 0.0005
    slippage: float = 0.0005
    freq: str = "D"
    train_ratio: float = 0.6


@dataclass
class GridConfig:
    metric: str = "sharpe_ratio"
    top_n: int = 3


@dataclass
class AssetConfig:
    """Configuration for a specific asset's strategy."""
    strategy: str
    params: Dict[str, Any]


@dataclass
class PortfolioConfig:
    """Configuration for multi-asset portfolio backtesting.
    
    Attributes:
        tickers: List of asset tickers to include in portfolio
        safe_haven_ticker: Ticker for safe haven asset (default "GLD")
        rsi_period: Period for RSI calculation (default 14)
        rsi_threshold: Minimum RSI value to qualify for selection (default 50.0)
        top_k: Number of top assets to select (default 5)
        assets: Per-asset strategy configuration (ticker -> AssetConfig)
        grid: Grid search space for portfolio parameters
    """
    tickers: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "AAPL", "GOOG", "GLD"])
    safe_haven_ticker: str = "GLD"
    rsi_period: int = 14
    rsi_threshold: float = 50.0
    top_k: int = 5
    assets: Dict[str, AssetConfig] = field(default_factory=dict)
    grid: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    portfolio: Optional[PortfolioConfig] = None


def _parse_grid_value(value):
    """Parse grid value - supports both lists and range strings like '4:40:2'.
    
    Examples:
        "4:40:2" -> [4, 6, 8, ..., 40]  (range from 4 to 40 with step 2)
        [5, 15, 20, 30, 35] -> [5, 15, 20, 30, 35]  (explicit list, used as-is)
    """
    if isinstance(value, str) and ":" in value:
        # Range notation: "start:end:step"
        parts = value.split(":")
        if len(parts) == 3:
            try:
                start, end, step = map(int, parts)
                return list(range(start, end + 1, step))  # +1 to include end
            except ValueError:
                pass  # If parsing fails, treat as regular string (unlikely for grid params)
    elif isinstance(value, list):
        # Explicit list: use as-is
        return value
    return value


def load_config(path: Path) -> WorkflowConfig:
    if not Path(path).exists():
        return WorkflowConfig()
    
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    
    # Parse strategy grid ranges if present
    strategy_payload = payload.get("strategy", {})
    if "grid" in strategy_payload:
        strategy_payload["grid"] = {
            k: _parse_grid_value(v) 
            for k, v in strategy_payload["grid"].items()
        }
    
    # Parse portfolio config if present
    portfolio_payload = payload.get("portfolio")
    portfolio_config = None
    if portfolio_payload:
        # Parse portfolio grid ranges if present
        if "grid" in portfolio_payload:
            portfolio_payload["grid"] = {
                k: _parse_grid_value(v) 
                for k, v in portfolio_payload["grid"].items()
            }
        
        # Parse assets config if present
        if "assets" in portfolio_payload:
            assets_dict = {}
            for ticker, asset_data in portfolio_payload["assets"].items():
                assets_dict[ticker] = AssetConfig(**asset_data)
            portfolio_payload["assets"] = assets_dict
            
        portfolio_config = PortfolioConfig(**portfolio_payload)
    
    return WorkflowConfig(
        data=DataConfig(**payload.get("data", {})),
        strategy=StrategyConfig(**strategy_payload),
        backtest=BacktestConfig(**payload.get("backtest", {})),
        grid=GridConfig(**payload.get("grid", {})),
        portfolio=portfolio_config,
    )

"""Lightweight configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    ticker: str = "BTC-USD"
    start: str = "2018-01-01"
    end: Optional[str] = None
    interval: str = "1d"
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
            "ema_fast": list(range(5, 35, 5)),
            "ema_mid": list(range(30, 100, 10)),
            "ema_slow": list(range(100, 260, 20)),
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
class WorkflowConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    grid: GridConfig = field(default_factory=GridConfig)


def load_config(path: Path) -> WorkflowConfig:
    if not Path(path).exists():
        return WorkflowConfig()
    
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    
    return WorkflowConfig(
        data=DataConfig(**payload.get("data", {})),
        strategy=StrategyConfig(**payload.get("strategy", {})),
        backtest=BacktestConfig(**payload.get("backtest", {})),
        grid=GridConfig(**payload.get("grid", {})),
    )

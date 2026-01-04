"""Lightweight configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
PORTFOLIOS_DIR = PROJECT_ROOT / "portfolios"
REGISTRY_FILE = PROJECT_ROOT / "registry.yml"


@dataclass
class DataConfig:
    ticker: str = "BTC-USD"
    start: str = "2018-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    data_source: str = "yfinance"  # Options: "yfinance" or "alphavantage"
    asset_type: Optional[str] = None  # Options: "crypto", "stock", or None (auto-detect)
    local_csv: Optional[str] = None
    cache_csv: Optional[str] = None  # Will auto-resolve to assets/{TICKER}/cache.csv if exists
    warmup_bars: int = 0  # Bars to pre-fetch before start date (0 = auto-calculate from grid)


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
    train_ratio: float = 0.6  # Legacy: used when val/test ratios not specified


@dataclass
class OptimizationConfig:
    """Configuration for the 3-phase optimization pipeline.
    
    Attributes:
        train_ratio: Fraction of data for training (grid search)
        val_ratio: Fraction of data for validation (overfit filtering)
        test_ratio: Fraction of data for testing (sensitivity analysis)
        top_percent_phase1: Keep top X% from grid search (default 1%)
        transfer_ratio_threshold: Min val_sharpe/train_sharpe to keep (default 0.6)
        sensitivity_step: Parameter perturbation step for neighbors (default 2)
        final_candidates: Number of final candidates to return (default 3)
        min_trades_per_year: Minimum trades per year filter (default 1.5)
    """
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    top_percent_phase1: float = 0.01  # Keep top 1%
    transfer_ratio_threshold: float = 0.6
    sensitivity_step: int = 2
    final_candidates: int = 3
    min_trades_per_year: float = 2


@dataclass
class GridConfig:
    metric: str = "sharpe_ratio"
    top_n: int = 3


@dataclass
class AssetConfig:
    """Configuration for a specific asset's strategy."""
    strategy: str
    params: Dict[str, Any]
    cache_csv: Optional[str] = None  # Optional per-asset cache CSV override


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
    optimization: Optional[OptimizationConfig] = None


def _parse_grid_value(value):
    """Parse grid value - supports both lists and range strings like '4:40:2'.
    
    Examples:
        "4:40:2" -> [4, 6, 8, ..., 40]  (range from 4 to 40 with step 2)
        [5, 15, 20, 30, 35] -> [5, 15, 20, 30, 35]  (explicit list, used as-is)
    """
    # If it's already a list, use as-is
    if isinstance(value, list):
        return value
    
    # Convert to string for parsing (handles cases where YAML parses as other types)
    value_str = str(value)
    
    # Check if it's a range notation: "start:end:step"
    if ":" in value_str:
        parts = value_str.split(":")
        if len(parts) == 3:
            try:
                start, end, step = map(int, parts)
                return list(range(start, end + 1, step))  # +1 to include end
            except (ValueError, TypeError):
                # If parsing fails, treat as regular string (unlikely for grid params)
                pass
    
    # If it's a number (int/float), return as single-item list
    if isinstance(value, (int, float)):
        return [int(value)]
    
    # Fallback: return as single-item list
    return [value]


def calculate_warmup_from_grid(grid: Dict[str, Any], strategy_name: str = "") -> int:
    """Auto-calculate required warmup bars from grid parameters.
    
    Analyzes the grid to find the maximum indicator period that requires
    historical data for proper calculation.
    
    Args:
        grid: Parameter grid dictionary (may contain range strings or lists)
        strategy_name: Strategy name for strategy-specific logic
        
    Returns:
        Maximum warmup bars needed (0 if cannot be determined)
    """
    if not grid:
        return 0
    
    max_period = 0
    
    # EMA-based strategies: ema_slow is typically the longest
    for key in ['ema_slow', 'ema_mid', 'ema_fast']:
        if key in grid:
            values = _parse_grid_value(grid[key])
            if values:
                max_period = max(max_period, max(values))
    
    # MACD: requires slowperiod + signalperiod for full indicator warmup
    if 'slowperiod' in grid:
        slow_values = _parse_grid_value(grid['slowperiod'])
        if slow_values:
            slow_max = max(slow_values)
            signal_max = 0
            if 'signalperiod' in grid:
                signal_values = _parse_grid_value(grid['signalperiod'])
                if signal_values:
                    signal_max = max(signal_values)
            max_period = max(max_period, slow_max + signal_max)
    
    # RSI period (for portfolio strategies)
    if 'rsi_period' in grid:
        rsi_values = _parse_grid_value(grid['rsi_period'])
        if rsi_values:
            max_period = max(max_period, max(rsi_values))
    
    return max_period


def calculate_warmup_from_params(params: Dict[str, Any], strategy_name: str = "") -> int:
    """Calculate warmup bars from strategy parameters (single backtest, no grid).
    
    Args:
        params: Strategy parameter dictionary
        strategy_name: Strategy name for strategy-specific logic
        
    Returns:
        Maximum warmup bars needed
    """
    max_period = 0
    
    # EMA-based strategies
    for key in ['ema_slow', 'ema_mid', 'ema_fast']:
        if key in params:
            max_period = max(max_period, int(params[key]))
    
    # MACD strategies
    if 'slowperiod' in params:
        slow = int(params['slowperiod'])
        signal = int(params.get('signalperiod', 0))
        max_period = max(max_period, slow + signal)
    
    # RSI
    if 'rsi_period' in params:
        max_period = max(max_period, int(params['rsi_period']))
    
    return max_period


def extend_start_date(start: str, warmup_bars: int, interval: str = "1d") -> str:
    """Extend start date backwards to accommodate warmup period.
    
    Args:
        start: Original start date string (YYYY-MM-DD)
        warmup_bars: Number of trading bars needed for warmup
        interval: Data interval (used to calculate calendar days)
        
    Returns:
        Extended start date string (YYYY-MM-DD)
    """
    import pandas as pd
    
    start_dt = pd.to_datetime(start)
    
    # Add buffer for weekends/holidays (1.5x for daily data)
    if interval.lower() in ("1d", "d"):
        calendar_days = int(warmup_bars * 1.5)
    else:
        calendar_days = warmup_bars
    
    extended_dt = start_dt - pd.Timedelta(days=calendar_days)
    return extended_dt.strftime('%Y-%m-%d')


# =============================================================================
# Warmup Error Classes
# =============================================================================

class WarmupError(Exception):
    """Base class for warmup-related errors."""
    pass


class DataDownloadError(WarmupError):
    """Raised when data download fails (network, API, empty response)."""
    def __init__(self, message: str, ticker: str = None):
        self.ticker = ticker
        super().__init__(message)


class InsufficientWarmupDataError(WarmupError):
    """Raised when not enough historical data exists for warmup period."""
    def __init__(self, ticker: str, required: int, available: int, earliest_available: str = None):
        self.ticker = ticker
        self.required = required
        self.available = available
        self.shortfall = required - available
        self.earliest_available = earliest_available
        
        msg = (
            f"Insufficient warmup data for {ticker}: "
            f"need {required} bars, have {available} (short by {self.shortfall})"
        )
        if earliest_available:
            msg += f". Earliest available: {earliest_available}"
        super().__init__(msg)


class InsufficientBacktestDataError(WarmupError):
    """Raised when warmup leaves too few bars for meaningful backtesting."""
    def __init__(self, total: int, warmup: int, remaining: int, minimum: int):
        self.total = total
        self.warmup = warmup
        self.remaining = remaining
        self.minimum = minimum
        super().__init__(
            f"After {warmup} warmup bars, only {remaining} bars remain "
            f"(minimum: {minimum}). Reduce indicator periods or extend data range."
        )


# =============================================================================
# Warmup Validation Functions
# =============================================================================

def validate_warmup_coverage(
    data_start: str,
    required_warmup: int,
    original_start: str,
    ticker: str
) -> Tuple[bool, str]:
    """Validate that we have sufficient warmup data.
    
    Args:
        data_start: Actual start date of data (YYYY-MM-DD)
        required_warmup: Number of bars required for warmup
        original_start: User's intended start date (YYYY-MM-DD)
        ticker: Asset ticker for error messages
        
    Returns:
        (is_valid, message) - True if valid, False with explanation if not
    """
    import pandas as pd
    
    data_start_dt = pd.to_datetime(data_start)
    original_start_dt = pd.to_datetime(original_start)
    
    # Estimate available warmup days (rough estimate)
    # Actual bar count will be validated in DataFetcher after loading
    available_days = (original_start_dt - data_start_dt).days
    # Approximate trading days (5/7 of calendar days)
    estimated_bars = int(available_days * 5 / 7)
    
    if estimated_bars < required_warmup:
        shortfall = required_warmup - estimated_bars
        return False, (
            f"Insufficient warmup data for {ticker}. "
            f"Required: {required_warmup} bars, Available (est): {estimated_bars} bars. "
            f"Shortfall: ~{shortfall} bars. "
            f"Options: 1) Reduce max indicator period in grid, "
            f"2) Use a later start date, "
            f"3) Use --allow-partial-warmup flag to proceed with warning."
        )
    
    return True, f"Warmup OK: ~{estimated_bars} bars available (required: {required_warmup})"


def validate_sufficient_backtest_data(
    total_bars: int,
    warmup_bars: int,
    min_backtest_bars: int = 252
) -> Tuple[bool, str]:
    """Ensure enough data remains after warmup for meaningful backtest.
    
    Args:
        total_bars: Total number of bars in dataset
        warmup_bars: Bars used for warmup
        min_backtest_bars: Minimum bars required (default 252 = ~1 year)
        
    Returns:
        (is_valid, message) - True if valid, False with explanation if not
    """
    remaining_bars = total_bars - warmup_bars
    
    if remaining_bars < min_backtest_bars:
        return False, (
            f"After {warmup_bars} warmup bars, only {remaining_bars} bars remain for backtesting. "
            f"Minimum required: {min_backtest_bars} bars. "
            f"Options: 1) Reduce max indicator period, 2) Extend data range, "
            f"3) Use --min-backtest-bars to override."
        )
    
    return True, f"Backtest data OK: {remaining_bars} bars after warmup"


def print_warmup_analysis(
    ticker: str,
    required_warmup: int,
    max_indicator: str,
    original_start: str,
    extended_start: str,
    available_warmup: int,
    backtest_bars: int
) -> None:
    """Print warmup analysis summary to console.
    
    Args:
        ticker: Asset ticker
        required_warmup: Bars required for warmup
        max_indicator: Name of indicator requiring most warmup (e.g., "ema_slow=250")
        original_start: User's intended start date
        extended_start: Extended start date for data fetch
        available_warmup: Actual bars available for warmup
        backtest_bars: Bars remaining for backtesting
    """
    warmup_ok = available_warmup >= required_warmup
    backtest_ok = backtest_bars >= 252
    
    warmup_status = "[OK]" if warmup_ok else "[X]"
    backtest_status = "[OK]" if backtest_ok else "[X]"
    
    print(f"\nWarmup Analysis for {ticker}:")
    print(f"  Required warmup: {required_warmup} bars (max indicator: {max_indicator})")
    print(f"  Extended start: {extended_start} (original: {original_start})")
    print(f"  Available warmup: {available_warmup} bars {warmup_status}")
    print(f"  Backtest data: {backtest_bars} bars {backtest_status}")


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
    
    # Parse optimization config if present
    opt_payload = payload.get("optimization", {})
    opt_config = OptimizationConfig(**opt_payload) if opt_payload else None
    
    return WorkflowConfig(
        data=DataConfig(**payload.get("data", {})),
        strategy=StrategyConfig(**strategy_payload),
        backtest=BacktestConfig(**payload.get("backtest", {})),
        grid=GridConfig(**payload.get("grid", {})),
        portfolio=portfolio_config,
        optimization=opt_config,
    )


# =============================================================================
# Registry - Single source of truth for optimized asset parameters
# =============================================================================

@dataclass
class RegistryEntry:
    """Registry entry for an optimized asset.
    
    Attributes:
        strategy: Strategy name (e.g., "ensemble_unconstrained")
        params: Optimized parameter dictionary
        last_optimized: Date of last optimization
        train_sharpe: Sharpe ratio on training data
        val_sharpe: Sharpe ratio on validation data
        test_sharpe: Sharpe ratio on test data
        stability_score: Sensitivity analysis stability score
        data_source: Data source used (yfinance/alphavantage)
        start_date: Data start date
    """
    strategy: str
    params: Dict[str, Any]
    last_optimized: Optional[str] = None
    train_sharpe: Optional[float] = None
    val_sharpe: Optional[float] = None
    test_sharpe: Optional[float] = None
    stability_score: Optional[float] = None
    data_source: str = "yfinance"
    start_date: str = "2018-01-01"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "strategy": self.strategy,
            "params": self.params,
            "last_optimized": self.last_optimized,
            "train_sharpe": self.train_sharpe,
            "val_sharpe": self.val_sharpe,
            "test_sharpe": self.test_sharpe,
            "stability_score": self.stability_score,
            "data_source": self.data_source,
            "start_date": self.start_date,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryEntry":
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "triple_ema"),
            params=data.get("params", {}),
            last_optimized=data.get("last_optimized"),
            train_sharpe=data.get("train_sharpe"),
            val_sharpe=data.get("val_sharpe"),
            test_sharpe=data.get("test_sharpe"),
            stability_score=data.get("stability_score"),
            data_source=data.get("data_source", "yfinance"),
            start_date=data.get("start_date", "2018-01-01"),
        )


def load_registry(path: Optional[Path] = None) -> Dict[str, RegistryEntry]:
    """Load asset registry from YAML file.
    
    Args:
        path: Path to registry file (default: PROJECT_ROOT/registry.yml)
        
    Returns:
        Dictionary mapping ticker -> RegistryEntry
    """
    registry_path = path or REGISTRY_FILE
    
    if not registry_path.exists():
        return {}
    
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    assets = data.get("assets", {})
    registry = {}
    
    for ticker, entry_data in assets.items():
        registry[ticker] = RegistryEntry.from_dict(entry_data)
    
    return registry


def save_registry(registry: Dict[str, RegistryEntry], path: Optional[Path] = None) -> None:
    """Save asset registry to YAML file.
    
    Args:
        registry: Dictionary mapping ticker -> RegistryEntry
        path: Path to registry file (default: PROJECT_ROOT/registry.yml)
    """
    registry_path = path or REGISTRY_FILE
    
    data = {
        "assets": {
            ticker: entry.to_dict() 
            for ticker, entry in registry.items()
        }
    }
    
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_registry_entry(ticker: str, path: Optional[Path] = None) -> Optional[RegistryEntry]:
    """Get registry entry for a specific ticker.
    
    Args:
        ticker: Asset ticker symbol
        path: Path to registry file (default: PROJECT_ROOT/registry.yml)
        
    Returns:
        RegistryEntry if found, None otherwise
    """
    registry = load_registry(path)
    return registry.get(ticker.upper())


def update_registry_entry(
    ticker: str,
    entry: RegistryEntry,
    path: Optional[Path] = None
) -> None:
    """Update or add a registry entry for a ticker.
    
    Args:
        ticker: Asset ticker symbol
        entry: RegistryEntry to save
        path: Path to registry file (default: PROJECT_ROOT/registry.yml)
    """
    registry = load_registry(path)
    registry[ticker.upper()] = entry
    save_registry(registry, path)


def get_asset_cache_path(ticker: str) -> Path:
    """Get the cache CSV path for an asset.
    
    Args:
        ticker: Asset ticker symbol
        
    Returns:
        Path to assets/{TICKER}/cache.csv
    """
    return ASSETS_DIR / ticker.upper() / "cache.csv"


def get_asset_results_dir(ticker: str) -> Path:
    """Get the results directory for an asset.
    
    Args:
        ticker: Asset ticker symbol
        
    Returns:
        Path to assets/{TICKER}/results/
    """
    results_dir = ASSETS_DIR / ticker.upper() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_template(template_name: str) -> WorkflowConfig:
    """Load a template configuration.
    
    Args:
        template_name: Template filename (e.g., "wide_ensemble_grid.yml")
        
    Returns:
        WorkflowConfig from template
        
    Raises:
        FileNotFoundError: If template doesn't exist
    """
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return load_config(template_path)


def build_config_from_registry(ticker: str, path: Optional[Path] = None) -> WorkflowConfig:
    """Build a WorkflowConfig from a registry entry.
    
    This allows using optimized parameters from registry.yml directly
    without needing to create a separate config file.
    
    Args:
        ticker: Asset ticker symbol
        path: Path to registry file (default: PROJECT_ROOT/registry.yml)
        
    Returns:
        WorkflowConfig built from registry entry
        
    Raises:
        ValueError: If ticker not found in registry
    """
    entry = get_registry_entry(ticker, path)
    if entry is None:
        registry_path = path or REGISTRY_FILE
        raise ValueError(
            f"Ticker '{ticker}' not found in registry ({registry_path}). "
            f"Run 'python main.py optimize --ticker {ticker} --save' first."
        )
    
    # Build WorkflowConfig from registry entry
    # Use defaults for backtest settings, but can be overridden
    return WorkflowConfig(
        data=DataConfig(
            ticker=ticker.upper(),
            start=entry.start_date,
            data_source=entry.data_source,
            cache_csv=None,  # Will be auto-resolved by load_prices()
        ),
        strategy=StrategyConfig(
            name=entry.strategy,
            params=entry.params,
            grid={},  # No grid needed for single backtest
        ),
        backtest=BacktestConfig(
            init_cash=100_000,
            fees=0.0004,
            slippage=0.0002,
            freq="D",
            train_ratio=0.6,
        ),
        grid=GridConfig(),
        portfolio=None,
        optimization=None,
    )

# Multi-Asset Portfolio Backtesting Implementation

## Overview

Extend the existing single-asset backtesting system to support multi-asset portfolios with RSI momentum filtering and top-K selection. The system will use vectorbt for portfolio construction, calculate RSI using vectorbt indicators, and maintain full backward compatibility.

## Implementation Files

### 1. `backtesting/data/multi_asset_loader.py`

**Purpose**: Load multiple assets into a single DataFrame with aligned dates.

**Key Functions**:

- `load_multi_asset_prices(tickers: List[str], config: DataConfig, force_download: bool = False) -> pd.DataFrame`
- Reuse existing `DataFetcher` for each ticker
- Load close prices for all assets
- Inner join on dates (only dates where all assets have data)
- Return DataFrame: columns = tickers, index = dates
- Handle missing data gracefully (drop dates with any missing values)

**Implementation Notes**:

- Iterate through tickers, use `DataFetcher` for each
- Extract `close()` series for each
- Use `pd.concat()` with `axis=1` and `join='inner'`
- Ensure GLD is included if specified in tickers list

### 2. `backtesting/strategies/rsi_filter_portfolio.py`

**Purpose**: Generate weight allocation matrix based on RSI ranking and ensemble signals.

**Key Classes/Functions**:

- `RSIFilterPortfolioStrategy` dataclass:
- Parameters: `rsi_period: int`, `rsi_threshold: float`, `top_k: int`
- `generate_weights(prices: pd.DataFrame, ensemble_entries: pd.DataFrame, ensemble_exits: pd.DataFrame) -> pd.DataFrame`
- Input: prices DataFrame (columns = assets), ensemble signals per asset
- Calculate RSI for each asset using `vbt.RSI.run(prices[asset], window=rsi_period).rsi`
- For each timestamp:
a. Rank assets by RSI (descending)
b. Filter: RSI > threshold
c. Select top K assets
d. Check ensemble entry signals for top K assets
e. Allocate 1/K weight to each qualifying asset
f. Remaining weight → 50% GLD + 50% Cash (implicit)
- Output: Weight DataFrame (same shape as prices + GLD column if not already present)

**Entry Rules (ALL must be true)**:

- Ensemble entry signal = True
- RSI > threshold
- Asset is in top K by RSI ranking

**Exit Rules (ANY triggers exit)**:

- Ensemble exit signal = True, OR
- RSI < threshold, OR
- Asset falls out of top K

**Implementation Notes**:

- Use `pd.DataFrame.rank()` for RSI ranking
- Handle edge cases: no qualifying assets → all to safe haven
- Position swaps handled automatically via weight changes
- Ensure GLD column exists in prices DataFrame

### 3. `backtesting/backtest/portfolio_builder.py`

**Purpose**: Construct vectorbt Portfolio from weight allocation matrix.

**Key Classes/Functions**:

- `PortfolioBuilder` class:
- `build_from_weights(prices: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig) -> vbt.Portfolio`
- Convert weight matrix to orders/signals for vectorbt
- Use `vbt.Portfolio.from_orders()` or `vbt.Portfolio.from_signals()` with size_type='percent'
- Handle rebalancing on triggers (weight changes = rebalancing events)
- Include fees, slippage, rebalancing costs from config
- Ensure GLD is in prices DataFrame

**Implementation Notes**:

- Research vectorbt best practice: `from_orders()` vs `from_signals()` for weight-based portfolios
- Use `size_type='percent'` or calculate absolute sizes from weights
- Handle cash allocation implicitly (unallocated = cash)
- Rebalancing occurs when weights change (trigger-based)

### 4. `backtesting/config.py` Extensions

**Purpose**: Add portfolio configuration classes.

**Additions**:

- `PortfolioConfig` dataclass:
- `tickers: List[str]` - List of asset tickers
- `safe_haven_ticker: str = "GLD"` - Safe haven asset (default GLD)
- `rsi_period: int = 14` - RSI calculation period (optimizable)
- `rsi_threshold: float = 50.0` - RSI filter threshold (optimizable)
- `top_k: int = 5` - Number of top assets to select (optimizable)

- Extend `WorkflowConfig`:
- `portfolio: Optional[PortfolioConfig] = None` - Optional portfolio config

**Implementation Notes**:

- Make portfolio config optional to maintain single-asset compatibility
- RSI period, threshold, and top_k should be optimizable in grid search

### 5. `backtesting/pipeline.py` Extensions

**Purpose**: Add portfolio backtest function.

**Additions**:

- `run_portfolio_backtest(cfg: WorkflowConfig, return_portfolios: bool = False) -> Dict[str, Any]`
- Load multi-asset prices using `load_multi_asset_prices()`
- Generate ensemble signals for each asset (reuse `EnsembleStrategy`)
- Create `RSIFilterPortfolioStrategy` instance
- Generate weight allocation matrix
- Build portfolio using `PortfolioBuilder`
- Compute metrics using existing `compute_metrics()` function
- Return metrics dictionary (similar structure to `run_single_backtest()`)

**Implementation Notes**:

- Reuse existing `EnsembleStrategy` for signal generation per asset
- Reuse existing `compute_metrics()` function
- Handle train/validation split if needed
- Maintain similar output structure to `run_single_backtest()`

## Key Implementation Details

### RSI Calculation

- Use `vbt.RSI.run(prices[asset], window=rsi_period).rsi` for each asset
- Standard RSI calculation (not on cumulative returns)
- Handle NaN values (RSI requires sufficient history)

### Weight Allocation Logic

- For each timestamp:

1. Calculate RSI for all assets
2. Rank by RSI (descending)
3. Filter: RSI > threshold
4. Select top K
5. Check ensemble entry signals for top K
6. Allocate 1/K to each qualifying asset
7. Remaining weight → 50% GLD + 50% Cash

### Rebalancing Triggers

- Entry signal fires → check RSI > threshold + in top K → enter
- Exit signal fires → exit → safe haven
- Asset falls out of top K → exit → safe haven
- RSI < threshold → exit → safe haven
- Ranking changes → rebalance if needed

### Vectorbt Portfolio Construction

- Research: Use `Portfolio.from_orders()` with weight-based orders or `Portfolio.from_signals()` with size_type='percent'
- Ensure GLD column exists in prices DataFrame
- Handle cash allocation implicitly
- Apply fees, slippage from config

### Edge Cases

- No qualifying assets → 100% safe haven (50% GLD + 50% Cash)
- Fewer than K assets qualify → allocate 1/K to each, remainder to safe haven
- GLD data missing → handle gracefully (skip safe haven or use alternative)
- Position swaps → handled via weight changes (automatic rebalancing)

## Testing Considerations

- Test with single asset (backward compatibility)
- Test with multiple assets
- Test edge cases (no qualifying assets, fewer than K qualify)
- Test rebalancing triggers
- Test with/without GLD data

## Dependencies

- vectorbt (existing)
- talib (existing, for ensemble signals)
- pandas (existing)
- numpy (existing)
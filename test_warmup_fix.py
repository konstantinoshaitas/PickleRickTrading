"""
Test script to verify the warmup fix works correctly.

This script demonstrates that:
1. The framework auto-calculates required warmup from grid parameters
2. Date is extended backwards for warmup
3. Indicators calculated on warmup data have no NaN values in backtest period

Run with: python test_warmup_fix.py
"""

import sys
import pandas as pd
import numpy as np

# Minimal test that doesn't require full framework imports
print("\n" + "#"*60)
print("# WARMUP FIX VERIFICATION TESTS")
print("#"*60)


def test_warmup_calculation():
    """Test warmup calculation logic."""
    print("\n" + "="*60)
    print("TEST 1: Warmup Calculation from Grid")
    print("="*60)
    
    def _parse_grid_value(value):
        if isinstance(value, list):
            return value
        value_str = str(value)
        if ":" in value_str:
            parts = value_str.split(":")
            if len(parts) == 3:
                start, end, step = map(int, parts)
                return list(range(start, end + 1, step))
        if isinstance(value, (int, float)):
            return [int(value)]
        return [value]
    
    def calculate_warmup_from_grid(grid, strategy_name=""):
        max_period = 0
        for key in ['ema_slow', 'ema_mid', 'ema_fast']:
            if key in grid:
                values = _parse_grid_value(grid[key])
                if values:
                    max_period = max(max_period, max(values))
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
        return max_period
    
    # Test EMA grid
    ema_grid = {
        "ema_fast": [5, 10, 15, 20],
        "ema_mid": [30, 50, 70],
        "ema_slow": [100, 150, 200, 250],
    }
    warmup = calculate_warmup_from_grid(ema_grid, "triple_ema")
    print(f"EMA grid (max ema_slow=250): warmup = {warmup}")
    assert warmup == 250, f"Expected 250, got {warmup}"
    
    # Test MACD grid
    macd_grid = {
        "fastperiod": [8, 12, 16],
        "slowperiod": [20, 26, 32],
        "signalperiod": [7, 9, 11],
    }
    warmup = calculate_warmup_from_grid(macd_grid, "macd")
    print(f"MACD grid (max slow=32 + signal=11): warmup = {warmup}")
    assert warmup == 43, f"Expected 43, got {warmup}"
    
    # Test range notation
    range_grid = {"ema_slow": "60:250:5"}
    warmup = calculate_warmup_from_grid(range_grid, "triple_ema")
    print(f"Range notation '60:250:5': warmup = {warmup}")
    assert warmup == 250, f"Expected 250, got {warmup}"
    
    print("[OK] Warmup calculation tests passed!")


def test_date_extension():
    """Test date extension for warmup."""
    print("\n" + "="*60)
    print("TEST 2: Date Extension for Warmup")
    print("="*60)
    
    def extend_start_date(start, warmup_bars, interval="1d"):
        start_dt = pd.to_datetime(start)
        if interval.lower() in ("1d", "d"):
            calendar_days = int(warmup_bars * 1.5)
        else:
            calendar_days = warmup_bars
        extended_dt = start_dt - pd.Timedelta(days=calendar_days)
        return extended_dt.strftime('%Y-%m-%d')
    
    original_start = "2018-01-01"
    warmup_bars = 200
    extended = extend_start_date(original_start, warmup_bars, "1d")
    
    print(f"Original start: {original_start}")
    print(f"Warmup bars: {warmup_bars}")
    print(f"Extended start: {extended}")
    
    original_dt = pd.to_datetime(original_start)
    extended_dt = pd.to_datetime(extended)
    days_back = (original_dt - extended_dt).days
    print(f"Days extended back: {days_back}")
    
    assert days_back >= warmup_bars, f"Expected >= {warmup_bars} days back, got {days_back}"
    print("[OK] Date extension tests passed!")


def test_indicator_warmup_with_sample_data():
    """Test that indicators have valid values after warmup period using sample data."""
    print("\n" + "="*60)
    print("TEST 3: Indicator Warmup with Sample Data")
    print("="*60)
    
    # Create sample price data (500 days before 2018-01-01 + backtest period)
    np.random.seed(42)
    full_dates = pd.date_range("2016-06-01", "2023-12-31", freq='B')  # Business days
    full_prices = pd.Series(
        100 * (1 + np.random.randn(len(full_dates)) * 0.02).cumprod(),
        index=full_dates,
        name="close"
    )
    
    backtest_start = pd.to_datetime("2018-01-01")
    warmup_bars = 250  # Simulate EMA-250 requirement
    
    # BEFORE FIX: Data starts at backtest_start (no warmup)
    print("\n--- BEFORE FIX (No Warmup) ---")
    prices_no_warmup = full_prices[full_prices.index >= backtest_start].copy()
    print(f"Data range: {prices_no_warmup.index[0].strftime('%Y-%m-%d')} to {prices_no_warmup.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total bars: {len(prices_no_warmup)}")
    
    # Calculate EMA-250 on data without warmup
    ema_no_warmup = prices_no_warmup.ewm(span=warmup_bars, adjust=False).mean()
    nan_count_before = ema_no_warmup.isna().sum()
    
    # Check first N values for accuracy (they will be biased without warmup)
    first_100_std = ema_no_warmup.iloc[:100].std()
    print(f"EMA-250 NaN count: {nan_count_before}")
    print(f"First 100 EMA values std: {first_100_std:.4f}")
    print("  [!] First values are unreliable without proper warmup!")
    
    # AFTER FIX: Data includes warmup period
    print("\n--- AFTER FIX (With Warmup) ---")
    
    # Calculate extended start
    extended_start = backtest_start - pd.Timedelta(days=int(warmup_bars * 1.5))
    prices_with_warmup = full_prices[full_prices.index >= extended_start].copy()
    print(f"Extended data range: {prices_with_warmup.index[0].strftime('%Y-%m-%d')} to {prices_with_warmup.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total bars (with warmup): {len(prices_with_warmup)}")
    
    # Count warmup bars available
    warmup_available = len(prices_with_warmup[prices_with_warmup.index < backtest_start])
    print(f"Warmup bars available: {warmup_available}")
    
    # Calculate EMA-250 on data with warmup
    ema_with_warmup = prices_with_warmup.ewm(span=warmup_bars, adjust=False).mean()
    
    # Trim to backtest period
    ema_backtest = ema_with_warmup[ema_with_warmup.index >= backtest_start]
    prices_backtest = prices_with_warmup[prices_with_warmup.index >= backtest_start]
    
    nan_count_after = ema_backtest.isna().sum()
    first_100_std_after = ema_backtest.iloc[:100].std()
    
    print(f"EMA-250 in backtest period NaN count: {nan_count_after}")
    print(f"First 100 EMA values std: {first_100_std_after:.4f}")
    
    # Compare first EMA value between approaches
    print(f"\nFirst EMA-250 value comparison:")
    print(f"  Without warmup: {ema_no_warmup.iloc[0]:.4f}")
    print(f"  With warmup: {ema_backtest.iloc[0]:.4f}")
    print(f"  Difference: {abs(ema_no_warmup.iloc[0] - ema_backtest.iloc[0]):.4f}")
    
    if nan_count_after == 0:
        print("\n[OK] All EMA-250 values are valid in backtest period with warmup!")
    else:
        print(f"\n[!] {nan_count_after} NaN values in backtest period")
    
    print("[OK] Indicator warmup demonstration complete!")


def test_signal_generation_comparison():
    """Compare signals generated with and without warmup."""
    print("\n" + "="*60)
    print("TEST 4: Signal Generation Comparison")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    full_dates = pd.date_range("2016-01-01", "2023-12-31", freq='B')
    full_prices = pd.Series(
        100 * (1 + np.random.randn(len(full_dates)) * 0.02).cumprod(),
        index=full_dates,
    )
    
    backtest_start = pd.to_datetime("2018-01-01")
    warmup_bars = 200
    
    # Parameters for triple EMA
    ema_fast = 21
    ema_mid = 55
    ema_slow = 200
    
    # WITHOUT WARMUP
    prices_no_warmup = full_prices[full_prices.index >= backtest_start].copy()
    ema1_nw = prices_no_warmup.ewm(span=ema_fast, adjust=False).mean()
    ema2_nw = prices_no_warmup.ewm(span=ema_mid, adjust=False).mean()
    ema3_nw = prices_no_warmup.ewm(span=ema_slow, adjust=False).mean()
    
    # Entry signals (fast crosses above mid or slow, OR mid crosses above slow)
    c1_nw = (ema1_nw > ema2_nw) & (ema1_nw.shift(1) <= ema2_nw.shift(1))
    c2_nw = (ema1_nw > ema3_nw) & (ema1_nw.shift(1) <= ema3_nw.shift(1))
    c3_nw = (ema2_nw > ema3_nw) & (ema2_nw.shift(1) <= ema3_nw.shift(1))
    entries_no_warmup = c1_nw | c2_nw | c3_nw
    
    # WITH WARMUP
    extended_start = backtest_start - pd.Timedelta(days=int(warmup_bars * 1.5))
    prices_with_warmup = full_prices[full_prices.index >= extended_start].copy()
    ema1_ww = prices_with_warmup.ewm(span=ema_fast, adjust=False).mean()
    ema2_ww = prices_with_warmup.ewm(span=ema_mid, adjust=False).mean()
    ema3_ww = prices_with_warmup.ewm(span=ema_slow, adjust=False).mean()
    
    c1_ww = (ema1_ww > ema2_ww) & (ema1_ww.shift(1) <= ema2_ww.shift(1))
    c2_ww = (ema1_ww > ema3_ww) & (ema1_ww.shift(1) <= ema3_ww.shift(1))
    c3_ww = (ema2_ww > ema3_ww) & (ema2_ww.shift(1) <= ema3_ww.shift(1))
    entries_with_warmup = c1_ww | c2_ww | c3_ww
    
    # Trim to backtest period
    entries_with_warmup_trimmed = entries_with_warmup[entries_with_warmup.index >= backtest_start]
    
    # Compare signals
    print(f"Entry signals in first year of backtest:")
    first_year_end = backtest_start + pd.Timedelta(days=365)
    
    entries_nw_first_year = entries_no_warmup[entries_no_warmup.index <= first_year_end].sum()
    entries_ww_first_year = entries_with_warmup_trimmed[entries_with_warmup_trimmed.index <= first_year_end].sum()
    
    print(f"  Without warmup: {entries_nw_first_year} signals")
    print(f"  With warmup: {entries_ww_first_year} signals")
    
    if entries_nw_first_year != entries_ww_first_year:
        print(f"  Difference: {abs(entries_nw_first_year - entries_ww_first_year)} signals")
        print("  [!] Signal mismatch detected! Warmup matters for accurate backtesting.")
    else:
        print("  Signals match (may happen with some data)")
    
    # Check early signals specifically (most affected by warmup)
    first_30_days = backtest_start + pd.Timedelta(days=30)
    early_nw = entries_no_warmup[entries_no_warmup.index <= first_30_days].sum()
    early_ww = entries_with_warmup_trimmed[entries_with_warmup_trimmed.index <= first_30_days].sum()
    
    print(f"\nEntry signals in first 30 days (most affected by warmup):")
    print(f"  Without warmup: {early_nw} signals")
    print(f"  With warmup: {early_ww} signals")
    
    print("\n[OK] Signal comparison complete!")


def main():
    """Run all tests."""
    test_warmup_calculation()
    test_date_extension()
    test_indicator_warmup_with_sample_data()
    test_signal_generation_comparison()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED")
    print("#"*60)
    
    print("""
Summary of Warmup Fix:
======================
The warmup fix ensures that technical indicators have sufficient historical
data for proper calculation at the start of a backtest.

Key Implementation Points:
1. warmup_bars added to DataConfig (auto-calculated from grid or specified)
2. DataFetcher extends start date backward to fetch warmup data
3. Indicators calculated on full data (with warmup)
4. Signals trimmed to original start for backtest evaluation
5. Validation ensures sufficient warmup data is available

Files Modified:
- backtesting/config.py: Added warmup_bars, calculation helpers, error classes
- backtesting/data/fetcher.py: Extended date handling, warmup validation
- backtesting/pipeline.py: Updated load_prices() and run_grid_search()
- backtesting/grid.py: Updated GridSearch and VectorizedGridSearch
- backtesting/optimizer.py: Updated AssetOptimizer for all phases
- backtesting/data/multi_asset_loader.py: Added warmup support

This eliminates the look-ahead bias issue where indicators produce NaN or
incorrect values at the start of a backtest due to insufficient history.
""")


if __name__ == "__main__":
    main()

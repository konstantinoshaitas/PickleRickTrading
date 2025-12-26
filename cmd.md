# CLI Commands


## Fetch Data
```bash
python main.py --config config/default.yml fetch --force
```

## Run Backtest
```bash
python main.py --config config/default.yml backtest
python main.py --config config/default.yml backtest --plot

# Examples
python main.py --config config/ETH_test.yml backtest --plot
python main.py --config config/ETH.yml backtest --plot
# Backtest GOOG using optimized parameters from registry.yml
python main.py backtest --ticker GOOG --plot

# Save plots to directory
python main.py backtest --ticker AAPL --plot --plot-dir plots/AAPL

# Force refresh data
python main.py backtest --ticker ETH --plot --refresh 
```

## Grid Search
```bash
python main.py --config config/default.yml grid --top 3 --output data/grid_results.parquet
python main.py --config config/sensitivity.yml grid --top 3 --output data/BTC_grid_sensitivity_results.parquet

python main.py --config config/wide.yml grid --n-jobs 12 --top 3 --batch-size 20000 --min-trades 2.3 --output data/SOL_ensemble.parquet
python main.py --config config/wide_macd.yml grid --batch-size 5000 --min-trades 1.5 --output data/ETH_macd.parquet
python main.py --config config/wide_ema.yml grid --batch-size 5000 --min-trades 1.5 --output data/ETH_ema.parquet

# Portfolio grid
python main.py --config config/portfolio_example.yml portfolio-grid --top 3 --output data/portfolio_grid_results.parquet
```

---

## NEW: 3-Phase Optimization Pipeline

The `optimize` command automates the entire parameter selection process:

```bash
# Optimize a single asset (uses templates/wide_ensemble_grid.yml)
python main.py optimize --ticker BTC --strategy ensemble_unconstrained

# Optimize and save to registry
python main.py optimize --ticker ETH --strategy ensemble_unconstrained --save

# With custom ratios / custom config
python main.py optimize --ticker TQQQ --template wide_ema_grid.yml --strategy triple_ema_unconstrained `
    --train-ratio 0.6 --val-ratio 0.15 --test-ratio 0.25 `
    --transfer-threshold 1.2 --top-percent 0.3 --save


```

The pipeline:
1. **Phase 1 (TRAIN)**: Wide grid search → Keep top 1% by Sharpe
2. **Phase 2 (VAL)**: Filter overfit parameters (transfer ratio ≥ 0.6)
3. **Phase 3 (TEST)**: Sensitivity analysis → Select most stable parameters

## View Registry

```bash
# Show all optimized assets
python main.py registry

# Show specific asset details
python main.py registry --ticker GOOG
```

---

## Notes

Results are saved in Parquet format (much more efficient for large datasets). To read:
```python
import pandas as pd
df = pd.read_parquet('data/grid_results.parquet')
```

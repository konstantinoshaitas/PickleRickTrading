# CLI Commands

## Fetch Data
```bash
python main.py --config config/default.yml fetch --force
```

## Run Backtest
```bash
python main.py --config config/default.yml backtest
python main.py --config config/default.yml backtest --plot
python main.py backtest --plot --plot-dir plots/


```

## Grid Search
```bash
python main.py --config config/default.yml grid --top 3 --output data/grid_results.parquet
python main.py --config config/sensitivity.yml grid --top 3 --output data/QQQ_grid_sensitivity_results.parquet
python main.py --config config/wide.yml grid --n-jobs 8 --top 3 --output data/GOOG_grid_wide_results.parquet
```

Note: Results are saved in Parquet format (much more efficient for large datasets). To read:
```python
import pandas as pd
df = pd.read_parquet('data/grid_results.parquet')
```

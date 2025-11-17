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
python main.py --config config/default.yml grid --top 3 --output data/grid_results.csv
python main.py --config config/sensitivity.yml grid --top 3 --output data/grid_sensitivity_results.csv
```

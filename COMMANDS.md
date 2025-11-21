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

## Git Operations

### Switch to grid_optimise branch
```bash
git checkout grid_optimise
```

### Update local branches from remote
```bash
git fetch origin
```

### Option 1: Merge main into grid_optimise (brings main's changes into your branch)
```bash
# Make sure you're on grid_optimise
git checkout grid_optimise

# Fetch latest changes
git fetch origin

# Merge main into grid_optimise
git merge origin/main
```

### Option 2: Rebase grid_optimise onto main (replays your commits on top of main)
```bash
# Make sure you're on grid_optimise
git checkout grid_optimise

# Fetch latest changes
git fetch origin

# Rebase your branch onto main
git rebase origin/main
```

### Push your changes after merge/rebase
```bash
# After merge (normal push)
git push origin grid_optimise

# After rebase (force push - be careful!)
git push --force-with-lease origin grid_optimise
```

**Note:** Use `--force-with-lease` instead of `--force` for safety. It prevents overwriting remote changes you don't have locally.

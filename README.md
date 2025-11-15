# PickleRick Trading 

This is a trimmed-down Python scaffold that mirrors the Triple EMA notebook (`Template-3EMA.ipynb`) workflow. Everything lives in a handful of modules so you can read, tweak, and extend the workflow quickly.

## Layout

```
PickleRickTrading/
|-- backtesting/
|   |-- __init__.py        # lightweight exports
|   |-- backtest/engine.py # vectorbt wrapper
|   |-- config.py          # dataclasses + YAML loader
|   |-- data/fetcher.py    # yfinance/local CSV loader + split helper
|   |-- grid.py            # brute-force grid search
|   |-- metrics.py         # basic Sharpe/return/drawdown helpers
|   `-- strategies/        # Triple EMA implementation
|-- config/default.yml     # edit tickers/params/ranges here
|-- main.py                # CLI entry point (load -> fetch -> grid -> report)
`-- Template-3EMA.ipynb    # reference notebook (unchanged)
```

## Workflow

1. Install deps: `pip install -r requirements.txt`.
2. Update `config/default.yml` with your ticker/date/EMA settings.
3. Use the CLI subcommands:
   - `python main.py fetch` – download/cache the OHLCV data.
   - `python main.py backtest` – run the configured strategy over the train and validation period and compare with a buy & hold baseline.
   - `python main.py grid` – brute-force the EMA ranges defined in the YAML and (optionally) save results to `data/grid_results.csv`.
   - Add `--config path/to.yml` to point at a different configuration file. Use `--refresh` on `backtest`/`grid` if you need to refetch data, and `--force` on `fetch` to ignore cached CSVs.

Each subcommand uses the same YAML file, so you can mix/match: fetch once, grid-search overnight, then backtest the best params manually or via your IDE.

## Configuration Knobs

- **data** – ticker, start/end dates, interval, optional `local_csv`, optional `cache_csv`.
- **strategy** – `name`, default `params`, and `grid` ranges (only Triple EMA is wired up today, but the registry is ready for more strategies).
- **backtest** – cash, fees/slippage, frequency, and train/validation ratio.
- **grid** – metric used to sort (`sharpe_ratio`) and how many rows to display/save.

## Extending

- Add strategies inside `backtesting/strategies/` and register them in `StrategyFactory`; they instantly become available to CLI subcommands.
- Point `local_csv` at curated datasets or expand the fetcher to include cleaning steps.
- Notebook/IDE workflows can import the same functions from `backtesting.pipeline` to execute individual stages (fetch only, backtest only, etc.) without notebooks.



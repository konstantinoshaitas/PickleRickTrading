"""CLI with modular subcommands: fetch, backtest, grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.config import WorkflowConfig, load_config
from backtesting.pipeline import (
    load_prices,
    run_grid_search,
    run_single_backtest,
    save_grid_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtesting workflow CLI")
    parser.add_argument("--config", default="config/default.yml", help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    
    fetch = sub.add_parser("fetch", help="Download data defined in config")
    fetch.add_argument("--force", action="store_true", help="Ignore cached CSV and refetch")
    
    backtest = sub.add_parser("backtest", help="Run single backtest with current params")
    backtest.add_argument("--refresh", action="store_true", help="Refetch data before running")
    
    grid = sub.add_parser("grid", help="Run grid search on training window")
    grid.add_argument("--refresh", action="store_true", help="Refetch data before running")
    grid.add_argument("--top", type=int, default=5, help="Rows to display from sorted results")
    grid.add_argument("--output", type=Path, default=Path("data/grid_results.csv"), help="CSV path for results")
    
    return parser


def cmd_fetch(cfg: WorkflowConfig, force: bool):
    close, ohlcv = load_prices(cfg, force_download=force)
    print(f"Fetched {len(ohlcv)} rows for {cfg.data.ticker} ({ohlcv.index.min().date()} -> {ohlcv.index.max().date()})")
    if cfg.data.cache_csv:
        print(f"Cached at: {cfg.data.cache_csv}")
    print(f"Latest close: {close.iloc[-1]:.2f}")


def cmd_backtest(cfg: WorkflowConfig, refresh: bool):
    close, _ = load_prices(cfg, force_download=refresh)
    metrics = run_single_backtest(cfg, close)
    
    train_start, train_end = metrics["train_window"]
    print(f"\nTrain metrics ({train_start.date()} -> {train_end.date()})")
    _print_metrics(metrics["train"])
    
    if "validation" in metrics:
        val_start, val_end = metrics["validation_window"]
        print(f"\nValidation metrics ({val_start.date()} -> {val_end.date()})")
        _print_metrics(metrics["validation"])
        print(f"\nBuy & Hold baseline ({val_start.date()} -> {val_end.date()})")
        _print_metrics(metrics["benchmark"])


def cmd_grid(cfg: WorkflowConfig, refresh: bool, top: int, output: Path):
    close, _ = load_prices(cfg, force_download=refresh)
    search = run_grid_search(cfg, close)
    df = pd.DataFrame(search.results)
    if df.empty:
        print("Grid search produced no valid results.")
        return
    df = df.sort_values(cfg.grid.metric, ascending=False)
    print(df.head(top).to_string(index=False))
    if output:
        save_grid_results(search, output)
        print(f"\nSaved full results to {output}")


def _print_metrics(metrics: dict):
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    
    if args.command == "fetch":
        cmd_fetch(cfg, force=args.force)
    elif args.command == "backtest":
        cmd_backtest(cfg, refresh=args.refresh)
    elif args.command == "grid":
        cmd_grid(cfg, refresh=args.refresh, top=args.top, output=args.output)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

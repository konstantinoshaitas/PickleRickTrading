"""CLI with modular subcommands: fetch, backtest, grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from backtesting.config import WorkflowConfig, load_config
from backtesting.pipeline import (
    load_prices,
    run_grid_search,
    run_single_backtest,
    save_grid_results,
)
from backtesting.visualization import (
    plot_drawdowns,
    plot_equity_curves,
    plot_signals,
    plot_trade_returns,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtesting workflow CLI")
    parser.add_argument("--config", default="config/default.yml", help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    
    fetch = sub.add_parser("fetch", help="Download data defined in config")
    fetch.add_argument("--force", action="store_true", help="Ignore cached CSV and refetch")
    
    backtest = sub.add_parser("backtest", help="Run single backtest with current params")
    backtest.add_argument("--refresh", action="store_true", help="Refetch data before running")
    backtest.add_argument("--plot", action="store_true", help="Generate visualization plots")
    backtest.add_argument("--plot-dir", type=Path, default=None, help="Directory to save plots (default: display interactively)")
    
    grid = sub.add_parser("grid", help="Run grid search on training window")
    grid.add_argument("--refresh", action="store_true", help="Refetch data before running")
    grid.add_argument("--top", type=int, default=5, help="Rows to display from sorted results")
    grid.add_argument("--output", type=Path, default=Path("data/grid_results.parquet"), help="Parquet file path for results (default: .parquet)")
    grid.add_argument("--n-jobs", type=int, default=None, help="Number of parallel processes (default: CPU count - 1)")
    
    return parser


def cmd_fetch(cfg: WorkflowConfig, force: bool):
    close, ohlcv = load_prices(cfg, force_download=force)
    print(f"Fetched {len(ohlcv)} rows for {cfg.data.ticker} ({ohlcv.index.min().date()} -> {ohlcv.index.max().date()})")
    if cfg.data.cache_csv:
        print(f"Cached at: {cfg.data.cache_csv}")
    print(f"Latest close: {close.iloc[-1]:.2f}")


def cmd_backtest(cfg: WorkflowConfig, refresh: bool, plot: bool, plot_dir: Optional[Path]):
    # Print configuration parameters
    print("=" * 70)
    print("BACKTEST CONFIGURATION")
    print("=" * 70)
    print(f"Strategy: {cfg.strategy.name}")
    print(f"Strategy Parameters:")
    for key, value in cfg.strategy.params.items():
        print(f"  {key}: {value}")
    print(f"\nBacktest Settings:")
    print(f"  Initial Cash: ${cfg.backtest.init_cash:,.0f}")
    print(f"  Fees: {cfg.backtest.fees:.4f} ({cfg.backtest.fees*100:.2f}%)")
    print(f"  Slippage: {cfg.backtest.slippage:.4f} ({cfg.backtest.slippage*100:.2f}%)")
    print(f"  Frequency: {cfg.backtest.freq}")
    print(f"  Train Ratio: {cfg.backtest.train_ratio:.1%}")
    print(f"\nData Settings:")
    print(f"  Ticker: {cfg.data.ticker}")
    print(f"  Start Date: {cfg.data.start}")
    print(f"  End Date: {cfg.data.end or 'Latest'}")
    print(f"  Interval: {cfg.data.interval}")
    print("=" * 70)
    print()
    
    close, _ = load_prices(cfg, force_download=refresh)
    metrics = run_single_backtest(cfg, close, return_portfolios=plot)
    
    train_start, train_end = metrics["train_window"]
    print(f"\nTrain metrics ({train_start.date()} -> {train_end.date()})")
    _print_metrics(metrics["train"])
    
    if "validation" in metrics:
        val_start, val_end = metrics["validation_window"]
        print(f"\nValidation metrics ({val_start.date()} -> {val_end.date()})")
        _print_metrics(metrics["validation"])
        print(f"\nBuy & Hold baseline ({val_start.date()} -> {val_end.date()})")
        _print_metrics(metrics["benchmark"])
    
    # Generate plots if requested
    if plot:
        if plot_dir:
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nGenerating plots in {plot_dir}...")
        else:
            print("\nGenerating plots (displaying interactively)...")
        
        # 1. Equity curves (train/validation comparison)
        if "val_portfolio" in metrics:
            portfolios_dict = {
                "Train": metrics["train_portfolio"],
                "Validation": metrics["val_portfolio"],
            }
            close_dict = {
                "Train": metrics["train_close"],
                "Validation": metrics["val_close"],
            }
            save_path = (plot_dir / "equity_curves.png") if plot_dir else None
            plot_equity_curves(
                portfolios_dict,
                close_dict,
                title=f"Equity Curves - {cfg.strategy.name}",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved equity_curves.png")
        
        # 2. Drawdowns (validation set)
        if "val_portfolio" in metrics:
            save_path = (plot_dir / "drawdowns.png") if plot_dir else None
            plot_drawdowns(
                metrics["val_portfolio"],
                metrics["val_close"],
                cfg.backtest.freq,
                title=f"Drawdowns - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved drawdowns.png")
        
        # 3. Signals (validation set)
        if "val_entries" in metrics:
            save_path = (plot_dir / "signals.png") if plot_dir else None
            plot_signals(
                metrics["val_close"],
                metrics["val_entries"],
                metrics["val_exits"],
                title=f"Price & Signals - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved signals.png")
        
        # 4. Trade-by-trade returns (validation set)
        if "val_portfolio" in metrics:
            save_path = (plot_dir / "trade_returns.png") if plot_dir else None
            plot_trade_returns(
                metrics["val_portfolio"],
                title=f"Per-Trade Returns - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved trade_returns.png")
        
        if not plot_dir:
            print("\nPlots displayed. Close windows to continue.")
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                pass


def cmd_grid(cfg: WorkflowConfig, refresh: bool, top: int, output: Path, n_jobs: Optional[int] = None):
    close, _ = load_prices(cfg, force_download=refresh)
    search = run_grid_search(cfg, close, n_jobs=n_jobs)
    df = pd.DataFrame(search.results)
    if df.empty:
        print("Grid search produced no valid results.")
        return
    
    # Sort by configured metric (default: sharpe_ratio)
    sort_metric = cfg.grid.metric
    if sort_metric in df.columns:
        df = df.sort_values(sort_metric, ascending=False)
    else:
        # Fallback to first available metric
        metric_cols = [c for c in df.columns if c not in ['ema_fast', 'ema_mid', 'ema_slow', 
                                                          'fastperiod', 'slowperiod', 'signalperiod']]
        if metric_cols:
            sort_metric = metric_cols[0]
            df = df.sort_values(sort_metric, ascending=False)
    
        print(df.head(top).to_string(index=False))
        if output:
            saved_path = save_grid_results(search, output, sort_by=sort_metric)
            print(f"\nSaved full results to {saved_path} (sorted by {sort_metric})")


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
        cmd_backtest(cfg, refresh=args.refresh, plot=args.plot, plot_dir=args.plot_dir)
    elif args.command == "grid":
        cmd_grid(cfg, refresh=args.refresh, top=args.top, output=args.output, n_jobs=args.n_jobs)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

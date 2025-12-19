"""CLI with modular subcommands: fetch, backtest, grid, optimize."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from backtesting.config import WorkflowConfig, load_config, load_registry, get_registry_entry
from backtesting.optimizer import AssetOptimizer, optimize_asset
from backtesting.pipeline import (
    load_prices,
    run_grid_search,
    run_portfolio_backtest,
    run_portfolio_grid_search,
    run_single_backtest,
    save_grid_results,
)
from backtesting.visualization import (
    plot_cumulative_equity,
    plot_drawdowns,
    plot_equity_curves,
    plot_full_sample_equity,
    plot_rolling_sharpe,
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
    backtest.add_argument("--ticker", type=str, default=None, help="Load optimized params from registry.yml (alternative to --config)")
    backtest.add_argument("--refresh", action="store_true", help="Refetch data before running")
    backtest.add_argument("--plot", action="store_true", help="Generate visualization plots")
    backtest.add_argument("--plot-dir", type=Path, default=None, help="Directory to save plots (default: display interactively)")
    
    grid = sub.add_parser("grid", help="Run grid search on training window")
    grid.add_argument("--refresh", action="store_true", help="Refetch data before running")
    grid.add_argument("--top", type=int, default=5, help="Rows to display from sorted results")
    grid.add_argument("--output", type=Path, default=None, help="Parquet file path for results (default: assets/{TICKER}/results/grid_results.parquet)")
    grid.add_argument("--n-jobs", type=int, default=None, help="Number of parallel processes (default: CPU count - 1)")
    grid.add_argument("--min-trades", type=float, default=0.5, help="Minimum trades per year filter (default: 0.5)")
    grid.add_argument("--batch-size", type=int, default=5000, help="Batch size for vectorized grid search (default: 5000)")
    
    portfolio = sub.add_parser("portfolio", help="Run multi-asset portfolio backtest")
    portfolio.add_argument("--plot", action="store_true", help="Generate visualization plots")
    portfolio.add_argument("--plot-dir", type=Path, default=None, help="Directory to save plots (default: display interactively)")
    
    p_grid = sub.add_parser("portfolio-grid", help="Run portfolio-level grid search")
    p_grid.add_argument("--top", type=int, default=5, help="Rows to display from sorted results")
    p_grid.add_argument("--output", type=Path, default=None, help="Parquet file path for results (default: assets/PORTFOLIO/results/portfolio_grid_results.parquet)")
    
    # Optimize command - 3-phase automated optimization
    optimize = sub.add_parser("optimize", help="Run 3-phase optimization pipeline for single asset")
    optimize.add_argument("--ticker", type=str, required=True, help="Asset ticker symbol (e.g., GOOG, BTC)")
    optimize.add_argument("--strategy", type=str, default="ensemble_unconstrained", 
                         help="Strategy name (default: ensemble_unconstrained)")
    optimize.add_argument("--template", type=str, default="wide_ensemble_grid.yml",
                         help="Template config for grid ranges (default: wide_ensemble_grid.yml)")
    optimize.add_argument("--save", action="store_true", help="Save best result to registry.yml")
    optimize.add_argument("--train-ratio", type=float, default=0.60, help="Train data ratio (default: 0.60)")
    optimize.add_argument("--val-ratio", type=float, default=0.20, help="Validation data ratio (default: 0.20)")
    optimize.add_argument("--test-ratio", type=float, default=0.20, help="Test data ratio (default: 0.20)")
    optimize.add_argument("--top-percent", type=float, default=0.01, help="Top percent to keep from grid search (default: 0.01 = 1%%)")
    optimize.add_argument("--transfer-threshold", type=float, default=0.6, help="Min transfer ratio (default: 0.6)")
    optimize.add_argument("--final-candidates", type=int, default=3, help="Number of final candidates (default: 3)")
    optimize.add_argument("--n-jobs", type=int, default=None, help="Number of parallel processes for phases 2 & 3 (default: CPU count - 1)")
    
    # Registry command - view registry contents
    registry = sub.add_parser("registry", help="View optimized asset registry")
    registry.add_argument("--ticker", type=str, default=None, help="Show specific ticker (default: show all)")

    return parser


def cmd_fetch(cfg: WorkflowConfig, force: bool):
    close, ohlcv = load_prices(cfg, force_download=force)
    print(f"Fetched {len(ohlcv)} rows for {cfg.data.ticker} ({ohlcv.index.min().date()} -> {ohlcv.index.max().date()})")
    if cfg.data.cache_csv:
        print(f"Cached at: {cfg.data.cache_csv}")
    print(f"Latest close: {close.iloc[-1]:.2f}")


def cmd_backtest(cfg: WorkflowConfig, refresh: bool, plot: bool, plot_dir: Optional[Path], from_registry: bool = False):
    # Print configuration parameters
    print("=" * 70)
    print("BACKTEST CONFIGURATION")
    if from_registry:
        print("(Loaded from registry.yml)")
    print("=" * 70)
    
    # Show registry metrics if available
    if from_registry:
        entry = get_registry_entry(cfg.data.ticker)
        if entry:
            print(f"Registry Info:")
            if entry.last_optimized:
                print(f"  Last Optimized: {entry.last_optimized}")
            if entry.train_sharpe is not None:
                print(f"  Train Sharpe: {entry.train_sharpe:.4f}")
            if entry.val_sharpe is not None:
                print(f"  Val Sharpe: {entry.val_sharpe:.4f}")
            if entry.test_sharpe is not None:
                print(f"  Test Sharpe: {entry.test_sharpe:.4f}")
            if entry.stability_score is not None:
                print(f"  Stability: {entry.stability_score:.4f}")
            print()
    
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
        
        # 1. Full Sample Equity Curves (with buy & hold comparison)
        if "val_portfolio" in metrics and "train_portfolio" in metrics:
            # Combine train and val close for full sample
            full_close = pd.concat([metrics["train_close"], metrics["val_close"]])
            
            save_path = (plot_dir / "full_sample_equity.png") if plot_dir else None
            plot_full_sample_equity(
                train_portfolio=metrics["train_portfolio"],
                val_portfolio=metrics["val_portfolio"],
                train_close=metrics["train_close"],
                val_close=metrics["val_close"],
                full_close=full_close,
                strategy_name=cfg.strategy.name,
                strategy_params=cfg.strategy.params,
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved full_sample_equity.png")
        
        # 2. Equity curves (train/validation comparison - original view)
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
        
        # 3. Drawdowns (validation set)
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
        
        # 4. Signals (validation set)
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
        
        # 5. Trade-by-trade returns (validation set)
        if "val_portfolio" in metrics:
            save_path = (plot_dir / "trade_returns.png") if plot_dir else None
            plot_trade_returns(
                metrics["val_portfolio"],
                title=f"Per-Trade Returns - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved trade_returns.png")
        
        # 6. Rolling Sharpe ratio (validation set)
        if "val_portfolio" in metrics:
            save_path = (plot_dir / "rolling_sharpe_val.png") if plot_dir else None
            plot_rolling_sharpe(
                metrics["val_portfolio"],
                metrics["val_close"],
                cfg.backtest.freq,
                title=f"Rolling Sharpe - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved rolling_sharpe_val.png")
        
        # 7. Rolling Sharpe ratio (train set)
        if "train_portfolio" in metrics:
            save_path = (plot_dir / "rolling_sharpe_train.png") if plot_dir else None
            plot_rolling_sharpe(
                metrics["train_portfolio"],
                metrics["train_close"],
                cfg.backtest.freq,
                title=f"Rolling Sharpe - {cfg.strategy.name} (Train)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved rolling_sharpe_train.png")
        
        # 8. Cumulative equity per trade (validation set)
        if "val_portfolio" in metrics:
            save_path = (plot_dir / "cumulative_equity.png") if plot_dir else None
            plot_cumulative_equity(
                metrics["val_portfolio"],
                title=f"Cumulative Equity - {cfg.strategy.name} (Validation)",
                save_path=save_path,
            )
            if plot_dir:
                print("  ✓ Saved cumulative_equity.png")
        
        if not plot_dir:
            print("\nPlots displayed. Close windows to continue.")
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                pass


def cmd_grid(cfg: WorkflowConfig, refresh: bool, top: int, output: Optional[Path], n_jobs: Optional[int] = None, min_trades: float = 0.5, batch_size: int = 5000):
    from backtesting.config import get_asset_results_dir
    from datetime import datetime
    
    close, _ = load_prices(cfg, force_download=refresh)
    search = run_grid_search(cfg, close, n_jobs=n_jobs, min_trades_per_year=min_trades, batch_size=batch_size)
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
    
    print(df.head(top))
    
    # Auto-resolve output path if not provided
    if output is None:
        results_dir = get_asset_results_dir(cfg.data.ticker)
        timestamp = datetime.now().strftime("%Y%m%d")
        strategy_name = cfg.strategy.name
        output = results_dir / f"{cfg.data.ticker}_{strategy_name}_grid_{timestamp}.parquet"
    
    saved_path = save_grid_results(search, output, sort_by=sort_metric)
    print(f"\nSaved full results to {saved_path} (sorted by {sort_metric})")


def cmd_portfolio(cfg: WorkflowConfig, plot: bool, plot_dir: Optional[Path]):
    """Run multi-asset portfolio backtest."""
    print("=" * 70)
    print("MULTI-ASSET PORTFOLIO BACKTEST")
    print("=" * 70)
    
    if not cfg.portfolio:
        print("Error: Portfolio configuration not found in config file.")
        print("Please add a 'portfolio' section with tickers, rsi params, etc.")
        return
        
    print(f"Tickers: {', '.join(cfg.portfolio.tickers)}")
    print(f"Safe Haven: {cfg.portfolio.safe_haven_ticker}")
    print(f"RSI Params: Period={cfg.portfolio.rsi_period}, Threshold={cfg.portfolio.rsi_threshold}")
    print(f"Top K: {cfg.portfolio.top_k}")
    print(f"\nStrategy: {cfg.strategy.name}")
    print("=" * 70)
    print()
    
    try:
        metrics = run_portfolio_backtest(cfg, return_portfolios=plot)
        
        train_start, train_end = metrics["train_window"]
        print(f"\nTrain metrics ({train_start.date()} -> {train_end.date()})")
        _print_metrics(metrics["train"])
        
        if "validation" in metrics:
            val_start, val_end = metrics["validation_window"]
            print(f"\nValidation metrics ({val_start.date()} -> {val_end.date()})")
            _print_metrics(metrics["validation"])
            print(f"\nBenchmark (Equal Weight) ({val_start.date()} -> {val_end.date()})")
            _print_metrics(metrics["benchmark"])
        
        # Generate plots if requested
        if plot:
            if plot_dir:
                plot_dir.mkdir(parents=True, exist_ok=True)
                print(f"\nGenerating plots in {plot_dir}...")
            else:
                print("\nGenerating plots (displaying interactively)...")
            
            # Plot equity curves (Train vs Validation)
            if "val_portfolio" in metrics:
                # Create pseudo-close prices (normalized portfolio value) for plotting
                train_value = metrics["train_portfolio"].value()
                val_value = metrics["val_portfolio"].value()
                
                portfolios_dict = {
                    "Train": metrics["train_portfolio"],
                    "Validation": metrics["val_portfolio"],
                }
                # Use portfolio values as "close" for plotting relative performance
                close_dict = {
                    "Train": train_value,
                    "Validation": val_value,
                }
                
                save_path = (plot_dir / "portfolio_equity.png") if plot_dir else None
                plot_equity_curves(
                    portfolios_dict,
                    close_dict,
                    title=f"Portfolio Equity - Top {cfg.portfolio.top_k} RSI Strategy",
                    save_path=save_path,
                )
                if plot_dir:
                    print("  ✓ Saved portfolio_equity.png")
                
                # Plot drawdowns (Validation)
                save_path = (plot_dir / "portfolio_drawdowns.png") if plot_dir else None
                plot_drawdowns(
                    metrics["val_portfolio"],
                    val_value,
                    cfg.backtest.freq,
                    title="Portfolio Drawdowns (Validation)",
                    save_path=save_path,
                )
                if plot_dir:
                    print("  ✓ Saved portfolio_drawdowns.png")
            
            if not plot_dir:
                print("\nPlots displayed. Close windows to continue.")
                try:
                    import matplotlib.pyplot as plt
                    plt.show()
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"\nError running portfolio backtest: {e}")
        import traceback
        traceback.print_exc()


def cmd_portfolio_grid(cfg: WorkflowConfig, top: int, output: Optional[Path]):
    """Run portfolio-level grid search."""
    from backtesting.config import get_asset_results_dir
    from datetime import datetime
    
    print("=" * 70)
    print("PORTFOLIO GRID SEARCH")
    print("=" * 70)
    
    if not cfg.portfolio:
        print("Error: Portfolio config required.")
        return

    if not cfg.portfolio.grid:
        print("Error: No portfolio grid defined in config (portfolio.grid).")
        return

    try:
        results = run_portfolio_grid_search(cfg)
        if not results:
            print("No results found.")
            return

        df = pd.DataFrame(results)
        
        # Determine sort metric
        sort_metric = cfg.grid.metric if cfg.grid.metric in df.columns else "sharpe_ratio"
        if sort_metric not in df.columns:
             # Fallback to first numeric col
             numerics = df.select_dtypes(include='number').columns
             if len(numerics) > 0:
                 sort_metric = numerics[0]

        df = df.sort_values(sort_metric, ascending=False)
        
        print(f"\nTop {top} Results (sorted by {sort_metric}):")
        print(df.head(top))
        
        # Auto-resolve output path if not provided
        if output is None:
            results_dir = get_asset_results_dir("PORTFOLIO")
            timestamp = datetime.now().strftime("%Y%m%d")
            output = results_dir / f"portfolio_grid_results_{timestamp}.parquet"
        
        saved_path = save_grid_results(results, output, sort_by=sort_metric)
        print(f"\nSaved full results to {saved_path}")

    except Exception as e:
        print(f"\nError running portfolio grid search: {e}")
        import traceback
        traceback.print_exc()


def _print_metrics(metrics: dict):
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def cmd_optimize(
    ticker: str,
    strategy: str,
    template: str,
    save: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    top_percent: float,
    transfer_threshold: float,
    final_candidates: int,
    n_jobs: Optional[int] = None,
):
    """Run 3-phase optimization pipeline for a single asset."""
    from backtesting.config import OptimizationConfig
    
    # Create optimization config from CLI args
    opt_config = OptimizationConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_percent_phase1=top_percent,
        transfer_ratio_threshold=transfer_threshold,
        final_candidates=final_candidates,
    )
    
    # Run optimization
    optimizer = AssetOptimizer(
        ticker=ticker,
        strategy=strategy,
        template=template,
        opt_config=opt_config,
        verbose=True,
        n_jobs=n_jobs,
    )
    
    try:
        result = optimizer.run()
        
        # Print detailed results for top candidates
        if result.candidates:
            print("\n" + "=" * 70)
            print("TOP CANDIDATES")
            print("=" * 70)
            
            for i, candidate in enumerate(result.candidates, 1):
                print(f"\n--- Candidate {i} ---")
                print(f"Parameters: {candidate.params}")
                print(f"Train Sharpe:  {candidate.train_sharpe:.4f}" if candidate.train_sharpe else "")
                print(f"Val Sharpe:    {candidate.val_sharpe:.4f}" if candidate.val_sharpe else "")
                print(f"Test Sharpe:   {candidate.test_sharpe:.4f}" if candidate.test_sharpe else "")
                print(f"Transfer Ratio: {candidate.transfer_ratio:.4f}" if candidate.transfer_ratio else "")
                print(f"Stability:     {candidate.stability_score:.4f}" if candidate.stability_score else "")
                print(f"Consistency:   {candidate.consistency:.4f}" if candidate.consistency else "")
                print(f"COMPOSITE:     {candidate.composite_score:.4f}" if candidate.composite_score else "")
                
                if candidate.train_return is not None:
                    print(f"\nReturns: Train={candidate.train_return:.2%}, Val={candidate.val_return:.2%}, Test={candidate.test_return:.2%}")
                if candidate.train_max_dd is not None:
                    print(f"Max DD:  Train={candidate.train_max_dd:.2%}, Val={candidate.val_max_dd:.2%}, Test={candidate.test_max_dd:.2%}")
        
        # Save to registry if requested
        if save and result.best_candidate:
            optimizer.save_to_registry()
            print(f"\n✓ Saved best candidate to registry.yml")
        elif save:
            print("\n✗ No candidates to save")
            
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()


def cmd_registry(ticker: Optional[str] = None):
    """View optimized asset registry."""
    registry = load_registry()
    
    if not registry:
        print("Registry is empty. Run 'optimize --ticker TICKER --save' to add assets.")
        return
    
    print("=" * 70)
    print("OPTIMIZED ASSET REGISTRY")
    print("=" * 70)
    
    if ticker:
        # Show specific ticker
        ticker = ticker.upper()
        if ticker not in registry:
            print(f"Ticker '{ticker}' not found in registry.")
            print(f"Available: {', '.join(registry.keys())}")
            return
        
        entry = registry[ticker]
        print(f"\n{ticker}:")
        print(f"  Strategy: {entry.strategy}")
        print(f"  Params: {entry.params}")
        print(f"  Last Optimized: {entry.last_optimized}")
        print(f"  Train Sharpe: {entry.train_sharpe:.4f}" if entry.train_sharpe else "  Train Sharpe: N/A")
        print(f"  Val Sharpe: {entry.val_sharpe:.4f}" if entry.val_sharpe else "  Val Sharpe: N/A")
        print(f"  Test Sharpe: {entry.test_sharpe:.4f}" if entry.test_sharpe else "  Test Sharpe: N/A")
        print(f"  Stability: {entry.stability_score:.4f}" if entry.stability_score else "  Stability: N/A")
    else:
        # Show all tickers
        print(f"\n{len(registry)} assets registered:\n")
        
        # Create summary table
        rows = []
        for t, entry in registry.items():
            rows.append({
                "Ticker": t,
                "Strategy": entry.strategy,
                "Train": f"{entry.train_sharpe:.3f}" if entry.train_sharpe else "N/A",
                "Val": f"{entry.val_sharpe:.3f}" if entry.val_sharpe else "N/A",
                "Test": f"{entry.test_sharpe:.3f}" if entry.test_sharpe else "N/A",
                "Stability": f"{entry.stability_score:.3f}" if entry.stability_score else "N/A",
                "Last Updated": entry.last_optimized or "N/A",
            })
        
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(f"\nUse 'python main.py registry --ticker TICKER' for details.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Commands that don't need config file
    if args.command == "optimize":
        cmd_optimize(
            ticker=args.ticker,
            strategy=args.strategy,
            template=args.template,
            save=args.save,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            top_percent=args.top_percent,
            transfer_threshold=args.transfer_threshold,
            final_candidates=args.final_candidates,
            n_jobs=args.n_jobs,
        )
        return
    elif args.command == "registry":
        cmd_registry(ticker=args.ticker)
        return
    
    # Commands that need config file (or can use registry)
    from_registry = False
    if args.command == "backtest" and args.ticker:
        # Load from registry.yml if --ticker is provided
        from backtesting.config import build_config_from_registry
        try:
            cfg = build_config_from_registry(args.ticker)
            from_registry = True
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        # Use config file (default behavior)
        cfg = load_config(Path(args.config))
    
    if args.command == "fetch":
        cmd_fetch(cfg, force=args.force)
    elif args.command == "backtest":
        cmd_backtest(cfg, refresh=args.refresh, plot=args.plot, plot_dir=args.plot_dir, from_registry=from_registry)
    elif args.command == "grid":
        cmd_grid(cfg, refresh=args.refresh, top=args.top, output=args.output, n_jobs=args.n_jobs, min_trades=args.min_trades, batch_size=args.batch_size)
    elif args.command == "portfolio":
        cmd_portfolio(cfg, plot=args.plot, plot_dir=args.plot_dir)
    elif args.command == "portfolio-grid":
        cmd_portfolio_grid(cfg, top=args.top, output=args.output)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

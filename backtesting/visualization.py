"""Visualization functions for backtest results analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt


def plot_rolling_sharpe(
    portfolio: vbt.Portfolio,
    close: pd.Series,
    freq: str,
    window: int = 252,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio time series.
    
    Args:
        portfolio: vectorbt Portfolio object
        close: Price series (for date index)
        freq: Frequency string for annualization
        window: Rolling window size (default: 252 for daily)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    ret = portfolio.returns()
    
    # Calculate rolling window (ensure it's reasonable)
    rolling_window = max(20, min(window, max(1, len(ret) // 4)))
    
    if len(ret) <= rolling_window:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "Not enough data for rolling Sharpe calculation", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or "Rolling Sharpe Ratio")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    # Calculate rolling Sharpe
    rolling_sharpe = ret.rolling(window=rolling_window).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() and x.std() != 0 else np.nan,
        raw=False
    )
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='blue', alpha=0.85)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.6, label='Sharpe = 1.0')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Sharpe = 0.5')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.6, label='Sharpe = 0.0')
    ax.set_title(title or f'Rolling Sharpe (window={rolling_window})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdowns(
    portfolio: vbt.Portfolio,
    close: pd.Series,
    freq: str,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot drawdown charts (underwater chart).
    
    Args:
        portfolio: vectorbt Portfolio object
        close: Price series (for date index)
        freq: Frequency string (unused but kept for API consistency)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    ret = portfolio.returns()
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Cumulative returns and drawdown area
    ax1.plot(eq.index, (eq - 1) * 100, color='black', linewidth=1.5, label='Cumulative Return %')
    ax1.plot(peak.index, (peak - 1) * 100, color='green', linewidth=1.0, alpha=0.7, label='Peak %')
    ax1.fill_between(eq.index, (eq - 1) * 100, (peak - 1) * 100, color='red', alpha=0.25, label='Drawdown Area')
    ax1.set_title(title or 'Cumulative Returns & Drawdowns', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Underwater chart
    ax2.fill_between(dd.index, dd * 100, 0, color='red', alpha=0.7, label='Drawdown %')
    ax2.plot(dd.index, dd * 100, color='darkred', linewidth=1)
    ax2.set_title('Underwater Chart (Drawdown from Peak)', fontsize=13)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_signals(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    indicators: Optional[Dict[str, pd.Series]] = None,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot price chart with indicators and buy/sell signals.
    
    Args:
        close: Price series
        entries: Boolean series indicating entry signals
        exits: Boolean series indicating exit signals
        indicators: Optional dict of indicator names to Series (e.g., {'EMA1': ema1_series})
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price
    ax.plot(close.index, close.values, label='Close', color='black', linewidth=1.5, alpha=0.7)
    
    # Plot indicators if provided
    if indicators:
        colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, (name, series) in enumerate(indicators.items()):
            color = colors[i % len(colors)]
            ax.plot(series.index, series.values, label=name, color=color, alpha=0.8, linewidth=1.2)
    
    # Plot buy/sell signals
    buy_idx = close.index[entries]
    sell_idx = close.index[exits]
    if len(buy_idx) > 0:
        ax.scatter(buy_idx, close.reindex(buy_idx).values, marker='^', color='green', s=80, label='Buy', zorder=5)
    if len(sell_idx) > 0:
        ax.scatter(sell_idx, close.reindex(sell_idx).values, marker='v', color='red', s=80, label='Sell', zorder=5)
    
    ax.set_title(title or 'Price Chart with Signals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_equity_curves(
    portfolios_dict: Dict[str, vbt.Portfolio],
    close_dict: Dict[str, pd.Series],
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot multiple equity curves for comparison (e.g., train/validation/full).
    
    Args:
        portfolios_dict: Dict mapping labels to Portfolio objects (e.g., {'train': train_pf, 'val': val_pf})
        close_dict: Dict mapping labels to price Series (for date index)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['black', 'blue', 'orange', 'green', 'purple', 'red']
    for i, (label, portfolio) in enumerate(portfolios_dict.items()):
        close = close_dict.get(label)
        if close is None:
            continue
        
        ret = portfolio.returns()
        eq = (1 + ret).cumprod()
        color = colors[i % len(colors)]
        ax.plot(close.index, eq.values, label=label, color=color, linewidth=2 if i == 0 else 1.5, alpha=0.8)
    
    ax.set_title(title or 'Equity Curves Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns (normalized to 1)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trade_returns(
    portfolio: vbt.Portfolio,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot trade-by-trade returns as a bar chart.
    
    Args:
        portfolio: vectorbt Portfolio object
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    trades = portfolio.trades
    trade_returns = trades.returns.values if hasattr(trades.returns, 'values') else np.array(trades.returns)
    trade_returns = np.asarray(trade_returns).ravel()
    
    if trade_returns.size == 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, "No trades to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or 'Per-Trade Returns')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    # Calculate statistics
    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]
    
    total_trades = len(trade_returns)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    avg_win_pct = (winning_trades.mean() * 100) if len(winning_trades) > 0 else 0
    avg_loss_pct = (losing_trades.mean() * 100) if len(losing_trades) > 0 else 0
    max_win_pct = (winning_trades.max() * 100) if len(winning_trades) > 0 else 0
    max_loss_pct = (losing_trades.min() * 100) if len(losing_trades) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Per-trade returns bar chart
    x = np.arange(1, trade_returns.size + 1)
    colors = np.where(trade_returns >= 0, 'green', 'red')
    ax.bar(x, trade_returns * 100.0, color=colors, alpha=0.85, width=0.8)
    ax.axhline(0, color='black', linewidth=1, alpha=0.6)
    
    # Add statistics text box
    stats_text = (
        f'Win Rate: {win_rate:.1f}% ({win_count}W/{loss_count}L)\n'
        f'Avg Win: {avg_win_pct:.2f}% | Avg Loss: {avg_loss_pct:.2f}%\n'
        f'Max Win: {max_win_pct:.2f}% | Max Loss: {max_loss_pct:.2f}%'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title or 'Per-Trade Returns (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Trade #', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cumulative_equity(
    portfolio: vbt.Portfolio,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot cumulative equity progression per trade.
    
    Args:
        portfolio: vectorbt Portfolio object
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    trades = portfolio.trades
    trade_returns = trades.returns.values if hasattr(trades.returns, 'values') else np.array(trades.returns)
    trade_returns = np.asarray(trade_returns).ravel()
    
    if trade_returns.size == 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, "No trades to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or 'Cumulative Equity per Trade')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    equity_per_trade = np.cumprod(1.0 + trade_returns)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(1, trade_returns.size + 1)
    ax.plot(x, equity_per_trade, color='black', linewidth=2)
    ax.set_title(title or 'Cumulative Equity per Trade (Trade Domain)', fontsize=13)
    ax.set_xlabel('Trade #', fontsize=12)
    ax.set_ylabel('Equity (x)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_full_sample_equity(
    train_portfolio: vbt.Portfolio,
    val_portfolio: vbt.Portfolio,
    train_close: pd.Series,
    val_close: pd.Series,
    full_close: pd.Series,
    strategy_name: str = "",
    strategy_params: Optional[Dict] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot equity curves with full sample, train, validation, and buy & hold comparison.
    
    Creates a chart similar to the notebook's equity curve visualization showing:
    - Full Sample (black): Strategy performance over entire period
    - Train (blue): In-sample performance
    - Validation (orange): Out-of-sample performance
    - Buy & Hold (gray dashed): Benchmark comparison
    - Train/Val Split (red dashed): Vertical line marking the split
    
    Args:
        train_portfolio: vectorbt Portfolio for training period
        val_portfolio: vectorbt Portfolio for validation period
        train_close: Price series for training period
        val_close: Price series for validation period
        full_close: Price series for full period (used for buy & hold)
        strategy_name: Optional strategy name for title
        strategy_params: Optional dict of strategy params for title
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate equity curves
    train_ret = train_portfolio.returns()
    val_ret = val_portfolio.returns()
    
    train_eq = (1 + train_ret).cumprod()
    val_eq = (1 + val_ret).cumprod()
    
    # Create full sample equity curve by chaining train and validation
    # Validation starts where train ended (multiply by train's final value)
    train_final = train_eq.iloc[-1] if len(train_eq) > 0 else 1.0
    val_eq_chained = val_eq * train_final
    
    # Combine into full sample
    full_eq = pd.concat([train_eq, val_eq_chained])
    
    # Calculate buy & hold equity curve
    full_returns = full_close.pct_change().fillna(0.0)
    bh_eq = (1 + full_returns).cumprod()
    
    # Plot Full Sample (black, thickest line)
    ax.plot(full_eq.index, full_eq.values, 
            label='Full Sample', color='black', linewidth=2.5)
    
    # Plot Train (blue)
    ax.plot(train_eq.index, train_eq.values, 
            label='Train (In-Sample)', color='blue', linewidth=1.5, alpha=0.8)
    
    # Plot Validation (orange) - using chained values for continuity
    ax.plot(val_eq_chained.index, val_eq_chained.values, 
            label='Validation (Out-of-Sample)', color='orange', linewidth=1.5, alpha=0.8)
    
    # Plot Buy & Hold (gray dashed)
    ax.plot(bh_eq.index, bh_eq.values, 
            label='Buy & Hold', color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Add vertical line at train/val split
    split_date = train_close.index[-1]
    ax.axvline(x=split_date, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Train/Val Split')
    
    # Build title
    if strategy_params:
        param_str = ', '.join(f'{k}={v}' for k, v in strategy_params.items())
        title = f"Best {strategy_name}({param_str}) - Equity Curves"
    elif strategy_name:
        title = f"{strategy_name} - Equity Curves"
    else:
        title = "Equity Curves (Full Sample)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns (normalized to 1)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

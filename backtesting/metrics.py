"""Minimal metrics helpers used by CLI + notebook."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import vectorbt as vbt


def compute_metrics(portfolio: vbt.Portfolio, close: pd.Series, freq: str) -> Dict[str, float]:
    """Compute comprehensive metrics matching notebook exactly."""
    total_return = float(portfolio.total_return())
    
    # Portfolio metrics
    metrics = {
        "total_return": total_return,
        "annualized_return": float(portfolio.annualized_return(freq=freq)),
        "sharpe_ratio": float(portfolio.sharpe_ratio(freq=freq)),
        "sortino_ratio": float(portfolio.sortino_ratio(freq=freq)),
        "max_drawdown": float(portfolio.max_drawdown()),
        "volatility": float(portfolio.annualized_volatility(freq=freq)),
        "total_trades": int(portfolio.trades.count()),
    }
    
    # Trade metrics (matching notebook logic)
    trades = portfolio.trades
    total_trades = len(trades)
    
    win_rate_pct = np.nan
    profit_factor = np.nan
    expectancy = 0.0
    avg_win_amount = 0.0
    avg_loss_amount = 0.0
    
    if total_trades > 0:
        tr = trades.returns.values if hasattr(trades.returns, 'values') else np.array(trades.returns)
        if tr.size > 0:
            tr = np.asarray(tr).ravel()
            pos = tr[tr > 0]
            neg = tr[tr < 0]
            win_rate_pct = (len(pos) / len(tr)) * 100.0 if len(tr) else np.nan
            gains = pos.sum() if len(pos) else 0.0
            losses = abs(neg.sum()) if len(neg) else 0.0
            profit_factor = (gains / losses) if losses > 0 else np.inf
            expectancy = float(tr.mean())
            avg_win_amount = float(pos.mean()) if len(pos) else 0.0
            avg_loss_amount = float(abs(neg.mean())) if len(neg) else 0.0
    
    metrics.update({
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win_amount": avg_win_amount,
        "avg_loss_amount": avg_loss_amount,
    })
    
    return metrics


def buy_and_hold(close: pd.Series, config) -> Dict[str, float]:
    entries = pd.Series(False, index=close.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=close.index)
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=config.init_cash,
        fees=config.fees,
        slippage=config.slippage,
        freq=config.freq,
    )
    return compute_metrics(portfolio, close, config.freq)

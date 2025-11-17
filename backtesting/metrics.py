"""Minimal metrics helpers used by CLI + notebook."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import vectorbt as vbt

# Suppress numpy RuntimeWarnings for division by zero/invalid operations
# These are handled by explicit checks in the code, so warnings are safe to ignore
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')


def compute_metrics(portfolio: vbt.Portfolio, close: pd.Series, freq: str) -> Dict[str, float]:
    """Compute comprehensive metrics matching notebook exactly."""
    total_return = float(portfolio.total_return())
    annualized_return = float(portfolio.annualized_return(freq=freq))
    max_drawdown = float(portfolio.max_drawdown())
    volatility = float(portfolio.annualized_volatility(freq=freq))
    sharpe_ratio = float(portfolio.sharpe_ratio(freq=freq))
    sortino_ratio = float(portfolio.sortino_ratio(freq=freq))
    
    # Portfolio metrics with try/except for optional vectorbt methods
    # Note: Some methods may not be available in all vectorbt versions or may require additional parameters
    information_ratio = np.nan
    tail_ratio = np.nan
    deflated_sharpe_ratio = np.nan
    try:
        information_ratio = float(portfolio.information_ratio(freq=freq))
    except Exception:
        pass
    try:
        tail_ratio = float(portfolio.tail_ratio(freq=freq))
    except Exception:
        pass
    try:
        # deflated_sharpe_ratio may not be available or may require number of trials parameter
        # for multiple testing adjustment - if unavailable, remains np.nan
        deflated_sharpe_ratio = float(portfolio.deflated_sharpe_ratio(freq=freq))
    except Exception:
        pass
    
    # Ulcer Index
    returns = portfolio.returns()
    ulcer_index = np.nan
    if len(returns) > 0:
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ulcer_index = float(np.sqrt((dd.pow(2)).mean())) if len(dd) > 0 else np.nan
    
    # Calmar Ratio: Annualized Return / Max Drawdown
    calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else np.nan
    
    # Trade metrics (matching notebook logic)
    trades = portfolio.trades
    total_trades = len(trades)
    
    win_rate_pct = np.nan
    profit_factor = np.nan
    expectancy = 0.0
    avg_win_amount = 0.0
    avg_loss_amount = 0.0
    payoff_ratio = np.nan
    largest_win = np.nan
    largest_loss = np.nan
    winning_streak = np.nan
    losing_streak = np.nan
    gain_to_pain_ratio = np.nan
    recovery_factor = np.nan
    total_profit = 0.0
    sqn = np.nan
    omega_ratio = np.nan
    serenity_index = np.nan
    
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
            
            # Payoff ratio
            payoff_ratio = (avg_win_amount / avg_loss_amount) if avg_loss_amount not in (0.0, np.nan) else np.inf
            
            # Largest win/loss
            largest_win = float(pos.max()) if len(pos) > 0 else np.nan
            largest_loss = float(neg.min()) if len(neg) > 0 else np.nan
            
            # Gain to pain ratio
            gain_to_pain_ratio = (gains / losses) if losses > 0 else np.inf
            
            # Total profit (absolute)
            total_profit = float(tr.sum())
            
            # System Quality Number (SQN): Expectancy / StdDev of returns
            if len(tr) > 1:
                tr_std = float(np.std(tr))
                sqn = (expectancy / tr_std) if tr_std != 0 else np.nan
            
            # Omega Ratio (probability-weighted gains above threshold / losses below threshold)
            # Using threshold of 0 (risk-free rate)
            threshold = 0.0
            gains_above = pos[pos > threshold].sum() if len(pos) > 0 else 0.0
            losses_below = abs(neg[neg < threshold].sum()) if len(neg) > 0 else 0.0
            omega_ratio = (gains_above / losses_below) if losses_below > 0 else np.inf
            
            # Serenity Index: (Total Return / Max Drawdown) * (Win Rate / 100)
            serenity_index = ((total_return / abs(max_drawdown)) * (win_rate_pct / 100.0)) if max_drawdown != 0 and not np.isnan(win_rate_pct) else np.nan
        
    # Winning/Losing streaks using vectorbt methods
    # Note: These are calculated outside the total_trades > 0 block to ensure they're always in the dict
    # If methods fail or no trades exist, they remain np.nan (which pandas writes as empty string in CSV)
    try:
        if total_trades > 0:
            winning_streak = int(trades.winning_streak())
        else:
            winning_streak = np.nan
    except Exception:
        winning_streak = np.nan
    try:
        if total_trades > 0:
            losing_streak = int(trades.losing_streak())
        else:
            losing_streak = np.nan
    except Exception:
        losing_streak = np.nan
    
    # Recovery Factor: Net Profit / Max Drawdown
    if max_drawdown != 0:
        recovery_factor = (total_profit / abs(max_drawdown))
    else:
        recovery_factor = np.nan
    
    # Trades per year
    years = max((close.index[-1] - close.index[0]).days / 365.25, 1e-9)
    trades_per_year = total_trades / years
    
    # Build metrics dictionary
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "total_profit": total_profit,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "omega_ratio": omega_ratio,
        "information_ratio": information_ratio,
        "tail_ratio": tail_ratio,
        "deflated_sharpe_ratio": deflated_sharpe_ratio,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "ulcer_index": ulcer_index,
        "win_rate": win_rate_pct,
        "total_trades": total_trades,
        "trades_per_year": trades_per_year,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sqn": sqn,
        "payoff_ratio": payoff_ratio,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_win_amount": avg_win_amount,
        "avg_loss_amount": avg_loss_amount,
        "winning_streak": winning_streak,
        "losing_streak": losing_streak,
        "recovery_factor": recovery_factor,
        "gain_to_pain_ratio": gain_to_pain_ratio,
        "serenity_index": serenity_index,
    }
    
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

"""
3D Grid Search Visualization for Parquet Results (GPU-Accelerated via WebGL)
Works with any parquet file from Python backtesting grid search with ≤3 parameters.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List, Tuple


# Known metric columns from backtesting/metrics.py
METRIC_COLUMNS = {
    'total_return', 'annualized_return', 'max_drawdown', 'volatility',
    'annualized_volatility', 'sharpe_ratio', 'sortino_ratio', 'information_ratio',
    'tail_ratio', 'deflated_sharpe_ratio', 'ulcer_index', 'calmar_ratio',
    'total_trades', 'win_rate_pct', 'profit_factor', 'expectancy',
    'avg_win_amount', 'avg_loss_amount', 'payoff_ratio', 'largest_win',
    'largest_loss', 'winning_streak', 'losing_streak', 'gain_to_pain_ratio',
    'recovery_factor', 'net_profit', 'sqn', 'omega_ratio', 'serenity_index',
    'max_drawdown_dollars', 'trades_per_year', 'n_trades', 'win_rate',
}


def load_parquet_grid(filename: str) -> pd.DataFrame:
    """Load parquet file and return DataFrame."""
    df = pd.read_parquet(filename)
    print(f"Loaded {len(df)} rows from {filename}")
    print(f"Columns: {list(df.columns)}")
    return df


def detect_param_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect parameter columns (non-metric columns)."""
    # Known parameter patterns for different strategies
    PARAM_PATTERNS = [
        'ema_fast', 'ema_mid', 'ema_slow',  # Triple EMA
        'fastperiod', 'slowperiod', 'signalperiod',  # Triple MACD
        'fast', 'slow', 'signal',  # Generic MACD/oscillator
        'period', 'window', 'threshold',  # Generic
    ]
    
    param_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Skip if it's a known metric
        if col_lower in {m.lower() for m in METRIC_COLUMNS}:
            continue
        # Check if it matches parameter patterns or is numeric int-like
        if any(pat in col_lower for pat in PARAM_PATTERNS):
            param_cols.append(col)
        elif df[col].dtype in ['int64', 'int32', 'float64']:
            # If numeric and not a metric, likely a parameter
            if col_lower not in {m.lower() for m in METRIC_COLUMNS}:
                # Check if values look like discrete parameters (small range of integers)
                unique_vals = df[col].nunique()
                if unique_vals < len(df) * 0.5 and df[col].min() >= 0:
                    param_cols.append(col)
    
    return param_cols[:3]  # Limit to 3 parameters for 3D plotting


def detect_metric_columns(df: pd.DataFrame) -> List[str]:
    """Detect available metric columns in DataFrame."""
    available = []
    for col in df.columns:
        if col.lower() in {m.lower() for m in METRIC_COLUMNS}:
            available.append(col)
    return available


def plot_3d_scatter(
    df: pd.DataFrame,
    param_cols: List[str],
    metric: str = 'sharpe_ratio',
    min_threshold: Optional[float] = None,
    max_points: int = 500000
) -> go.Figure:
    """
    GPU-accelerated 3D scatter plot using Plotly WebGL.
    
    Args:
        df: DataFrame with grid search results
        param_cols: List of 1-3 parameter column names
        metric: Metric column to use for color gradient
        min_threshold: Only show points above this metric value
        max_points: Limit points for performance
    """
    print(f"\nPreparing 3D plot with metric: {metric}")
    
    # Get parameter arrays
    n_params = len(param_cols)
    if n_params < 1:
        raise ValueError("Need at least 1 parameter column")
    
    x = df[param_cols[0]].values
    y = df[param_cols[1]].values if n_params >= 2 else np.zeros(len(df))
    z = df[param_cols[2]].values if n_params >= 3 else np.zeros(len(df))
    values = df[metric].values
    
    # Filter valid values (non-NaN, non-inf)
    mask = np.isfinite(values)
    x, y, z, values = x[mask], y[mask], z[mask], values[mask]
    print(f"Valid points: {len(values)}")
    print(f"{metric} range: {values.min():.4f} to {values.max():.4f}")
    
    # Optional threshold filter
    if min_threshold is not None:
        mask = values >= min_threshold
        x, y, z, values = x[mask], y[mask], z[mask], values[mask]
        print(f"After threshold filter (>= {min_threshold}): {len(values)} points")
    
    # Smart sampling: prioritize top performers + uniform coverage
    if len(values) > max_points:
        print(f"Smart sampling {len(values)} down to {max_points} points...")
        
        top_count = max_points // 2
        top_indices = np.argsort(values)[-top_count:]
        
        remaining_indices = np.setdiff1d(np.arange(len(values)), top_indices)
        random_count = max_points - top_count
        if len(remaining_indices) > random_count:
            random_indices = np.random.choice(remaining_indices, random_count, replace=False)
        else:
            random_indices = remaining_indices
        
        idx = np.concatenate([top_indices, random_indices])
        x, y, z, values = x[idx], y[idx], z[idx], values[idx]
    
    print(f"Rendering {len(values)} points with WebGL...")
    
    # Find best point
    best_idx = np.argmax(values)
    
    # Axis labels
    x_label = param_cols[0] if n_params >= 1 else "X"
    y_label = param_cols[1] if n_params >= 2 else "Y (dummy)"
    z_label = param_cols[2] if n_params >= 3 else "Z (dummy)"
    
    fig = go.Figure()
    
    # Main scatter (WebGL accelerated)
    hover_template = f'{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{z_label}: %{{z}}<br>{metric}: %{{marker.color:.4f}}<extra></extra>'
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=values,
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title=metric, tickfont=dict(color='white')),
        ),
        hovertemplate=hover_template,
        name='Grid Results'
    ))
    
    # Best point marker
    best_params = f'{x_label}={x[best_idx]}'
    if n_params >= 2:
        best_params += f' {y_label}={y[best_idx]}'
    if n_params >= 3:
        best_params += f' {z_label}={z[best_idx]}'
    
    fig.add_trace(go.Scatter3d(
        x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
        mode='markers',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        name=f'Best: {best_params} ({metric}={values[best_idx]:.4f})'
    ))
    
    # Dark theme styling
    fig.update_layout(
        title=dict(text=f'3D Grid Search - {metric} (WebGL)', font=dict(color='white')),
        scene=dict(
            xaxis=dict(title=x_label, backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            yaxis=dict(title=y_label, backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            zaxis=dict(title=z_label, backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        legend=dict(font=dict(color='white')),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig


def plot_2d_heatmap(
    df: pd.DataFrame,
    param_cols: List[str],
    metric: str = 'sharpe_ratio',
    fixed_param: Optional[Tuple[str, int]] = None
) -> go.Figure:
    """
    2D heatmap slice of grid search results.
    
    Args:
        df: DataFrame with grid search results
        param_cols: Parameter columns (first 2 used for axes)
        metric: Metric to visualize
        fixed_param: Optional (param_name, value) to filter 3rd dimension
    """
    if len(param_cols) < 2:
        raise ValueError("Need at least 2 parameter columns for heatmap")
    
    data = df.copy()
    
    # If we have 3 params and user specified a fixed value, filter
    if fixed_param and len(param_cols) >= 3:
        param_name, param_val = fixed_param
        data = data[data[param_name] == param_val]
        print(f"Filtered to {param_name}={param_val}: {len(data)} points")
    
    x_col = param_cols[0]
    y_col = param_cols[1]
    
    x_vals = np.array(data[x_col])
    y_vals = np.array(data[y_col])
    metric_vals = np.array(data[metric])
    
    # Filter valid
    mask = np.isfinite(metric_vals)
    x_vals, y_vals, metric_vals = x_vals[mask], y_vals[mask], metric_vals[mask]
    
    # Build grid
    x_unique = sorted(set(x_vals))
    y_unique = sorted(set(y_vals))
    
    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    x_map = {v: i for i, v in enumerate(x_unique)}
    y_map = {v: i for i, v in enumerate(y_unique)}
    
    for xv, yv, mv in zip(x_vals, y_vals, metric_vals):
        grid[y_map[yv], x_map[xv]] = mv
    
    title = f'Heatmap: {metric}'
    if fixed_param:
        title += f' ({fixed_param[0]}={fixed_param[1]})'
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=x_unique,
        y=y_unique,
        colorscale='Viridis',
        colorbar=dict(title=metric, tickfont=dict(color='white')),
        hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>',
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white')),
        xaxis=dict(title=x_col, color='white', gridcolor='#444'),
        yaxis=dict(title=y_col, color='white', gridcolor='#444'),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
    )
    
    return fig


def print_available_metrics(df: pd.DataFrame):
    """Print available metrics in the DataFrame."""
    metrics = detect_metric_columns(df)
    print("\n📊 Available metrics to visualize:")
    for m in sorted(metrics):
        if m in df.columns:
            valid = df[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 0:
                print(f"  - {m}: [{valid.min():.4f}, {valid.max():.4f}]")


if __name__ == '__main__':
    # =========================================================================
    # CONFIGURATION - Edit these values
    # =========================================================================
    
    # Path to your parquet file
    PARQUET_FILE = 'data/VOO_ema.parquet'  # Change this!
    
    # Metric to use for color gradient
    # Common options: 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return'
    METRIC = 'sharpe_ratio'
    
    # Optional: filter to show only points above this threshold (None = show all)
    MIN_THRESHOLD = 1  # e.g., 0.5 for Sharpe > 0.5
    
    # Plot mode: 1 = 3D scatter, 2 = 2D heatmap, 3 = both
    PLOT_MODE = 1
    
    # For 2D heatmap with 3 params: fix the 3rd param at this value (None = auto)
    # Example: ('signalperiod', 9) or ('ema_slow', 50)
    FIXED_PARAM = None
    
    # =========================================================================
    
    print(f"Loading {PARQUET_FILE}...")
    df = load_parquet_grid(PARQUET_FILE)
    
    # Auto-detect parameters
    param_cols = detect_param_columns(df)
    print(f"\n🔧 Detected parameter columns: {param_cols}")
    
    # Show available metrics
    print_available_metrics(df)
    
    # Validate metric exists
    if METRIC not in df.columns:
        print(f"\n❌ Metric '{METRIC}' not found! Available: {detect_metric_columns(df)}")
        exit(1)
    
    print(f"\n📈 Using metric: {METRIC}")
    
    if PLOT_MODE in [1, 3]:
        print("\n=== 3D SCATTER (WebGL) ===")
        fig1 = plot_3d_scatter(df, param_cols, metric=METRIC, min_threshold=MIN_THRESHOLD)
        print("Opening 3D plot in browser... (rotate with mouse)")
        fig1.show()
    
    if PLOT_MODE in [2, 3]:
        print("\n=== 2D HEATMAP ===")
        # Auto-select fixed param if we have 3 params and none specified
        fixed = FIXED_PARAM
        if fixed is None and len(param_cols) == 3:
            # Use median value of 3rd param
            third_param = param_cols[2]
            median_val = int(df[third_param].median())
            fixed = (third_param, median_val)
            print(f"Auto-fixing {third_param}={median_val} for 2D slice")
        
        fig2 = plot_2d_heatmap(df, param_cols, metric=METRIC, fixed_param=fixed)
        fig2.show()
    
    print("\n✅ Done! Check your browser.")


"""
3D Grid Search Visualization (GPU-Accelerated via WebGL)
Run after executing ultra_grid.go which generates cube_data.json
"""
import json
import numpy as np
import plotly.graph_objects as go

def load_cube_data(filename='cube_data.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_3d_scatter_gpu(data, metric='Martin', min_threshold=None, max_points=500000):
    """
    GPU-accelerated 3D scatter plot using Plotly WebGL.
    
    - min_threshold: only show points above this metric value (None = all)
    - max_points: limit points for performance (default 500k)
    """
    print("Preparing data for GPU rendering...")
    
    fast = np.array([p['Fast'] for p in data])
    slow = np.array([p['Slow'] for p in data])
    signal = np.array([p['Signal'] for p in data])
    values = np.array([p[metric] for p in data])
    
    # Filter valid values
    mask = values > -100
    fast, slow, signal, values = fast[mask], slow[mask], signal[mask], values[mask]
    print(f"Valid points: {len(values)}")
    print(f"Martin range: {values.min():.2f} to {values.max():.2f}")
    
    # Optional threshold filter
    if min_threshold is not None:
        mask = values >= min_threshold
        fast, slow, signal, values = fast[mask], slow[mask], signal[mask], values[mask]
        print(f"After threshold filter: {len(values)} points")
    
    # Smart sampling: prioritize top performers + uniform coverage
    if len(values) > max_points:
        print(f"Smart sampling {len(values)} down to {max_points} points...")
        
        # Take top 20% by Martin ratio (the interesting ones)
        top_count = max_points // 2
        top_indices = np.argsort(values)[-top_count:]
        
        # Random sample from the rest for coverage
        remaining_indices = np.setdiff1d(np.arange(len(values)), top_indices)
        random_count = max_points - top_count
        if len(remaining_indices) > random_count:
            random_indices = np.random.choice(remaining_indices, random_count, replace=False)
        else:
            random_indices = remaining_indices
        
        # Combine
        idx = np.concatenate([top_indices, random_indices])
        fast, slow, signal, values = fast[idx], slow[idx], signal[idx], values[idx]
        print(f"  - Top performers: {len(top_indices)}")
        print(f"  - Random coverage: {len(random_indices)}")
    
    print(f"Rendering {len(values)} points with WebGL...")
    
    # Find best point
    best_idx = np.argmax(values)
    
    # Create 3D scatter with WebGL
    fig = go.Figure()
    
    # Main scatter (WebGL accelerated)
    fig.add_trace(go.Scatter3d(
        x=slow, y=fast, z=signal,
        mode='markers',
        marker=dict(
            size=2,
            color=values,
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title=f'{metric} Ratio', tickfont=dict(color='white')),
        ),
        hovertemplate='Slow: %{x}<br>Fast: %{y}<br>Signal: %{z}<br>Martin: %{marker.color:.2f}<extra></extra>',
        name='Grid Results'
    ))
    
    # Best point marker
    fig.add_trace(go.Scatter3d(
        x=[slow[best_idx]], y=[fast[best_idx]], z=[signal[best_idx]],
        mode='markers',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        name=f'Best: F={fast[best_idx]} S={slow[best_idx]} Sig={signal[best_idx]}'
    ))
    
    # Dark theme styling
    fig.update_layout(
        title=dict(text=f'3D Grid Search - {metric} Ratio (WebGL)', font=dict(color='white')),
        scene=dict(
            xaxis=dict(title='Slow EMA', backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            yaxis=dict(title='Fast EMA', backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            zaxis=dict(title='Signal Len', backgroundcolor='#1a1a2e', gridcolor='#444', color='white'),
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        legend=dict(font=dict(color='white')),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig

def plot_2d_heatmap_gpu(data, fixed_signal=9):
    """GPU-accelerated 2D heatmap slice at fixed signal value"""
    fast = np.array([p['Fast'] for p in data])
    slow = np.array([p['Slow'] for p in data])
    signal = np.array([p['Signal'] for p in data])
    martin = np.array([p['Martin'] for p in data])
    
    # Filter to fixed signal
    mask = signal == fixed_signal
    if mask.sum() == 0:
        print(f"No data for signal={fixed_signal}")
        return None
    
    fast_f, slow_f, martin_f = fast[mask], slow[mask], martin[mask]
    
    # Build grid
    fast_unique = sorted(set(fast_f))
    slow_unique = sorted(set(slow_f))
    
    grid = np.full((len(fast_unique), len(slow_unique)), np.nan)
    fast_map = {v: i for i, v in enumerate(fast_unique)}
    slow_map = {v: i for i, v in enumerate(slow_unique)}
    
    for f, s, m in zip(fast_f, slow_f, martin_f):
        if m > -100:
            grid[fast_map[f], slow_map[s]] = m
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=slow_unique,
        y=fast_unique,
        colorscale='Viridis',
        colorbar=dict(title='Martin Ratio', tickfont=dict(color='white')),
        hovertemplate='Slow: %{x}<br>Fast: %{y}<br>Martin: %{z:.2f}<extra></extra>',
    ))
    
    fig.update_layout(
        title=dict(text=f'Heatmap (Signal={fixed_signal})', font=dict(color='white')),
        xaxis=dict(title='Slow EMA', color='white', gridcolor='#444'),
        yaxis=dict(title='Fast EMA', color='white', gridcolor='#444'),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
    )
    
    return fig

if __name__ == '__main__':
    print("Loading cube_data.json...")
    data = load_cube_data()
    print(f"Loaded {len(data)} points")
    
    # Choose what to plot:
    # 1 = 3D scatter only
    # 2 = 2D heatmap only  
    # 3 = both (two tabs)
    PLOT_MODE = 1
    
    if PLOT_MODE in [1, 3]:
        print("\n=== 3D SCATTER (WebGL) ===")
        # GPU-accelerated 3D scatter
        # min_threshold: filter by Martin ratio (None = show all)
        # max_points: defaults to 500k for smooth rendering
        fig1 = plot_3d_scatter_gpu(data)
        print("Opening 3D plot in browser... (rotate with mouse)")
        fig1.show()
    
    if PLOT_MODE in [2, 3]:
        print("\n=== 2D HEATMAP ===")
        fig2 = plot_2d_heatmap_gpu(data, fixed_signal=9)
        fig2.show()
    
    print("\nDone! Check your browser.")


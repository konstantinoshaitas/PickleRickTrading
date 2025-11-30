print("INITIATING TRIPLE EMA CROSSOVER GRID SEARCH (BATCHED VECTORIZATION)")
print("=" * 70)
print(f"Testing Strategy: Triple Exponential Moving Average Crossover")
print(f"Training Period: {train_close.index[0].date()} → {train_close.index[-1].date()}")
print(f"Initial Capital: $100,000")
print(f"Batch Processing: ENABLED")
print("=" * 70)

# --- CONFIGURATION ---
# Adjust this based on your RAM. 
# 5000 is safe for 16GB RAM. If you have 32GB+, try 10000. If 8GB, try 2000.
BATCH_SIZE = 5000 

# 1. PREPARE PARAMETERS
total_combinations = len(ema_combinations)
print(f"Total Combinations to Test: {total_combinations}")

# Container for final results
grid_search_results = []
successful_batches = 0
failed_batches = 0

# Calculate how many batches we need
num_batches = (total_combinations + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Starting execution in {num_batches} batches of {BATCH_SIZE}...\n")

# 2. START BATCH LOOP
for batch_idx, i in enumerate(range(0, total_combinations, BATCH_SIZE)):
    try:
        # Slice the current chunk of parameters
        current_batch_params = ema_combinations[i : i + BATCH_SIZE]
        current_batch_size = len(current_batch_params)
        
        print(f"Processing Batch {batch_idx + 1}/{num_batches} (Strategies {i} to {i + current_batch_size})...")

        # Unzip parameters for this specific batch
        p1_list, p2_list, p3_list = map(list, zip(*current_batch_params))

        # --- A. INDICATORS (Batch Vectorized) ---
        ema1 = vbt.MA.run(train_close, p1_list, ewm=True)
        ema2 = vbt.MA.run(train_close, p2_list, ewm=True)
        ema3 = vbt.MA.run(train_close, p3_list, ewm=True)

        # Standardize column names for this batch
        common_cols = pd.Index(range(current_batch_size), name='combo_id')
        df_ema1 = ema1.ma.copy(); df_ema1.columns = common_cols
        df_ema2 = ema2.ma.copy(); df_ema2.columns = common_cols
        df_ema3 = ema3.ma.copy(); df_ema3.columns = common_cols

        # --- B. SIGNALS (Batch Vectorized) ---
        # Calculate crossovers
        c1 = df_ema1.vbt.crossed_above(df_ema2)
        c2 = df_ema1.vbt.crossed_above(df_ema3)
        c3 = df_ema2.vbt.crossed_above(df_ema3)
        entries_raw = c1 | c2 | c3

        d1 = df_ema1.vbt.crossed_below(df_ema2)
        d2 = df_ema1.vbt.crossed_below(df_ema3)
        d3 = df_ema2.vbt.crossed_below(df_ema3)
        exits_raw = d1 | d2 | d3

        # Shift signals (Lookahead Fix)
        entries = entries_raw.shift(1).fillna(False).astype(bool)
        exits = exits_raw.shift(1).fillna(False).astype(bool)

        # --- C. BACKTEST (Batch Vectorized) ---
        pf = vbt.Portfolio.from_signals(
            close=train_close,
            entries=entries,
            exits=exits,
            init_cash=100_000,
            fees=0.0005,
            slippage=0.0005,
            freq='D'
        )

        # --- D. METRICS (Batch Vectorized) ---
        years = max((train_close.index[-1] - train_close.index[0]).days / 365.25, 1e-9)

        # Extract Raw Series
        total_returns = pf.total_return()
        annualized_returns = pf.annualized_return(freq='D')
        max_drawdowns = pf.max_drawdown()
        volatilities = pf.annualized_volatility(freq='D')
        sharpe_ratios = pf.sharpe_ratio(freq='D')
        sortino_ratios = pf.sortino_ratio(freq='D')

        # Advanced ratios with error handling
        try: info_ratios = pf.information_ratio(freq='D')
        except: info_ratios = pd.Series(np.nan, index=common_cols)
        try: tail_ratios = pf.tail_ratio(freq='D')
        except: tail_ratios = pd.Series(np.nan, index=common_cols)
        try: deflated_sharpes = pf.deflated_sharpe_ratio(freq='D')
        except: deflated_sharpes = pd.Series(np.nan, index=common_cols)

        # Ulcer Index
        ulcer_indices = (pf.drawdown().pow(2).mean()) ** 0.5

        # Trade Stats
        trades = pf.trades
        total_trades = trades.count()
        win_rates = (trades.win_rate() * 100).fillna(0.0)
        profit_factors = trades.profit_factor().replace([np.inf, -np.inf], np.nan)
        expectancies = trades.expectancy().fillna(0.0)
        avg_wins = trades.winning.pnl.mean().fillna(0.0)
        avg_losses = trades.losing.pnl.mean().abs().fillna(0.0)
        winning_streaks = trades.winning_streak.max().fillna(0)
        losing_streaks = trades.losing_streak.max().fillna(0)

        # --- E. STORE BATCH RESULTS ---
        batch_df = pd.DataFrame({
            "ema1_period": p1_list,
            "ema2_period": p2_list,
            "ema3_period": p3_list,
            "total_return": total_returns.values,
            "annualized_return": annualized_returns.values,
            "max_drawdown": max_drawdowns.values,
            "volatility": volatilities.values,
            "sharpe_ratio": sharpe_ratios.values,
            "sortino_ratio": sortino_ratios.values,
            "information_ratio": info_ratios.values,
            "tail_ratio": tail_ratios.values,
            "deflated_sharpe_ratio": deflated_sharpes.values,
            "ulcer_index": ulcer_indices.values,
            "total_trades": total_trades.values,
            "win_rate": win_rates.values,
            "profit_factor": profit_factors.values,
            "expectancy": expectancies.values,
            "avg_win_amount": avg_wins.values,
            "avg_loss_amount": avg_losses.values,
            "winning_streak": winning_streaks.values,
            "losing_streak": losing_streaks.values,
        })

        # Calculations & Filtering
        batch_df['trades_per_year'] = batch_df['total_trades'] / years
        batch_df['payoff_ratio'] = (batch_df['avg_win_amount'] / batch_df['avg_loss_amount']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Apply Logic Filter
        batch_df = batch_df[batch_df['trades_per_year'] >= 2]

        # Convert to dictionary records and extend master list
        grid_search_results.extend(batch_df.to_dict('records'))
        
        successful_batches += 1
        
        # --- F. MEMORY CLEANUP (CRITICAL) ---
        # Explicitly delete large objects to free RAM for the next batch
        del ema1, ema2, ema3, df_ema1, df_ema2, df_ema3
        del entries, exits, pf, trades
        del entries_raw, exits_raw, c1, c2, c3, d1, d2, d3
        
        # Force Python to release memory back to system
        gc.collect()

    except Exception as e:
        failed_batches += 1
        print(f"❌ Error in Batch {batch_idx + 1}: {str(e)}")
        continue

# SUMMARY
print("\n" + "="*40)
print("BATCHED GRID SEARCH COMPLETED!")
print("="*40)
print(f"Total Combinations: {total_combinations}")
print(f"Strategies Saved:   {len(grid_search_results)}")
print(f"Batches Success:    {successful_batches}")
print(f"Batches Failed:     {failed_batches}")
print("\nResults stored in 'grid_search_results'")
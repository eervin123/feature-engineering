import vectorbtpro as vbt
import pandas as pd
import numpy as np
from vwap_features import VWAPFeatureEngineering, VWAPStrategies

def backtest_strategy_vbt(df, strategy_name):
    portfolio = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=df['Signal'] > 0,
        exits=df['Signal'] < 0,
        init_cash=10000,
        fees=0.001,  # 0.1% fee per trade
        freq='D'
    )
    
    total_return = portfolio.total_return  # Removed parentheses
    sharpe_ratio = portfolio.sharpe_ratio
    max_drawdown = portfolio.max_drawdown
    
    return {
        'Strategy': strategy_name,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        # 'Number of Trades': portfolio.num_trades(),  # This one is a method, so keep the parentheses
    }

if __name__ == "__main__":
    # Fetch Bitcoin price data using vectorbtpro
    btc_data = vbt.YFData.fetch('BTC-USD', end='2024-01-01').get()

    # Prepare the DataFrame
    df = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Initialize the VWAP feature engineering class
    vwap_features = VWAPFeatureEngineering(df, price_col='Close', volume_col='Volume', lookback_windows=[5, 10, 20, 50])

    # Add VWAP features to the dataframe
    vwap_features.add_vwap_features()

    # Get the dataframe with added features
    df_with_features = vwap_features.get_features()

    # Calculate additional indicators using vbt
    df_with_features['RSI_14'] = vbt.RSI.run(df_with_features['Close'], window=14).rsi.to_numpy()
    
    macd = vbt.MACD.run(df_with_features['Close'], fast_window=12, slow_window=26, signal_window=9)
    df_with_features['MACD'] = macd.macd.to_numpy()
    df_with_features['MACD_Signal'] = macd.signal.to_numpy()
    
    df_with_features['ATR_14'] = vbt.ATR.run(df_with_features['High'], df_with_features['Low'], df_with_features['Close'], window=14).atr.to_numpy()

    # Initialize strategies
    strategies = VWAPStrategies(df_with_features)

    # List to store results
    results = []

    # Buy and Hold strategy as benchmark
    df_with_features['Signal'] = 1
    results.append(backtest_strategy_vbt(df_with_features, "Buy and Hold"))

    # Test different strategies
    strategy_functions = [
        (strategies.dual_vwap_crossover_strategy, ('VWAP_5', 'VWAP_20'), "Dual VWAP Crossover (5,20)"),
        (strategies.vwap_rsi_strategy, ('VWAP_20', 'RSI_14'), "VWAP-RSI (20)"),
        (strategies.vwap_macd_strategy, ('VWAP_20', 'MACD', 'MACD_Signal'), "VWAP-MACD (20)"),
        (strategies.vwap_breakout_strategy, ('VWAP_20',), "VWAP Breakout (20)"),
        (strategies.vwap_volatility_strategy, ('VWAP_20', 'ATR_14', 2), "VWAP Volatility (20)")
    ]

    sensitivities = [0.5, 1.0, 1.5, 2.0, 2.5]

    for strategy_func, args, name in strategy_functions:
        for sensitivity in sensitivities:
            # Long strategy
            df_with_features[['Signal', 'Position']] = strategies.apply_strategy(strategy_func, *args, sensitivity=sensitivity)
            results.append(backtest_strategy_vbt(df_with_features, f"Long {name} (Sensitivity: {sensitivity})"))

            # Short strategy
            df_with_features[['Signal', 'Position']] = strategies.apply_strategy(strategy_func, *args, short=True, sensitivity=sensitivity)
            results.append(backtest_strategy_vbt(df_with_features, f"Short {name} (Sensitivity: {sensitivity})"))

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print(results_df.sort_values('Total Return', ascending=False))

    # Save results to CSV
    csv_filename = 'vwap_strategy_results_vbt.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    # Plot the equity curve of the top strategy
    top_strategy = results_df.iloc[0]['Strategy']
    print(f"Top strategy: {top_strategy}")

    # Extract strategy name and sensitivity (if present)
    if '(Sensitivity:' in top_strategy:
        strategy_name, sensitivity = top_strategy.rsplit('(Sensitivity:', 1)
        strategy_name = strategy_name.strip()
        sensitivity = float(sensitivity.rstrip(')'))
    else:
        strategy_name = top_strategy
        sensitivity = 1.0  # default sensitivity

    print(f"Strategy name: {strategy_name}")
    print(f"Sensitivity: {sensitivity}")

    # Find the matching strategy function
    matching_strategy = None
    for func, args, name in strategy_functions:
        if name in strategy_name:
            matching_strategy = func
            matching_args = args
            break

    if matching_strategy:
        df_with_features[['Signal', 'Position']] = strategies.apply_strategy(
            matching_strategy, 
            *matching_args, 
            sensitivity=sensitivity, 
            short='Short' in strategy_name
        )
    else:
        print(f"Could not find matching strategy for {strategy_name}")
        df_with_features['Signal'] = 1  # Default to buy-and-hold

    portfolio = vbt.Portfolio.from_signals(
        close=df_with_features['Close'],
        entries=df_with_features['Signal'] > 0,
        exits=df_with_features['Signal'] < 0,
        init_cash=10000,
        fees=0.001,
        freq='D'
    )
    fig = portfolio.plot()
    fig.write_html("top_strategy_equity_curve.html")
    print(f"\nEquity curve plot saved for top strategy: {top_strategy}")

    # Print additional statistics for the top strategy
    print("\nAdditional Statistics for Top Strategy:")
    print(f"Total Return: {portfolio.total_return:.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {portfolio.sortino_ratio:.2f}")
    print(f"Max Drawdown: {portfolio.max_drawdown:.2%}")
    print(f"Win Rate: {portfolio.trades.win_rate:.2%}")
    print(f"Profit Factor: {portfolio.trades.profit_factor:.2f}")

    # Print basic statistics about the input data
    print("\nData range:", df_with_features.index.min(), "to", df_with_features.index.max())
    print("Number of rows:", len(df_with_features))
    print("Close price range:", df_with_features['Close'].min(), "to", df_with_features['Close'].max())
    print("VWAP_20 range:", df_with_features['VWAP_20'].min(), "to", df_with_features['VWAP_20'].max())
    print("RSI_14 range:", df_with_features['RSI_14'].min(), "to", df_with_features['RSI_14'].max())

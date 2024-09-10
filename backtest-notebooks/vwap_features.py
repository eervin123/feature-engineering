import pandas as pd
import numpy as np
import vectorbtpro as vbt
import datetime

class VWAPFeatureEngineering:
    def __init__(self, df, price_col='Close', volume_col='Volume', lookback_windows=[5, 10, 20, 50]):
        self.df = df.copy()
        self.price_col = price_col
        self.volume_col = volume_col
        self.lookback_windows = lookback_windows
        
    def calculate_vwap(self, window):
        pv = self.df[self.price_col] * self.df[self.volume_col]
        vwap = pv.rolling(window=window).sum() / self.df[self.volume_col].rolling(window=window).sum()
        return vwap

    def add_vwap_features(self):
        for window in self.lookback_windows:
            vwap = self.calculate_vwap(window)
            
            self.df[f'VWAP_{window}'] = vwap
            self.df[f'Price_Spread_{window}'] = self.df[self.price_col] - vwap
            self.df[f'VWAP_Crossover_{window}'] = np.where(self.df[self.price_col] > vwap, 1, 0)
            self.df[f'VWAP_Distance_{window}'] = (self.df[self.price_col] - vwap) / vwap
            
            std_dev = self.df[self.price_col].rolling(window=window).std()
            self.df[f'VWAP_Upper_Band_{window}'] = vwap + std_dev
            self.df[f'VWAP_Lower_Band_{window}'] = vwap - std_dev
            
            self.df[f'VWAP_Momentum_{window}'] = vwap.pct_change()

    def get_features(self):
        return self.df

class VWAPStrategies:
    def __init__(self, df):
        self.df = df

    def dual_vwap_crossover_strategy(self, short_vwap, long_vwap, short=False, sensitivity=1.0):
        if short:
            self.df['Signal'] = np.where(self.df[short_vwap] < self.df[long_vwap] * (1 - 0.01 * sensitivity), 1, 
                                 np.where(self.df[short_vwap] > self.df[long_vwap] * (1 + 0.01 * sensitivity), -1, 0))
        else:
            self.df['Signal'] = np.where(self.df[short_vwap] > self.df[long_vwap] * (1 + 0.01 * sensitivity), 1, 
                                 np.where(self.df[short_vwap] < self.df[long_vwap] * (1 - 0.01 * sensitivity), -1, 0))
        self.df['Position'] = self.df['Signal'].diff()
        print(f"Dual VWAP Crossover Strategy: {self.df['Signal'].value_counts()}")

    def vwap_rsi_strategy(self, vwap_column, rsi_column, rsi_overbought=70, rsi_oversold=30, short=False, sensitivity=1.0):
        rsi_overbought = 70 - (10 * sensitivity)
        rsi_oversold = 30 + (10 * sensitivity)
        if short:
            self.df['Signal'] = np.where((self.df['Close'] < self.df[vwap_column]) & (self.df[rsi_column] > rsi_overbought), 1,
                                 np.where((self.df['Close'] > self.df[vwap_column]) & (self.df[rsi_column] < rsi_oversold), -1, 0))
        else:
            self.df['Signal'] = np.where((self.df['Close'] > self.df[vwap_column]) & (self.df[rsi_column] < rsi_oversold), 1,
                                 np.where((self.df['Close'] < self.df[vwap_column]) & (self.df[rsi_column] > rsi_overbought), -1, 0))
        self.df['Position'] = self.df['Signal'].diff()
        print(f"VWAP-RSI Strategy: {self.df['Signal'].value_counts()}")

    def vwap_macd_strategy(self, vwap_column, macd_column, signal_column, short=False, sensitivity=1.0):
        if short:
            self.df['Signal'] = np.where((self.df['Close'] < self.df[vwap_column]) & (self.df[macd_column] < self.df[signal_column] * (1 - 0.1 * sensitivity)), 1,
                                 np.where((self.df['Close'] > self.df[vwap_column]) & (self.df[macd_column] > self.df[signal_column] * (1 + 0.1 * sensitivity)), -1, 0))
        else:
            self.df['Signal'] = np.where((self.df['Close'] > self.df[vwap_column]) & (self.df[macd_column] > self.df[signal_column] * (1 + 0.1 * sensitivity)), 1,
                                 np.where((self.df['Close'] < self.df[vwap_column]) & (self.df[macd_column] < self.df[signal_column] * (1 - 0.1 * sensitivity)), -1, 0))
        self.df['Position'] = self.df['Signal'].diff()
        print(f"VWAP-MACD Strategy: {self.df['Signal'].value_counts()}")

    def vwap_breakout_strategy(self, vwap_column, lookback=20, short=False, sensitivity=1.0):
        self.df['High_' + str(lookback)] = self.df['High'].rolling(window=lookback).max()
        self.df['Low_' + str(lookback)] = self.df['Low'].rolling(window=lookback).min()
        
        if short:
            self.df['Signal'] = np.where((self.df['Close'] < self.df['Low_' + str(lookback)]) | (self.df['Close'] < self.df[vwap_column]), 1,
                                 np.where((self.df['Close'] > self.df['High_' + str(lookback)]) | (self.df['Close'] > self.df[vwap_column]), -1, 0))
        else:
            self.df['Signal'] = np.where((self.df['Close'] > self.df['High_' + str(lookback)]) | (self.df['Close'] > self.df[vwap_column]), 1,
                                 np.where((self.df['Close'] < self.df['Low_' + str(lookback)]) | (self.df['Close'] < self.df[vwap_column]), -1, 0))
        self.df['Position'] = self.df['Signal'].diff()
        print(f"VWAP Breakout Strategy: {self.df['Signal'].value_counts()}")

    def vwap_volatility_strategy(self, vwap_column, atr_column, multiplier=2, short=False, sensitivity=1.0):
        self.df['Upper_Band'] = self.df[vwap_column] + (multiplier * sensitivity * self.df[atr_column])
        self.df['Lower_Band'] = self.df[vwap_column] - (multiplier * sensitivity * self.df[atr_column])
        
        if short:
            self.df['Signal'] = np.where(self.df['Close'] > self.df['Upper_Band'], -1,
                                 np.where(self.df['Close'] < self.df['Lower_Band'], 1, 0))
        else:
            self.df['Signal'] = np.where(self.df['Close'] > self.df['Upper_Band'], 1,
                                 np.where(self.df['Close'] < self.df['Lower_Band'], -1, 0))
        self.df['Position'] = self.df['Signal'].diff()
        print(f"VWAP Volatility Strategy: {self.df['Signal'].value_counts()}")

    def apply_strategy(self, strategy_func, *args, short=False, sensitivity=1.0):
        strategy_func(*args, short=short, sensitivity=sensitivity)
        return self.df[['Signal', 'Position']].copy()

def backtest_strategy(df, initial_capital=10000, strategy_name=""):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Profit'] = df['Cumulative_Returns'] * initial_capital

    total_return = df['Cumulative_Returns'].iloc[-1] - 1
    
    if df['Strategy_Returns'].std() != 0 and not np.isnan(df['Strategy_Returns'].std()):
        sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(365)
    else:
        sharpe_ratio = np.nan

    num_trades = (df['Position'] != 0).sum()

    return {
        'Strategy': strategy_name,
        'Total Return': total_return if num_trades > 0 else 0,
        'Sharpe Ratio': sharpe_ratio if num_trades > 0 else np.nan,
        'Final Profit': df['Cumulative_Profit'].iloc[-1] - initial_capital if num_trades > 0 else 0,
        'Number of Trades': num_trades
    }

# Example usage
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
    df_with_features['Position'] = 1  # Always fully invested
    buy_and_hold_result = backtest_strategy(df_with_features, strategy_name="Buy and Hold")
    results.append(buy_and_hold_result)
    print("Buy and Hold Strategy Result:")
    print(buy_and_hold_result)

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
            results.append(backtest_strategy(df_with_features, strategy_name=f"Long {name} (Sensitivity: {sensitivity})"))

            # Short strategy
            df_with_features[['Signal', 'Position']] = strategies.apply_strategy(strategy_func, *args, short=True, sensitivity=sensitivity)
            results.append(backtest_strategy(df_with_features, strategy_name=f"Short {name} (Sensitivity: {sensitivity})"))

    # Display results
    results_df = pd.DataFrame(results)
    print(results_df.sort_values('Total Return', ascending=False))

    # Print strategies with no trades
    no_trades = results_df[results_df['Number of Trades'] == 0]
    if not no_trades.empty:
        print("\nStrategies with no trades:")
        print(no_trades['Strategy'].tolist())

    # Print top 3 strategies
    print("\nTop 3 performing strategies:")
    print(results_df.sort_values('Total Return', ascending=False).head(3))

    # Print basic statistics about the input data
    print("\nData range:", df_with_features.index.min(), "to", df_with_features.index.max())
    print("Number of rows:", len(df_with_features))
    print("Close price range:", df_with_features['Close'].min(), "to", df_with_features['Close'].max())
    print("VWAP_20 range:", df_with_features['VWAP_20'].min(), "to", df_with_features['VWAP_20'].max())
    print("RSI_14 range:", df_with_features['RSI_14'].min(), "to", df_with_features['RSI_14'].max())

    # Save results to CSV
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    csv_filename = f'vwap_strategy_results_{date_time}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    # Diagnostic Information
    print("\nDiagnostic Information:")
    print(f"First Close price: {df_with_features['Close'].iloc[0]}")
    print(f"Last Close price: {df_with_features['Close'].iloc[-1]}")
    print(f"Total price change: {(df_with_features['Close'].iloc[-1] / df_with_features['Close'].iloc[0]) - 1:.2%}")

    # Calculate the buy and hold return manually
    manual_total_return = (df_with_features['Close'].iloc[-1] / df_with_features['Close'].iloc[0]) - 1
    print(f"Manually calculated total return: {manual_total_return:.2%}")


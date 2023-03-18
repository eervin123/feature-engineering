import pandas as pd
import numpy as np
import vectorbt as vbt

vbt.settings.set_theme("dark")

# Load historical Bitcoin data and preprocess it
def load_data():
    # Load hourly Bitcoin data, assuming you have a CSV file in your Documents folder
    file_path = '~/Documents/data/hourly_BTCUSDT.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df[['close']]  # We only need the closing price
    return df

# Define the strategy
def apply_strategy(df):
    # Calculate volatility
    hourly_returns = df['close'].pct_change()
    daily_returns = df['close'].pct_change(24) # lookst at the 24 hour returns
    last_seven_day_hourly_vol = hourly_returns.rolling(window=7 * 24).std()
    last_seven_day_daily_vol = daily_returns.rolling(window=7).std()
    # Calculate standard deviations these are in percentage terms
    # std1 = 1 * last_seven_day_daily_vol
    # std2 = 2 * last_seven_day_daily_vol
    # std3 = 3 * last_seven_day_daily_vol
    std1 = 1 * last_seven_day_hourly_vol
    std2 = 2 * last_seven_day_hourly_vol
    std3 = 3 * last_seven_day_hourly_vol

    # Calculate price thresholds
    upper1 = df['close'] + std1*df['close']
    upper2 = df['close'] + std2*df['close']
    upper3 = df['close'] + std3*df['close']
    lower1 = df['close'] - std1*df['close']
    lower2 = df['close'] - std2*df['close']
    lower3 = df['close'] - std3*df['close']

    # Filter rows at midnight
    midnight_returns = daily_returns.at_time('00:00') # this is the previous 24 hour returns as of midnight
    print(midnight_returns)
    

    # Determine long and short entries based on the previous day's returns
    long_entries = midnight_returns < 0 # if the previous day's returns are greater than 0, then we buy
    short_entries = midnight_returns > 0 # if the previous day's returns are less than 0, then we sell

    # Resample to the original index
    long_entries = long_entries.reindex(df.index).ffill().fillna(False)
    short_entries = short_entries.reindex(df.index).ffill().fillna(False)
    
    # Set up the long & short entries based on std deviations
    long_entries = np.where(df['close'] < lower1, True, long_entries)
    long_entries = np.where(df['close'] < lower2, True, long_entries)
    
    short_entries = np.where(df['close'] > upper1, True, short_entries)
    short_entries = np.where(df['close'] > upper2, True, short_entries)
  

    # Determine exits


    exits = pd.Series(False, index=df.index)
    # exits = (daily_returns > std3) | (daily_returns < -std3) # if the daily returns are greater than 3 std deviations, then we exit
    # Set exits to True at 20:00 UTC
    # print(df[df.index.hour == 23])
    exits = exits | (df.index.hour == 22) # if the hour is 22, then we exit on the close so we can re-enter the next day this basically closes at 23:00 UTC (closing candle)
    print(exits[exits==True])
    return long_entries, short_entries, exits


# Backtest the strategy using vectorbt
def backtest(df, long_entries, short_entries, exits):
    portfolio = vbt.Portfolio.from_signals(df['close'], long_entries, exits, short_entries, short_exits = exits, fees=0.001, freq='H')
    return portfolio

# Backtest using vectorbt from_order_func()




# Evaluate the results and perform walk-forward optimization
def analyze_results(portfolio, plot=True, trades_to_csv=True):
    # Display performance metrics
    print(portfolio.stats())
    # Plot default backtest dashboard
    if plot:
        portfolio.plot().show()
    # Save trades to CSV for later analysis
    if trades_to_csv:
        portfolio.trades.records_readable.to_csv('output/stdev_trades_analysis.csv')

    # Walk-forward optimization can be done using the vbt.WFOptimizer class
    # See https://vectorbt.io/docs/optimization.html for details

if __name__ == "__main__":
    df = load_data()
    long_entries, short_entries, exits = apply_strategy(df)
    portfolio = backtest(df, long_entries, short_entries, exits)
    analyze_results(portfolio, plot=True, trades_to_csv=True)


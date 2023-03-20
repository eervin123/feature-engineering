import numpy as np
import pandas as pd
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Stochastic strategy using Random Forest Regressor")
parser.add_argument('--k_period', type=int, default=14, help='K period for the Stochastic Oscillator (default: 60)')
parser.add_argument('--d_period', type=int, default=3, help='D period for the Stochastic Oscillator (default: 40)')
parser.add_argument('--forecast', type=int, default=5, help='Number of forecast periods into the future (default: 1)')
args = parser.parse_args()

# Load the historical price data
df = pd.read_csv('data/btc.csv', parse_dates=True, index_col=0)

# Calculate the stochastic oscillator
k_period = args.k_period
d_period = args.d_period
df['stoch_k'], df['stoch_d'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
df['stoch_signal'] = (df['stoch_k']/df['stoch_d'])-1 # percent that the stochastic k is above or below stochastic d

# Fit the random forest model
forecast = args.forecast # enter number of forecast minutes or periods in the future
logging.info(f"Forecasting {forecast} periods into the future based on {k_period} and {d_period} periods")
df = df.dropna(subset=['stoch_signal'])
X = df['stoch_signal'].iloc[:-forecast].values.reshape(-1, 1)
y = df['Close'].pct_change(forecast).dropna().iloc[:-forecast].values.reshape(-1, 1)

min_length = min(len(X), len(y))
X = X[:min_length]
y = y[:min_length]

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X, y.ravel())

# Calculate the prediction accuracy
y_val = df['Close'].shift(-forecast).pct_change(forecast).dropna()
X_val = df['stoch_signal'][-len(y_val):]
X_val = X_val.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)

y_pred = model.predict(X_val)

valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_val.ravel())
y_pred = y_pred[valid_indices]
y_val = y_val[valid_indices]

mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
threshold = 0.0005
correct_predictions = np.abs(y_pred - y_val.ravel()) <= threshold
accuracy = np.mean(correct_predictions) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Accuracy: {accuracy}%")

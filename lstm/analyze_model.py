import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import load_model

# Load the saved model
model = load_model('models/lstm_50_64.h5')

# Load and preprocess the data as before
data = pd.read_csv('data/btcusdt_1m.csv')
data['Open time'] = pd.to_datetime(data['Open time'])
data = data.set_index('Open time')
data = data['Open']
data_hourly = data.resample('1H').first()
data_hourly = data_hourly.fillna(method='ffill')
print(data_hourly)
data_reshaped = data_hourly.values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped)

# Prepare the dataset with the desired time lags
def create_lagged_dataset(data, n_in, intervals):
    X, y = [], []
    n_out = max(intervals)
    
    for i in range(n_in, len(data) - n_out):
        X.append(data[i - n_in:i])
        y.append([data[i + interval - 1] for interval in intervals])

    return np.array(X), np.array(y)

n_in = 240  # Number of time steps to look back
intervals = [1, 2, 4, 6, 8]  # Desired prediction intervals in hours

X, y = create_lagged_dataset(data_scaled, n_in, intervals)

# Train-test split
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Assume you already have X_train, X_test, y_train, and y_test from your previous script
# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Inverse transform the predictions and targets
y_train_inv = np.array([scaler.inverse_transform(y_train[:, i].reshape(-1, 1)).flatten() for i in range(y_train.shape[1])]).T
y_test_inv = np.array([scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten() for i in range(y_test.shape[1])]).T
y_pred_train_inv = np.array([scaler.inverse_transform(y_pred_train[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_train.shape[1])]).T
y_pred_test_inv = np.array([scaler.inverse_transform(y_pred_test[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_test.shape[1])]).T

import matplotlib.dates as mdates

# Plot the actual vs. predicted values for a selected interval
selected_interval = 0
start_datetime = data_hourly.index[n_in:-max(intervals)]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
# Training data and predictions
train_start_datetime = data_hourly.index[n_in: n_in + train_size]
ax1.plot(train_start_datetime, y_train_inv[:, selected_interval], label='Train Actual', color='blue')
ax1.plot(train_start_datetime, y_pred_train_inv[:, selected_interval], label='Train Predicted', color='green')

test_start_datetime = data_hourly.index[n_in + train_size: -max(intervals)]
ax1.plot(test_start_datetime, y_test_inv[:, selected_interval], label='Test Actual', color='red')
ax1.plot(test_start_datetime, y_pred_test_inv[:, selected_interval], label='Test Predicted', color='orange')
ax1.set_title(f'BTC Hourly Prices - Actual vs. Predicted ({intervals[selected_interval]} hour interval)')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

# Difference plot for both training and test data as a percentage
train_difference_percentage = ((y_pred_train_inv[:, selected_interval] - y_train_inv[:, selected_interval]) / y_train_inv[:, selected_interval]) * 100
test_difference_percentage = ((y_pred_test_inv[:, selected_interval] - y_test_inv[:, selected_interval]) / y_test_inv[:, selected_interval]) * 100

ax2.plot(train_start_datetime, train_difference_percentage, label='Train Difference', color='g')
ax2.plot(test_start_datetime, test_difference_percentage, label='Test Difference', color='r')
ax2.set_xlabel('Hours')
ax2.set_ylabel('Difference (%)')
ax2.legend(loc='upper left')

from numpy import mean
# Calculate the average difference of the test data
avg_difference_test = mean(test_difference_percentage)

# Estimate the profits based on the average difference
profits = 0
num_trades = len(test_start_datetime)

for i in range(num_trades):
    if test_difference_percentage[i] > 0:  # Predicted increase in price
        profits += (test_difference_percentage[i])
    else:  # Predicted decrease in price
        profits += (test_difference_percentage[i])

# Calculate average profit per trade
average_profit_per_trade = profits / num_trades

print(f"Average difference in test data: {avg_difference_test:.2f}%")
print(f"Estimated average profit per trade: {average_profit_per_trade:.2f}%")

# Initialize the investment and equity curve
initial_investment = 1000
equity_curve = [initial_investment]

# Calculate the equity curve
for i in range(num_trades - 1):  # Change the range to 'num_trades - 1' to avoid IndexError
    actual_diff = (y_test_inv[i + 1, selected_interval] - y_test_inv[i, selected_interval]) / y_test_inv[i, selected_interval] * 100
    predicted_diff = test_difference_percentage[i]
    
    # If the prediction and actual price movement have the same sign, the trade is successful
    if np.sign(predicted_diff) == np.sign(actual_diff):
        profit = equity_curve[-1] * abs(predicted_diff) / 100
    else:  # If the prediction and actual price movement have opposite signs, the trade is a loss
        profit = -equity_curve[-1] * abs(predicted_diff) / 100
    
    equity_curve.append(equity_curve[-1] + profit)

# Plot the equity curve
plt.figure(figsize=(15, 6))
plt.plot(test_start_datetime[:-1], equity_curve[1:])  # Remove the last datetime to match the length of equity_curve
plt.title('Hypothetical Equity Curve ($1000 Initial Investment)')
plt.xlabel('Hours')
plt.ylabel('Equity ($)')
plt.show()

print(f"Final equity: ${equity_curve[-1]:.2f}")


# Format the x-axis ticks
date_fmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax2.xaxis.set_major_formatter(date_fmt)
fig.autofmt_xdate()

fig.tight_layout()
plt.show()

# Plot the mean absolute error for each interval
train_mae = [mean_absolute_error(y_train_inv[:, i], y_pred_train_inv[:, i]) for i in range(y_train_inv.shape[1])]
test_mae = [mean_absolute_error(y_test_inv[:, i], y_pred_test_inv[:, i]) for i in range(y_test_inv.shape[1])]

plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(intervals)), train_mae, width=0.4, label='Train MAE')
plt.bar(np.arange(len(intervals)) + 0.4, test_mae, width=0.4, label='Test MAE')
plt.xticks(np.arange(len(intervals)) + 0.2, [f'{interval} hours' for interval in intervals])
plt.title('Mean Absolute Error for Different Prediction Intervals')
plt.ylabel('MAE')
plt.legend()
plt.show()

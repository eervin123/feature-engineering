import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import load_model

def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    data['Open time'] = pd.to_datetime(data['Open time'])
    data = data.set_index('Open time')
    data = data['Open']
    data_hourly = data.resample('1H').first()
    data_hourly = data_hourly.fillna(method='ffill')
    
    return data_hourly

def scale_data(data_reshaped):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    return data_scaled, scaler

def create_lagged_dataset(data, n_in, intervals):
    X, y = [], []
    n_out = max(intervals)
    
    for i in range(n_in, len(data) - n_out):
        X.append(data[i - n_in:i])
        y.append([data[i + interval - 1] for interval in intervals])

    return np.array(X), np.array(y)

def train_test_split(X, y, train_size):
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, X_test, y_train, y_test

def calculate_mae(y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv):
    train_mae = [mean_absolute_error(y_train_inv[:, i], y_pred_train_inv[:, i]) for i in range(y_train_inv.shape[1])]
    test_mae = [mean_absolute_error(y_test_inv[:, i], y_pred_test_inv[:, i]) for i in range(y_test_inv.shape[1])]
    return train_mae, test_mae

def plot_mae(train_mae, test_mae, intervals):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(intervals)), train_mae, width=0.4, label='Train MAE')
    plt.bar(np.arange(len(intervals)) + 0.4, test_mae, width=0.4, label='Test MAE')
    plt.xticks(np.arange(len(intervals)) + 0.2, [f'{interval} hours' for interval in intervals])
    plt.title('Mean Absolute Error for Different Prediction Intervals')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

def make_predictions(model, X_train, X_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test

def inverse_transform(scaler, y_train, y_test, y_pred_train, y_pred_test):
    y_train_inv = np.array([scaler.inverse_transform(y_train[:, i].reshape(-1, 1)).flatten() for i in range(y_train.shape[1])]).T
    y_test_inv = np.array([scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten() for i in range(y_test.shape[1])]).T
    y_pred_train_inv = np.array([scaler.inverse_transform(y_pred_train[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_train.shape[1])]).T
    y_pred_test_inv = np.array([scaler.inverse_transform(y_pred_test[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_test.shape[1])]).T
    return y_train_inv, y_test_inv, y_pred_train_inv, y_pred_test_inv

def plot_actual_vs_predicted(start_datetime, selected_interval, train_size, y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    ax1.plot(start_datetime[:train_size], y_train_inv[:, selected_interval], label='Actual')
    ax1.plot(start_datetime[:train_size], y_pred_train_inv[:, selected_interval], label='Predicted')
    ax1.set_title('Training Set - Actual vs Predicted')
    ax1.legend()
    
    ax2.plot(start_datetime[train_size + n_in:], y_test_inv[n_in:, selected_interval], label='Actual')


    ax2.plot(start_datetime[train_size:], y_pred_test_inv[:, selected_interval], label='Predicted')
    ax2.set_title('Test Set - Actual vs Predicted')
    ax2.legend()
    
    plt.show()
    
    return fig, (ax1, ax2)

def calculate_metrics(y_true, y_pred, selected_interval):
    test_difference_percentage = (y_pred[:, selected_interval] - y_true[:, selected_interval]) / y_true[:, selected_interval] * 100
    avg_difference_test = np.mean(np.abs(test_difference_percentage))
    
    trades = len(test_difference_percentage)
    average_difference_per_trade = (test_difference_percentage.sum() / trades)
    
    return avg_difference_test, average_difference_per_trade


def calculate_equity_curve(test_difference_percentage, initial_balance=1_000):
    equity_curve = [initial_balance]
    balance = initial_balance

    for i, percentage_difference in enumerate(test_difference_percentage):
        pnl = balance * (percentage_difference / 100)
        balance += pnl
        equity_curve.append(balance)
        print(f"Step {i}: Percentage difference: {percentage_difference:.2f}%, PnL: {pnl:.2f}, Balance: {balance:.2f}")

    return equity_curve

def plot_equity_curve(test_start_datetime, equity_curve):
    plt.figure(figsize=(15, 5))
    
    min_length = min(len(test_start_datetime), len(equity_curve))
    print(f'Minimum length: {min_length}')
    
    test_start_datetime = test_start_datetime[:min_length]
    equity_curve = equity_curve[:min_length]

    plt.plot(test_start_datetime, equity_curve)
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.show()

def calculate_raw_differences(y_true, y_pred, selected_interval):
    raw_differences = y_pred[:, selected_interval] - y_true[:, selected_interval]
    return raw_differences

def plot_raw_differences(test_start_datetime, raw_differences):
    plt.figure(figsize=(15, 5))

    min_length = min(len(test_start_datetime), len(raw_differences))
    print(f'Minimum length: {min_length}')
    
    test_start_datetime = test_start_datetime[:min_length]
    raw_differences = raw_differences[:min_length]

    plt.plot(test_start_datetime, raw_differences)
    plt.xlabel('Time')
    plt.ylabel('Raw Difference')
    plt.title('Raw Differences')
    plt.show()

def save_results_to_dataframe(start_datetime, y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv, intervals):
    actual_train_cols = [f'actual_{interval}' for interval in intervals]
    predicted_train_cols = [f'predicted_{interval}' for interval in intervals]

    actual_test_cols = [f'actual_{interval}' for interval in intervals]
    predicted_test_cols = [f'predicted_{interval}' for interval in intervals]

    train_index = start_datetime[:len(y_train_inv)]
    test_index = start_datetime[len(y_train_inv)+max(intervals):]

    train_actual = pd.DataFrame(y_train_inv[:, :len(intervals)], index=train_index, columns=actual_train_cols)
    train_predicted = pd.DataFrame(y_pred_train_inv[:, :len(intervals)], index=train_index, columns=predicted_train_cols)

    test_actual = pd.DataFrame(y_test_inv[max(intervals):, :len(intervals)], index=test_index, columns=actual_test_cols)
    test_predicted = pd.DataFrame(y_pred_test_inv[max(intervals):, :len(intervals)], index=test_index, columns=predicted_test_cols)

    train_results = pd.concat([train_actual, train_predicted], axis=1)
    test_results = pd.concat([test_actual, test_predicted], axis=1)

    return train_results, test_results



def save_to_csv(df_train, df_test, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_train.to_csv(os.path.join(output_dir, 'train_results.csv'))
    df_test.to_csv(os.path.join(output_dir, 'test_results.csv'))

# Load the saved model
model = load_model('models/lstm_50_64.h5')

# Load and preprocess the data as before
data_hourly = load_and_preprocess_data('data/btcusdt_1m.csv')
data_reshaped = data_hourly.values.reshape(-1, 1)
data_scaled, scaler = scale_data(data_reshaped)

# Prepare the dataset with the desired time lags
n_in = 240  # Number of time steps to look back
intervals = [1, 2, 4, 6, 8]  # Desired prediction intervals in hours

X, y = create_lagged_dataset(data_scaled, n_in, intervals)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size)

# Make predictions
y_pred_train, y_pred_test = make_predictions(model, X_train, X_test)

# Inverse transform the predictions and targets
y_train_inv, y_test_inv, y_pred_train_inv, y_pred_test_inv = inverse_transform(scaler, y_train, y_test, y_pred_train, y_pred_test)

# Plot the actual vs. predicted values for a selected interval
selected_interval = 0
start_datetime = data_hourly.index[n_in:-max(intervals)]
start_datetime = start_datetime[:len(X_train) + len(X_test)]
test_start_datetime = start_datetime[train_size+n_in:-max(intervals)]
print(f'Start datetime length: {len(start_datetime)}')
print(f'Test start datetime length: {len(test_start_datetime)}')

fig, (ax1, ax2) = plot_actual_vs_predicted(start_datetime, selected_interval, train_size, y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv)

# Plot the mean absolute error for each interval
train_mae, test_mae = calculate_mae(y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv)
plot_mae(train_mae, test_mae, intervals)

# Calculate and display performance metrics
avg_difference_test, average_difference_per_trade = calculate_metrics(y_test_inv, y_pred_test_inv, selected_interval)


print(f"Estimated average difference per trade: {average_difference_per_trade:.2f}%")
print(f"Average difference in test data: {avg_difference_test:.2f}%")


# Calculate and plot the equity curve
test_difference_percentage = (y_pred_test_inv[:, selected_interval] - y_test_inv[:, selected_interval]) / y_test_inv[:, selected_interval] * 100
raw_differences = calculate_raw_differences(y_test_inv, y_pred_test_inv, selected_interval)
plot_raw_differences(test_start_datetime, raw_differences)

equity_curve = calculate_equity_curve(test_difference_percentage, initial_balance=1_000)

plot_equity_curve(test_start_datetime[:len(equity_curve)-1], equity_curve[:-1])

print(f"Final equity: ${equity_curve[-1]:.2f}")

# Save the results to DataFrames and then export them to CSV files
train_results, test_results = save_results_to_dataframe(start_datetime, y_train_inv, y_pred_train_inv, y_test_inv, y_pred_test_inv, intervals)
save_to_csv(train_results, test_results)
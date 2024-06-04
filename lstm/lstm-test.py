import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Load and preprocess the data
data = pd.read_csv('data/btcusdt_1m.csv')
data['Open time'] = pd.to_datetime(data['Open time'])
data = data.set_index('Open time')
data = data['Open']
data_hourly = data.resample('1H').first()
# print how many NaN values are in the data
print(f'Number of NaN values: {data_hourly.isna().sum()}')
# fill the NaN values with the previous value
data_hourly = data_hourly.fillna(method='ffill')
# Check for NaN and Inf values
assert not np.any(np.isnan(data_hourly)), "Data contains NaN values"
assert not np.any(np.isinf(data_hourly)), "Data contains Inf values"

# Reshape the data to a 2D array
data_reshaped = data_hourly.values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped)  # Use the reshaped data here

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

# Build and train the LSTM model
def create_lstm_model(input_shape, n_out):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(n_out))
    opt = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(loss='mse', optimizer=opt)
    return model

model = create_lstm_model(X_train.shape[1:], y_train.shape[1])
epochs = 50
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])


# Make predictions and evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Inverse transform the predictions and targets
y_train_inv = np.array([scaler.inverse_transform(y_train[:, i].reshape(-1, 1)).flatten() for i in range(y_train.shape[1])]).T
y_test_inv = np.array([scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten() for i in range(y_test.shape[1])]).T
y_pred_train_inv = np.array([scaler.inverse_transform(y_pred_train[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_train.shape[1])]).T
y_pred_test_inv = np.array([scaler.inverse_transform(y_pred_test[:, i].reshape(-1, 1)).flatten() for i in range(y_pred_test.shape[1])]).T

# Calculate the mean absolute error for each interval
train_mae = [mean_absolute_error(y_train_inv[:, i], y_pred_train_inv[:, i]) for i in range(y_train_inv.shape[1])]
test_mae = [mean_absolute_error(y_test_inv[:, i], y_pred_test_inv[:, i]) for i in range(y_test_inv.shape[1])]

for i, (interval, mae_train, mae_test) in enumerate(zip(intervals, train_mae, test_mae)):
    print(f'Interval {i + 1} ({interval} hours): Train MAE = {mae_train:.2f}, Test MAE = {mae_test:.2f}')

model.save(f'models/lstm_{epochs}_{batch_size}.h5')
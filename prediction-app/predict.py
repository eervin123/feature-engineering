import logging
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

def make_prediction(data, forecast_period, model_file: str):
    # Load the model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Use the model to make predictions for the given forecast period
    forecast = forecast_period # minutes into the future to predict
    data_filter = len(data) * 1.00 # 100% of the data will be used for training
    df = data.iloc[:int(data_filter)] # full data
    cols = df.columns

    # Create the features (X) and target (y) data using numpy
    X = df[cols].iloc[:-forecast].values # signal and volume
    y = df['Close'].pct_change(forecast).dropna().iloc[:-forecast].values.reshape(-1, 1) # future

    # Split the data into training and testing sets and make sure they are the same length
    min_length = min(len(X), len(y))
    X = X[:min_length]
    y = y[:min_length]

    # Use the model on the full dataset
    logging.info(f"Applying model to full dataset")
    y_val = df['Close'].shift(-forecast).pct_change(forecast).dropna() # future
    X_val = df[cols][-len(y_val):] # signal and volume
    y_val = y_val.values.reshape(-1, 1) # reshape to 2D
    # make predictions
    logging.info(f"Making predictions")
    y_pred = model.predict(X_val.values)
    logging.info(f"Predictions made")

    # remove NaNs and infinities
    valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_val.ravel()) # remove NaNs
    y_pred = y_pred[valid_indices]
    y_val = y_val[valid_indices]
    
    # Calculate the predicted price change
    current_price = df['Close'].iloc[-1]
    logging.info(f"Current price: {current_price} the last several predictions are {y_pred[-10:]}")
    forecast_price = y_pred[-1] * current_price + current_price
    price_change = y_pred[-1] * current_price
    logging.info(f"Forecast price: {forecast_price} the price change is {price_change}")

    # Return the predictions, actual values, and predicted price change
    return y_pred, y_val, price_change

def calculate_and_print_results(y_val, y_pred, threshold=0.0005, forecast_period: int = 10):

    mse = mean_squared_error(y_pred, y_val)
    rmse = np.sqrt(mse)
    correct_predictions = np.abs(y_pred - y_val.ravel()) <= threshold
    direction_correct = (np.sign(y_pred) == np.sign(y_val.ravel())).astype(int)
    direction_accuracy = direction_correct.mean() * 100
    precision_accuracy = np.mean(correct_predictions) * 100
    r2 = r2_score(y_val, y_pred)

    print(f"This model was trained on all of 2022 and \
        through March 2023. It is updated nightly. Beware of overfitting. \n")
    print("The model statistics are as follows: \n")
    print(f"For a threshold of: {threshold}")
    print(f"Forecast period: {forecast_period} minutes")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse:.4f}")
    # print accuracy within 4 decimal places
    print(f"Precision Accuracy: {precision_accuracy:.4f}% Percent of observations within the threshold of {threshold}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}% This happened {direction_correct.sum()} times out of {len(direction_correct)}")
    print(f"R-squared: {r2:.4f}")
    results = {
        'threshold': threshold,
        'forecast period': forecast_period,
        'mse': mse,
        'rmse': rmse,
        'precision_accuracy': precision_accuracy,
        'direction_accuracy': direction_accuracy,
        'r2': r2
    }
    return results

def print_forecast(forecast, price_change, data):
    if price_change >= 0:
        up_or_down = 'up'
    else:
        up_or_down = 'down'
    forecast_change_in_percent = abs(price_change / data['Close'].iloc[-1]) * 100
    print(f"In the next {forecast} minutes, the model predicts bitcoin will go {up_or_down} {forecast_change_in_percent:.4f}%")
    print(f"The current price is {data['Close'].iloc[-1]} and the forecast price is {data['Close'].iloc[-1] + price_change}")


if __name__ == "__main__":
    forecast = 60 # minutes into the future to predict
    file_path = 'data/btc.csv'
    model_folder_path = 'prediction-app/models'
    # Look through the models folder for the model with the correct forecast period
    model_file = [f for f in os.listdir(model_folder_path) if f.startswith(f'rf_model_forecast_{forecast}')]
    model_file = os.path.join(model_folder_path, model_file[0])
    print(model_file)
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    filter = int(len(data) * 1.00) # % of the data will be used for training
    filtered_df = data.iloc[-filter:] # slice of the data
    y_pred, y_val, price_change = make_prediction(filtered_df, forecast_period=forecast, model_file=model_file)
    print('\n') # new line
    print_forecast(forecast, price_change, filtered_df)
    print('\n') # new line
    stats = calculate_and_print_results(y_val=y_val, y_pred=y_pred, threshold=0.001, forecast_period=forecast)

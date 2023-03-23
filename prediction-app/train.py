# train.py
'''
This file is used to train the models and save them to disk
The idea is to save the models to disk and then load them in the app.py file
We will set up a cron job to run this file every evening. It takes about 90 minutes to train 6 models
'''
import logging
import pandas as pd
import pandas_ta as ta
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading data")
    data = pd.read_csv('data/btc.csv', index_col=0, parse_dates=True)
    return data

def train_models(data, forecast_periods):
    cols = data.columns

    for forecast_period in forecast_periods:
        logging.info(f"Training model for {forecast_period} mins forward")

        # Create the features (X) and target (y) data using numpy
        X = data[cols].iloc[:-forecast_period].values
        y = data['Close'].pct_change(forecast_period).dropna().iloc[:-forecast_period].values

        # Split the data into training and testing sets and make sure they are the same length
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]

        # Create the model and fit it to the data
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

        # Save the model
        now = datetime.utcnow()
        timestamp = now.strftime('%Y-%m-%dT%H-%M-%SZ')
        file_name = f"prediction-app/models/rf_model_forecast_{forecast_period}_timestamp_{timestamp}.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)

        logging.info(f"Model saved to {file_name}")

if __name__ == '__main__':
    data = load_data()
    forecast_periods = [5, 10, 15, 20, 30, 60]
    train_models(data, forecast_periods)

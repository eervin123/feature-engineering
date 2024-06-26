'''
This file loads the data updates it with new data, adds features and then saves it back to disk
This is not efficient but it is working for now. 
TODO: change the save file to be `btc_with_features.csv` and then load that file and append the new data to it
we may also want to leave all the features on the original data rather than recreating them every time we get new data. 
for now it is quick and easy to just recreate them every time we get new data.
'''

import logging
import pandas_ta as ta
import pandas as pd
import vectorbtpro as vbt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BTC_DATA_PATH = 'data/btc.csv'

def preprocess_data(data: pd.DataFrame):
    try:
        logging.info("Preprocessing data")
        momo_bands_sma_ta = [
            {"kind": "sma", "length": 20000},
            {"kind": "sma", "length": 1000},
            {"kind": "ema", "length": 10000},
            {"kind": "ema", "length": 5000},
            {"kind": "bbands", "length": 2000, "ddof": 0},
            {"kind": "macd"},
            {"kind": "rsi"},
            {"kind": "log_return", "cumulative": True},
            {"kind": "sma", "close": "CUMLOGRET_1", "length": 50, "suffix": "CUMLOGRET"},
        ]
        momo_bands_sma_strategy = ta.Strategy(
            "Momo, Bands, MAs and Cumulative Log Returns",
            momo_bands_sma_ta,
            "MACD, RSI, Momo with BBANDS and EMAs, SMAs and Cumulative Log Returns"
        )
        data.ta.strategy(momo_bands_sma_strategy, append=True)
        data.dropna(inplace=True)
        logging.info(f"Feature Data Columns: {data.columns}")
        return data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None

def load_data(from_file: bool = True, start_date: str = '2023-01-01', end_date: str = 'now UTC', file_path: str = BTC_DATA_PATH):
    try:
        if from_file:
            logging.info("Loading data from file...")
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            logging.info(f"Downloading data from Binance...start_date: {start_date}, end_date: {end_date}")
            data = vbt.BinanceData.download('BTCUSDT', interval='1m', start=start_date, end=end_date).get(['Close', 'Volume'])

        data = data[['Close', 'Volume']]
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def save_data(data: pd.DataFrame, file_path: str = BTC_DATA_PATH):
    logging.info(f"Saving data to {file_path}")
    data.to_csv(file_path)
    logging.info("Data saved")

def preprocess_and_save_data(data: pd.DataFrame, file_path: str = BTC_DATA_PATH):
    data = preprocess_data(data)
    save_data(data, file_path)
    return data

def load_preprocess_and_save_data(load_data=load_data, preprocess_and_save_data=preprocess_and_save_data):
    file_path = BTC_DATA_PATH

    # Load the data
    data = load_data(from_file=True, file_path=file_path)
    last_date = data.index[-1]
    logging.info(f"Last date in data is {last_date}")

    # Get the new data
    new_data = load_data(from_file=False, start_date=last_date, end_date='now UTC', file_path=file_path)
    
    # Update the data and drop duplicates 
    data = pd.concat([data, new_data])
    logging.info(f"The index range is now {data.index[0]} to {data.index[-1]}")
    data = data[~data.index.duplicated(keep='first')]  # drop the duplicates if there are any
    logging.info(f"After dropped duplicates func, index is now {data.index[0]} to {data.index[-1]}")
    
    # Preprocess and save the data
    data = preprocess_and_save_data(data, file_path)

if __name__ == '__main__':
    load_preprocess_and_save_data(load_data, preprocess_and_save_data)

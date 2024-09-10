from vectorbtpro import *
from regimes_nb import MarketRegimeDetector_1d
import logging
from tuneta.tune_ta import TuneTA

import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Filter out warnings
warnings.filterwarnings('ignore')
# Fetch data separately for BTC and ETH
btc_data = vbt.YFData.fetch('BTC-USD', end='2024-01-01').get()
eth_data = vbt.YFData.fetch('ETH-USD', end='2024-01-01').get()

# Save regimes to csv
btc_data_with_regimes = pd.read_csv('backtest-notebooks/notebook-results/btc_data_with_regimes.csv', index_col=0)
eth_data_with_regimes = pd.read_csv('backtest-notebooks/notebook-results/eth_data_with_regimes.csv', index_col=0)
btc_data_with_regimes.index = pd.to_datetime(btc_data_with_regimes.index)
eth_data_with_regimes.index = pd.to_datetime(eth_data_with_regimes.index)
# Drop the stock splits and dividends from the data
btc_data_with_regimes = btc_data_with_regimes.drop(columns=['Stock Splits', 'Dividends'])
eth_data_with_regimes = eth_data_with_regimes.drop(columns=['Stock Splits', 'Dividends'])
print(btc_data_with_regimes.columns)
print(btc_data_with_regimes.index)
btc_X = btc_data_with_regimes.drop(columns=['Market Regime'])
btc_y = btc_data_with_regimes['Market Regime'] 

eth_X = eth_data_with_regimes.drop(columns=['Market Regime'])
eth_y = eth_data_with_regimes['Market Regime'] 


# Drop the first row because Return is NaN
X = btc_X.iloc[1:] 
y = btc_y.iloc[1:]

tt = TuneTA(n_jobs=8, verbose=True)

tt.fit(X, y,
       indicators=['tta'],
       ranges=[(4,100)],
       trials=20,
       early_stop=5,
       )
tt.report(target_corr=True, features_corr=True)
features = tt.transform(X)
features.to_csv('backtest-notebooks/notebook-results/btc_tta_features.csv')
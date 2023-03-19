from ta_optimizer import TechnicalIndicatorOptimizer
import logging
import os
import pandas_ta as ta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data into a Pandas DataFrame
df = pd.read_csv('btc.csv', parse_dates=True, index_col=0)
model_store = 'models/'
os.makedirs(model_store, exist_ok=True) # create the models directory if it doesn't exist


# Example usage with SMA
sma_optimizer = TechnicalIndicatorOptimizer(
    df, 
    ta.sma, 
    "sma", 
    "length", 
    [5, 10, 20, 50, 100], 
    [5, 30, 60, 90, 120]
)
sma_best_result = sma_optimizer.optimize()
print("SMA Best Result:", sma_best_result)

# Example usage with EMA
ema_optimizer = TechnicalIndicatorOptimizer(
    df, 
    ta.ema, 
    "ema", 
    "length", 
    [5, 10, 20, 50, 100], 
    [5, 30, 60, 90, 120]
)
ema_best_result = ema_optimizer.optimize()
print("EMA Best Result:", ema_best_result)

sma_results = [sma_optimizer.evaluate_indicator_param(param_value) for param_value in sma_optimizer.param_values]
ema_results = [ema_optimizer.evaluate_indicator_param(param_value) for param_value in ema_optimizer.param_values]

sma_optimizer.plot_analysis_dashboard(sma_results, plot_title='SMA Optimization Dashboard')
ema_optimizer.plot_analysis_dashboard(ema_results, plot_title='EMA Optimization Dashboard')

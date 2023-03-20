from ta_optimizer import TechnicalIndicatorOptimizer
import logging
import os
import pandas_ta as ta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data into a Pandas DataFrame
df = pd.read_csv('data/btc.csv', parse_dates=True, index_col=0)
model_store = 'models/'
os.makedirs(model_store, exist_ok=True) # create the models directory if it doesn't exist


# SMA
logging.info("Optimizing SMA")
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

# EMA
logging.info("Optimizing EMA")
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

# Stochastic
logging.info("Optimizing Stochastic")
stoch_optimizer = TechnicalIndicatorOptimizer(
    df,
    None,
    "stoch",
    ["k_period", "d_period"],
    [(5, 3), (14, 3), (14, 5)],
    [5, 30, 60, 90, 120],
    indicator_type='stochastic'
)
stoch_best_result = stoch_optimizer.optimize()
print("Stochastic Best Result:", stoch_best_result)

# Plot the analysis results for the two different moving average indicators
sma_results = [sma_optimizer.evaluate_indicator_param(param_value) for param_value in sma_optimizer.param_values]
ema_results = [ema_optimizer.evaluate_indicator_param(param_value) for param_value in ema_optimizer.param_values]
# Generate the analysis results for the different Stochastic parameter values
stoch_results = [stoch_optimizer.evaluate_indicator_param(param_value) for param_value in stoch_optimizer.param_values]


logging.info("Plotting SMA Analysis Dashboard")
sma_optimizer.plot_analysis_dashboard(
    sma_results, title='SMA Optimization Dashboard')
logging.info("Plotting EMA Analysis Dashboard")
ema_optimizer.plot_analysis_dashboard(
    ema_results, title='EMA Optimization Dashboard')
logging.info("Plotting Stochastic Analysis Dashboard")
stoch_optimizer.plot_analysis_dashboard(
    stoch_results, title='Stochastic Optimization Dashboard')

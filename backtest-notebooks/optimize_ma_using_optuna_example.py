import optuna
import yfinance as yf
import numpy as np
import vectorbtpro as vbt
import pandas as pd
vbt.settings.set_theme('dark')
import warnings
warnings.filterwarnings("ignore")

def backtest_strategy(data, short_window, long_window, vol_threshold):
    # Apply the strategy
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()
    
    # Conditions 
    high_vol = data['Volatility'] > vol_threshold
    low_vol = ~high_vol
    up_trend = data['Close'] > short_ma
    down_trend = data['Close'] < short_ma
    
    entries = up_trend & low_vol
    short_entries = down_trend & low_vol
    exits = down_trend | high_vol
    short_exits = up_trend | high_vol
    
    pf = vbt.Portfolio.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )
    
    return pf

def objective(trial, data):
    short_window = trial.suggest_int('short_window', 1, 50)
    long_window = trial.suggest_int('long_window', 20, 200)
    
    if short_window >= long_window:
        return float('inf')
    
    vol_threshold = trial.suggest_float('vol_threshold', data['Volatility'].min(), data['Volatility'].max())
    
    pf = backtest_strategy(data, short_window, long_window, vol_threshold)
    
    return -pf.total_return

def create_labels(data, short_window, long_window, vol_threshold):
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()
    
    high_vol = data['Volatility'] > vol_threshold
    low_vol = ~high_vol
    up_trend = data['Close'] > short_ma
    down_trend = data['Close'] < short_ma
    
    labels = pd.Series(0, index=data.index)
    
    labels[up_trend & low_vol] = 1          # Low Volatility Uptrend
    labels[down_trend & low_vol] = -1       # Low Volatility Downtrend
    labels[up_trend & high_vol] = 0.5       # High Volatility Uptrend
    labels[down_trend & high_vol] = -0.5    # High Volatility Downtrend
    
    return labels

def plot_charts_with_heatmap(pf, data):
    # Create the subplots layout
    fig = pf.plot(subplots=[
        'orders',
        'trade_pnl',
        'cum_returns',
        'drawdowns',
        'underwater',
        'gross_exposure',
    ], make_subplots_kwargs=dict(
        rows=4,  # Adding an extra row for the heatmap
        cols=2,
        shared_xaxes=True,
        specs=[[{}, {}], [{}, {}], [{}, {}], [{"secondary_y": True}, None]]  # Secondary y-axis enabled in the last row, first column
    ),
    subplot_settings=dict(
        cum_returns=dict(pct_scale=True),
        underwater=dict(pct_scale=True),
    ),  
    )

    fig['layout'].update(height=800, width=1200)
    
    # Create the heatmap overlay
    data['Close'].vbt.overlay_with_heatmap(data['Labels'], title='Regime Overlay')

    fig.add_trace(
        data['Close'].vbt.overlay_with_heatmap(data['Labels'], title='Regime Overlay').data[0],
        row=4, col=1
    )
    fig.add_trace(
        data['Close'].vbt.overlay_with_heatmap(data['Labels'], title='Regime Overlay').data[1],
        row=4, col=1, 
        secondary_y=True
    )
    fig.show()



# Data Download and Preparation
data = yf.download("BTC-USD", period="5y", interval="1d", auto_adjust=True)
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std() * np.sqrt(365)

# Optimize the strategy
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, data), n_trials=1000)

# Print the best parameters
best_params = study.best_params
print('Best parameters: ', best_params)
print('Best value: ', -study.best_value) #using a negative to cancel out the negative sign in the objective function

# Backtest with the best parameters
best_pf = backtest_strategy(
    data,
    best_params['short_window'],
    best_params['long_window'],
    best_params['vol_threshold']
)

# Create and assign labels
data['Labels'] = create_labels(data, 
                               best_params['short_window'], 
                               best_params['long_window'], 
                               best_params['vol_threshold'])

# Compare it to a buy and hold strategy
buy_hold_pf = vbt.Portfolio.from_holding(data['Close'])

comparison = pd.concat({
    'Optimized': best_pf.stats(),
    'Buy and Hold': buy_hold_pf.stats()
}, axis=1)

print("Label Counts")
print(data['Labels'].value_counts())

print(comparison)

# Plot the charts with the heatmap overlay
plot_charts_with_heatmap(best_pf, data)

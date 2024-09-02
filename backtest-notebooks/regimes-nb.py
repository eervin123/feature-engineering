from vectorbtpro import *
import numpy as np
import pandas as pd
from numba import njit
import traceback

@njit
def rolling_mean_nb(arr, window):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window - 1:
            out[i] = np.nan
        else:
            out[i] = np.mean(arr[i - window + 1:i + 1])
    return out

@njit
def annualized_volatility_nb(returns, window):
    out = np.empty_like(returns)
    for i in range(len(returns)):
        if i < window - 1:
            out[i] = np.nan
        else:
            out[i] = np.std(returns[i - window + 1:i + 1]) * np.sqrt(365)
    return out

@njit
def determine_regime_nb(price, ma_short, ma_long, vol_short, avg_vol_threshold):
    regimes = np.empty_like(price, dtype=np.int32)
    for i in range(len(price)):
        if np.isnan(ma_short[i]) or np.isnan(ma_long[i]) or np.isnan(vol_short[i]):
            regimes[i] = -1  # Unknown
        elif price[i] > ma_short[i] and price[i] > ma_long[i]:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 0  # Above Avg Vol Bull Trend
            else:
                regimes[i] = 1  # Below Avg Vol Bull Trend
        elif price[i] < ma_short[i] and price[i] < ma_long[i]:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 2  # Above Avg Vol Bear Trend
            else:
                regimes[i] = 3  # Below Avg Vol Bear Trend
        else:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 4  # Above Avg Vol Sideways
            else:
                regimes[i] = 5  # Below Avg Vol Sideways
    return regimes

@njit
def calculate_regimes_nb(price, returns, ma_short_window, ma_long_window, vol_short_window, avg_vol_window):
    ma_short = rolling_mean_nb(price, ma_short_window)
    ma_long = rolling_mean_nb(price, ma_long_window)
    vol_short = annualized_volatility_nb(returns, vol_short_window)
    avg_vol_threshold = np.nanmean(annualized_volatility_nb(returns, avg_vol_window))
    regimes = determine_regime_nb(price, ma_short, ma_long, vol_short, avg_vol_threshold)
    return regimes

class MarketRegimeDetector:
    def __init__(self):
        self.RegimeIndicator = vbt.IndicatorFactory(
            class_name='RegimeIndicator',
            input_names=['price', 'returns'],
            param_names=['ma_short_window', 'ma_long_window', 'vol_short_window', 'avg_vol_window'],
            output_names=['regimes']
        ).with_apply_func(calculate_regimes_nb)

    def detect_regimes(self, data, ma_short_window=21, ma_long_window=88, vol_short_window=21, avg_vol_window=365):
        if isinstance(data, str):
            data = vbt.YFData.fetch(data, end='2024-01-01').get()
        
        data['Return'] = data['Close'].pct_change()
        
        regime_indicator = self.RegimeIndicator.run(
            data['Close'].values,
            data['Return'].values,
            ma_short_window=ma_short_window,
            ma_long_window=ma_long_window,
            vol_short_window=vol_short_window,
            avg_vol_window=avg_vol_window
        )
        
        data['Market Regime'] = regime_indicator.regimes.values
        return data

    def plot_regimes(self, data):
        return data['Close'].vbt.overlay_with_heatmap(data['Market Regime']).show()

    @staticmethod
    def create_signals_for_regime(data, regime):
        regime_changes = data['Market Regime'].ne(data['Market Regime'].shift())
        entries = (data['Market Regime'] == regime) & regime_changes
        exits = (data['Market Regime'] != regime) & regime_changes.shift(-1)
        exits = exits.fillna(False).astype(bool)
        return entries.astype(bool), exits

    def analyze_regimes(self, data, init_cash=10000, fees=0.001):
        regime_names = [
            'Above Avg Vol Bull Trend',
            'Below Avg Vol Bull Trend',
            'Above Avg Vol Bear Trend',
            'Below Avg Vol Bear Trend',
            'Above Avg Vol Sideways',
            'Below Avg Vol Sideways'
        ]

        all_stats = {}
        for regime, regime_name in enumerate(regime_names):
            entries, exits = self.create_signals_for_regime(data, regime)
            pf = vbt.Portfolio.from_signals(
                data['Close'],
                entries,
                exits,
                init_cash=init_cash,
                fees=fees
            )
            all_stats[regime_name] = pf.stats()

        all_stats_df = pd.concat(all_stats, axis=1)
        return all_stats_df

# Test code
if __name__ == "__main__":
    try:
        print("Creating MarketRegimeDetector instance")
        detector = MarketRegimeDetector()
        print("MarketRegimeDetector instance created successfully")
        
        print("Fetching BTC-USD data")
        btc_data = detector.detect_regimes('BTC-USD')
        print("Data fetched successfully")
        print(btc_data.head())
        
        print("Analyzing regimes")
        try:
            stats = detector.analyze_regimes(btc_data)
            print("Regime analysis completed")
            
            print("\nTotal Return for each regime:")
            for regime in stats.columns:
                total_return = stats.loc['Total Return [%]', regime]
                print(f"{regime}: {total_return:.2f}%")
        except Exception as e:
            print(f"Error during regime analysis: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()


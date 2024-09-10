import vectorbtpro as vbt
from numba import njit
import numpy as np
import pandas as pd


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

class MarketRegimeDetector_1d:
    def __init__(self):
        self.RegimeIndicator = vbt.IndicatorFactory(
            class_name='RegimeIndicator',
            input_names=['price', 'returns'],
            param_names=['ma_short_window', 'ma_long_window', 'vol_short_window', 'avg_vol_window'],
            output_names=['regimes']
        ).with_apply_func(calculate_regimes_nb)

    def detect_regimes(self, data, ma_short_window=21, ma_long_window=88, vol_short_window=21, avg_vol_window=365):
        if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
            raise ValueError("Input data must be a DataFrame with a 'Close' column")
        
        data = data.copy()
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
            'High Vol Bull Trend',
            'Low Vol Bull Trend',
            'High Vol Bear Trend',
            'Low Vol Bear Trend',
            'High Vol Sideways',
            'Low Vol Sideways'
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
    # Fetch data separately for BTC and ETH
    btc_data = vbt.YFData.fetch('BTC-USD', end='2024-01-01').get()
    eth_data = vbt.YFData.fetch('ETH-USD', end='2024-01-01').get()

    detector = MarketRegimeDetector_1d()

    # Analyze BTC
    btc_data_with_regimes = detector.detect_regimes(btc_data)
    btc_stats = detector.analyze_regimes(btc_data_with_regimes)

    # Analyze ETH
    eth_data_with_regimes = detector.detect_regimes(eth_data)
    eth_stats = detector.analyze_regimes(eth_data_with_regimes)

    # Print results for BTC
    print("\nBTC Total Return for each regime:")
    for regime in btc_stats.columns:
        total_return = btc_stats.loc['Total Return [%]', regime]
        pos_coverage = btc_stats.loc['Position Coverage [%]', regime]
        print(f"{regime}: TR: {total_return:.2f}% Time in Regime: {pos_coverage:.2f}%")

    # Print results for ETH
    print("\nETH Total Return for each regime:")
    for regime in eth_stats.columns:
        total_return = eth_stats.loc['Total Return [%]', regime]
        pos_coverage = eth_stats.loc['Position Coverage [%]', regime]
        print(f"{regime}: TR: {total_return:.2f}% Time in Regime: {pos_coverage:.2f}%")

    # Save statistics to CSV
    btc_csv_filename = 'backtest-notebooks/notebook-results/btc_regime_statistics.csv'
    eth_csv_filename = 'backtest-notebooks/notebook-results/eth_regime_statistics.csv'
    btc_stats.to_csv(btc_csv_filename)
    eth_stats.to_csv(eth_csv_filename)
    print(f"\nBTC Statistics have been saved to {btc_csv_filename}")
    print(f"ETH Statistics have been saved to {eth_csv_filename}")

    # Optional: If you want to keep the plotting functionality
    # detector.plot_regimes(btc_data_with_regimes)
    # detector.plot_regimes(eth_data_with_regimes)


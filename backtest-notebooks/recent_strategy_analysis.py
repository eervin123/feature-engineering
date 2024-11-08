import vectorbtpro as vbt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from numba import njit

# Import strategy functions and regimes from your main file
from regimes_multi_strat_pf import (
    run_bbands_strategy,
    run_ma_strategy,
    run_rsi_divergence_strategy,
    run_macd_divergence_strategy,
    run_psar_strategy,
    calculate_regimes_nb,
    # Import regime definitions
    simple_ma_long_only_btc,
    simple_ma_long_only_eth,
    simple_ma_short_only_btc,
    simple_ma_short_only_eth,
    simple_macd_long_only_btc,
    simple_macd_long_only_eth,
    simple_macd_short_only_btc,
    simple_macd_short_only_eth,
    simple_rsi_divergence_long_only_btc,
    simple_bbands_limits_long_only_btc,
    simple_bbands_limits_long_only_eth,
    simple_bbands_limits_short_only_btc,
    simple_bbands_limits_short_only_eth,
    simple_psar_long_only_btc,
    simple_psar_long_only_eth,
    simple_psar_short_only_btc,
    simple_psar_short_only_eth,
)

# Create RegimeIndicator
RegimeIndicator = vbt.IndicatorFactory(
    class_name="RegimeIndicator",
    input_names=["price", "returns"],
    param_names=[
        "ma_short_window",
        "ma_long_window",
        "vol_short_window",
        "avg_vol_window",
    ],
    output_names=["regimes"],
).with_apply_func(calculate_regimes_nb)


def load_recent_data(lookback_days=365):
    """
    Load recent data from local pickle file with enough history for strategy warmup.

    Parameters:
    lookback_days (int): Number of days of historical data to load (minimum 365 for regime calculation)

    Returns:
    tuple: Recent BTC and ETH data in both 1h and daily timeframes
    """
    if lookback_days < 365:
        print(
            "Warning: lookback_days should be at least 365 for proper regime calculation"
        )
        lookback_days = 365

    def load_binance_data(data_path):
        data = vbt.BinanceData.load(data_path)
        btc_1h = data.resample("1H").data["BTCUSDT"]
        btc_daily = data.resample("1D").data["BTCUSDT"]
        btc_daily["Return"] = btc_daily["Close"].pct_change()
        eth_daily = data.resample("1D").data["ETHUSDT"]
        eth_daily["Return"] = eth_daily["Close"].pct_change()
        eth_1h = data.resample("1H").data["ETHUSDT"]
        return btc_1h, btc_daily, eth_1h, eth_daily

    # Load all data
    btc_1h, btc_daily, eth_1h, eth_daily = load_binance_data("data/m1_data.pkl")

    # Calculate the start date based on lookback_days
    end_idx = btc_daily.index[-1]  # Most recent date in the data
    start_idx = end_idx - pd.Timedelta(days=lookback_days)

    # Slice the data for the recent period
    btc_1h = btc_1h[btc_1h.index >= start_idx]
    btc_daily = btc_daily[btc_daily.index >= start_idx]
    eth_1h = eth_1h[eth_1h.index >= start_idx]
    eth_daily = eth_daily[eth_daily.index >= start_idx]

    return btc_1h, btc_daily, eth_1h, eth_daily


def main():
    # Load recent data
    btc_1h, btc_daily, eth_1h, eth_daily = load_recent_data()

    # Calculate regimes
    regime_indicator = RegimeIndicator.run(
        btc_daily["Close"],
        btc_daily["Return"],
        ma_short_window=21,
        ma_long_window=88,
        vol_short_window=21,
        avg_vol_window=365,
    )

    # Resample regime data to hourly
    btc_daily_regime_data = pd.Series(
        regime_indicator.regimes.values, index=btc_daily.index
    )
    # Calculate ETH regimes separately
    eth_regime_indicator = RegimeIndicator.run(
        eth_daily["Close"],
        eth_daily["Return"], 
        ma_short_window=21,
        ma_long_window=88,
        vol_short_window=21,
        avg_vol_window=365,
    )
    eth_daily_regime_data = pd.Series(
        eth_regime_indicator.regimes.values, index=eth_daily.index
    )

    btc_hourly_regime_data = btc_daily_regime_data.resample("1h").ffill()
    eth_hourly_regime_data = eth_daily_regime_data.resample("1h").ffill()

    # Run strategies with recent data
    strategies = {
        "BTC BBands Long": run_bbands_strategy(
            btc_1h, btc_hourly_regime_data, simple_bbands_limits_long_only_btc, "long",
            bb_window=14, bb_alpha=2.0, atr_window=14, atr_multiplier=2.0
        ),
        "BTC BBands Short": run_bbands_strategy(
            btc_1h, btc_hourly_regime_data, simple_bbands_limits_short_only_btc, "short",
            bb_window=14, bb_alpha=2.0, atr_window=14, atr_multiplier=2.0
        ),
        "ETH BBands Long": run_bbands_strategy(
            eth_1h, eth_hourly_regime_data, simple_bbands_limits_long_only_eth, "long",
            bb_window=14, bb_alpha=2.0, atr_window=14, atr_multiplier=2.0
        ),
        "ETH BBands Short": run_bbands_strategy(
            eth_1h, eth_hourly_regime_data, simple_bbands_limits_short_only_eth, "short",
            bb_window=14, bb_alpha=2.0, atr_window=14, atr_multiplier=2.0
        ),
        "BTC MA Long": run_ma_strategy(
            btc_1h, btc_hourly_regime_data, simple_ma_long_only_btc, 
            fast_ma=21, slow_ma=55, direction="long", fees=0.001
        ),
        "BTC MA Short": run_ma_strategy(
            btc_1h, btc_hourly_regime_data, simple_ma_short_only_btc, 
            fast_ma=21, slow_ma=55, direction="short", fees=0.001
        ),
        "ETH MA Long": run_ma_strategy(
            eth_1h, eth_hourly_regime_data, simple_ma_long_only_eth, 
            fast_ma=21, slow_ma=55, direction="long", fees=0.001
        ),
        "ETH MA Short": run_ma_strategy(
            eth_1h, eth_hourly_regime_data, simple_ma_short_only_eth, 
            fast_ma=21, slow_ma=55, direction="short", fees=0.001
        ),
        "BTC RSI Long": run_rsi_divergence_strategy(
            btc_1h, btc_hourly_regime_data, simple_rsi_divergence_long_only_btc,
            rsi_window=14, rsi_threshold=30, lookback_window=25,
            atr_window=14, atr_multiplier=2.0, direction="long"
        ),
        "BTC MACD Long": run_macd_divergence_strategy(
            btc_1h, btc_hourly_regime_data, simple_macd_long_only_btc,
            fast_window=12, slow_window=26, signal_window=9,
            fees=0.001, direction="long"
        ),
        "BTC MACD Short": run_macd_divergence_strategy(
            btc_1h, btc_hourly_regime_data, simple_macd_short_only_btc,
            fast_window=12, slow_window=26, signal_window=9,
            fees=0.001, direction="short"
        ),
        "ETH MACD Long": run_macd_divergence_strategy(
            eth_1h, eth_hourly_regime_data, simple_macd_long_only_eth,
            fast_window=12, slow_window=26, signal_window=9,
            fees=0.001, direction="long"
        ),
        "ETH MACD Short": run_macd_divergence_strategy(
            eth_1h, eth_hourly_regime_data, simple_macd_short_only_eth,
            fast_window=12, slow_window=26, signal_window=9,
            fees=0.001, direction="short"
        ),
        "BTC PSAR Long": run_psar_strategy(
            btc_1h, btc_hourly_regime_data, simple_psar_long_only_btc,
            direction="long", fees=0.001, af0=0.02, af_increment=0.02, max_af=0.2
        ),
        "BTC PSAR Short": run_psar_strategy(
            btc_1h, btc_hourly_regime_data, simple_psar_short_only_btc,
            direction="short", fees=0.001, af0=0.02, af_increment=0.02, max_af=0.2
        ),
        "ETH PSAR Long": run_psar_strategy(
            eth_1h, eth_hourly_regime_data, simple_psar_long_only_eth,
            direction="long", fees=0.001, af0=0.02, af_increment=0.02, max_af=0.2
        ),
        "ETH PSAR Short": run_psar_strategy(
            eth_1h, eth_hourly_regime_data, simple_psar_short_only_eth,
            direction="short", fees=0.001, af0=0.02, af_increment=0.02, max_af=0.2
        ),
    }

    # Create combined portfolio analysis
    combined_portfolio = vbt.Portfolio.column_stack(
        list(strategies.values()),
        cash_sharing=True,
        group_by=True,
        init_cash=len(strategies) * 100,  # Adjust initial cash as needed
    )

    # Display combined portfolio metrics
    print("\n=== Combined Portfolio Analysis ===")
    print(combined_portfolio.stats())

    # Plot recent performance
    combined_portfolio.plot_cum_returns().show()


if __name__ == "__main__":
    main()

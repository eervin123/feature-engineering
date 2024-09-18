import vectorbtpro as vbt
import pandas as pd
from regimes_multi_strat_pf import (
    calculate_regimes_nb,
    run_psar_strategy,
    psar_nb_with_next,
    simple_bbands_limits_long_only_btc,
    simple_bbands_limits_long_only_eth,
    simple_psar_long_only_btc,
    simple_psar_long_only_eth,
    simple_psar_short_only_btc,
    simple_psar_short_only_eth,
)

# Load data
data = vbt.BinanceData.from_hdf("data/m1_data.h5")
btc_1h = data.resample("1H").data["BTCUSDT"]
eth_1h = data.resample("1H").data["ETHUSDT"]

# Calculate regimes
btc_daily = data.resample("1D").data["BTCUSDT"]
btc_daily["Return"] = btc_daily["Close"].pct_change()
eth_daily = data.resample("1D").data["ETHUSDT"]
eth_daily["Return"] = eth_daily["Close"].pct_change()

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

btc_regime_indicator = RegimeIndicator.run(
    btc_daily["Close"],
    btc_daily["Return"],
    ma_short_window=21,
    ma_long_window=88,
    vol_short_window=21,
    avg_vol_window=365,
)
eth_regime_indicator = RegimeIndicator.run(
    eth_daily["Close"],
    eth_daily["Return"],
    ma_short_window=21,
    ma_long_window=88,
    vol_short_window=21,
    avg_vol_window=365,
)

btc_daily["Market Regime"] = btc_regime_indicator.regimes.values
eth_daily["Market Regime"] = eth_regime_indicator.regimes.values

btc_daily_regime_data = btc_daily["Market Regime"]
btc_hourly_regime_data = btc_daily_regime_data.resample("1h").ffill()
eth_daily_regime_data = eth_daily["Market Regime"]
eth_hourly_regime_data = eth_daily_regime_data.resample("1h").ffill()

btc_aligned_regime_data = btc_hourly_regime_data.reindex(btc_1h.index, method="ffill")
eth_aligned_regime_data = eth_hourly_regime_data.reindex(eth_1h.index, method="ffill")

# Calculate PSAR for both BTC and ETH
btc_psar = run_psar_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_bbands_limits_long_only_btc,
    direction="long",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.2,
)
eth_psar = run_psar_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_bbands_limits_long_only_eth,
    direction="long",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.2,
)

# Plot PSAR for visual inspection
def plot_psar(price, psar, symbol):
    # Calculate PSAR values
    high = price['High'].values
    low = price['Low'].values
    close = price['Close'].values
    psarl, psars, _, _, _, _ = psar_nb_with_next(high, low, close, af0=0.02, af_increment=0.02, max_af=0.2)
    
    # Create DataFrame with PSAR values
    psar_df = pd.DataFrame({
        'psar_long': psarl,
        'psar_short': psars
    }, index=price.index)
    
    # Plot
    fig = price['Close'].vbt.plot(trace_kwargs=dict(name=f'{symbol} Close'))
    psar_df['psar_long'].vbt.plot(trace_kwargs=dict(name='PSAR Long', line=dict(color='green')), fig=fig)
    psar_df['psar_short'].vbt.plot(trace_kwargs=dict(name='PSAR Short', line=dict(color='red')), fig=fig)
    fig.show()
    
    return psar_df  # Return the PSAR values for potential further use

# plot_psar(btc_1h, btc_psar, "BTC")
# plot_psar(eth_1h, eth_psar, "ETH")

# Test the PSAR strategy
btc_psar_long_only_pf = run_psar_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_psar_long_only_btc,
    direction="long",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.05,
)

eth_psar_long_only_pf = run_psar_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_psar_long_only_eth,
    direction="long",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.05,
)

btc_psar_short_only_pf = run_psar_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_psar_short_only_btc,
    direction="short",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.05,
)

eth_psar_short_only_pf = run_psar_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_psar_short_only_eth,
    direction="short",
    fees=0.001,
    af0=0.02,
    af_increment=0.02,
    max_af=0.05,
)

psar_stats = pd.concat(
    [
        btc_psar_long_only_pf.stats(),
        eth_psar_long_only_pf.stats(),
        btc_psar_short_only_pf.stats(),
        eth_psar_short_only_pf.stats(),
    ],
    axis=1,
    keys=[
        "btc_psar_long_only",
        "eth_psar_long_only",
        "btc_psar_short_only",
        "eth_psar_short_only",
    ],
)
print(psar_stats)
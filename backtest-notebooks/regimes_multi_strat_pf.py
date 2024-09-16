# %%
import vectorbtpro as vbt
import numpy as np
import pandas as pd
from numba import njit
import calendar

# Enable Plotly Resampler globally
vbt.settings.plotting.use_resampler = True
# dark theme
vbt.settings.set_theme("dark")


@njit
def rolling_mean_nb(arr, window):
    """
    Calculate the rolling mean of an array with a given window size.

    Parameters:
    arr (np.ndarray): Input array.
    window (int): Window size for the rolling mean.

    Returns:
    np.ndarray: Array of rolling means.
    """
    out = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window - 1:
            out[i] = np.nan
        else:
            out[i] = np.mean(arr[i - window + 1 : i + 1])
    return out


@njit
def annualized_volatility_nb(returns, window):
    """
    Calculate the annualized volatility of an array of returns with a given window size.

    Parameters:
    returns (np.ndarray): Array of returns.
    window (int): Window size for the volatility calculation.

    Returns:
    np.ndarray: Array of annualized volatilities.
    """
    out = np.empty_like(returns)
    for i in range(len(returns)):
        if i < window - 1:
            out[i] = np.nan
        else:
            out[i] = np.std(returns[i - window + 1 : i + 1]) * np.sqrt(365)
    return out


@njit
def determine_regime_nb(price, ma_short, ma_long, vol_short, avg_vol_threshold):
    """
    Determine the market regime based on price, moving averages, and volatility.

    Parameters:
    price (np.ndarray): Array of prices.
    ma_short (np.ndarray): Array of short moving averages.
    ma_long (np.ndarray): Array of long moving averages.
    vol_short (np.ndarray): Array of short volatilities.
    avg_vol_threshold (float): Threshold for average volatility.

    Returns:
    np.ndarray: Array of market regimes.
    """
    regimes = np.empty_like(price, dtype=np.int32)
    for i in range(len(price)):
        if np.isnan(ma_short[i]) or np.isnan(ma_long[i]) or np.isnan(vol_short[i]):
            regimes[i] = -1  # Unknown
        elif price[i] > ma_short[i] and price[i] > ma_long[i]:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 1  # Above Avg Vol Bull Trend
            else:
                regimes[i] = 2  # Below Avg Vol Bull Trend
        elif price[i] < ma_short[i] and price[i] < ma_long[i]:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 3  # Above Avg Vol Bear Trend
            else:
                regimes[i] = 4  # Below Avg Vol Bear Trend
        else:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 5  # Above Avg Vol Sideways
            else:
                regimes[i] = 6  # Below Avg Vol Sideways
    return regimes


@njit
def calculate_regimes_nb(
    price, returns, ma_short_window, ma_long_window, vol_short_window, avg_vol_window
):
    """
    Calculate market regimes based on price, returns, and moving average and volatility parameters.

    Parameters:
    price (np.ndarray): Array of prices.
    returns (np.ndarray): Array of returns.
    ma_short_window (int): Window size for the short moving average.
    ma_long_window (int): Window size for the long moving average.
    vol_short_window (int): Window size for the short volatility calculation.
    avg_vol_window (int): Window size for the average volatility calculation.

    Returns:
    np.ndarray: Array of market regimes.
    """
    ma_short = rolling_mean_nb(price, ma_short_window)
    ma_long = rolling_mean_nb(price, ma_long_window)
    vol_short = annualized_volatility_nb(returns, vol_short_window)
    avg_vol_threshold = np.nanmean(annualized_volatility_nb(returns, avg_vol_window))
    regimes = determine_regime_nb(
        price, ma_short, ma_long, vol_short, avg_vol_threshold
    )
    return regimes


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


# %% [markdown]
#

# %%
# Strategy Regimes
# These names should match the config names of the strategies in config/strategy_configs
simple_ma_long_only_btc = [1, 2]
simple_ma_long_only_eth = [1, 2]
simple_ma_short_only_btc = [5, 6]
simple_ma_short_only_eth = [5, 6]
simple_macd_long_only_btc = [1, 2, 3]
simple_macd_long_only_eth = [1, 2]
simple_macd_short_only_btc = [4, 5, 6]
simple_macd_short_only_eth = [5, 6]
simple_rsi_divergence_long_only_btc = [1, 2, 3]
simple_bbands_limits_long_only_btc = [2]
simple_bbands_limits_long_only_eth = [2]
simple_bbands_limits_short_only_btc = [3, 4]
simple_bbands_limits_short_only_eth = [3, 4]

# %%
# Get the data
data = vbt.BinanceData.from_hdf("data/m1_data.h5")


# %% [markdown]
# ### Calc the regimes on daily data
# set up the hourly data for strategies and calculate regimes for daily data
#
#

# %%
btc_1h = data.resample("1H").data["BTCUSDT"]
btc_daily = data.resample("1D").data["BTCUSDT"]
btc_daily["Return"] = btc_daily["Close"].pct_change()
eth_daily = data.resample("1D").data["ETHUSDT"]
eth_daily["Return"] = eth_daily["Close"].pct_change()
eth_1h = data.resample("1H").data["ETHUSDT"]

# Add regimes to DataFrame
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

# %% [markdown]
# # Run the bbands
#
#

# %%
# Define allowed regimes for long and short strategies
allowed_regimes_long = [2]  # Bullish regime
allowed_regimes_short = [3, 4]  # Bearish

# Note: We've defined separate regime lists for long and short strategies.
# This allows us to use different market conditions for each strategy type,
# similar to the approach used in the moving average strategy.
# Long positions are entered in bullish regimes (2),
# while short positions are entered in bearish (5) and volatile (6) regimes.


# Resample the regime data to hourly frequency
btc_daily_regime_data = btc_daily["Market Regime"]
btc_hourly_regime_data = btc_daily_regime_data.resample("1h").ffill()
eth_daily_regime_data = eth_daily["Market Regime"]
eth_hourly_regime_data = eth_daily_regime_data.resample("1h").ffill()

# Align the hourly regime data with the btc and eth DataFrames
btc_aligned_regime_data = btc_hourly_regime_data.reindex(btc_1h.index, method="ffill")
eth_aligned_regime_data = eth_hourly_regime_data.reindex(eth_1h.index, method="ffill")


def run_bbands_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",  # or 'short'
    fees: float = 0.001,
    bb_window: int = 14,
    bb_alpha: float = 2,
    atr_window: int = 14,
    atr_multiplier: int = 10,  # @GRANT YOU CAN CHANGE THIS
):
    """
    Run a Bollinger Bands strategy on a given symbol's OHLCV data.

    Parameters:
    symbol_ohlcv_df (pd.DataFrame): OHLCV data for the symbol.
    regime_data (pd.Series): Market regime data.
    allowed_regimes (list): List of allowed market regimes for the strategy.
    direction (str): Direction of the strategy ('long' or 'short').
    fees (float): Transaction fees.
    bb_window (int): Window size for Bollinger Bands.
    bb_alpha (float): Alpha value for Bollinger Bands.
    atr_window (int): Window size for Average True Range (ATR).
    atr_multiplier (int): Multiplier for ATR to set stop loss and take profit levels.

    Returns:
    vbt.Portfolio: Portfolio object containing the strategy results.
    """
    # Calculate Bollinger Bands and ATR
    bbands_run = vbt.BBANDS.run(
        close=symbol_ohlcv_df["Close"], window=bb_window, alpha=bb_alpha
    )
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df["High"],
        low=symbol_ohlcv_df["Low"],
        close=symbol_ohlcv_df["Close"],
        window=atr_window,
    )

    # Determine long entries
    long_entries = (symbol_ohlcv_df["Close"] < bbands_run.lower) & (
        regime_data.isin(allowed_regimes)
    )
    short_entries = (symbol_ohlcv_df["Close"] > bbands_run.upper) & (
        regime_data.isin(allowed_regimes)
    )

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)

    # Calculate stop loss and take profit levels
    long_sl_stop = symbol_ohlcv_df["Close"] - atr_multiplier * atr.atr
    long_tp_stop = symbol_ohlcv_df["Close"] + atr_multiplier * atr.atr
    short_sl_stop = symbol_ohlcv_df["Close"] + atr_multiplier * atr.atr
    short_tp_stop = symbol_ohlcv_df["Close"] - atr_multiplier * atr.atr

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df["Close"],
            entries=long_entries,
            exits=regime_exits,  # Exit when leaving allowed regimes
            fees=fees,
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            delta_format="target",
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df["Close"],
            short_entries=short_entries,
            short_exits=regime_exits,  # Exit when leaving allowed regimes
            fees=fees,
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            delta_format="target",
        )
    return pf


# Test the function
btc_bbands_long_only_pf = run_bbands_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=allowed_regimes_long,
    direction="long",
)
eth_bbands_long_only_pf = run_bbands_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=allowed_regimes_long,
    direction="long",
)

btc_bbands_short_only_pf = run_bbands_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=allowed_regimes_short,
    direction="short",
)
eth_bbands_short_only_pf = run_bbands_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=allowed_regimes_short,
    direction="short",
)

# Create a new series with only allowed regimes highlighted for BTC
btc_allowed_regimes_long_only = btc_daily["Market Regime"].copy()
btc_allowed_regimes_long_only[
    ~btc_allowed_regimes_long_only.isin(allowed_regimes_long)
] = np.nan

# Create a new series with only allowed regimes highlighted for ETH
eth_allowed_regimes_long_only = eth_daily["Market Regime"].copy()
eth_allowed_regimes_long_only[
    ~eth_allowed_regimes_long_only.isin(allowed_regimes_long)
] = np.nan

# Plot with only allowed regimes highlighted, using daily data for BTC
btc_daily["Close"].vbt.overlay_with_heatmap(
    btc_allowed_regimes_long_only, title="BTC BBands Strategy Regimes 2", height=200
).show()

# Plot with only allowed regimes highlighted, using daily data for ETH
eth_daily["Close"].vbt.overlay_with_heatmap(
    eth_allowed_regimes_long_only, title="ETH BBands Strategy Regimes 2", height=200
).show()

# Create a new series with only allowed regimes highlighted for BTC (short version)
btc_allowed_regimes_short = btc_daily["Market Regime"].copy()
btc_allowed_regimes_short[~btc_allowed_regimes_short.isin(allowed_regimes_short)] = (
    np.nan
)

# Create a new series with only allowed regimes highlighted for ETH (short version)
eth_allowed_regimes_short = eth_daily["Market Regime"].copy()
eth_allowed_regimes_short[~eth_allowed_regimes_short.isin(allowed_regimes_short)] = (
    np.nan
)

# Plot with only allowed regimes highlighted for short positions, using daily data for BTC
btc_daily["Close"].vbt.overlay_with_heatmap(
    btc_allowed_regimes_short, title="BTC BBands Strategy Short Regimes", height=200
).show()

# Plot with only allowed regimes highlighted for short positions, using daily data for ETH
eth_daily["Close"].vbt.overlay_with_heatmap(
    eth_allowed_regimes_short, title="ETH BBands Strategy Short Regimes", height=200
).show()


height = 600
btc_bbands_long_only_pf.plot(
    height=height, title=f"BTC BBands Strategy Long regimes {allowed_regimes_long}"
).show()
eth_bbands_long_only_pf.plot(
    height=height, title=f"ETH BBands Strategy Long regimes {allowed_regimes_long}"
).show()
btc_bbands_short_only_pf.plot(
    height=height, title=f"BTC BBands Strategy Short regimes {allowed_regimes_short}"
).show()
eth_bbands_short_only_pf.plot(
    height=height, title=f"ETH BBands Strategy Short regimes {allowed_regimes_short}"
).show()


pd.concat(
    [
        btc_bbands_long_only_pf.stats(),
        eth_bbands_long_only_pf.stats(),
        btc_bbands_short_only_pf.stats(),
        eth_bbands_short_only_pf.stats(),
    ],
    axis=1,
    keys=[
        "btc_bbands_long_only",
        "eth_bbands_long_only",
        "btc_bbands_short_only",
        "eth_bbands_short_only",
    ],
)


# %% [markdown]
# ## Okay now let's do the moving average strategies
#

# %%
# Define allowed regimes at the top
allowed_regimes_long = [1, 2]
allowed_regimes_short = [
    3,
    4,
]  # Note this disagree's with the regimes travis has. different number but same regimes
# Resample the regime data to hourly frequency
btc_daily_regime_data = btc_daily["Market Regime"]
btc_hourly_regime_data = btc_daily_regime_data.resample("1h").ffill()
eth_daily_regime_data = eth_daily["Market Regime"]
eth_hourly_regime_data = eth_daily_regime_data.resample("1h").ffill()

# Align the hourly regime data with the btc and eth DataFrames
btc_aligned_regime_data = btc_hourly_regime_data.reindex(btc_1h.index, method="ffill")
eth_aligned_regime_data = eth_hourly_regime_data.reindex(eth_1h.index, method="ffill")


def run_ma_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_ma: int,
    slow_ma: int,
    direction: str = "long",
):
    """
    Run a moving average strategy on a given symbol's OHLCV data.

    Parameters:
    symbol_ohlcv_df (pd.DataFrame): OHLCV data for the symbol.
    regime_data (pd.Series): Market regime data.
    allowed_regimes (list): List of allowed market regimes for the strategy.
    fast_ma (int): Window size for the fast moving average.
    slow_ma (int): Window size for the slow moving average.
    direction (str): Direction of the strategy ('long' or 'short').

    Returns:
    vbt.Portfolio: Portfolio object containing the strategy results.
    """

    fast_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=fast_ma).ma
    slow_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=slow_ma).ma

    long_entries = fast_ma > slow_ma
    long_exits = fast_ma < slow_ma

    short_entries = fast_ma < slow_ma
    short_exits = fast_ma > slow_ma

    # Add regime filter
    long_entries = long_entries & regime_data.isin(allowed_regimes_long)
    short_entries = short_entries & regime_data.isin(allowed_regimes_short)
    # Create exit signals when leaving allowed regimes
    long_regime_exits = ~regime_data.isin(allowed_regimes_long)
    short_regime_exits = ~regime_data.isin(allowed_regimes_short)

    # Combine regime exits with other exit conditions
    long_exits = long_exits | long_regime_exits
    short_exits = short_exits | short_regime_exits

    # Run the simulation
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df.Close,
            entries=long_entries,
            exits=long_exits,
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df.Close,
            short_entries=short_entries,
            short_exits=short_exits,
        )

    return pf


btc_ma_long_only_pf = run_ma_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=allowed_regimes_long,
    fast_ma=21,
    slow_ma=55,
)
eth_ma_long_only_pf = run_ma_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=allowed_regimes_long,
    fast_ma=21,
    slow_ma=55,
)
btc_ma_short_only_pf = run_ma_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=allowed_regimes_short,
    fast_ma=21,
    slow_ma=55,
    direction="short",
)
eth_ma_short_only_pf = run_ma_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=allowed_regimes_short,
    fast_ma=21,
    slow_ma=55,
    direction="short",
)
# Create separate series for long and short allowed regimes for BTC
btc_allowed_regimes_long = btc_daily["Market Regime"].copy()
btc_allowed_regimes_long[~btc_allowed_regimes_long.isin(allowed_regimes_long)] = np.nan

btc_allowed_regimes_short = btc_daily["Market Regime"].copy()
btc_allowed_regimes_short[~btc_allowed_regimes_short.isin(allowed_regimes_short)] = (
    np.nan
)

# Create separate series for long and short allowed regimes for ETH
eth_allowed_regimes_long = eth_daily["Market Regime"].copy()
eth_allowed_regimes_long[~eth_allowed_regimes_long.isin(allowed_regimes_long)] = np.nan

eth_allowed_regimes_short = eth_daily["Market Regime"].copy()
eth_allowed_regimes_short[~eth_allowed_regimes_short.isin(allowed_regimes_short)] = (
    np.nan
)
height = 200
# Plot with only long allowed regimes highlighted for BTC
btc_daily["Close"].vbt.overlay_with_heatmap(
    btc_allowed_regimes_long, title="BTC MA Strategy Long Regimes", height=height
).show()

# Plot with only short allowed regimes highlighted for BTC
btc_daily["Close"].vbt.overlay_with_heatmap(
    btc_allowed_regimes_short, title="BTC MA Strategy Short Regimes", height=height
).show()

# Plot with only long allowed regimes highlighted for ETH
eth_daily["Close"].vbt.overlay_with_heatmap(
    eth_allowed_regimes_long, title="ETH MA Strategy Long Regimes", height=height
).show()

# Plot with only short allowed regimes highlighted for ETH
eth_daily["Close"].vbt.overlay_with_heatmap(
    eth_allowed_regimes_short, title="ETH MA Strategy Short Regimes", height=height
).show()

height = 600
btc_ma_long_only_pf.plot(height=height, title="BTC MA Strategy LONG ONLY").show()
eth_ma_long_only_pf.plot(height=height, title="ETH MA Strategy LONG ONLY").show()
btc_ma_short_only_pf.plot(height=height, title="BTC MA Strategy SHORT ONLY").show()
eth_ma_short_only_pf.plot(height=height, title="ETH MA Strategy SHORT ONLY").show()

pd.concat(
    [
        btc_ma_long_only_pf.stats(),
        eth_ma_long_only_pf.stats(),
        btc_ma_short_only_pf.stats(),
        eth_ma_short_only_pf.stats(),
    ],
    axis=1,
    keys=[
        "btc_ma_long_only",
        "eth_ma_long_only",
        "btc_ma_short_only",
        "eth_ma_short_only",
    ],
)


# %% [markdown]
# # For a break let's have a look at a blended portfolio of strategies
#
# Here we compare some blends with long and short strategies and some with only long or only short strategies

# %%
# First create a combined dataframe with the 2 symbols' closing prices
btc_eth_combined = pd.concat(
    [btc_1h["Close"], eth_1h["Close"]], axis=1, keys=["BTCUSDT", "ETHUSDT"]
)

# Now we can run the portfolio on the combined dataframe
btc_eth_pf_long_short_blend = vbt.Portfolio.column_stack(
    [
        btc_bbands_long_only_pf,
        btc_bbands_short_only_pf,
        eth_bbands_long_only_pf,
        eth_bbands_short_only_pf,
        btc_ma_long_only_pf,
        eth_ma_long_only_pf,
        btc_ma_short_only_pf,
        eth_ma_short_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=800,
)  # 800 is 8 times the initial capital because we are using 8 strategies
btc_eth_pf_long_only_blend = vbt.Portfolio.column_stack(
    [
        btc_bbands_long_only_pf,
        btc_ma_long_only_pf,
        eth_bbands_long_only_pf,
        eth_ma_long_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=400,
)  # 400 is 4 times the initial capital because we are using 4 strategies
btc_eth_pf_short_only_blend = vbt.Portfolio.column_stack(
    [
        btc_bbands_short_only_pf,
        eth_bbands_short_only_pf,
        eth_ma_short_only_pf,
        btc_ma_short_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=400,
)  # 400 is 4 times the initial capital because we are using 4 strategies
pd.concat(
    [
        btc_eth_pf_long_short_blend.stats(),
        btc_eth_pf_long_only_blend.stats(),
        btc_eth_pf_short_only_blend.stats(),
    ],
    axis=1,
    keys=[
        "btc_eth_pf_long_short_blend",
        "btc_eth_pf_long_only_blend",
        "btc_eth_pf_short_only_blend",
    ],
)

# %%
vbt.settings.plotting.use_resampler = False
btc_eth_pf_long_short_blend.plot_allocations(
    height=height, line_visible=False, title="BTC ETH Blended Portfolio Allocation"
).show()

# %%
vbt.settings.plotting.use_resampler = True
btc_eth_pf_long_short_blend.plot(height=200, title="BTC ETH Blended Portfolio").show()
btc_eth_pf_long_only_blend.plot(
    height=200, title="BTC ETH Long Only Blended Portfolio"
).show()
btc_eth_pf_short_only_blend.plot(
    height=200, title="BTC ETH Short Only Blended Portfolio"
).show()

# %% [markdown]
# # Plot the monthly returns as a heatmap


# Assuming `pf` is your portfolio object
# Resample the entire portfolio to the monthly frequency and compute the returns
monthly_returns = btc_eth_pf_long_short_blend.resample("M").returns

# Create a matrix of monthly returns
monthly_return_matrix = pd.Series(
    monthly_returns.values * 100,  # Convert to percentage
    index=pd.MultiIndex.from_arrays(
        [monthly_returns.index.year, monthly_returns.index.month],
        names=["year", "month"],
    ),
).unstack("month")

# Rename the columns to month abbreviations
monthly_return_matrix.columns = monthly_return_matrix.columns.map(
    lambda x: calendar.month_abbr[x]
)

# Calculate annual returns and add as a new column
annual_returns = monthly_return_matrix.sum(axis=1)
monthly_return_matrix["Annual"] = annual_returns

# Replace NaN values with an empty string for display
text_matrix = monthly_return_matrix.round(2).astype(str).replace("nan", "")

# Plot the monthly returns as a heatmap
fig = monthly_return_matrix.vbt.heatmap(
    is_x_category=True,
    trace_kwargs=dict(
        zmid=0,
        colorscale="temps_r",
        text=text_matrix + "%",  # Add percentage text
        texttemplate="%{text}",  # Template for displaying text
        textfont=dict(color="black"),  # Default font color to gray
    ),
)
# Add a title to the plot
fig.update_layout(
    title={
        "text": "Monthly Returns Heatmap - BTC ETH Blended Portfolio",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)

# Update layout to position y-axis labels at the top
fig.update_layout(yaxis=dict(side="top"))  # Position y-axis labels at the top

fig.show()

# %%

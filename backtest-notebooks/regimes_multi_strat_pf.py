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
                regimes[i] = 5  # Above Avg Vol Bear Trend
            else:
                regimes[i] = 6  # Below Avg Vol Bear Trend
        else:
            if vol_short[i] > avg_vol_threshold:
                regimes[i] = 3  # Above Avg Vol Sideways
            else:
                regimes[i] = 4  # Below Avg Vol Sideways
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


def plot_strategy_results(strategies, strategy_name, plot_func='plot_cum_returns', height=1200, width=1200):
    """
    Create a 2x2 subplot for different strategy results.

    Parameters:
    strategies (list): List of 4 vbt.Portfolio objects to plot
    strategy_name (str): Name of the strategy for the main title
    plot_func (str): Name of the plotting function to use (e.g., 'plot_cum_returns', 'plot_value')
    height (int): Height of the plot
    width (int): Width of the plot

    Returns:
    plotly.graph_objs.Figure: The created figure
    """
    assert len(strategies) == 4, "Must provide exactly 4 strategies"

    titles = [f"{name} Strategy" for name in ["BTC Long-Only", "BTC Short-Only", "ETH Long-Only", "ETH Short-Only"]]
    fig = vbt.make_subplots(rows=2, cols=2, vertical_spacing=0.1, subplot_titles=titles)

    for i, pf in enumerate(strategies):
        row = i // 2 + 1
        col = i % 2 + 1
        getattr(pf, plot_func)(
            add_trace_kwargs=dict(row=row, col=col),
            fig=fig
        )

    fig.update_layout(height=height, width=width, title=f"{strategy_name} Strategies")
    return fig

def create_monthly_returns_heatmap(portfolio, title):
    """
    Create a monthly returns heatmap for a given portfolio.

    Parameters:
    portfolio (vbt.Portfolio): The portfolio to analyze
    title (str): The title for the heatmap

    Returns:
    plotly.graph_objs.Figure: The created heatmap figure
    """
    # Resample the portfolio to monthly frequency and compute returns
    monthly_returns = portfolio.resample("M").returns

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
    monthly_heatmap = monthly_return_matrix.vbt.heatmap(
        is_x_category=True,
        trace_kwargs=dict(
            zmid=0,
            colorscale="temps_r",
            text=text_matrix + "%",  # Add percentage text
            texttemplate="%{text}",  # Template for displaying text
            textfont=dict(color="black"),  # Default font color to black
        ),
    )

    # Add a title to the plot
    monthly_heatmap.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis=dict(side="top")  # Position y-axis labels at the top
    )

    return monthly_heatmap

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
simple_bbands_limits_short_only_btc = [5, 6]
simple_bbands_limits_short_only_eth = [5, 6]

# %%
# Get the data
data = vbt.BinanceData.from_hdf("data/m1_data.h5")
btc_1h = data.resample("1H").data["BTCUSDT"]
btc_daily = data.resample("1D").data["BTCUSDT"]
btc_daily["Return"] = btc_daily["Close"].pct_change()
eth_daily = data.resample("1D").data["ETHUSDT"]
eth_daily["Return"] = eth_daily["Close"].pct_change()
eth_1h = data.resample("1H").data["ETHUSDT"]

# Set up the RegimeIndicator
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

# Plot heatmap overlay for BTC Market Regimes
btc_daily["Close"].vbt.overlay_with_heatmap(
    btc_daily["Market Regime"],
    title="BTC Market Regimes",
    height=200
).show()

# Plot heatmap overlay for ETH Market Regimes
eth_daily["Close"].vbt.overlay_with_heatmap(
    eth_daily["Market Regime"],
    title="ETH Market Regimes",
    height=200
).show()

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
    allowed_regimes=simple_bbands_limits_long_only_btc,
    direction="long",
)
eth_bbands_long_only_pf = run_bbands_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_bbands_limits_long_only_eth,
    direction="long",
)

btc_bbands_short_only_pf = run_bbands_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_bbands_limits_short_only_btc,
    direction="short",
)
eth_bbands_short_only_pf = run_bbands_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_bbands_limits_short_only_eth,
    direction="short",
)

bbands_stats = pd.concat(
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
print(bbands_stats)

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
    long_entries = long_entries & regime_data.isin(allowed_regimes)
    short_entries = short_entries & regime_data.isin(allowed_regimes)
    # Create exit signals when leaving allowed regimes
    long_regime_exits = ~regime_data.isin(allowed_regimes)
    short_regime_exits = ~regime_data.isin(allowed_regimes)

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
    allowed_regimes=simple_ma_long_only_btc,
    fast_ma=21,
    slow_ma=55,
)
eth_ma_long_only_pf = run_ma_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_ma_long_only_eth,
    fast_ma=21,
    slow_ma=55,
)
btc_ma_short_only_pf = run_ma_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_ma_short_only_btc,
    fast_ma=21,
    slow_ma=55,
    direction="short",
)
eth_ma_short_only_pf = run_ma_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_ma_short_only_eth,
    fast_ma=21,
    slow_ma=55,
    direction="short",
)

ma_stats =pd.concat(
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
print(ma_stats)

def run_rsi_divergence_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    rsi_window: int = 14,
    rsi_threshold: int = 30,
    lookback_window: int = 25,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
    direction: str = "long",  # or 'short'
):
    """
    Run an RSI divergence strategy on a given symbol's OHLCV data.

    Parameters:
    symbol_ohlcv_df (pd.DataFrame): OHLCV data for the symbol.
    regime_data (pd.Series): Market regime data.
    allowed_regimes (list): List of allowed market regimes for the strategy.
    rsi_window (int): Window size for RSI calculation.
    rsi_threshold (int): RSI threshold for oversold condition.
    lookback_window (int): Window size for price and RSI low comparison.
    atr_window (int): Window size for ATR calculation.
    atr_multiplier (float): Multiplier for ATR to set stop loss and take profit levels.
    fees (float): Transaction fees.

    Returns:
    vbt.Portfolio: Portfolio object containing the strategy results.
    """
    # Calculate RSI
    rsi = vbt.RSI.run(symbol_ohlcv_df['Close'], window=rsi_window).rsi

    # Calculate ATR
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df['High'],
        low=symbol_ohlcv_df['Low'],
        close=symbol_ohlcv_df['Close'],
        window=atr_window
    ).atr

    # Calculate rolling minimum for price and RSI
    price_min = symbol_ohlcv_df['Close'].rolling(window=lookback_window).min()
    rsi_min = rsi.rolling(window=lookback_window).min()

    # Generate entry signals
    entries = (
        (symbol_ohlcv_df['Close'] == price_min) &  # New price low
        (rsi < rsi_threshold) &  # RSI below threshold
        (rsi > rsi_min) &  # RSI not at new low
        (regime_data.isin(allowed_regimes))  # In allowed regime
    )
    
    short_entries = (
        (symbol_ohlcv_df['Close'] == price_min) &  # New price low
        (rsi > rsi_threshold) &  # RSI above threshold
        (rsi < rsi_min) &  # RSI not at new low
        (regime_data.isin(allowed_regimes))  # In allowed regime
    )

    # Calculate stop loss and take profit levels
    entry_price = symbol_ohlcv_df['Close']
    long_sl_stop = entry_price - atr_multiplier * atr
    long_tp_stop = entry_price + atr_multiplier * atr
    short_sl_stop = entry_price + atr_multiplier * atr
    short_tp_stop = entry_price - atr_multiplier * atr

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            entries=entries,
            exits=regime_exits,  # Exit when leaving allowed regimes
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            short_entries=short_entries,
            short_exits=regime_exits,  # Exit when leaving allowed regimes
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            fees=fees,
            delta_format='target',
        )
    return pf

# Test the function
btc_rsi_divergence_pf_long = run_rsi_divergence_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_rsi_divergence_long_only_btc,
    direction="long",
)


# Plot the results
height = 600
btc_rsi_divergence_pf_long.plot(
    height=height,
    title=f"BTC RSI Divergence Strategy (regimes {simple_rsi_divergence_long_only_btc})"
).show()

print("BTC RSI Divergence Long Only")
print(btc_rsi_divergence_pf_long.stats())

def run_macd_divergence_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
    direction: str = "long",  # or 'short'
):
    """
    Run a MACD divergence strategy on a given symbol's OHLCV data.

    Parameters:
    symbol_ohlcv_df (pd.DataFrame): OHLCV data for the symbol.
    regime_data (pd.Series): Market regime data.
    allowed_regimes (list): List of allowed market regimes for the strategy.
    fast_window (int): Fast EMA window for MACD calculation.
    slow_window (int): Slow EMA window for MACD calculation.
    signal_window (int): Signal line window for MACD calculation.
    atr_window (int): Window size for ATR calculation.
    atr_multiplier (float): Multiplier for ATR to set stop loss and take profit levels.
    fees (float): Transaction fees.
    direction (str): Direction of the strategy ('long' or 'short').

    Returns:
    vbt.Portfolio: Portfolio object containing the strategy results.
    """
    # Calculate MACD
    macd = vbt.MACD.run(
        symbol_ohlcv_df['Close'],
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window
    )


    # Generate entry signals
    if direction == "long":
        entries = (macd.macd > macd.signal) & (macd.macd.shift(1) <= macd.signal.shift(1))
    else:  # short
        entries = (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)

    # Generate exit signals
    if direction == "long":
        exits = (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))
    else:  # short
        exits = (macd.macd > macd.signal) & (macd.macd.shift(1) <= macd.signal.shift(1))

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)

    # Combine regime exits with other exit conditions
    exits = exits | regime_exits

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            entries=entries,
            exits=exits,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            short_entries=entries,
            short_exits=exits,
            fees=fees,
            delta_format='target',
        )
    return pf

# Test the function for BTC MACD long-only strategy
btc_macd_long_only_pf = run_macd_divergence_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_macd_long_only_btc,
    direction="long",
)

# Test the function for BTC MACD short-only strategy
btc_macd_short_only_pf = run_macd_divergence_strategy(
    btc_1h,
    btc_aligned_regime_data,
    allowed_regimes=simple_macd_short_only_btc,
    direction="short",
)

# Test the function for ETH MACD long-only strategy
eth_macd_long_only_pf = run_macd_divergence_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_macd_long_only_eth,
    direction="long",
)

# Test the function for ETH MACD short-only strategy
eth_macd_short_only_pf = run_macd_divergence_strategy(
    eth_1h,
    eth_aligned_regime_data,
    allowed_regimes=simple_macd_short_only_eth,
    direction="short",
)

# Print statistics
macd_stats = pd.concat(
    [
        btc_macd_long_only_pf.stats(),
        btc_macd_short_only_pf.stats(),
        eth_macd_long_only_pf.stats(),
        eth_macd_short_only_pf.stats(),
    ],
    axis=1,
    keys=[
        "btc_macd_long_only",
        "btc_macd_short_only",
        "eth_macd_long_only",
        "eth_macd_short_only",
    ],
)
print(macd_stats)

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
        btc_rsi_divergence_pf_long,
        btc_macd_long_only_pf,
        # btc_macd_short_only_pf,
        eth_macd_long_only_pf,
        # eth_macd_short_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=1100,  # 1300 is 13 times the initial capital because we are using 13 strategies
)
btc_eth_pf_long_only_blend = vbt.Portfolio.column_stack(
    [
        btc_bbands_long_only_pf,
        btc_ma_long_only_pf,
        eth_bbands_long_only_pf,
        eth_ma_long_only_pf,
        btc_rsi_divergence_pf_long,
        btc_macd_long_only_pf,
        eth_macd_long_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=700,  # 700 is 7 times the initial capital because we are using 7 strategies
)
btc_eth_pf_short_only_blend = vbt.Portfolio.column_stack(
    [
        btc_bbands_short_only_pf,
        eth_bbands_short_only_pf,
        eth_ma_short_only_pf,
        btc_ma_short_only_pf,
        btc_macd_short_only_pf,
        eth_macd_short_only_pf,
    ],
    cash_sharing=True,
    group_by=True,
    init_cash=600,  # 600 is 6 times the initial capital because we are using 6 strategies
)
blended_stats = pd.concat(
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
print(blended_stats)
# %%


# %%
vbt.settings.plotting.use_resampler = False

# Usage for MACD strategies
macd_strategies = [btc_macd_long_only_pf, btc_macd_short_only_pf, eth_macd_long_only_pf, eth_macd_short_only_pf]
macd_plots = plot_strategy_results(macd_strategies, "MACD Divergence")
macd_plots.show()

# Usage for BBands strategies
bbands_strategies = [btc_bbands_long_only_pf, btc_bbands_short_only_pf, eth_bbands_long_only_pf, eth_bbands_short_only_pf]
bbands_plots = plot_strategy_results(bbands_strategies, "Bollinger Bands")
bbands_plots.show()

# Usage for MA strategies
ma_strategies = [btc_ma_long_only_pf, btc_ma_short_only_pf, eth_ma_long_only_pf, eth_ma_short_only_pf]
ma_plots = plot_strategy_results(ma_strategies, "Moving Average")
ma_plots.show()

# Usage example:
blended_heatmap = create_monthly_returns_heatmap(
    btc_eth_pf_long_short_blend, 
    "Monthly Returns Heatmap - BTC ETH Blended Portfolio"
)
blended_heatmap.show()

# You can easily create heatmaps for other portfolios:
long_only_heatmap = create_monthly_returns_heatmap(
    btc_eth_pf_long_only_blend, 
    "Monthly Returns Heatmap - BTC ETH Long Only Portfolio"
)
long_only_heatmap.show()

short_only_heatmap = create_monthly_returns_heatmap(
    btc_eth_pf_short_only_blend, 
    "Monthly Returns Heatmap - BTC ETH Short Only Portfolio"
)
short_only_heatmap.show()

# %%

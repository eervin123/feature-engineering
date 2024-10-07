from regimes_multi_strat_pf import (
    calculate_regimes_nb,
    run_bbands_strategy,
    psar_nb_with_next,
)

import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
import vectorbtpro as vbt
from tabulate import tabulate
import plotly.io as pio

# Set the default renderer to 'browser' to open plots in your default web browser
pio.renderers.default = "browser"

vbt.settings.set_theme("dark")


data = vbt.BinanceData.from_hdf("data/m1_data.h5")
btc_1h = data.resample("1H").data["BTCUSDT"]
btc_daily = data.resample("1D").data["BTCUSDT"]
btc_daily["Return"] = btc_daily["Close"].pct_change()
eth_daily = data.resample("1D").data["ETHUSDT"]
eth_daily["Return"] = eth_daily["Close"].pct_change()
eth_1h = data.resample("1H").data["ETHUSDT"]


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
simple_psar_long_only_btc = [1, 2]
simple_psar_long_only_eth = [1, 2]
simple_psar_short_only_btc = [5, 6]
simple_psar_short_only_eth = [5, 6]

# Resample the regime data to hourly frequency
btc_daily_regime_data = btc_daily["Market Regime"]
btc_hourly_regime_data = btc_daily_regime_data.resample("1h").ffill()
eth_daily_regime_data = eth_daily["Market Regime"]
eth_hourly_regime_data = eth_daily_regime_data.resample("1h").ffill()

# Align the hourly regime data with the btc and eth DataFrames
btc_aligned_regime_data = btc_hourly_regime_data.reindex(btc_1h.index, method="ffill")
eth_aligned_regime_data = eth_hourly_regime_data.reindex(eth_1h.index, method="ffill")


def optimize_strategy(strategy_func, strategy_params, symbol_ohlcv_df, regime_data, allowed_regimes, n_trials=500):
    def objective(trial):
        params = {}
        for k, v in strategy_params.items():
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], (int, float)):
                if isinstance(v[0], int):
                    params[k] = trial.suggest_int(k, v[0], v[1])
                else:
                    params[k] = trial.suggest_float(k, v[0], v[1])
            elif isinstance(v, list):
                params[k] = trial.suggest_categorical(k, v)
            else:
                params[k] = v

        pf = strategy_func(
            symbol_ohlcv_df=symbol_ohlcv_df,
            regime_data=regime_data,
            allowed_regimes=allowed_regimes,
            **params
        )

        # We're minimizing the negative of the total return
        return -pf.total_return

    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=25, interval_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    def early_stopping_callback(study, trial):
        if study.best_trial.number + 100 < trial.number:
            study.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[early_stopping_callback])

    best_params = study.best_params
    best_direction = best_params['direction']
    
    print(f"Best parameters for {strategy_func.__name__} ({best_direction}): ", best_params)
    print("Best value: ", study.best_value)

    # Backtest with the best parameters
    best_pf = strategy_func(
        symbol_ohlcv_df=symbol_ohlcv_df,
        regime_data=regime_data,
        allowed_regimes=allowed_regimes,
        **best_params,
    )

    return best_params, best_pf, best_direction


# Define parameter ranges for each strategy
bbands_params = {
    "bb_window": (5, 300),
    "bb_alpha": (0.5, 4.0),
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

ma_params = {
    "fast_ma": (3, 100),
    "slow_ma": (10, 500),
    "direction": ["long", "short"],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

rsi_params = {
    "rsi_window": (5, 50),
    "rsi_threshold": (10, 90),
    "lookback_window": (5, 100),
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

macd_params = {
    "fast_window": (5, 100),
    "slow_window": (10, 300),
    "signal_window": (5, 50),
    "direction": ["long", "short"],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

psar_params = {
    "af0": (0.01, 0.1),
    "af_increment": (0.01, 0.1),
    "max_af": (0.1, 0.5),
    "direction": ["long", "short"],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

rsi_mean_reversion_params = {
    "rsi_window": (5, 200),
    "rsi_lower": (20, 40),
    "rsi_upper": (60, 80),
    "atr_window": (5, 100),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

mean_reversion_params = {
    'bb_window': (5, 300),
    'bb_alpha': (0.5, 4.0),
    'timeframe_1': ['1h', '4H', '8H', '12H', '24H'],
    'timeframe_2': ['24H', '48H', '72H', '1W'],
    'direction': ['long', 'short'],
    'atr_window': (5, 50),
    'atr_multiplier': (0.5, 10.0),
}


def run_rsi_mean_reversion_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",  # or 'short'
    fees: float = 0.001,
    rsi_window: int = 14,
    rsi_lower: int = 30,
    rsi_upper: int = 70,
    atr_window: int = 14,
    atr_multiplier: int = 3,
):
    """
    Run an RSI mean reversion strategy on a given symbol's OHLCV data.

    Parameters:
    symbol_ohlcv_df (pd.DataFrame): OHLCV data for the symbol.
    regime_data (pd.Series): Market regime data.
    allowed_regimes (list): List of allowed market regimes for the strategy.
    direction (str): Direction of the strategy ('long' or 'short').
    fees (float): Transaction fees.
    rsi_window (int): Window size for RSI.
    rsi_lower (int): Lower threshold for RSI.
    rsi_upper (int): Upper threshold for RSI.
    atr_window (int): Window size for Average True Range (ATR).
    atr_multiplier (int): Multiplier for ATR to set stop loss and take profit levels.

    Returns:
    vbt.Portfolio: Portfolio object containing the strategy results.
    """
    # Calculate RSI and ATR
    rsi = vbt.RSI.run(close=symbol_ohlcv_df["Close"], window=rsi_window)
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df["High"],
        low=symbol_ohlcv_df["Low"],
        close=symbol_ohlcv_df["Close"],
        window=atr_window,
    )

    # Determine long and short entries
    long_entries = (rsi.rsi < rsi_lower) & (regime_data.isin(allowed_regimes))
    short_entries = (rsi.rsi > rsi_upper) & (regime_data.isin(allowed_regimes))

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)
    
    # Create and return the portfolio
    if direction == "long":
        
        long_sl_stop = symbol_ohlcv_df["Close"] - atr_multiplier * atr.atr
        long_tp_stop = symbol_ohlcv_df["Close"] + atr_multiplier * atr.atr
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
        short_sl_stop = symbol_ohlcv_df["Close"] + atr_multiplier * atr.atr
        short_tp_stop = symbol_ohlcv_df["Close"] - atr_multiplier * atr.atr
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

def mean_reversion_strategy(symbol_ohlcv_df, regime_data, allowed_regimes, direction="long", bb_window=21, bb_alpha=2.0, timeframe_1='4H', timeframe_2='24H', atr_window=14, atr_multiplier=3.0, **kwargs):
    # Convert to vbt.BinanceData
    data = vbt.BinanceData.from_data(symbol_ohlcv_df)
    
    # Calculate Bollinger Bands on multiple timeframes
    bbands_tf1 = vbt.talib("BBANDS").run(data.close, timeperiod=bb_window, nbdevup=bb_alpha, nbdevdn=bb_alpha, timeframe=timeframe_1)
    bbands_tf2 = vbt.talib("BBANDS").run(data.close, timeperiod=bb_window, nbdevup=bb_alpha, nbdevdn=bb_alpha, timeframe=timeframe_2)
    
    # Calculate ATR
    atr = vbt.ATR.run(data.high, data.low, data.close, window=atr_window).atr
    
    # Generate long and short entry conditions
    long_entries = (
        (data.close < bbands_tf2.middleband) &
        (data.close < bbands_tf1.lowerband)
    ) | (
        (data.close > bbands_tf2.lowerband) &
        (data.close < bbands_tf1.lowerband)
    )
    
    # Short entries are the inverse of long entries with upper and lower bands swapped
    short_entries = (
        (data.close > bbands_tf2.middleband) &
        (data.close > bbands_tf1.upperband)
    ) | (
        (data.close < bbands_tf2.upperband) &
        (data.close > bbands_tf1.upperband)
    )

    short_entries = (
        (data.close < bbands_tf2.upperband) &
        (data.close > bbands_tf1.middleband)
    ) | (
        (data.close < bbands_tf2.upperband) &
        (data.close < bbands_tf1.upperband)
    )

    # Ensure we're only trading in allowed regimes
    allowed_regime_mask = regime_data.isin(allowed_regimes)
    long_entries = long_entries & allowed_regime_mask
    short_entries = short_entries & allowed_regime_mask
    
    # Exit when leaving allowed regimes
    regime_change_exits = allowed_regime_mask.shift(1) & ~allowed_regime_mask
    
    # Create the portfolio based on the direction parameter
    if direction == "long":
        sl_stop = data.close - atr_multiplier * atr
        tp_stop = data.close + atr_multiplier * atr
        pf = vbt.Portfolio.from_signals(
            close=data.close,
            entries=long_entries,
            exits=regime_change_exits | short_entries,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            init_cash=10000,
            fees=0.001
        )
    else:  # short
        sl_stop = data.close + atr_multiplier * atr
        tp_stop = data.close - atr_multiplier * atr
        pf = vbt.Portfolio.from_signals(
            close=data.close,
            short_entries=short_entries,
            short_exits=regime_change_exits | long_entries,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            init_cash=10000,
            fees=0.001
        )
    
    return pf

def run_ma_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_ma: int,
    slow_ma: int,
    direction: str = "long",
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    fast_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=fast_ma).ma
    slow_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=slow_ma).ma

    long_entries = fast_ma > slow_ma
    short_entries = fast_ma < slow_ma

    # Add regime filter
    long_entries = long_entries & regime_data.isin(allowed_regimes)
    short_entries = short_entries & regime_data.isin(allowed_regimes)

    # Calculate ATR
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df['High'],
        low=symbol_ohlcv_df['Low'],
        close=symbol_ohlcv_df['Close'],
        window=atr_window
    ).atr

    # Calculate stop loss and take profit levels
    long_sl_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr
    long_tp_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_sl_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_tp_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr

    # Run the simulation
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df.Close,
            entries=long_entries,
            exits=~regime_data.isin(allowed_regimes),
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df.Close,
            short_entries=short_entries,
            short_exits=~regime_data.isin(allowed_regimes),
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            fees=fees,
            delta_format='target',
        )

    return pf

def run_macd_divergence_strategy_with_stops(
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

    # Calculate ATR
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df['High'],
        low=symbol_ohlcv_df['Low'],
        close=symbol_ohlcv_df['Close'],
        window=atr_window
    ).atr

    # Calculate stop loss and take profit levels
    long_sl_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr
    long_tp_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_sl_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_tp_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            entries=entries,
            exits=~regime_data.isin(allowed_regimes),
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            short_entries=entries,
            short_exits=~regime_data.isin(allowed_regimes),
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            fees=fees,
            delta_format='target',
        )
    return pf

def run_rsi_divergence_strategy_with_stops(
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
    if direction == "long":
        entries = (
            (symbol_ohlcv_df['Close'] == price_min) &  # New price low
            (rsi < rsi_threshold) &  # RSI below threshold
            (rsi > rsi_min) &  # RSI not at new low
            (regime_data.isin(allowed_regimes))  # In allowed regime
        )
    else:  # short
        entries = (
            (symbol_ohlcv_df['Close'] == symbol_ohlcv_df['Close'].rolling(window=lookback_window).max()) &  # New price high
            (rsi > 100 - rsi_threshold) &  # RSI above inverse threshold
            (rsi < rsi.rolling(window=lookback_window).max()) &  # RSI not at new high
            (regime_data.isin(allowed_regimes))  # In allowed regime
        )

    # Calculate stop loss and take profit levels
    long_sl_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr
    long_tp_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_sl_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_tp_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            entries=entries,
            exits=~regime_data.isin(allowed_regimes),
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            short_entries=entries,
            short_exits=~regime_data.isin(allowed_regimes),
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            fees=fees,
            delta_format='target',
        )
    return pf

def run_psar_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    af0: float = 0.02,
    af_increment: float = 0.02,
    max_af: float = 0.2,
    direction: str = "long",
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    # Calculate PSAR using the custom function
    long, short, _, _, _, _ = psar_nb_with_next(
        symbol_ohlcv_df['High'].values,
        symbol_ohlcv_df['Low'].values,
        symbol_ohlcv_df['Close'].values,
        af0=af0,
        af_increment=af_increment,
        max_af=max_af
    )

    # Generate entry signals
    if direction == "long":
        entries = pd.Series(long < symbol_ohlcv_df['Low'].values, index=symbol_ohlcv_df.index)
    else:  # short
        entries = pd.Series(short > symbol_ohlcv_df['High'].values, index=symbol_ohlcv_df.index)

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)

    # Calculate ATR
    atr = vbt.ATR.run(
        high=symbol_ohlcv_df['High'],
        low=symbol_ohlcv_df['Low'],
        close=symbol_ohlcv_df['Close'],
        window=atr_window
    ).atr

    # Calculate stop loss and take profit levels
    long_sl_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr
    long_tp_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_sl_stop = symbol_ohlcv_df['Close'] + atr_multiplier * atr
    short_tp_stop = symbol_ohlcv_df['Close'] - atr_multiplier * atr

    # Create and return the portfolio
    if direction == "long":
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            entries=entries,
            exits=~regime_data.isin(allowed_regimes),
            sl_stop=long_sl_stop,
            tp_stop=long_tp_stop,
            fees=fees,
            delta_format='target',
        )
    else:
        pf = vbt.PF.from_signals(
            close=symbol_ohlcv_df['Close'],
            short_entries=entries,
            short_exits=~regime_data.isin(allowed_regimes),
            sl_stop=short_sl_stop,
            tp_stop=short_tp_stop,
            fees=fees,
            delta_format='target',
        )
    return pf

def optimize_wrapper(name, func, params):
    print(f"Optimizing {name}...")
    
    # Optimize for long
    long_params = params.copy()
    long_params['direction'] = ['long']
    best_long_params, long_pf, _ = optimize_strategy(func, long_params, btc_1h, btc_aligned_regime_data, [3, 4])
    
    # Optimize for short
    short_params = params.copy()
    short_params['direction'] = ['short']
    best_short_params, short_pf, _ = optimize_strategy(func, short_params, btc_1h, btc_aligned_regime_data, [3, 4])
    
    # Create stats for both
    long_stats = create_stats(name, "long", long_pf, best_long_params)
    short_stats = create_stats(name, "short", short_pf, best_short_params)
    
    return long_stats, short_stats, long_pf, short_pf

def create_stats(name, direction, pf, params):
    return {
        "Strategy": f"{name} ({direction.capitalize()})",
        "Direction": direction,
        "Total Return": pf.total_return,
        "Sharpe Ratio": pf.sharpe_ratio,
        "Sortino Ratio": pf.sortino_ratio,
        "Win Rate": pf.trades.win_rate,
        "Max Drawdown": pf.max_drawdown,
        "Calmar Ratio": pf.calmar_ratio,
        "Omega Ratio": pf.omega_ratio,
        "Profit Factor": pf.trades.profit_factor,
        "Expectancy": pf.trades.expectancy,
        "Total Trades": pf.trades.count(),
        **params
    }

def run_optimized_strategies():
    strategies = [
        ("Moving Average", run_ma_strategy_with_stops, ma_params),
        ("MACD Divergence", run_macd_divergence_strategy_with_stops, macd_params),
        ("RSI Divergence", run_rsi_divergence_strategy_with_stops, rsi_params),
        ("Bollinger Bands", run_bbands_strategy, bbands_params),
        ("Parabolic SAR", run_psar_strategy_with_stops, psar_params),
        ("RSI Mean Reversion", run_rsi_mean_reversion_strategy, rsi_mean_reversion_params),
        ("Mean Reversion", mean_reversion_strategy, mean_reversion_params),
    ]

    results = Parallel(n_jobs=-1)(delayed(optimize_wrapper)(name, func, params) for name, func, params in strategies)

    all_stats = [item for sublist in results for item in sublist[:2]]  # Flatten the stats
    all_portfolios = [item for sublist in results for item in sublist[2:]]  # Flatten the portfolios
    strategy_names = [f"{name} (Long)" for name, _, _ in strategies] + [f"{name} (Short)" for name, _, _ in strategies]

    return pd.DataFrame(all_stats), all_portfolios, strategy_names

if __name__ == "__main__":
    optimized_results, portfolios, strategy_names = run_optimized_strategies()
    optimized_results.to_csv("optimized_results.csv", index=False)
    
    # Display results in a formatted table
    print(tabulate(optimized_results, headers='keys', tablefmt='pipe', floatfmt='.4f'))

    # Plot the performance for each strategy
    for pf, strat_name in zip(portfolios, strategy_names):
        pf.plot(title=strat_name).show()
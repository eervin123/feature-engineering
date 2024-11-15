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


def validate_timeframe_params(tf1_list, tf2_list):
    """Validate that timeframe lists have no overlap and tf2 values are larger than tf1."""
    # Convert timeframe strings to hours
    def to_hours(tf):
        if isinstance(tf, (int, float)):
            return tf
        num = int(''.join(filter(str.isdigit, tf)))
        unit = ''.join(filter(str.isalpha, tf)).upper()
        if unit == 'H':
            return num
        elif unit == 'D':
            return num * 24
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    # Get the maximum value from tf1_list and minimum value from tf2_list
    max_tf1 = max(to_hours(tf) for tf in tf1_list)
    min_tf2 = min(to_hours(tf) for tf in tf2_list)
    
    if min_tf2 <= max_tf1:
        raise ValueError("All timeframe_2 values must be larger than timeframe_1 values")

def optimize_strategy(strategy_func, strategy_params, symbol_ohlcv_df, regime_data, allowed_regimes, n_trials=1000):
    # Add validation for timeframe parameters if they exist
    if 'timeframe_1' in strategy_params and 'timeframe_2' in strategy_params:
        validate_timeframe_params(strategy_params['timeframe_1'], strategy_params['timeframe_2'])
    
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

        # Objective function options to maximize:
        
        # Current objective: Balance between trade frequency and returns
        # - Weights number of trades (20%) to avoid overfitting on few trades
        objective = (pf.trades.count()*0.20) * pf.total_return
        
        # Alternative objectives to consider:
        # pf.total_return              # Simple returns - good for pure performance
        # pf.sharpe_ratio             # Returns/volatility - good for risk-adjusted performance
        # pf.sortino_ratio            # Similar to Sharpe but only penalizes downside volatility
        # pf.omega_ratio              # Probability weighted ratio of gains vs losses
        # pf.trades.win_rate          # Pure win rate - but beware of small gains vs large losses
        # pf.calmar_ratio             # Returns/max drawdown - good for drawdown-sensitive strategies
        # pf.trades.profit_factor     # Gross profits/gross losses - good for consistent profitability
        
        return float('-inf') if pd.isna(objective) else objective  # Return -inf for invalid strategies

    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=25, interval_steps=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    def early_stopping_callback(study, trial):
        if study.best_trial.number + 200 < trial.number:
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
    "use_sl_tp": [True, False],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

ma_params = {
    "fast_ma": (15, 200),    
    "slow_ma": (100, 500),
    "direction": ["long", "short"],
    "use_sl_tp": [False],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

rsi_params = {
    "rsi_window": (5, 50),
    "rsi_threshold": (10, 90),
    "lookback_window": (5, 100),
    "use_sl_tp": [True, False],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

macd_params = {
    "fast_window": (5, 100),
    "slow_window": (10, 300),
    "signal_window": (5, 50),
    "direction": ["long", "short"],
    "use_sl_tp": [True, False],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

psar_params = {
    "af0": (0.01, 0.1),
    "af_increment": (0.01, 0.1),
    "max_af": (0.1, 0.5),
    "direction": ["long", "short"],
    "use_sl_tp": [True, False],
    "atr_window": (5, 50),
    "atr_multiplier": (0.5, 10.0),
}

rsi_mean_reversion_params = {
    "rsi_window": (5, 200),
    "rsi_lower": (20, 40),
    "rsi_upper": (60, 80),
    "use_sl_tp": [True, False],
    "atr_window": (5, 100),
    "atr_multiplier": (0.5, 10.0),
    "direction": ["long", "short"],
}

mean_reversion_params = {
    'bb_window': (5, 300),
    'bb_alpha': (0.5, 4.0),
    'timeframe_1': ['4H', '8H', '12H'],  # Back to using string lists
    'timeframe_2': ['16H', '24H', '32H', '48H', '72H'],  # Back to using string lists
    'direction': ['long', 'short'],
    'use_sl_tp': [True, False],
    'atr_window': (5, 50),
    'atr_multiplier': (0.5, 10.0),
}


def run_rsi_mean_reversion_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    fees: float = 0.001,
    rsi_window: int = 14,
    rsi_lower: int = 30,
    rsi_upper: int = 70,
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: int = 3,
):
    rsi = vbt.RSI.run(close=symbol_ohlcv_df["Close"], window=rsi_window)
    
    # Determine entries
    long_entries = (rsi.rsi < rsi_lower) & (regime_data.isin(allowed_regimes))
    short_entries = (rsi.rsi > rsi_upper) & (regime_data.isin(allowed_regimes))
    regime_exits = ~regime_data.isin(allowed_regimes)
    
    pf_kwargs = {
        'close': symbol_ohlcv_df["Close"],
        'fees': fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr
        
        if direction == "long":
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                'delta_format': 'target'
            })
        else:
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                'delta_format': 'target'
            })

    if direction == "long":
        pf_kwargs.update({
            'entries': long_entries,
            'exits': regime_exits
        })
    else:
        pf_kwargs.update({
            'short_entries': short_entries,
            'short_exits': regime_exits
        })

    return vbt.PF.from_signals(**pf_kwargs)

def mean_reversion_strategy(
    symbol_ohlcv_df,
    regime_data,
    allowed_regimes,
    direction="long",
    bb_window=21,
    bb_alpha=2.0,
    timeframe_1='4H',
    timeframe_2='24H',
    use_sl_tp: bool = True,
    atr_window=14,
    atr_multiplier=3.0,
    **kwargs
):
    
    data = vbt.BinanceData.from_data(symbol_ohlcv_df)
    
    bbands_tf1 = vbt.talib("BBANDS").run(data.close, timeperiod=bb_window, nbdevup=bb_alpha, nbdevdn=bb_alpha, timeframe=timeframe_1)
    bbands_tf2 = vbt.talib("BBANDS").run(data.close, timeperiod=bb_window, nbdevup=bb_alpha, nbdevdn=bb_alpha, timeframe=timeframe_2)
    
    # Generate entry conditions
    long_entries = (
        (data.close < bbands_tf2.middleband) &
        (data.close < bbands_tf1.lowerband)
    ) | (
        (data.close > bbands_tf2.lowerband) &
        (data.close < bbands_tf1.lowerband)
    )
    
    short_entries = (
        (data.close > bbands_tf2.middleband) &
        (data.close > bbands_tf1.upperband)
    ) | (
        (data.close < bbands_tf2.upperband) &
        (data.close > bbands_tf1.upperband)
    )

    allowed_regime_mask = regime_data.isin(allowed_regimes)
    long_entries = long_entries & allowed_regime_mask
    short_entries = short_entries & allowed_regime_mask
    regime_change_exits = allowed_regime_mask.shift(1) & ~allowed_regime_mask

    pf_kwargs = {
        'close': data.close,
        'init_cash': 10000,
        'fees': 0.001
    }

    if use_sl_tp:
        atr = vbt.ATR.run(data.high, data.low, data.close, window=atr_window).atr
        
        if direction == "long":
            pf_kwargs.update({
                'sl_stop': data.close - atr_multiplier * atr,
                'tp_stop': data.close + atr_multiplier * atr
            })
        else:
            pf_kwargs.update({
                'sl_stop': data.close + atr_multiplier * atr,
                'tp_stop': data.close - atr_multiplier * atr
            })

    if direction == "long":
        pf_kwargs.update({
            'entries': long_entries,
            'exits': regime_change_exits | short_entries
        })
    else:
        pf_kwargs.update({
            'short_entries': short_entries,
            'short_exits': regime_change_exits | long_entries
        })

    return vbt.Portfolio.from_signals(**pf_kwargs)

def run_ma_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_ma: int = 21,
    slow_ma: int = 55,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    fast_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=fast_ma).ma
    slow_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=slow_ma).ma

    long_entries = fast_ma > slow_ma
    long_exits = fast_ma < slow_ma
    short_entries = fast_ma < slow_ma
    short_exits = fast_ma > slow_ma

    # Add regime filter
    long_entries = long_entries & regime_data.isin(allowed_regimes)
    short_entries = short_entries & regime_data.isin(allowed_regimes)
    long_regime_exits = ~regime_data.isin(allowed_regimes)
    short_regime_exits = ~regime_data.isin(allowed_regimes)

    # Combine regime exits with other exit conditions
    long_exits = long_exits | long_regime_exits
    short_exits = short_exits | short_regime_exits

    pf_kwargs = {
        'close': symbol_ohlcv_df.Close,
        'open': symbol_ohlcv_df.Open,
        'high': symbol_ohlcv_df.High,
        'low': symbol_ohlcv_df.Low,
        'fees': fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df.High,
            low=symbol_ohlcv_df.Low,
            close=symbol_ohlcv_df.Close,
            window=atr_window
        ).atr

        if direction == "long":
            pf_kwargs.update({
                'entries': long_entries,
                'exits': long_exits,
                'sl_stop': symbol_ohlcv_df.Close - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df.Close + atr_multiplier * atr,
                'delta_format': 'target'
            })
        else:
            pf_kwargs.update({
                'short_entries': short_entries,
                'short_exits': short_exits,
                'sl_stop': symbol_ohlcv_df.Close + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df.Close - atr_multiplier * atr,
                'delta_format': 'target'
            })
    else:
        if direction == "long":
            pf_kwargs.update({
                'entries': long_entries,
                'exits': long_exits
            })
        else:
            pf_kwargs.update({
                'short_entries': short_entries,
                'short_exits': short_exits
            })

    return vbt.PF.from_signals(**pf_kwargs)

def run_macd_divergence_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
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

    pf_kwargs = {
        'close': symbol_ohlcv_df['Close'],
        'fees': fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df['High'],
            low=symbol_ohlcv_df['Low'],
            close=symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr

        if direction == "long":
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'delta_format': 'target'
            })
        else:
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'delta_format': 'target'
            })

    if direction == "long":
        pf_kwargs.update({
            'entries': entries,
            'exits': ~regime_data.isin(allowed_regimes)
        })
    else:
        pf_kwargs.update({
            'short_entries': entries,
            'short_exits': ~regime_data.isin(allowed_regimes)
        })

    return vbt.PF.from_signals(**pf_kwargs)

def run_rsi_divergence_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    rsi_window: int = 14,
    rsi_threshold: int = 30,
    lookback_window: int = 25,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    # Calculate RSI
    rsi = vbt.RSI.run(symbol_ohlcv_df['Close'], window=rsi_window).rsi

    # Calculate rolling minimum for price and RSI
    price_min = symbol_ohlcv_df['Close'].rolling(window=lookback_window).min()
    rsi_min = rsi.rolling(window=lookback_window).min()

    # Generate entry signals
    if direction == "long":
        entries = (
            (symbol_ohlcv_df['Close'] == price_min) &
            (rsi < rsi_threshold) &
            (rsi > rsi_min) &
            (regime_data.isin(allowed_regimes))
        )
    else:  # short
        entries = (
            (symbol_ohlcv_df['Close'] == symbol_ohlcv_df['Close'].rolling(window=lookback_window).max()) &
            (rsi > 100 - rsi_threshold) &
            (rsi < rsi.rolling(window=lookback_window).max()) &
            (regime_data.isin(allowed_regimes))
        )

    pf_kwargs = {
        'close': symbol_ohlcv_df['Close'],
        'fees': fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df['High'],
            low=symbol_ohlcv_df['Low'],
            close=symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr

        if direction == "long":
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'delta_format': 'target'
            })
        else:
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'delta_format': 'target'
            })

    if direction == "long":
        pf_kwargs.update({
            'entries': entries,
            'exits': ~regime_data.isin(allowed_regimes)
        })
    else:
        pf_kwargs.update({
            'short_entries': entries,
            'short_exits': ~regime_data.isin(allowed_regimes)
        })

    return vbt.PF.from_signals(**pf_kwargs)

def run_psar_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    af0: float = 0.02,
    af_increment: float = 0.02,
    max_af: float = 0.2,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    # Calculate PSAR
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

    pf_kwargs = {
        'close': symbol_ohlcv_df['Close'],
        'fees': fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df['High'],
            low=symbol_ohlcv_df['Low'],
            close=symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr

        if direction == "long":
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'delta_format': 'target'
            })
        else:
            pf_kwargs.update({
                'sl_stop': symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                'delta_format': 'target'
            })

    if direction == "long":
        pf_kwargs.update({
            'entries': entries,
            'exits': ~regime_data.isin(allowed_regimes)
        })
    else:
        pf_kwargs.update({
            'short_entries': entries,
            'short_exits': ~regime_data.isin(allowed_regimes)
        })

    return vbt.PF.from_signals(**pf_kwargs)

def run_bbands_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    fees: float = 0.001,
    bb_window: int = 14,
    bb_alpha: float = 2,
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: int = 5,
):
    # Calculate Bollinger Bands
    bbands_run = vbt.BBANDS.run(
        close=symbol_ohlcv_df["Close"], 
        window=bb_window, 
        alpha=bb_alpha
    )

    # Determine entries
    long_entries = (symbol_ohlcv_df["Close"] < bbands_run.lower) & (
        regime_data.isin(allowed_regimes)
    )
    short_entries = (symbol_ohlcv_df["Close"] > bbands_run.upper) & (
        regime_data.isin(allowed_regimes)
    )

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)

    # Common portfolio parameters
    pf_kwargs = {
        'close': symbol_ohlcv_df["Close"],
        'open': symbol_ohlcv_df["Open"],
        'high': symbol_ohlcv_df["High"],
        'low': symbol_ohlcv_df["Low"],
        'fees': fees,
    }

    if use_sl_tp:
        # Calculate ATR and stops
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update({
                'entries': long_entries,
                'exits': regime_exits,
                'sl_stop': symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                'delta_format': "target",
            })
        else:
            pf_kwargs.update({
                'short_entries': short_entries,
                'short_exits': regime_exits,
                'sl_stop': symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                'tp_stop': symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                'delta_format': "target",
            })
    else:
        if direction == "long":
            pf_kwargs.update({
                'entries': long_entries,
                'exits': regime_exits,
            })
        else:
            pf_kwargs.update({
                'short_entries': short_entries,
                'short_exits': regime_exits,
            })

    return vbt.PF.from_signals(**pf_kwargs)

def create_stats(name, symbol, direction, pf, params):
    return {
        "Symbol": symbol,
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
        "Portfolio": pf,  # Store the Portfolio object
        **params
    }

def optimize_wrapper(name, func, params, target_regimes):
    print(f"Optimizing {name} for regimes {target_regimes}...")
    
    results = []
    for symbol, ohlcv_df, regime_data in [("BTC", btc_1h, btc_aligned_regime_data), 
                                          ("ETH", eth_1h, eth_aligned_regime_data)]:
        # Optimize for long
        long_params = params.copy()
        long_params['direction'] = ['long']
        best_long_params, long_pf, _ = optimize_strategy(func, long_params, ohlcv_df, regime_data, target_regimes)
        
        # Optimize for short
        short_params = params.copy()
        short_params['direction'] = ['short']
        best_short_params, short_pf, _ = optimize_strategy(func, short_params, ohlcv_df, regime_data, target_regimes)
        
        # Create stats for both
        long_stats = create_stats(name, symbol, "long", long_pf, best_long_params)
        short_stats = create_stats(name, symbol, "short", short_pf, best_short_params)
        
        results.extend([long_stats, short_stats])
    
    return results

def run_optimized_strategies(target_regimes):
    strategies = [
        ("Moving Average", run_ma_strategy_with_stops, ma_params),
        ("MACD Divergence", run_macd_divergence_strategy_with_stops, macd_params),
        ("RSI Divergence", run_rsi_divergence_strategy_with_stops, rsi_params),
        ("Bollinger Bands", run_bbands_strategy_with_stops, bbands_params),
        ("Parabolic SAR", run_psar_strategy_with_stops, psar_params),
        ("RSI Mean Reversion", run_rsi_mean_reversion_strategy, rsi_mean_reversion_params),
        ("Mean Reversion", mean_reversion_strategy, mean_reversion_params),
    ]

    results = Parallel(n_jobs=-1)(delayed(optimize_wrapper)(name, func, params, target_regimes) 
                                 for name, func, params in strategies)
    all_stats = [item for sublist in results for item in sublist]  # Flatten the stats
    return pd.DataFrame(all_stats)

if __name__ == "__main__":
    # Define target regimes for optimization
    target_regimes = [1,2]  # Default regimes, can be modified here
    
    # You could also make it interactive:
    # print("Available regimes: 1 (Bull), 2 (Bear), 3 (Range), 4 (Volatility), 5 (Bull Volatile), 6 (Bear Volatile)")
    # target_regimes = [int(x) for x in input("Enter target regimes (space-separated numbers): ").split()]
    
    optimized_results = run_optimized_strategies(target_regimes)
    
    # Save to CSV with strategies as columns
    csv_df = optimized_results.drop('Portfolio', axis=1).set_index(['Symbol', 'Strategy', 'Direction'])
    csv_df = csv_df.unstack(['Strategy', 'Direction'])
    csv_df.to_csv(f"optimized_results_regimes_{'_'.join(map(str, target_regimes))}.csv")
    
    # Alternative flatter structure
    csv_df = optimized_results.drop('Portfolio', axis=1)
    csv_df['Strategy'] = csv_df['Strategy'] + ' (' + csv_df['Direction'] + ')'
    csv_df = csv_df.drop('Direction', axis=1).set_index(['Symbol', 'Strategy']).transpose()
    csv_df.to_csv(f"optimized_results_regimes_{'_'.join(map(str, target_regimes))}.csv")
    
    # Display results in formatted tables
    btc_results = optimized_results[optimized_results['Symbol'] == 'BTC']
    eth_results = optimized_results[optimized_results['Symbol'] == 'ETH']
    
    # Function to format results table
    def format_results_table(df):
        # Drop Portfolio column and set Strategy as index
        df = df.drop('Portfolio', axis=1).set_index('Strategy')
        
        # Move Symbol column to index if it exists
        if 'Symbol' in df.columns:
            df = df.drop('Symbol', axis=1)
        
        # Format numeric columns
        format_dict = {
            # Metrics
            'Total Return': '{:.2f}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Win Rate': '{:.2%}',
            'Max Drawdown': '{:.2%}',
            'Calmar Ratio': '{:.2f}',
            'Omega Ratio': '{:.2f}',
            'Profit Factor': '{:.2f}',
            'Expectancy': '{:.2f}',
            'Total Trades': '{:.0f}',
            
            # Strategy Parameters
            'fast_ma': '{:.0f}',
            'slow_ma': '{:.0f}',
            'atr_window': '{:.0f}',
            'atr_multiplier': '{:.2f}',
            'fast_window': '{:.0f}',
            'slow_window': '{:.0f}',
            'signal_window': '{:.0f}',
            'rsi_window': '{:.0f}',
            'rsi_threshold': '{:.0f}',
            'lookback_window': '{:.0f}',
            'bb_window': '{:.0f}',
            'bb_alpha': '{:.2f}',
            'af0': '{:.3f}',
            'af_increment': '{:.3f}',
            'max_af': '{:.3f}',
            'rsi_lower': '{:.0f}',
            'rsi_upper': '{:.0f}'
        }
        
        # Transpose the dataframe
        df = df.transpose()
        
        # Apply formatting
        for col in df.columns:
            for metric, fmt in format_dict.items():
                if metric in df.index and not pd.isna(df.loc[metric, col]):
                    try:
                        df.loc[metric, col] = fmt.format(float(df.loc[metric, col]))
                    except (ValueError, TypeError):
                        # Keep original value if formatting fails
                        pass
        
        return df
    
    print("BTC Results:")
    print(tabulate(format_results_table(btc_results), headers='keys', tablefmt='pipe', floatfmt='.4f'))
    
    print("\nETH Results:")
    print(tabulate(format_results_table(eth_results), headers='keys', tablefmt='pipe', floatfmt='.4f'))

    # Create subplot figure for all strategies
    n_strategies = len(optimized_results)
    n_cols = 4  # Number of columns in the grid
    n_rows = (n_strategies + n_cols - 1) // n_cols  # Calculate required rows

    # Create subplot layout
    fig = vbt.make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=[f"{row['Symbol']} - {row['Strategy']}" for _, row in optimized_results.iterrows()]
    )

    # Plot each strategy's cumulative returns
    for idx, row in optimized_results.iterrows():
        row_idx = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        # Get cumulative returns plot
        strategy_fig = row['Portfolio'].plot_cum_returns()
        for trace in strategy_fig.data:
            fig.add_trace(trace, row=row_idx, col=col_idx)

    # Update layout
    fig.update_layout(
        height=300 * n_rows,  # Adjust height based on number of rows
        width=1200,
        showlegend=False,
        title="Optimized Strategy Performance",
    )

    # Show the figure
    fig.show()
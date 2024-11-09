import vectorbtpro as vbt
import pandas as pd
import numpy as np

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


def build_allocation_array(strategies, strategy_weights=None, asset_weights=None):
    """
    Build an allocation array based on both strategy type and asset weights.

    Parameters:
    strategies (dict): Dictionary of strategy names to strategy objects
    strategy_weights (dict): Weights by strategy type, e.g.:
        {
            'BBands': 0.3,  # 30% to Bollinger Bands strategies
            'MA': 0.3,      # 30% to Moving Average strategies
            'MACD': 0.2,    # 20% to MACD strategies
            'RSI': 0.1,     # 10% to RSI strategies
            'PSAR': 0.1     # 10% to PSAR strategies
        }
    asset_weights (dict): Weights by asset, e.g.:
        {
            'BTC': 0.6,  # 60% to BTC strategies
            'ETH': 0.4   # 40% to ETH strategies
        }
    """
    if strategy_weights is None and asset_weights is None:
        # Default to equal weights if none provided
        n_strategies = len(strategies)
        return np.array([1 / n_strategies] * n_strategies)

    # Count strategies by type and asset
    strategy_counts = {
        "BBands": {
            "BTC": sum(1 for s in strategies.keys() if "BBands" in s and "BTC" in s),
            "ETH": sum(1 for s in strategies.keys() if "BBands" in s and "ETH" in s),
        },
        "MA": {
            "BTC": sum(1 for s in strategies.keys() if "MA " in s and "BTC" in s),
            "ETH": sum(1 for s in strategies.keys() if "MA " in s and "ETH" in s),
        },
        "MACD": {
            "BTC": sum(1 for s in strategies.keys() if "MACD" in s and "BTC" in s),
            "ETH": sum(1 for s in strategies.keys() if "MACD" in s and "ETH" in s),
        },
        "RSI": {
            "BTC": sum(1 for s in strategies.keys() if "RSI" in s and "BTC" in s),
            "ETH": sum(1 for s in strategies.keys() if "RSI" in s and "ETH" in s),
        },
        "PSAR": {
            "BTC": sum(1 for s in strategies.keys() if "PSAR" in s and "BTC" in s),
            "ETH": sum(1 for s in strategies.keys() if "PSAR" in s and "ETH" in s),
        },
    }

    # Calculate weights for each strategy
    weights = []
    for strategy_name in strategies.keys():
        asset = "BTC" if "BTC" in strategy_name else "ETH"
        asset_weight = asset_weights.get(asset, 0.5) if asset_weights else 1.0

        if "BBands" in strategy_name:
            strat_weight = (
                strategy_weights.get("BBands", 0.2) if strategy_weights else 1.0
            )
            count = strategy_counts["BBands"][asset]
            weight = (asset_weight * strat_weight) / count if count > 0 else 0
        elif "MA " in strategy_name:
            strat_weight = strategy_weights.get("MA", 0.2) if strategy_weights else 1.0
            count = strategy_counts["MA"][asset]
            weight = (asset_weight * strat_weight) / count if count > 0 else 0
        elif "MACD" in strategy_name:
            strat_weight = (
                strategy_weights.get("MACD", 0.2) if strategy_weights else 1.0
            )
            count = strategy_counts["MACD"][asset]
            weight = (asset_weight * strat_weight) / count if count > 0 else 0
        elif "RSI" in strategy_name:
            strat_weight = strategy_weights.get("RSI", 0.2) if strategy_weights else 1.0
            count = strategy_counts["RSI"][asset]
            weight = (asset_weight * strat_weight) / count if count > 0 else 0
        elif "PSAR" in strategy_name:
            strat_weight = (
                strategy_weights.get("PSAR", 0.2) if strategy_weights else 1.0
            )
            count = strategy_counts["PSAR"][asset]
            weight = (asset_weight * strat_weight) / count if count > 0 else 0
        else:
            raise ValueError(f"Unknown strategy type: {strategy_name}")
        weights.append(weight)

    # Convert to numpy array and normalize
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Ensure weights sum to 1.0

    # Validate weights sum to 1.0
    weight_sum = np.sum(weights)
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(f"Weights sum to {weight_sum}, expected approximately 1.0")

    return weights


def find_warmup_end(portfolio):
    """
    Find the end of the warmup period by identifying when all strategies have started trading.
    
    Parameters:
    portfolio: vbt.Portfolio object
    
    Returns:
    pd.Timestamp: The timestamp when the warmup period ends
    """
    # Get first trade for each strategy
    first_trades = []
    for col in portfolio.trades.records_readable['Column'].unique():
        trades = portfolio.trades.records_readable.query('Column == @col')
        if not trades.empty:
            first_trades.append(trades['Entry Index'].iloc[0])
    
    if not first_trades:
        raise ValueError("No trades found in portfolio")
    
    # Return the latest 'first trade' timestamp
    return max(first_trades)

def combine_stats(strategies, start_date=None):
    """
    Combine performance statistics from multiple strategy portfolios into a single DataFrame.
    
    Parameters:
    strategies (dict): Dictionary of strategy names to vbt.Portfolio objects
    
    Returns:
    pd.DataFrame: Combined statistics for all strategies
    """
    # Initialize list to store individual strategy stats
    all_stats = []
    
    # Calculate stats for each strategy
    for name, portfolio in strategies.items():
        # Get strategy stats
        stats = portfolio[start_date:].stats()
        # Convert to DataFrame if it's a Series
        if isinstance(stats, pd.Series):
            stats = stats.to_frame(name=name)
        else:
            stats = stats.rename(columns={0: name})
        all_stats.append(stats)
    
    # Combine all stats into a single DataFrame
    combined_stats = pd.concat(all_stats, axis=1)
    
    # Transpose so strategies are rows and metrics are columns
    return combined_stats.T


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
            btc_1h,
            btc_hourly_regime_data,
            simple_bbands_limits_long_only_btc,
            "long",
            bb_window=14,
            bb_alpha=2.0,
            atr_window=14,
            atr_multiplier=2.0,
        ),
        "BTC BBands Short": run_bbands_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_bbands_limits_short_only_btc,
            "short",
            bb_window=14,
            bb_alpha=2.0,
            atr_window=14,
            atr_multiplier=2.0,
        ),
        "ETH BBands Long": run_bbands_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_bbands_limits_long_only_eth,
            "long",
            bb_window=14,
            bb_alpha=2.0,
            atr_window=14,
            atr_multiplier=2.0,
        ),
        "ETH BBands Short": run_bbands_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_bbands_limits_short_only_eth,
            "short",
            bb_window=14,
            bb_alpha=2.0,
            atr_window=14,
            atr_multiplier=2.0,
        ),
        "BTC MA Long": run_ma_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_ma_long_only_btc,
            fast_ma=21,
            slow_ma=55,
            direction="long",
            fees=0.001,
        ),
        "BTC MA Short": run_ma_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_ma_short_only_btc,
            fast_ma=21,
            slow_ma=55,
            direction="short",
            fees=0.001,
        ),
        "ETH MA Long": run_ma_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_ma_long_only_eth,
            fast_ma=21,
            slow_ma=55,
            direction="long",
            fees=0.001,
        ),
        "ETH MA Short": run_ma_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_ma_short_only_eth,
            fast_ma=21,
            slow_ma=55,
            direction="short",
            fees=0.001,
        ),
        "BTC RSI Long": run_rsi_divergence_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_rsi_divergence_long_only_btc,
            rsi_window=14,
            rsi_threshold=30,
            lookback_window=25,
            atr_window=14,
            atr_multiplier=2.0,
            direction="long",
        ),
        "BTC MACD Long": run_macd_divergence_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_macd_long_only_btc,
            fast_window=12,
            slow_window=26,
            signal_window=9,
            fees=0.001,
            direction="long",
        ),
        "BTC MACD Short": run_macd_divergence_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_macd_short_only_btc,
            fast_window=12,
            slow_window=26,
            signal_window=9,
            fees=0.001,
            direction="short",
        ),
        "ETH MACD Long": run_macd_divergence_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_macd_long_only_eth,
            fast_window=12,
            slow_window=26,
            signal_window=9,
            fees=0.001,
            direction="long",
        ),
        "ETH MACD Short": run_macd_divergence_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_macd_short_only_eth,
            fast_window=12,
            slow_window=26,
            signal_window=9,
            fees=0.001,
            direction="short",
        ),
        "BTC PSAR Long": run_psar_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_psar_long_only_btc,
            direction="long",
            fees=0.001,
            af0=0.02,
            af_increment=0.02,
            max_af=0.2,
        ),
        "BTC PSAR Short": run_psar_strategy(
            btc_1h,
            btc_hourly_regime_data,
            simple_psar_short_only_btc,
            direction="short",
            fees=0.001,
            af0=0.02,
            af_increment=0.02,
            max_af=0.2,
        ),
        "ETH PSAR Long": run_psar_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_psar_long_only_eth,
            direction="long",
            fees=0.001,
            af0=0.02,
            af_increment=0.02,
            max_af=0.2,
        ),
        "ETH PSAR Short": run_psar_strategy(
            eth_1h,
            eth_hourly_regime_data,
            simple_psar_short_only_eth,
            direction="short",
            fees=0.001,
            af0=0.02,
            af_increment=0.02,
            max_af=0.2,
        ),
    }

    # Define strategy weights
    strategy_weights = {
        "BBands": 0.1,  # 30% to Bollinger Bands strategies
        "MA": 0.5,  # 30% to Moving Average strategies
        "MACD": 0.1,  # 20% to MACD strategies
        "RSI": 0.1,  # 10% to RSI strategies
        "PSAR": 0.2,  # 10% to PSAR strategies
    }

    asset_weights = {
        "BTC": 0.4,  # 50% to BTC strategies
        "ETH": 0.6,  # 50% to ETH strategies
    }

    # Build allocation array
    weights = build_allocation_array(
        strategies, strategy_weights=strategy_weights, asset_weights=asset_weights
    )

    # Create combined portfolio with weights
    combined_portfolio = vbt.Portfolio.column_stack(
        list(strategies.values()),
        cash_sharing=True,
        group_by=True,
        init_cash=len(strategies) * 100,
        weights=weights,
    )

    # Find end of warmup period
    warmup_end = find_warmup_end(combined_portfolio)
    print(f"\nWarmup period ends at: {warmup_end}")

    # Trim the portfolio data using date slicing
    trimmed_portfolio = combined_portfolio[warmup_end::]

    print("\n=== Overall Portfolio Performance ===")
    print(trimmed_portfolio.stats())

    # Create visualization with subplots for portfolio overview
    portfolio_fig = vbt.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # Plot cumulative returns using trimmed portfolio
    cum_returns_fig = trimmed_portfolio.plot_cum_returns()
    for trace in cum_returns_fig.data:
        portfolio_fig.add_trace(trace, row=1, col=1)

    # Plot asset allocations using trimmed portfolio
    allocations_fig = trimmed_portfolio.plot_allocations()
    for trace in allocations_fig.data:
        portfolio_fig.add_trace(trace, row=2, col=1)

    # Plot asset value using trimmed portfolio
    asset_value_fig = trimmed_portfolio.plot_asset_value()
    for trace in asset_value_fig.data:
        portfolio_fig.add_trace(trace, row=3, col=1)

    # Update layout for portfolio overview
    portfolio_fig.update_layout(
        height=1200,
        width=1000,
        showlegend=True,
        title="Portfolio Overview",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Show the combined portfolio figure
    portfolio_fig.show()

    print("\n=== Individual Strategy Performance ===")
    stats_df = combine_stats(strategies, start_date=warmup_end)
    pd.set_option('display.max_columns', None)
    print(stats_df)

    # Create figure for individual strategy performance
    n_strategies = len(strategies)
    n_cols = 4  # Number of columns in the grid
    n_rows = (n_strategies + n_cols - 1) // n_cols  # Calculate required rows (ceiling division)
    
    individual_fig = vbt.make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=list(strategies.keys())
    )

    # Plot each strategy's cumulative returns
    for idx, (name, portfolio) in enumerate(strategies.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Get cumulative returns plot for trimmed portfolio
        strategy_fig = portfolio[warmup_end::].plot_cum_returns()
        for trace in strategy_fig.data:
            individual_fig.add_trace(trace, row=row, col=col)

    # Update layout
    individual_fig.update_layout(
        height=300 * n_rows,  # Adjust height based on number of rows
        width=1200,
        showlegend=False,
        title="Individual Strategy Performance",
    )

    # Show the individual strategies figure
    individual_fig.show()

if __name__ == "__main__":
    main()

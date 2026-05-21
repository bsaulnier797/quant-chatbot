import pandas as pd
import numpy as np


def simulate_strategy(predictions, prices: pd.DataFrame) -> pd.Series:
    """
    Simulate a simple long/hold strategy based on model predictions.

    Logic: if the model predicted up (1) for day t, capture that day's
    return. If it predicted down (0), stay in cash (return = 0).

    Args:
        predictions: array-like of 0/1 predictions aligned to prices
        prices:      DataFrame with a 'Close' column

    Returns:
        Series of daily strategy returns (not cumulative)
    """
    df = pd.DataFrame({
        'prediction': predictions,
        'close': prices['Close']
    }).dropna()

    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['daily_return'] * df['prediction']

    return df['strategy_return'].dropna()


def compute_strategy_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05
) -> dict:
    """
    Compute performance metrics for the strategy vs a buy-and-hold benchmark.

    Args:
        strategy_returns:  daily strategy returns from simulate_strategy()
        benchmark_returns: daily buy-and-hold returns (prices['Close'].pct_change())
        risk_free_rate:    annualized risk-free rate (default 5%)

    Returns:
        Dict with metrics for both strategy and benchmark
    """
    def _metrics(returns, label):
        total_return = (1 + returns).prod() - 1
        annualized = (1 + total_return) ** (252 / len(returns)) - 1

        # Volatility from daily returns, annualized
        volatility = returns.std() * np.sqrt(252)

        # Sharpe: excess return over risk-free rate
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

        # Max drawdown from equity curve
        equity = (1 + returns).cumprod()
        drawdown = (equity.cummax() - equity) / equity.cummax()
        max_drawdown = drawdown.max()

        # Win rate: % of trading days with positive return
        active_days = returns[returns != 0]
        win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0

        return {
            f'{label}_total_return': round(total_return, 4),
            f'{label}_annualized_return': round(annualized, 4),
            f'{label}_volatility': round(volatility, 4),
            f'{label}_sharpe': round(sharpe, 4),
            f'{label}_max_drawdown': round(max_drawdown, 4),
            f'{label}_win_rate': round(win_rate, 4),
        }

    result = {}
    result.update(_metrics(strategy_returns, 'strategy'))
    result.update(_metrics(benchmark_returns, 'benchmark'))
    return result


def build_equity_curve(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Build an equity curve starting at $1 for both strategy and benchmark.

    Args:
        strategy_returns:  daily strategy returns from simulate_strategy()
        benchmark_returns: daily buy-and-hold returns aligned to same period

    Returns:
        DataFrame with columns: strategy, benchmark (cumulative growth of $1)
    """
    df = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Start both at 1.0 -- growth of $1 invested
    df['strategy'] = (1 + df['strategy']).cumprod()
    df['benchmark'] = (1 + df['benchmark']).cumprod()

    return df


if __name__ == "__main__":
    # Quick sanity check -- strategy that's always right should beat benchmark
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    prices = pd.DataFrame({'Close': 100 + np.random.randn(252).cumsum()}, index=dates)
    daily_returns = prices['Close'].pct_change().dropna()
    perfect_predictions = (daily_returns > 0).astype(int).values

    strategy = simulate_strategy(perfect_predictions[:-1], prices.iloc[1:])
    benchmark = daily_returns.iloc[1:]

    metrics = compute_strategy_metrics(strategy, benchmark)
    print("Metrics:", metrics)

    curve = build_equity_curve(strategy, benchmark)
    print("\nEquity curve tail:\n", curve.tail())
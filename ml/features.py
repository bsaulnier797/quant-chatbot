import pandas as pd
import numpy as np
from data.stock_data import get_price_history


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index over a rolling window.
    RSI = 100 - (100 / (1 + avg_gain / avg_loss))
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_feature_matrix(ticker: str, period: str = '2y') -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Pull price history and engineer features for ML training.

    Returns:
        X: feature DataFrame
        y: target Series (1 if tomorrow's close > today's, else 0)
        prices: raw price DataFrame (needed later for backtest)
    """
    prices = get_price_history(ticker, period=period)
    if prices.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    df = prices.copy()
    returns = df['Close'].pct_change()

    # Lagged returns
    df['lag_return_1'] = returns.shift(1)
    df['lag_return_5'] = returns.shift(5)
    df['lag_return_10'] = returns.shift(10)
    df['lag_return_20'] = returns.shift(20)

    # Rolling volatility
    df['rolling_vol_10'] = returns.rolling(window=10).std() * np.sqrt(252)
    df['rolling_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)

    # Rolling Sharpe (20-day, risk-free rate of 5%)
    excess = returns - 0.05 / 252
    df['rolling_sharpe'] = (
        excess.rolling(window=20).mean() / excess.rolling(window=20).std()
    ) * np.sqrt(252)

    # RSI
    df['rsi_14'] = compute_rsi(df['Close'], window=14)

    # Moving average crossover signal — use MA-20 and MA-50 instead of MA-50/MA-200.
    # MA-200 burns the first 200 rows on dropna, leaving too little data for walk-forward CV.
    ma_20 = df['Close'].rolling(window=20).mean()
    ma_50 = df['Close'].rolling(window=50).mean()
    df['ma_signal'] = (ma_20 > ma_50).astype(int)

    # Price relative to MA-50 (momentum feature)
    df['price_vs_ma50'] = df['Close'] / ma_50 - 1

    # Volume change vs 5-day average
    df['volume_change'] = df['Volume'] / df['Volume'].rolling(window=5).mean() - 1

    # Target: did tomorrow's close go up?
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'lag_return_1', 'lag_return_5', 'lag_return_10', 'lag_return_20',
        'rolling_vol_10', 'rolling_vol_20', 'rolling_sharpe',
        'rsi_14', 'ma_signal', 'price_vs_ma50', 'volume_change'
    ]

    # Drop rows with NaN from rolling windows and the last row (no target)
    df = df.dropna(subset=feature_cols + ['target'])

    X = df[feature_cols]
    y = df['target']
    aligned_prices = df[['Close']]

    return X, y, aligned_prices


if __name__ == "__main__":
    # Quick sanity check
    X, y, prices = build_feature_matrix("SPY")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class balance:\n{y.value_counts(normalize=True)}")
    print(f"\nFirst 5 rows:\n{X.head()}")
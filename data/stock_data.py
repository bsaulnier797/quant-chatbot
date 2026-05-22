import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import random

try:
    import pandas_datareader as pdr
    STOOQ_AVAILABLE = True
except ImportError:
    STOOQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect Yahoo Finance 429 / rate-limit errors."""
    msg = str(e).lower()
    return any(x in msg for x in ["429", "rate limit", "too many requests", "no data found"])


def _backoff(attempt: int, base: float = 2.0, cap: float = 30.0):
    """Exponential backoff with full jitter so parallel calls don't collide."""
    delay = min(cap, base ** attempt)
    time.sleep(delay + random.uniform(0, delay * 0.5))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns and drop rows with no close price."""
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if "Close" not in df.columns:
        return pd.DataFrame()
    # Keep only OHLCV columns that exist
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def _period_to_days(period: str) -> int:
    """Convert a yfinance-style period string to approximate number of calendar days."""
    mapping = {
        "1mo": 35,
        "3mo": 95,
        "6mo": 185,
        "1y": 370,
        "2y": 740,
        "5y": 1830,
        "ytd": 365,
        "max": 7300,
    }
    return mapping.get(period.lower(), 370)


def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Stooq via pandas_datareader.
    Stooq returns data newest-first, so we reverse it.
    No API key required, no rate limits.
    """
    if not STOOQ_AVAILABLE:
        return pd.DataFrame()

    try:
        days = _period_to_days(period)
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days)

        df = pdr.get_data_stooq(ticker, start=start, end=end)
        if df.empty:
            return pd.DataFrame()

        df = df.sort_index()  # Stooq returns newest-first
        return _clean(df)

    except Exception as e:
        print(f"[stock_data] Stooq fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with retry and backoff.
    Used as fallback when Stooq fails.
    """
    for attempt in range(4):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
            if not df.empty:
                return _clean(df)
            return pd.DataFrame()

        except Exception as e:
            if attempt < 3:
                if _is_rate_limit_error(e):
                    time.sleep(10 + random.uniform(0, 5))
                else:
                    _backoff(attempt)
            else:
                print(f"[stock_data] yfinance failed for {ticker} after 4 attempts: {e}")

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV price history for a ticker.
    Tries Stooq first (no rate limits, no API key), falls back to yfinance.
    Results are cached in memory for 1 hour.
    """
    # Primary: Stooq
    df = _fetch_stooq(ticker, period)
    if not df.empty:
        return df

    # Fallback: yfinance
    print(f"[stock_data] Stooq returned no data for {ticker}, trying yfinance...")
    return _fetch_yfinance(ticker, period)


@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> dict:
    """
    Fetch the current/latest price for a ticker.
    Uses the most recent close from Stooq first, then falls back to
    yfinance fast_info for a more real-time value.
    Results cached for 60 seconds.
    """
    # Try Stooq last close first (end-of-day, reliable)
    try:
        df = _fetch_stooq(ticker, "5d")
        if not df.empty:
            return {
                "ticker": ticker,
                "price": round(float(df["Close"].iloc[-1]), 2),
                "currency": "USD",
                "company_name": ticker,
            }
    except Exception:
        pass

    # Fallback: yfinance fast_info for more real-time price
    for attempt in range(3):
        try:
            fast = yf.Ticker(ticker).fast_info
            price = getattr(fast, "last_price", None) or getattr(fast, "previous_close", None)
            if price is not None:
                return {
                    "ticker": ticker,
                    "price": round(float(price), 2),
                    "currency": getattr(fast, "currency", "USD"),
                    "company_name": ticker,
                }
        except Exception as e:
            if attempt < 2:
                if _is_rate_limit_error(e):
                    time.sleep(10 + random.uniform(0, 5))
                else:
                    _backoff(attempt)

    return {
        "ticker": ticker,
        "price": "unavailable",
        "currency": "USD",
        "company_name": ticker,
        "error": "Could not fetch price from Stooq or Yahoo Finance — try again in a moment",
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_returns(price_history: pd.DataFrame) -> pd.Series:
    if price_history.empty or "Close" not in price_history.columns:
        return pd.Series()
    return price_history["Close"].pct_change().dropna()


def compute_total_return(price_history: pd.DataFrame) -> float | None:
    if price_history.empty or "Close" not in price_history.columns:
        return None
    if len(price_history) < 2:
        return None
    return (price_history["Close"].iloc[-1] - price_history["Close"].iloc[0]) / price_history["Close"].iloc[0]


def compute_annualized_volatility(price_history: pd.DataFrame) -> float | None:
    returns = compute_returns(price_history)
    if returns.empty:
        return None
    return returns.std() * np.sqrt(252)


def compute_sharpe_ratio(price_history: pd.DataFrame, risk_free_rate: float = 0.05) -> float | None:
    returns = compute_returns(price_history)
    if returns.empty:
        return None
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if std == 0:
        return None
    return excess.mean() / std * np.sqrt(252)


def compute_max_drawdown(price_history: pd.DataFrame) -> float | None:
    if price_history.empty or "Close" not in price_history.columns:
        return None
    cumulative_max = price_history["Close"].cummax()
    drawdown = (price_history["Close"] - cumulative_max) / cumulative_max
    return drawdown.min()


def compute_rolling_volatility(price_history: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = compute_returns(price_history)
    if returns.empty:
        return pd.Series()
    return returns.rolling(window=window).std() * np.sqrt(252)


def compare_tickers(ticker1: str, ticker2: str, period: str = "1y", interval: str = "1d") -> dict | str:
    history1 = get_price_history(ticker1, period)
    history2 = get_price_history(ticker2, period)

    if history1.empty:
        return f"Could not fetch data for {ticker1}."
    if history2.empty:
        return f"Could not fetch data for {ticker2}."

    return {
        ticker1: {
            "total_return": compute_total_return(history1),
            "annualized_volatility": compute_annualized_volatility(history1),
            "sharpe_ratio": compute_sharpe_ratio(history1),
            "max_drawdown": compute_max_drawdown(history1),
        },
        ticker2: {
            "total_return": compute_total_return(history2),
            "annualized_volatility": compute_annualized_volatility(history2),
            "sharpe_ratio": compute_sharpe_ratio(history2),
            "max_drawdown": compute_max_drawdown(history2),
        },
    }
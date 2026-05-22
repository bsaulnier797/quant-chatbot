import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import random


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
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV price history for a ticker from yfinance.
    Results are cached in memory for 1 hour so repeated in-session calls
    for the same ticker are instant with no extra network requests.
    """
    for attempt in range(4):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
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
                print(f"[stock_data] Failed to fetch {ticker} after 4 attempts: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> dict:
    """
    Fetch the current price for a ticker.
    Uses fast_info (lightweight endpoint) with a 60-second cache.
    Falls back to the last close from price history if fast_info fails.
    """
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

    # Fallback: last close from cached price history
    try:
        df = get_price_history(ticker, "5d")
        if not df.empty:
            return {
                "ticker": ticker,
                "price": round(float(df["Close"].iloc[-1]), 2),
                "currency": "USD",
                "company_name": ticker,
            }
    except Exception:
        pass

    return {
        "ticker": ticker,
        "price": "unavailable",
        "currency": "USD",
        "company_name": ticker,
        "error": "Yahoo Finance rate limit — try again in a moment",
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
    history1 = get_price_history(ticker1, period, interval)
    history2 = get_price_history(ticker2, period, interval)

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
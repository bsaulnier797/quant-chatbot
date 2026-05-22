import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import random
from data.database import (
    save_price_history,
    load_price_history,
    is_data_fresh,
    FRESHNESS_DAILY_HISTORY
)
from pipeline.transform import clean_price_history
from pipeline.ingest import add_to_watchlist


# ---------------------------------------------------------------------------
# Internal retry helper
# ---------------------------------------------------------------------------

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect Yahoo Finance 429 / rate-limit errors."""
    msg = str(e).lower()
    return any(x in msg for x in ["429", "rate limit", "too many requests", "no data found"])


def _backoff(attempt: int, base: float = 2.0, cap: float = 30.0):
    """Exponential backoff with full jitter so parallel calls don't collide."""
    delay = min(cap, base ** attempt)
    time.sleep(delay + random.uniform(0, delay * 0.5))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_price_history(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch price history for a ticker. Checks the local SQLite cache first.
    If the data is stale or missing, fetches from yfinance and caches it.
    Results are also cached in Streamlit's memory cache for 1 hour to avoid
    hammering yfinance on repeated in-session calls.
    """
    if is_data_fresh(ticker, "daily_history", FRESHNESS_DAILY_HISTORY):
        df = load_price_history(ticker, "daily_history")
        if not df.empty:
            return df

    for attempt in range(4):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval=interval)

            if not history.empty:
                cleaned = clean_price_history(history, ticker)
                save_price_history(ticker, cleaned, "daily_history")
                add_to_watchlist(ticker)
                return cleaned

            # Empty response — might be a bad ticker, don't retry
            return pd.DataFrame()

        except Exception as e:
            if attempt < 3:
                if _is_rate_limit_error(e):
                    # Longer pause for explicit rate-limit signals
                    time.sleep(10 + random.uniform(0, 5))
                else:
                    _backoff(attempt)
                continue
            print(f"[stock_data] Failed to fetch {ticker} after 4 attempts: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> dict:
    """
    Fetch the current price for a ticker with a 60-second cache.
    Uses fast_info instead of info — much less rate-limited.
    Falls back to the last close from cached history if yfinance is unavailable.
    """
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            # fast_info is a lightweight endpoint — avoids the heavy /quoteSummary call
            fast = stock.fast_info
            price = getattr(fast, "last_price", None) or getattr(fast, "previous_close", None)

            if price is not None:
                return {
                    "ticker": ticker,
                    "price": round(float(price), 2),
                    "currency": getattr(fast, "currency", "USD"),
                    "company_name": ticker,  # fast_info doesn't expose longName
                }

        except Exception as e:
            if attempt < 2:
                if _is_rate_limit_error(e):
                    time.sleep(10 + random.uniform(0, 5))
                else:
                    _backoff(attempt)
                continue

    # Final fallback: last close from cached history
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
        "error": "Yahoo Finance rate limit — try again in a moment"
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_returns(price_history: pd.DataFrame) -> pd.Series:
    if price_history.empty or 'Close' not in price_history.columns:
        return pd.Series()
    return price_history['Close'].pct_change().dropna()


def compute_total_return(price_history: pd.DataFrame) -> float | None:
    if price_history.empty or 'Close' not in price_history.columns:
        return None
    if len(price_history) < 2:
        return None
    return (price_history['Close'].iloc[-1] - price_history['Close'].iloc[0]) / price_history['Close'].iloc[0]


def compute_annualized_volatility(price_history: pd.DataFrame) -> float | None:
    returns = compute_returns(price_history)
    if returns.empty:
        return None
    return returns.std() * np.sqrt(252)


def compute_sharpe_ratio(price_history: pd.DataFrame, risk_free_rate: float = 0.05) -> float | None:
    returns = compute_returns(price_history)
    if returns.empty:
        return None
    excess_returns = returns - risk_free_rate / 252
    std = excess_returns.std()
    if std == 0:
        return None
    return excess_returns.mean() / std * np.sqrt(252)


def compute_max_drawdown(price_history: pd.DataFrame) -> float | None:
    if price_history.empty or 'Close' not in price_history.columns:
        return None
    cumulative_max = price_history['Close'].cummax()
    drawdown = (price_history['Close'] - cumulative_max) / cumulative_max
    return drawdown.min()


def compute_rolling_volatility(price_history: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = compute_returns(price_history)
    if returns.empty:
        return pd.Series()
    return returns.rolling(window=window).std() * np.sqrt(252)


def compare_tickers(ticker1: str, ticker2: str, period: str = '1y', interval: str = '1d') -> dict | str:
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
        }
    }
import yfinance as yf
import pandas as pd 
import numpy as np
import streamlit as st
import time

@st.cache_data(ttl=300)
def get_price_history(ticker, period='1y', interval='1d'):
    """
    Fetch historical price data for a given stock ticker.
    """
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval=interval)
            return history
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> dict:
    """
    Fetch current price for a given stock ticker.
    Falls back to price history if .info is rate limited.
    """
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = (info.get("currentPrice") or 
                     info.get("regularMarketPrice") or 
                     info.get("previousClose"))
            return {
                "ticker": ticker,
                "price": price,
                "currency": info.get("currency", "USD"),
                "company_name": info.get("longName", ticker),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            # Fallback — get last close from price history
            try:
                df = get_price_history(ticker, "5d")
                if not df.empty:
                    price = round(float(df["Close"].iloc[-1]), 2)
                    return {
                        "ticker": ticker,
                        "price": price,
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

def compute_returns(price_history):
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return pd.Series()
    returns = price_history['Close'].pct_change().dropna()
    return returns

def compute_total_return(price_history):
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return None
    initial_price = price_history['Close'].iloc[0]
    final_price = price_history['Close'].iloc[-1]
    total_return = (final_price - initial_price) / initial_price
    return total_return

def compute_annualized_volatility(price_history):
    returns = compute_returns(price_history)
    if returns.empty:
        print("No returns to compute volatility.")
        return None
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def compute_sharpe_ratio(price_history, risk_free_rate=0.01):
    returns = compute_returns(price_history)
    if returns.empty:
        print("No returns to compute Sharpe ratio.")
        return None
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

def compute_max_drawdown(price_history):
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return None
    cumulative_max = price_history['Close'].cummax()
    drawdown = (price_history['Close'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def compute_rolling_volatility(price_history, window=20):
    returns = compute_returns(price_history)
    if returns.empty:
        print("No returns to compute rolling volatility.")
        return pd.Series()
    rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_volatility

def compare_tickers(ticker1, ticker2, period='1y', interval='1d'):
    history1 = get_price_history(ticker1, period, interval)
    history2 = get_price_history(ticker2, period, interval)
    metrics = {
        ticker1: {
            "total_return": compute_total_return(history1),
            "annualized_volatility": compute_annualized_volatility(history1),
            "sharpe_ratio": compute_sharpe_ratio(history1),
            "max_drawdown": compute_max_drawdown(history1)
        },
        ticker2: {
            "total_return": compute_total_return(history2),
            "annualized_volatility": compute_annualized_volatility(history2),
            "sharpe_ratio": compute_sharpe_ratio(history2),
            "max_drawdown": compute_max_drawdown(history2)
        }
    }
    return metrics
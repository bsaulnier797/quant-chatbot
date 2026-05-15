import yfinance as yf
import pandas as pd 
import numpy as np

def get_price_history(ticker, period='1y', interval='1d'):
    """
    Fetch historical price data for a given stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL').
    period (str): The period for which to fetch data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').
    interval (str): The data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d').

    Returns:
    pd.DataFrame: A DataFrame containing the historical price data.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        return history
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    
def get_current_price(ticker: str) -> dict:
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

def compute_returns(price_history):
    """
    Compute the returns from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.

    Returns:
    pd.Series: A Series containing the returns.
    """
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return pd.Series()
    
    returns = price_history['Close'].pct_change().dropna()
    return returns

def compute_total_return(price_history):
    """
    Compute the total return from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.

    Returns:
    float: The total return over the period.
    """
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return None
    
    initial_price = price_history['Close'].iloc[0]
    final_price = price_history['Close'].iloc[-1]
    
    total_return = (final_price - initial_price) / initial_price
    return total_return

def compute_annualized_volatility(price_history):
    """
    Compute the total annualized volatility from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.

    Returns:
    float: The total annualized volatility over the period.
    """
    returns = compute_returns(price_history)
    
    if returns.empty:
        print("No returns to compute volatility.")
        return None
    
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days in a year
    return annualized_volatility

def compute_sharpe_ratio(price_history, risk_free_rate=0.01):
    """
    Compute the Sharpe ratio from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.
    risk_free_rate (float): The risk-free rate to use in the calculation (default is 0.01 for 1%).

    Returns:
    float: The Sharpe ratio over the period.
    """
    returns = compute_returns(price_history)
    
    if returns.empty:
        print("No returns to compute Sharpe ratio.")
        return None
    
    excess_returns = returns - risk_free_rate / 252  # Adjusting risk-free rate for daily returns
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualize the Sharpe ratio
    return sharpe_ratio

def compute_max_drawdown(price_history):
    """
    Compute the maximum drawdown from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.

    Returns:
    float: The maximum drawdown over the period.
    """
    if 'Close' not in price_history.columns:
        print("Price history must contain a 'Close' column.")
        return None
    
    cumulative_max = price_history['Close'].cummax()
    drawdown = (price_history['Close'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return max_drawdown


def compute_rolling_volatility(price_history, window=20):
    """
    Compute the rolling volatility from historical price data.

    Parameters:
    price_history (pd.DataFrame): A DataFrame containing historical price data with a 'Close' column.
    window (int): The window size for calculating rolling volatility (default is 20).

    Returns:
    pd.Series: A Series containing the rolling volatility.
    """
    returns = compute_returns(price_history)
    
    if returns.empty:
        print("No returns to compute rolling volatility.")
        return pd.Series()
    
    rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize the rolling volatility
    return rolling_volatility

def compare_tickers(ticker1, ticker2, period='1y', interval='1d'):
    """
    Compare two stock tickers by fetching their historical price data and computing key metrics.

    Parameters:
    ticker1 (str): The first stock ticker symbol (e.g., 'AAPL').
    ticker2 (str): The second stock ticker symbol (e.g., 'MSFT').
    period (str): The period for which to fetch data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').
    interval (str): The data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d').

    Returns:
    dict: A dictionary containing the comparison metrics for both tickers.
    """
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
# pipeline/transform.py
# Cleans and standardizes raw yfinance DataFrames before storing in SQLite

import pandas as pd
import numpy as np


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only OHLCV columns and drops any extra columns
    yfinance returns like Dividends and Stock Splits.
    """
    keep_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    existing = [col for col in keep_columns if col in df.columns]
    return df[existing]


def clean_price_history(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Applies all cleaning steps to a raw yfinance DataFrame
    before it gets stored in SQLite.

    Args:
        df:     raw DataFrame from yfinance
        ticker: ticker symbol used for print statements

    Returns:
        cleaned pandas DataFrame with DatetimeIndex
    """
    if df.empty:
        print(f"[transform] Empty DataFrame for {ticker} — skipping clean")
        return pd.DataFrame()

    df = standardize_columns(df)

    df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    df = df.ffill(limit=3)

    df = df.dropna(subset=['Close'])

    df = df[df['Close'] > 0]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    print(f"[transform] Cleaned {len(df)} rows for {ticker}")
    return df


def period_to_days(period: str) -> int:
    """
    Converts a yfinance period string to number of days.
    Used by database.py to calculate query start dates.

    Args:
        period: yfinance period string e.g. "6mo", "1y", "ytd"

    Returns:
        int number of days
    """
    from datetime import date
    today = date.today()

    period_map = {
        "1mo":  30,
        "3mo":  90,
        "6mo":  180,
        "1y":   365,
        "2y":   730,
        "5y":   1825,
        "max":  9999,
    }

    if period == "ytd":
        return (today - today.replace(month=1, day=1)).days

    return period_map.get(period, 180)
import yfinance as yf
import time
from datetime import datetime
from data.database import initialize_database, save_price_history, is_data_fresh
from data.database import FRESHNESS_DAILY_HISTORY
from pipeline.transform import clean_price_history

WATCHLIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "TSLA", "GOOGL", "AMZN", "META", "BRK-B"
]
WATCHLIST_PERIOD = "1y"


def fetch_and_store(ticker, period):
    print(f"Fetching data for {ticker}...")
    if is_data_fresh(ticker, "daily_history", FRESHNESS_DAILY_HISTORY):
        print(f"Data for {ticker} is fresh. Skipping fetch.")
        return

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            print(f"No data found for {ticker}.")
            return

        cleaned_hist = clean_price_history(hist, ticker)
        save_price_history(ticker, cleaned_hist, "daily_history")
        print(f"Data for {ticker} saved successfully.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")


def run_watchlist_ingestion():
    for ticker in WATCHLIST:
        fetch_and_store(ticker, WATCHLIST_PERIOD)
        time.sleep(1)


if __name__ == "__main__":
    initialize_database()
    run_watchlist_ingestion()
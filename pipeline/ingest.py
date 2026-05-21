import yfinance as yf
import json
import time
from data.database import initialize_database, save_price_history, is_data_fresh, FRESHNESS_DAILY_HISTORY
from pipeline.transform import clean_price_history
import os
 
WATCHLIST_PATH = "watchlist.json"
WATCHLIST_PERIOD = "1y"
 
 
def load_watchlist() -> list[str]:
    if not os.path.exists(WATCHLIST_PATH):
        default = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", 
                   "TSLA", "GOOGL", "AMZN", "META", "BRK-B"]
        with open(WATCHLIST_PATH, "w") as f:
            json.dump({"tickers": default}, f, indent=2)
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)["tickers"]
 
 
def add_to_watchlist(ticker: str):
    """Add a ticker to the watchlist if it isn't already there."""
    with open(WATCHLIST_PATH, "r") as f:
        data = json.load(f)
 
    if ticker not in data["tickers"]:
        data["tickers"].append(ticker)
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[watchlist] Added {ticker}.")
 
 
def fetch_and_store(ticker: str, period: str):
    """Fetch price history for a ticker and store it in the database."""
    if is_data_fresh(ticker, "daily_history", FRESHNESS_DAILY_HISTORY):
        print(f"[ingest] {ticker} is fresh. Skipping.")
        return
 
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
 
        if hist.empty:
            print(f"[ingest] No data returned for {ticker}.")
            return
 
        cleaned = clean_price_history(hist, ticker)
        save_price_history(ticker, cleaned, "daily_history")
        print(f"[ingest] Saved {ticker}.")
 
    except Exception as e:
        print(f"[ingest] Error fetching {ticker}: {e}")
 
 
def run_watchlist_ingestion():
    """Ingest all tickers currently in the watchlist."""
    watchlist = load_watchlist()
    print(f"[ingest] Running ingestion for {len(watchlist)} tickers...")
    for ticker in watchlist:
        fetch_and_store(ticker, WATCHLIST_PERIOD)
        time.sleep(1)
    print("[ingest] Done.")
 
 
if __name__ == "__main__":
    initialize_database()
    run_watchlist_ingestion()
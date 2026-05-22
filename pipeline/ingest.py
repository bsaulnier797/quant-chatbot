import yfinance as yf
import json
import time
import random
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
    if not os.path.exists(WATCHLIST_PATH):
        load_watchlist()  # creates the file with defaults

    with open(WATCHLIST_PATH, "r") as f:
        data = json.load(f)

    if ticker not in data["tickers"]:
        data["tickers"].append(ticker)
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[watchlist] Added {ticker}.")


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ["429", "rate limit", "too many requests", "no data found"])


def fetch_and_store(ticker: str, period: str):
    """
    Fetch price history for a ticker and store it in the database.
    Retries up to 3 times with exponential backoff and jitter.
    """
    if is_data_fresh(ticker, "daily_history", FRESHNESS_DAILY_HISTORY):
        print(f"[ingest] {ticker} is fresh. Skipping.")
        return

    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                print(f"[ingest] No data returned for {ticker}.")
                return

            cleaned = clean_price_history(hist, ticker)
            save_price_history(ticker, cleaned, "daily_history")
            print(f"[ingest] Saved {ticker}.")
            return

        except Exception as e:
            if attempt < 2:
                if _is_rate_limit_error(e):
                    wait = 15 + random.uniform(0, 5)
                    print(f"[ingest] Rate limit hit for {ticker}. Waiting {wait:.1f}s...")
                else:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[ingest] Error fetching {ticker} (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[ingest] Failed to fetch {ticker} after 3 attempts: {e}")


def run_watchlist_ingestion():
    """
    Ingest all tickers in the watchlist.
    Sleeps between requests with jitter to avoid triggering Yahoo Finance rate limits.
    """
    watchlist = load_watchlist()
    print(f"[ingest] Running ingestion for {len(watchlist)} tickers...")
    for ticker in watchlist:
        fetch_and_store(ticker, WATCHLIST_PERIOD)
        # Jittered sleep: 1–2.5s between tickers to spread out requests
        time.sleep(1 + random.uniform(0, 1.5))
    print("[ingest] Done.")


if __name__ == "__main__":
    initialize_database()
    run_watchlist_ingestion()
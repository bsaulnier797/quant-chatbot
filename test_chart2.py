from data.stock_data import get_price_history
ticker = "NVDA"
try:
    df = get_price_history(ticker, "1mo")
    print("DataFrame empty:", df.empty)
    print("Rows:", len(df))
except Exception as e:
    print("Error:", e)

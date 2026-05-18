from langchain.tools import tool
from data.stock_data import (
    get_price_history,
    get_current_price,
    compute_total_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compare_tickers
)
import plotly.graph_objects as go


@tool
def stock_performance_tool(input: str) -> str:
    """
    Use this tool when the user asks how a stock has performed, its returns,
    volatility, or price history over a time period.
    Input format: TICKER PERIOD (e.g. NVDA 6mo or AAPL 1y)
    Valid periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    parts = input.strip().split()
    if not parts:
        return "Please provide a ticker symbol."
    ticker = parts[0].upper()
    period = parts[1] if len(parts) > 1 else "6mo"

    df = get_price_history(ticker, period)
    if df.empty:
        return f"Could not fetch data for {ticker}. Please check the ticker symbol."

    total_return = compute_total_return(df)
    volatility = compute_annualized_volatility(df)
    sharpe = compute_sharpe_ratio(df)
    drawdown = compute_max_drawdown(df)

    if any(v is None for v in [total_return, volatility, sharpe, drawdown]):
        return f"Could not compute metrics for {ticker}. Please try again."

    return (
        f"{ticker} over {period}:\n"
        f"  Total Return: {total_return:.2%}\n"
        f"  Annualized Volatility: {volatility:.2%}\n"
        f"  Sharpe Ratio: {sharpe:.2f}\n"
        f"  Max Drawdown: {drawdown:.2%}"
    )


@tool
def plot_current_price(input: str) -> str:
    """
    Use this tool when the user asks for the current or live price of a stock.
    Also use for questions like 'what is Apple trading at' or 'how much is TSLA'.
    Input format: 'TICKER' (e.g. 'AAPL')
    """
    ticker = input.strip().upper()
    price_data = get_current_price(ticker)

    if price_data is None:
        return f"Could not fetch current price for {ticker}."

    if price_data.get("price") == "unavailable":
        return "Yahoo Finance is rate limiting requests. Try again in a moment."

    return (
        f"{price_data['company_name']} ({ticker}): "
        f"${price_data['price']} {price_data['currency']}"
    )


@tool
def plot_price_history(input: str) -> str:
    """
    Use this tool to generate a price history chart for a stock over a specified period.
    Input format: 'TICKER PERIOD' (e.g. 'NVDA 6mo')
    Valid periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    parts = input.strip().split()
    if not parts:
        return "Please provide a ticker symbol."
    ticker = parts[0].upper()
    period = parts[1] if len(parts) > 1 else "6mo"

    df = get_price_history(ticker, period)
    if df.empty:
        return f"Could not fetch data for {ticker}. Please check the ticker symbol."

    start = df["Close"].iloc[0]
    end = df["Close"].iloc[-1]
    total_return = (end - start) / start

    return (
        f"{ticker} price history over {period}:\n"
        f"  Start: ${start:.2f}\n"
        f"  End: ${end:.2f}\n"
        f"  Total Return: {total_return:.2%}"
    )


@tool
def plot_comparison_table(input: str) -> str:
    """
    Use this tool to compare multiple stocks side by side over a specified period.
    Input format: 'TICKER1 TICKER2 PERIOD' (e.g. 'NVDA AAPL 6mo')
    Valid periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    parts = input.strip().split()
    if len(parts) < 2:
        return "Please provide at least two ticker symbols."

    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}
    if parts[-1].lower() in valid_periods:
        period = parts[-1].lower()
        tickers = [t.upper() for t in parts[:-1]]
    else:
        period = "6mo"
        tickers = [t.upper() for t in parts]

    if len(tickers) < 2:
        return "Please provide at least two ticker symbols to compare."

    lines = [f"Comparison table over {period}:"]
    for ticker in tickers:
        df = get_price_history(ticker, period)
        if df.empty:
            lines.append(f"{ticker}: Could not fetch data.")
            continue

        tr = compute_total_return(df)
        vol = compute_annualized_volatility(df)
        sharpe = compute_sharpe_ratio(df)
        dd = compute_max_drawdown(df)

        if any(v is None for v in [tr, vol, sharpe, dd]):
            lines.append(f"{ticker}: Could not compute metrics.")
            continue

        lines.append(
            f"{ticker}:\n"
            f"  Total Return: {tr:.2%}\n"
            f"  Volatility: {vol:.2%}\n"
            f"  Sharpe: {sharpe:.2f}\n"
            f"  Max Drawdown: {dd:.2%}"
        )

    return "\n".join(lines)
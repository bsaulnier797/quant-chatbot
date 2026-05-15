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


@tool
def stock_performance_tool(input: str) -> str:
    """
    Use this tool when the user asks how a stock has performed, its returns,
    volatility, or price history over a time period.
    Input format: 'TICKER PERIOD' (e.g. 'NVDA 6mo' or 'AAPL 1y')
    Valid periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    parts = input.strip().split()
    ticker = parts[0].upper()
    period = parts[1] if len(parts) > 1 else "6mo"

    df = get_price_history(ticker, period)
    total_return = compute_total_return(df)
    volatility = compute_annualized_volatility(df)
    sharpe = compute_sharpe_ratio(df)
    drawdown = compute_max_drawdown(df)

    return (
        f"{ticker} over {period}: "
        f"Total Return: {total_return:.2%}, "
        f"Annualized Volatility: {volatility:.2%}, "
        f"Sharpe Ratio: {sharpe:.2f}, "
        f"Max Drawdown: {drawdown:.2%}"
    )


@tool
def stock_comparison_tool(input: str) -> str:
    """
    Use this tool when the user wants to compare two stocks or compare a stock
    against a benchmark like the S&P 500 (SPY).
    Input format: 'TICKER1 TICKER2 PERIOD' (e.g. 'NVDA SPY 6mo')
    If no second ticker is given, default to SPY as the benchmark.
    Valid periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    parts = input.strip().split()
    ticker1 = parts[0].upper()
    ticker2 = parts[1].upper() if len(parts) > 1 else "SPY"
    period = parts[2] if len(parts) > 2 else "6mo"

    result = compare_tickers(ticker1, ticker2, period)

    lines = [f"Comparison over {period}:"]
    for ticker, metrics in result.items():
        lines.append(
            f"{ticker}: Return={metrics['total_return']:.2%}, "
            f"Volatility={metrics['annualized_volatility']:.2%}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"Max Drawdown={metrics['max_drawdown']:.2%}"
        )
    return "\n".join(lines)


@tool
def current_price_tool(input: str) -> str:
    """
    Use this tool when the user asks for the current or live price of a stock.
    Input format: 'TICKER' (e.g. 'AAPL')
    """
    ticker = input.strip().upper()
    price = get_current_price(ticker)

    if price is None:
        return f"Could not fetch current price for {ticker}."

    return f"{ticker} current price: ${price}"

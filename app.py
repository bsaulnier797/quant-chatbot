import streamlit as st
import plotly.graph_objects as go
from agent.agent import build_agent, ask
from data.stock_data import get_price_history
import re
from charts.plotting import (
    plot_candlestick_chart,
    plot_with_moving_averages,
    plot_returns_distribution,
    plot_correlation_heatmap
)

# Page Configuration
st.set_page_config(page_title="Quant Chatbot", page_icon="📈", layout="wide")

st.title("📈 Finance & Quant Chatbot")
st.caption("Ask natural language questions about stocks, returns, volatility, and more.")

# learning vs expert mode
mode = st.radio("Select Mode", ["Expert", "Learning"], horizontal=True)

# Sidebar
with st.sidebar:
    st.header("Example Questions")
    st.markdown("""
    - How has NVDA performed over the last 6 months?
    - Compare TSLA and SPY over the last year
    - What is the current price of Apple?
    - What is the Sharpe ratio of MSFT over 1 year?
    - Compare AAPL and GOOGL over 3 months
    """)
    st.divider()
    st.caption("Data sourced from Yahoo Finance via yfinance.")


# try_render_chart function
def try_render_chart(question: str):
    import re

    # Don't render for price-only questions
    price_only_keywords = ["what is the price", "current price", "how much is", "how much does"]
    if any(keyword in question.lower() for keyword in price_only_keywords):
        return None

    # Find tickers
    tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', question.upper())
    tickers = [t[0] or t[1] for t in tickers]

    stopwords = {
        "HOW", "HAS", "THE", "AND", "FOR", "OVER", "LAST", "VS",
        "WHAT", "IS", "OF", "IN", "A", "AN", "TO", "COMPARE", "AM",
        "YEAR", "MONTH", "MONTHS", "YEARS", "DAY", "DAYS", "ME", "DO",
        "MY", "TODAY", "MARKET", "STOCK", "PRICE", "CURRENT", "GET",
        "CAN", "YOU", "TELL", "ABOUT", "SHOW", "GIVE", "HELP", "IT",
        "AT", "BY", "BE", "ARE", "WAS", "ITS", "ETF", "CEO", "CFO",
        "USA", "GDP", "IMF", "FED", "SEC", "IPO", "YTD", "EPS", "PE",
        "LEARN", "GOOD", "SOME", "TOOLS", "STILL", "GAVE", "THIS",
        "PROMPT", "GRAPH", "FINANCE", "THESE", "THAT", "WITH", "BEST",
        "USE", "USING", "USED", "MORE", "MOST", "LESS", "ALSO", "JUST",
        "LIKE", "WANT", "NEED", "KNOW", "DOES", "DID", "WILL", "WOULD",
        "GIVE", "FULL", "COMPLETE", "ANALYSIS", "INCLUDING", "TELL"
    }

    tickers = [t for t in tickers if t not in stopwords and len(t) >= 2]

    # Validate tickers have real data
    validated_tickers = []
    for ticker in tickers[:3]:
        try:
            df = get_price_history(ticker, "5d")
            if not df.empty:
                validated_tickers.append(ticker)
        except Exception:
            pass

    tickers = validated_tickers

    if not tickers:
        return None

    # Determine period
    period = "6mo"
    if "1 year" in question.lower() or "1y" in question.lower():
        period = "1y"
    elif "3 month" in question.lower():
        period = "3mo"
    elif "ytd" in question.lower():
        period = "ytd"

    question_lower = question.lower()

    try:
        # Correlation heatmap — multiple tickers
        if len(tickers) > 1 and any(word in question_lower for word in ["correlat", "heatmap", "portfolio", "compare"]):
            return plot_correlation_heatmap(None, tickers)

        # Returns distribution
        if any(word in question_lower for word in ["distribution", "returns distribution", "histogram"]):
            df = get_price_history(tickers[0], period)
            return plot_returns_distribution(df, tickers[0])

        # Moving averages
        if any(word in question_lower for word in ["moving average", "ma50", "ma20", "trend"]):
            df = get_price_history(tickers[0], period)
            return plot_with_moving_averages(df, tickers[0])

        # Candlestick — default for single ticker performance questions
        if len(tickers) == 1:
            df = get_price_history(tickers[0], period)
            return plot_candlestick_chart(df, tickers[0])

        # Normalized comparison — multiple tickers
        if len(tickers) > 1:
            fig = go.Figure()
            for ticker in tickers:
                df = get_price_history(ticker, period)
                if df.empty:
                    continue
                normalized = (df["Close"] / df["Close"].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode="lines",
                    name=ticker
                ))
            fig.update_layout(
                title="Normalized Price Performance (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode="x unified"
            )
            return fig

    except Exception:
        return None

    return None

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = build_agent()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart" in message and message["chart"] is not None:
            st.plotly_chart(message["chart"], use_container_width=True)


# Chat input and response
# Chat input and response
if prompt := st.chat_input("Ask a question about stocks or markets..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, prompt, mode.lower())

        st.markdown(response)

        # Render multiple charts for full analysis questions
        question_lower = prompt.lower()
        local_stopwords = {
            "HOW", "HAS", "THE", "AND", "FOR", "OVER", "LAST", "VS",
            "WHAT", "IS", "OF", "IN", "A", "AN", "TO", "COMPARE", "AM",
            "YEAR", "MONTH", "MONTHS", "YEARS", "DAY", "DAYS", "ME", "DO",
            "MY", "TODAY", "MARKET", "STOCK", "PRICE", "CURRENT", "GET",
            "CAN", "YOU", "TELL", "ABOUT", "SHOW", "GIVE", "HELP", "IT",
            "AT", "BY", "BE", "ARE", "WAS", "ITS", "ETF", "CEO", "CFO",
            "USA", "GDP", "IMF", "FED", "SEC", "IPO", "YTD", "EPS", "PE",
            "LEARN", "GOOD", "SOME", "TOOLS", "STILL", "GAVE", "THIS",
            "PROMPT", "GRAPH", "FINANCE", "THESE", "THAT", "WITH", "BEST",
            "USE", "USING", "USED", "MORE", "MOST", "LESS", "ALSO", "JUST",
            "LIKE", "WANT", "NEED", "KNOW", "DOES", "DID", "WILL", "WOULD",
            "GIVE", "FULL", "COMPLETE", "ANALYSIS", "INCLUDING", "TELL"
        }

        if "full analysis" in question_lower or "complete analysis" in question_lower:
            validated = []
            for t in re.findall(r'\b([A-Z]{2,5})\b', prompt.upper()):
                if t not in local_stopwords:
                    try:
                        df = get_price_history(t, "5d")
                        if not df.empty:
                            validated.append(t)
                    except Exception:
                        pass
                if len(validated) == 2:
                    break

            if validated:
                period = "1y" if "year" in question_lower else "6mo"
                df = get_price_history(validated[0], period)

                st.subheader("📊 Price Chart")
                st.plotly_chart(plot_candlestick_chart(df, validated[0]), use_container_width=True)

                st.subheader("📈 Moving Averages")
                st.plotly_chart(plot_with_moving_averages(df, validated[0]), use_container_width=True)

                st.subheader("📉 Returns Distribution")
                st.plotly_chart(plot_returns_distribution(df, validated[0]), use_container_width=True)

                if len(validated) > 1:
                    st.subheader("🔥 Correlation Heatmap")
                    st.plotly_chart(plot_correlation_heatmap(None, validated), use_container_width=True)

            chart = None

        else:
            chart = try_render_chart(prompt)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chart": chart
    })
import streamlit as st
import plotly.graph_objects as go
from agent.agent import build_agent, ask
from data.stock_data import get_price_history
import re


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
    
    # Only match tickers that are explicitly mentioned with context
    # Look for patterns like "NVDA", "$NVDA", or known index ETFs
    tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', question)
    tickers = [t[0] or t[1] for t in tickers]
    
    stopwords = {
        "HOW", "HAS", "THE", "AND", "FOR", "OVER", "LAST", "VS",
        "WHAT", "IS", "OF", "IN", "A", "AN", "TO", "COMPARE", "AM",
        "YEAR", "MONTH", "MONTHS", "YEARS", "DAY", "DAYS", "ME", "DO",
        "MY", "TODAY", "MARKET", "STOCK", "PRICE", "CURRENT", "GET",
        "CAN", "YOU", "TELL", "ABOUT", "SHOW", "GIVE", "HELP", "IT",
        "AT", "BY", "BE", "ARE", "WAS", "ITS", "ETF", "CEO", "CFO",
        "USA", "GDP", "IMF", "FED", "SEC", "IPO", "YTD", "EPS", "PE"
    }
    
    tickers = [t for t in tickers if t not in stopwords and len(t) >= 2]
    
    # Only render chart if we found at least one plausible ticker
    if not tickers:
        return None
    
    # Verify the ticker actually has data before rendering
    try:
        test_df = get_price_history(tickers[0], "1mo")
        if test_df.empty:
            return None
    except Exception:
        return None
    
    period = "6mo"
    if "1 year" in question.lower() or "1y" in question.lower():
        period = "1y"
    elif "3 month" in question.lower():
        period = "3mo"
    elif "ytd" in question.lower():
        period = "ytd"
    
    try:
        fig = go.Figure()
        for ticker in tickers[:3]:
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
        if not fig.data:
            return None
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        return fig
    except Exception:
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
if prompt := st.chat_input("Ask a question about stocks or markets..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, prompt)

        st.markdown(response)

        chart = try_render_chart(prompt)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chart": chart
    })
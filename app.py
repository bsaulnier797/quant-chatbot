import streamlit as st
import plotly.graph_objects as go
from agent.agent import build_agent, ask
from data.stock_data import get_price_history


# Page Configuration
st.set_page_config(page_title="Quant Chatbot", layout="wide")

# Sidebar for user input
st.sidebar.title("Quant Chatbot")
user_input = st.sidebar.text_input("Ask a question about stocks (e.g. 'How has NVDA performed over the last 6 months?')")
# Credit for yahoo finance data
st.sidebar.markdown("Data source: [Yahoo Finance](https://finance.yahoo.com/)")

# Session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize agent if not already done
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

# spinner while waiting for response
if st.sidebar.button("Ask"):
    if user_input:
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

def try_render_chart(prompt):
    raise NotImplementedError

# Chat input and response display 

if prompt := st.chat_input("Ask a question about stocks or markets..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, prompt)
        
        st.markdown(response)
        
        # 5. Try to render a chart
        chart = try_render_chart(prompt)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            chart = None
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chart": chart
    })


def try_render_chart(question: str):
    import re
    
    tickers = re.findall(r'\b([A-Z]{1,5})\b', question.upper())
    
    stopwords = {"HOW", "HAS", "THE", "AND", "FOR", "OVER", "LAST", "VS",
                 "WHAT", "IS", "OF", "IN", "A", "AN", "TO", "COMPARE",
                 "YEAR", "MONTH", "MONTHS", "YEARS", "DAY", "DAYS", "ME",
                 "MY", "TODAY", "MARKET", "STOCK", "PRICE", "CURRENT"}
    tickers = [t for t in tickers if t not in stopwords]
    
    if not tickers:
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
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode="x unified"
        )
        return fig
    except Exception:
        return None


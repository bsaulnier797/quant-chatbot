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
    # Learn about finance with clickable sections for volatility, Sharpe ratio, drawdown, etc.
    st.header("Quick Learning")
    st.divider()
    # clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []


    selected_concept = st.selectbox("Select a concept to learn about:", [
        "None",
        "Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Total Return",
        "Correlation",
        "Beta",
        "Alpha"
    ],
    index=0
    )
    
    

    st.divider()
    st.caption("Data sourced from Yahoo Finance via yfinance.")

# Generate Follow-up Questions
def generate_followups(client, question: str, response: str, mode: str) -> list:
    """
    Generate 3 relevant follow-up questions based on the last exchange.
    Returns a list of 3 short question strings.
    """
    try:
        prompt = f"""The user asked: "{question}"
The assistant responded: "{response[:600]}"

Generate exactly 3 short follow-up questions the user might want to ask next.
They should be specific, natural, and directly related to what was just discussed.
Return only a JSON array of 3 strings, nothing else.
Example: ["What is its Sharpe ratio?", "How does that compare to SPY?", "Show me the returns distribution"]"""

        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="You generate follow-up questions. Return only a JSON array of 3 strings.",
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        text = result.content[0].text.strip()
        # Strip markdown code fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        questions = json.loads(text)
        return questions[:3] if isinstance(questions, list) else []

    except Exception:
        return []


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


if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = build_agent()

if "followups" not in st.session_state:
    st.session_state.followups = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None






# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart" in message and message["chart"] is not None:
            st.plotly_chart(message["chart"], use_container_width=True)


# follow ups
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    st.session_state.followups = []

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, prompt, mode.lower(),
                           history=st.session_state.messages)
        st.markdown(response)

        chart = try_render_chart(prompt)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chart": chart
    })

    st.session_state.followups = generate_followups(
        st.session_state.agent, prompt, response, mode.lower()
    )
    st.rerun()  

# Show follow-up question buttons
if st.session_state.followups:
    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(3)
    for i, question in enumerate(st.session_state.followups):
        with cols[i]:
            if st.button(question, key=f"followup_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.session_state.followups = []
                st.rerun()
# Chat input and response
# Chat input and response
if prompt := st.chat_input("Ask a question about stocks or markets..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state.agent, prompt, mode.lower(), history=st.session_state.messages)

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


if selected_concept and selected_concept != "None":
    with st.expander(f"Learn about {selected_concept}"):
        if selected_concept == "Volatility":
            st.markdown("""
            **Volatility** is a measure of how much a stock's price fluctuates over time. 
            A stock with high volatility has large price swings, while a stock with low volatility has more stable prices. 
            Volatility is often measured using the standard deviation of returns or the annualized volatility formula.
            """)
        elif selected_concept == "Sharpe Ratio":
            st.markdown("""
            The **Sharpe Ratio** is a measure of risk-adjusted return. It tells you how much excess return you are getting for each unit of risk you take on. 
            A higher Sharpe ratio indicates better risk-adjusted performance. The formula is: 
            Sharpe Ratio = (Return of the Portfolio - Risk-Free Rate) / Standard Deviation of the Portfolio's Excess Return.
            """)
        elif selected_concept == "Max Drawdown":
            st.markdown("""
            **Max Drawdown** is the largest percentage drop from a peak to a trough in the value of an investment. 
            It measures the worst-case loss an investor could have experienced during a specific period. 
            A smaller max drawdown indicates less severe losses during downturns.
            """)
        elif selected_concept == "Total Return":
            st.markdown("""
            **Total Return** is the overall return on an investment, including both capital gains and dividends. 
            It is calculated as: Total Return = (Ending Value - Beginning Value + Dividends) / Beginning Value.
            """)
        elif selected_concept == "Correlation":
            st.markdown("""
            **Correlation** measures the strength and direction of the linear relationship between two variables. 
            It ranges from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.
            """)
        elif selected_concept == "Beta":
            st.markdown("""
            **Beta** is a measure of a stock's volatility in relation to the overall market. 
            A beta of 1 indicates that the stock's price tends to move in line with the market, while a beta greater than 1 indicates higher volatility and a beta less than 1 indicates lower volatility.
            """)
        elif selected_concept == "Alpha":
            st.markdown("""
            **Alpha** represents the excess return of an investment relative to the return of a benchmark index. 
            It measures the skill of the portfolio manager in generating returns above the market average.
            """)
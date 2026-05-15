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


    # try render helper function
def try_render_chart(prompt):
    # This is a placeholder implementation. You would need to implement logic to parse the prompt,
    # determine if a chart is needed, and then generate the appropriate chart using Plotly or another library.
    # For example, if the prompt asks for the price history of a stock, you could fetch the data and create a line chart.
    try:
        if "price history" in prompt.lower():
            ticker = prompt.split()[0]  # This is a very naive way to extract the ticker
            df = get_price_history(ticker, "6mo")  # Fetch 6 months of data as an example
            fig = go.Figure(data=go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price")
            return fig
    except Exception as e:
        print(f"Error rendering chart for prompt '{prompt}': {e}")
        return None


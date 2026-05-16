# Imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data.stock_data import (
    get_price_history,
    compute_returns,
    compute_rolling_volatility
)

# candlesticks on charts



def plot_candlestick_chart(data, ticker):
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    return fig

# Moving averages overlay

def plot_with_moving_averages(data, ticker):
    import plotly.graph_objects as go
    
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='20-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-day MA'))


    fig.update_layout(
        title=f"{ticker} Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig

# Returns distribution
def plot_returns_distribution(data, ticker):
    import plotly.graph_objects as go
    
    returns = compute_returns(data)
    
    fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
    fig.update_layout(
        title=f"{ticker} Returns Distribution",
        xaxis_title="Returns",
        yaxis_title="Frequency"
    )
    return fig

# Correlation heatmap
def plot_correlation_heatmap(data, tickers):
    import plotly.graph_objects as go
    
    returns = pd.DataFrame({ticker: compute_returns(get_price_history(ticker, "1y")) for ticker in tickers})
    corr = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Ticker",
        yaxis_title="Ticker"
    )
    return fig
# Finance & Quant Chatbot

A conversational AI app that lets you ask plain English questions about stocks, returns, volatility, and other quantitative finance topics. It also includes a full ML forecasting tab that trains an XGBoost model on historical price data and backtests its signals against buy-and-hold.

[**Try the live app**](https://bsaulnier797-quant-chatbot-app-magfyv.streamlit.app)

---

## Why I built this

Most finance tools either dump raw data on you or require you to already know what you're looking for. I wanted something where you could type "How has NVDA performed over the last six months?" and get back a real answer with charts, calculations, and context. The ML tab came out of wanting to go one step further and ask whether a model trained on price features could actually generate edge vs just holding.

---

## What you can do with it

**Chat tab**
- Ask questions in plain English, no special syntax needed
- Get candlestick charts, moving averages, returns distributions, and correlation heatmaps generated automatically based on what you asked
- See quant metrics including Sharpe ratio, max drawdown, annualized volatility, and total return
- Compare multiple tickers side by side with normalized performance charts
- Switch between **Expert mode** (direct analysis with the numbers) and **Learning mode** (plain English explanations for people newer to finance)
- Look up finance concepts like beta, correlation, and max drawdown in the sidebar without losing your place in the conversation
- Get AI-suggested follow-up questions after each response

**ML Forecast tab**
- Enter any ticker and click Train Model
- The pipeline builds a feature matrix from price history, runs walk-forward cross-validation, trains an XGBoost classifier, and backtests the resulting signals
- See model accuracy, precision, recall, and F1 from both walk-forward CV and a held-out test set
- Compare strategy vs buy-and-hold on an equity curve with Sharpe ratio, max drawdown, and win rate
- View feature importance so you can see what the model is actually using
- Get a next-day Buy/Hold signal with confidence percentage

---

## How it works

**Chat flow**

When you send a message, a Claude Haiku agent reads your question and decides which data tools to call. Those tools hit yfinance for real market data, run the relevant calculations, and send results back to the agent. The agent writes a response and the frontend determines whether to render a chart alongside it. Price data is cached in memory with `@st.cache_data` so repeated queries for the same ticker are instant.

```
User message
    -> Agent interprets intent
    -> Calls tools (price history, Sharpe ratio, drawdown, etc.)
    -> Gets results back
    -> Writes response
    -> Frontend renders chart if relevant
```

**ML pipeline flow**

```
Ticker input
    -> build_feature_matrix()  (lagged returns, rolling vol, RSI, MA signals, volume)
    -> walk_forward_validate() (rolling train/test splits, never shuffles)
    -> train_model()           (XGBoost on first half of data)
    -> evaluate_model()        (held-out test set metrics)
    -> simulate_strategy()     (trade next-day returns using predictions)
    -> compute_strategy_metrics() + build_equity_curve()
    -> render to Streamlit
```

Walk-forward validation is used instead of a random train/test split because shuffling time series data leaks future information into training, which inflates accuracy scores. Each fold trains on all data before it and tests on the next window forward.

---

## Project structure

```
quant-chatbot/
├── app.py                  # Streamlit UI, chat logic, and ML tab
├── agent/
│   ├── agent.py            # Anthropic API agent with tool-calling loop
│   └── tools.py            # Tool definitions wrapping the data layer
├── data/
│   └── stock_data.py       # Price history, returns, and quant metric calculations
├── charts/
│   └── plotting.py         # Plotly chart functions
├── ml/
│   ├── features.py         # Feature engineering from price history
│   ├── model.py            # XGBoost training, walk-forward CV, evaluation
│   └── backtest.py         # Strategy simulation and equity curve
├── memory/                 # Conversation memory utilities
├── notebooks/              # EDA and exploration
└── requirements.txt
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| AI agent | Anthropic API (Claude Haiku) |
| Market data | yfinance |
| ML model | XGBoost |
| Charts | Plotly |
| Language | Python 3.12 |
| Deployment | Streamlit Community Cloud |

---

## ML features used

| Feature | Description |
|---|---|
| `lag_return_1/5/10/20` | Lagged daily returns over 1, 5, 10, and 20 days |
| `rolling_vol_10/20` | Annualized rolling volatility over 10 and 20 days |
| `rolling_sharpe` | 20-day rolling Sharpe ratio |
| `rsi_14` | 14-day Relative Strength Index |
| `ma_signal` | MA-20 vs MA-50 crossover signal |
| `price_vs_ma50` | Price relative to 50-day moving average |
| `volume_change` | Volume relative to 5-day average |

---

## Running it locally

**1. Clone the repo**
```bash
git clone https://github.com/bsaulnier797/quant-chatbot.git
cd quant-chatbot
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your Anthropic API key**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

Get an API key at [console.anthropic.com](https://console.anthropic.com).

**4. Run the app**
```bash
streamlit run app.py
```

---

## Things worth knowing

- yfinance occasionally has API issues with Yahoo Finance. If you see ticker fetch errors, make sure you are on `yfinance>=0.2.51`.
- Streamlit Cloud requires secrets to be set in the dashboard under **Settings > Secrets**. The local `.env` file does not carry over automatically.
- The ML model trains on demand when you click Train Model. The first run for a ticker takes 20-40 seconds depending on the walk-forward fold count. Results are cached for the session after the first run.
- This is a research and portfolio project. The Buy/Hold signal is not financial advice.

---

## About

Built by Brett Saulnier, data science and mathematics student at the University of Wisconsin-Madison.

[GitHub](https://github.com/bsaulnier797) 

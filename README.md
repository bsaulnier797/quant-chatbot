# Finance and Quant Chatbot

A conversational AI app that lets you ask plain English questions about stocks, returns, volatility, and other quantitative finance concepts. Built with Python, Streamlit, and the Anthropic API.

[**Try the live app**](https://bsaulnier797-quant-chatbot-app-magfyv.streamlit.app)

---

## Why I built this

I wanted a tool where you could type something like "How has NVDA performed over the last six months?" and get back a real answer with charts and the math behind it, not just a number. Most finance tools either require you to already know what you're looking for or give you raw data with no explanation. This tries to do both at once.

---

## What you can do with it

- Ask questions in plain English, no special syntax needed
- Get candlestick charts, moving averages, returns distributions, and correlation heatmaps generated automatically based on what you asked
- See quant metrics including Sharpe ratio, max drawdown, annualized volatility, and total return
- Compare multiple tickers side by side with normalized performance charts
- Switch between **Expert mode** (direct analysis with the numbers) and **Learning mode** (plain English explanations for people newer to finance)
- Look up finance concepts like beta, correlation, and max drawdown in the sidebar without losing your place in the conversation
- Get AI-suggested follow-up questions after each response so you can keep exploring

---

## How it works

When you send a message, a Claude Haiku agent reads your question and decides which data tools to call. Those tools pull real market data from yfinance, run the relevant calculations, and send the results back to the agent. The agent writes a response and the frontend determines whether to render a chart alongside it.

There is also a data pipeline that runs on startup and ingests price data for a default watchlist into a local SQLite database, so repeated queries for the same tickers are faster.

```
User message
    -> Agent interprets intent
    -> Calls tools (price history, Sharpe ratio, drawdown, etc.)
    -> Gets results back
    -> Writes response
    -> Frontend renders chart if relevant
```

---

## Project structure

```
quant-chatbot/
├── app.py                  # Streamlit UI and chat logic
├── agent/
│   ├── agent.py            # Anthropic API agent with tool-calling loop
│   └── tools.py            # Tool definitions wrapping the data layer
├── data/
│   ├── stock_data.py       # Price history, returns, and quant metric calculations
│   └── database.py         # SQLite database setup
├── charts/
│   └── plotting.py         # Plotly chart functions
├── memory/                 # Conversation memory utilities
├── pipeline/
│   └── ingest.py           # Watchlist data ingestion pipeline
├── notebooks/              # EDA and exploration notebooks
└── requirements.txt
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| AI agent | Anthropic API (Claude Haiku) |
| Market data | yfinance |
| Charts | Plotly |
| Local cache | SQLite |
| Language | Python 3.12 |
| Deployment | Streamlit Community Cloud |

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

You can get an API key at [console.anthropic.com](https://console.anthropic.com).

**4. Run the app**
```bash
streamlit run app.py
```

---

## Things worth knowing

- The app uses `yfinance` to fetch market data. Occasionally Yahoo Finance changes their API and data fetches may fail temporarily. If you see ticker errors, check that you are on `yfinance>=0.2.51`.
- The SQLite cache lives locally and is not committed to the repo. It gets rebuilt on first run.
- Streamlit Cloud deployments require secrets to be configured in the dashboard under **Settings > Secrets**. The local `.env` file does not transfer automatically.

---

## About

Built by Brett Saulnier, data science and mathematics student at the University of Wisconsin-Madison. This project was built as a portfolio piece covering data engineering, AI agent design, and frontend deployment.

[GitHub](https://github.com/bsaulnier797) | [LinkedIn](#)

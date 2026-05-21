# Finance & Quant Chatbot 📈

Ask plain English questions about stocks and get back real analysis with interactive charts. No spreadsheets, no digging through financial sites -- just type what you want to know.

[**Try it live**](https://bsaulnier797-quant-chatbot-app-magfyv.streamlit.app/) 

---

## The idea

I wanted a tool where you could type something like "how has NVDA performed over the last 6 months?" and actually get a useful answer -- not just a number, but context, charts, and the math behind it. So I built one.

It pulls real market data, runs quantitative calculations, and responds through an AI agent that understands what you're actually asking. Whether you're checking Sharpe ratios or just curious how two stocks have moved relative to each other, it handles it conversationally.

---

## What you can do with it

- Ask questions in plain English -- no special syntax needed
- Get candlestick charts, moving averages, returns distributions, and correlation heatmaps auto-generated based on what you asked
- See quant metrics like Sharpe ratio, max drawdown, annualized volatility, and total return
- Compare multiple tickers side by side with normalized performance charts
- Switch between **Expert mode** (direct analysis) and **Learning mode** (plain English explanations)
- Use the sidebar to look up finance concepts like beta, correlation, or max drawdown without losing your place in the conversation
- Get AI-suggested follow-up questions after each response so you can keep exploring

---

## How it works

When you send a message, an AI agent (Claude Haiku via the Anthropic API) reads your question and decides which data tools to call. Those tools hit yfinance for real market data, run the relevant calculations, and send the results back to the agent. The agent then writes a response and the frontend figures out whether to render a chart alongside it.

There's also a data pipeline that runs on startup. It ingests price data for a default watchlist into a local SQLite database so repeated queries don't have to re-fetch the same data from scratch.

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
├── app.py              # Streamlit UI and chat logic
├── agent/
│   ├── agent.py        # Anthropic API agent with tool-calling loop
│   └── tools.py        # Tool definitions wrapping the data layer
├── data/
│   ├── stock_data.py   # Price history, returns, and quant metrics
│   └── database.py     # SQLite setup
├── charts/
│   └── plotting.py     # Plotly chart functions
├── memory/             # Conversation memory
├── pipeline/
│   └── ingest.py       # Watchlist ingestion pipeline
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
| Language | Python |
| Deployment | Streamlit Community Cloud |

---

## Running it locally

**1. Clone and enter the repo**
```bash
git clone https://github.com/bsaulnier797/quant-chatbot.git
cd quant-chatbot
```

**2. Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the root:
```
ANTHROPIC_API_KEY=your_key_here
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

**4. Start the app**
```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Connect your repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `ANTHROPIC_API_KEY` under **Secrets** in the app settings
4. Deploy

One gotcha: your local `.env` file doesn't carry over to Streamlit Cloud. You have to add the key manually through their secrets manager or the app won't be able to call the API.

---

## Things worth trying

- "How has NVDA performed over the last 6 months?"
- "What is the Sharpe ratio of MSFT over 1 year?"
- "Compare AAPL and GOOGL over 3 months"
- "Show me the returns distribution for TSLA"
- "Compare TSLA and SPY and show me a correlation heatmap"

---

## A couple things I ran into

**LangChain wasn't worth the trouble.** I started with LangChain for the agent layer but hit version incompatibilities with the Anthropic integration (v1.3.0) that weren't easy to work around. Scrapping it and calling the Anthropic API directly made everything more predictable and a lot easier to debug.

**Chart rendering is trickier than it looks.** Deciding when to show a chart, and which type, based on a free-text question took more iteration than expected. Regex-based ticker extraction needs a solid stopword list or you end up treating words like "GDP" and "CEO" as stock tickers.

---

## Author

Brett Saulnier | [GitHub](https://github.com/bsaulnier797)

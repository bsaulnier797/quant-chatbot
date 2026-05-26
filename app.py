import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
from agent.agent import build_agent, ask
from data.stock_data import get_price_history
from charts.plotting import (
    plot_candlestick_chart,
    plot_with_moving_averages,
    plot_returns_distribution,
    plot_correlation_heatmap,
)
from ml.model import (
    build_features,
    walk_forward_validate,
    train_model,
    evaluate_model,
    get_feature_importance,
    predict_next_day,
)
from ml.backtest import simulate_strategy, compute_strategy_metrics, build_equity_curve


# =============================================================================
# CONSTANTS
# =============================================================================

STOPWORDS = {
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
    "FULL", "COMPLETE", "ANALYSIS", "INCLUDING",
}

CONCEPT_EXPLANATIONS = {
    "Volatility": "**Volatility** is a measure of how much a stock's price fluctuates over time. "
                  "A stock with high volatility has large price swings, while a stock with low volatility "
                  "has more stable prices. Volatility is often measured using the standard deviation of "
                  "returns or the annualized volatility formula.",
    "Sharpe Ratio": "The **Sharpe Ratio** is a measure of risk-adjusted return. It tells you how much "
                    "excess return you are getting for each unit of risk you take on. A higher Sharpe "
                    "ratio indicates better risk-adjusted performance. Formula: "
                    "Sharpe = (Portfolio Return - Risk-Free Rate) / Standard Deviation of Excess Return.",
    "Max Drawdown": "**Max Drawdown** is the largest percentage drop from a peak to a trough in the value "
                    "of an investment. It measures the worst-case loss an investor could have experienced "
                    "during a specific period.",
    "Total Return": "**Total Return** is the overall return on an investment, including both capital gains "
                    "and dividends. Total Return = (Ending Value - Beginning Value + Dividends) / Beginning Value.",
    "Correlation": "**Correlation** measures the strength and direction of the linear relationship between "
                   "two variables. Ranges from -1 (perfect negative) to 1 (perfect positive), with 0 meaning "
                   "no linear relationship.",
    "Beta": "**Beta** is a measure of a stock's volatility relative to the overall market. A beta of 1 "
            "moves in line with the market, > 1 means more volatile, < 1 means less volatile.",
    "Alpha": "**Alpha** represents the excess return of an investment relative to a benchmark index. "
             "It measures the skill of generating returns above the market average.",
}


# =============================================================================
# HELPERS
# =============================================================================

def get_api_key() -> str:
    """Retrieve the Anthropic API key from Streamlit secrets or environment."""
    try:
        return st.secrets["anthropic"]["api_key"]
    except Exception:
        import os
        return os.getenv("ANTHROPIC_API_KEY", "")


def extract_tickers(text: str, max_tickers: int = 3) -> list[str]:
    """Pull validated tickers out of a free-text question."""
    raw = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', text.upper())
    candidates = [t[0] or t[1] for t in raw]
    candidates = [t for t in candidates if t not in STOPWORDS and len(t) >= 2]

    validated = []
    for ticker in candidates[:max_tickers * 2]:
        if len(validated) >= max_tickers:
            break
        try:
            df = get_price_history(ticker, "5d")
            if not df.empty:
                validated.append(ticker)
        except Exception:
            pass
    return validated


def detect_period(text: str) -> str:
    """Infer chart period from a free-text question."""
    text = text.lower()
    if "1 year" in text or "1y" in text:
        return "1y"
    if "3 month" in text:
        return "3mo"
    if "ytd" in text:
        return "ytd"
    return "6mo"


def generate_followups(question: str, response: str, mode: str) -> list:
    import anthropic
    import json
    try:
        client = anthropic.Anthropic(api_key=get_api_key())
        prompt = f"""The user asked: "{question}"
The assistant responded: "{response[:600]}"

Generate exactly 3 short follow-up questions the user might want to ask next.
They should be specific, natural, and directly related to what was just discussed.
Return only a JSON array of 3 strings, nothing else."""

        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="You generate follow-up questions. Return only a JSON array of 3 strings.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = result.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        questions = json.loads(text)
        return questions[:3] if isinstance(questions, list) else []
    except Exception as e:
        print(f"Follow-up error: {e}")
        return []


def try_render_chart(question: str):
    """Decide which chart (if any) to render for a given question."""
    if any(k in question.lower() for k in ["what is the price", "current price", "how much is", "how much does"]):
        return None

    tickers = extract_tickers(question)
    if not tickers:
        return None

    period = detect_period(question)
    q = question.lower()

    try:
        if len(tickers) > 1 and any(w in q for w in ["correlat", "heatmap", "portfolio"]):
            return plot_correlation_heatmap(None, tickers)

        if any(w in q for w in ["distribution", "histogram"]):
            return plot_returns_distribution(get_price_history(tickers[0], period), tickers[0])

        if any(w in q for w in ["moving average", "ma50", "ma20", "trend"]):
            return plot_with_moving_averages(get_price_history(tickers[0], period), tickers[0])

        if len(tickers) == 1:
            return plot_candlestick_chart(get_price_history(tickers[0], period), tickers[0])

        # Multiple tickers — normalized comparison line chart
        fig = go.Figure()
        for ticker in tickers:
            df = get_price_history(ticker, period)
            if df.empty:
                continue
            normalized = (df["Close"] / df["Close"].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=df.index, y=normalized, mode="lines", name=ticker))
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date", yaxis_title="Normalized Price", hovermode="x unified"
        )
        return fig

    except Exception:
        return None


def render_full_analysis(question: str):
    """Render the full multi-chart breakdown when user asks for 'full analysis'."""
    tickers = extract_tickers(question, max_tickers=2)
    if not tickers:
        return

    period = "1y" if "year" in question.lower() else "6mo"
    df = get_price_history(tickers[0], period)

    st.subheader("📊 Price Chart")
    st.plotly_chart(plot_candlestick_chart(df, tickers[0]), use_container_width=True)

    st.subheader("📈 Moving Averages")
    st.plotly_chart(plot_with_moving_averages(df, tickers[0]), use_container_width=True)

    st.subheader("📉 Returns Distribution")
    st.plotly_chart(plot_returns_distribution(df, tickers[0]), use_container_width=True)

    if len(tickers) > 1:
        st.subheader("🔥 Correlation Heatmap")
        st.plotly_chart(plot_correlation_heatmap(None, tickers), use_container_width=True)


def handle_chat_message(prompt: str, mode: str):
    """Render a user message, get the agent response, and render any charts."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(
                st.session_state.agent, prompt, mode.lower(),
                history=st.session_state.messages
            )
        st.markdown(response)

        if "full analysis" in prompt.lower() or "complete analysis" in prompt.lower():
            render_full_analysis(prompt)
            chart = None
        else:
            chart = try_render_chart(prompt)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": response, "chart": chart})
    st.session_state.followups = generate_followups(prompt, response, mode.lower())


# =============================================================================
# ML FORECAST TAB
# =============================================================================

@st.cache_data(show_spinner=False)
def run_ml_pipeline(ticker: str, period: str = "2y"):
    """Full ML pipeline: features -> walk-forward CV -> train -> evaluate -> backtest."""
    X, y, prices = build_features(ticker, period=period)
    if X.empty:
        return {"error": f"Could not build features for {ticker}. Check that the ticker is valid and has sufficient price history."}

    cv_results = walk_forward_validate(X, y)
    if not cv_results:
        return {"error": f"Not enough data to run walk-forward validation for {ticker} "
                         f"({len(X)} rows after feature engineering). Try a ticker with more history."}

    # Split into train/test before fitting the final model
    split = len(X) // 2
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_prices = prices.iloc[split:]

    # Train on first half, evaluate on second half (true out-of-sample)
    model = train_model(X_train, y_train)
    held_out_metrics = evaluate_model(model, X_test, y_test)

    importance = get_feature_importance(model, X.columns.tolist())

    predictions = model.predict(X_test.ffill().bfill())
    strategy_returns = simulate_strategy(predictions, test_prices)
    benchmark_returns = test_prices["Close"].pct_change().dropna()

    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    backtest_metrics = compute_strategy_metrics(strategy_returns, benchmark_returns)
    equity = build_equity_curve(strategy_returns, benchmark_returns)
    next_pred = predict_next_day(model, X)

    avg_cv_metrics = {
        "accuracy": sum(r["accuracy"] for r in cv_results) / len(cv_results),
        "precision": sum(r["precision"] for r in cv_results) / len(cv_results),
        "recall": sum(r["recall"] for r in cv_results) / len(cv_results),
        "f1": sum(r["f1_score"] for r in cv_results) / len(cv_results),
    }

    return {
        "cv_metrics": avg_cv_metrics,
        "held_out_metrics": held_out_metrics,
        "importance": importance,
        "metrics": backtest_metrics,
        "equity": equity,
        "next_pred": next_pred,
    }


def render_ml_tab():
    st.header("ML Forecast")
    st.caption("XGBoost classifier predicting next-day direction, validated with walk-forward CV.")

    col_input, col_button = st.columns([3, 1])
    with col_input:
        ticker = st.text_input("Ticker", value="SPY", key="ml_ticker").upper()
    with col_button:
        st.write("")
        st.write("")
        train = st.button("Train Model", use_container_width=True)

    if not train and "ml_results" not in st.session_state:
        st.info("Enter a ticker and click 'Train Model' to run the ML pipeline.")
        return

    if train:
        # Clear previous results so stale data never shows for a new ticker
        st.session_state.pop("ml_results", None)
        with st.spinner(f"Building features, validating, and backtesting for {ticker}..."):
            st.session_state.ml_results = run_ml_pipeline(ticker)
            st.session_state.ml_ticker_cached = ticker

    results = st.session_state.get("ml_results")
    if results is None:
        st.error(f"Pipeline returned no results for {ticker}. Try rebooting the app.")
        return
    if "error" in results:
        st.error(results["error"])
        return

    cached_ticker = st.session_state.get("ml_ticker_cached", ticker)

    # Next-day prediction card
    pred = results["next_pred"]
    signal_color = "🟢" if pred["signal"] == "Buy" else "🟡"
    st.subheader(f"{signal_color} Next-Day Signal: **{pred['signal']}**")
    st.caption(f"Confidence: {pred['confidence']}% | As of {pred['date']} | Ticker: {cached_ticker}")
    st.caption("⚠️ This is a research project, not financial advice.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Walk-Forward CV (avg)**")
        cv = results["cv_metrics"]
        st.metric("Accuracy", f"{cv['accuracy']*100:.1f}%")
        st.metric("Precision", f"{cv['precision']*100:.1f}%")
        st.metric("Recall", f"{cv['recall']*100:.1f}%")
        st.metric("F1 Score", f"{cv['f1']*100:.1f}%")

        # Held-out test set evaluation
        st.markdown("**Held-Out Test Set**")
        ho = results["held_out_metrics"]
        st.metric("Test Accuracy", f"{ho['accuracy']*100:.1f}%")
        st.metric("Test F1", f"{ho['f1_score']*100:.1f}%")

    with col2:
        st.markdown("**Strategy vs Benchmark**")
        m = results["metrics"]
        st.metric("Strategy Total Return", f"{m['strategy_total_return']*100:.1f}%",
                  delta=f"{(m['strategy_total_return'] - m['benchmark_total_return'])*100:.1f}% vs B&H")
        st.metric("Strategy Sharpe", f"{m['strategy_sharpe']:.2f}",
                  delta=f"{m['strategy_sharpe'] - m['benchmark_sharpe']:.2f} vs B&H")
        st.metric("Max Drawdown", f"{m['strategy_max_drawdown']*100:.1f}%")
        st.metric("Win Rate", f"{m['strategy_win_rate']*100:.1f}%")

    with col3:
        st.markdown("**Feature Importance**")
        imp = results["importance"]
        fig = px.bar(imp, x="importance", y="feature", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("**Equity Curve: Strategy vs Buy-and-Hold**")
    equity = results["equity"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["strategy"], mode="lines", name="ML Strategy"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity["benchmark"], mode="lines", name="Buy & Hold"))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Growth of $1",
        hovermode="x unified", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# CHAT TAB
# =============================================================================

def render_chat_tab(mode: str):
    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)

    # Handle pending prompt from follow-up button clicks
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        st.session_state.followups = []
        handle_chat_message(prompt, mode)
        st.rerun()

    # Render follow-up suggestion buttons
    if st.session_state.followups:
        st.markdown("**Suggested follow-ups:**")
        cols = st.columns(3)
        for i, question in enumerate(st.session_state.followups):
            with cols[i]:
                if st.button(question, key=f"followup_{i}", use_container_width=True):
                    st.session_state.pending_prompt = question
                    st.session_state.followups = []
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question about stocks or markets..."):
        handle_chat_message(prompt, mode)
        st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

st.set_page_config(page_title="Quant Chatbot", page_icon="📈", layout="wide")

st.title("📈 Finance & Quant Chatbot")
st.caption("Ask natural language questions about stocks, returns, volatility, and more. "
           "Or head to the ML Forecast tab to see a model predict next-day direction.")

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

    st.header("Quick Learning")
    selected_concept = st.selectbox(
        "Select a concept to learn about:",
        ["None"] + list(CONCEPT_EXPLANATIONS.keys()),
        index=0
    )
    if selected_concept != "None":
        with st.expander(f"About {selected_concept}", expanded=True):
            st.markdown(CONCEPT_EXPLANATIONS[selected_concept])

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.followups = []
        st.rerun()

    st.divider()
    st.caption("Data sourced from Yahoo Finance via yfinance.")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = build_agent()
if "followups" not in st.session_state:
    st.session_state.followups = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Tabs
chat_tab, ml_tab = st.tabs(["💬 Chat", "🤖 ML Forecast"])

with chat_tab:
    render_chat_tab(mode)

with ml_tab:
    render_ml_tab()
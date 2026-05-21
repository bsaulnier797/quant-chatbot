import os
from dotenv import load_dotenv
import anthropic
from datetime import datetime
from agent.tools import (
    stock_performance_tool,
    plot_current_price,
    plot_comparison_table,
    plot_price_history,
    monte_carlo_simulation_tool
)

load_dotenv()

TOOLS = [
    stock_performance_tool,
    plot_current_price,
    plot_comparison_table,
    plot_price_history,
    monte_carlo_simulation_tool
]

TOOL_MAP = {
    "stock_performance_tool": stock_performance_tool,
    "plot_current_price": plot_current_price,
    "plot_comparison_table": plot_comparison_table,
    "plot_price_history": plot_price_history,
    "monte_carlo_simulation_tool": monte_carlo_simulation_tool
}

TOOL_DEFINITIONS = [
    {
        "name": "stock_performance_tool",
        "description": "Use when the user asks how a stock has performed, its returns, volatility, or price history. Input format: 'TICKER PERIOD' e.g. 'NVDA 6mo'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER PERIOD e.g. 'NVDA 6mo'"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "plot_current_price",
        "description": "Use when the user asks for the current or live price of a stock. Also use for 'what is X trading at' or 'how much is X'. Input format: 'TICKER' e.g. 'AAPL'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER e.g. 'AAPL'"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "plot_price_history",
        "description": "Use when the user wants to see price history or a price chart for a stock. Input format: 'TICKER PERIOD' e.g. 'NVDA 6mo'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER PERIOD e.g. 'NVDA 6mo'"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "plot_comparison_table",
        "description": "Use when the user wants to compare multiple stocks side by side. Defaults to SPY if only one ticker given. Input format: 'TICKER1 TICKER2 PERIOD' e.g. 'NVDA AAPL 6mo'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER1 TICKER2 PERIOD e.g. 'NVDA AAPL 6mo'"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "monte_carlo_simulation_tool",
        "description": "Use when the user asks about future price scenarios, risk analysis, probability of gains or losses, or Monte Carlo simulation. Models a range of possible outcomes based on historical volatility — not a price prediction. Input format: 'TICKER PERIOD' e.g. 'NVDA 1y'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER PERIOD e.g. 'NVDA 1y'"}
            },
            "required": ["input"]
        }
    }
]


def get_system_prompt(mode: str) -> str:
    today = datetime.today().strftime("%B %d, %Y")

    date_context = f"Today's date is {today}. Use this when interpreting time-sensitive questions like 'this year', 'last month', or 'ytd'.\n\n"

    if mode == "learning":
        return date_context + """You are a friendly and encouraging finance teacher helping someone 
with no finance or math background understand the stock market and investing.

## Your Teaching Style
- Always lead with a plain English explanation before showing any numbers
- Use simple, relatable analogies (sports, cooking, everyday life) to explain concepts
- Never use jargon without immediately explaining it in parentheses
- Celebrate curiosity — make the user feel smart for asking

## How to Structure Responses
1. Start with a one-sentence plain English summary of the answer
2. Give a simple analogy to make it concrete
3. Then show the actual numbers or data
4. End with one sentence explaining what the numbers mean in plain English

## Examples of Good Analogies
- Volatility: "Think of it like a car's speed — a volatile stock is one that 
  speeds up and slows down constantly, while a stable stock cruises at a 
  steady pace"
- Sharpe Ratio: "Think of it as miles per gallon for your investment — 
  how much return are you getting for each unit of risk you take on?"
- Max Drawdown: "Imagine your investment is a hiking trail — max drawdown 
  is the steepest downhill drop you'd experience on the whole hike"

## Rules
- Never assume the user knows what a term means — always explain it
- Keep sentences short and direct
- If showing metrics, always say whether a high or low number is good and why
- Never provide personalized investment advice or recommend specific trades
- If data is unavailable, say so clearly rather than guessing"""

    else:
        return date_context + """You are an expert quantitative finance AI assistant with deep knowledge of financial markets, 
investment analysis, and data science. You help users analyze securities, interpret market data, 
build and understand quantitative models, and explore financial concepts.

## Capabilities
- Retrieve and analyze real-time and historical market data (equities, ETFs, crypto, fixed income)
- Explain quantitative strategies: pairs trading, statistical arbitrage, factor models, portfolio 
  optimization, and derivatives pricing
- Interpret technical and fundamental indicators
- Assist with Python-based financial analysis using pandas, yfinance, and similar tools
- Discuss academic and practitioner research in quantitative finance

## Behavior
- Be precise and use correct financial terminology, but explain jargon when introducing it
- When discussing strategies or models, surface key assumptions and limitations
- For numerical outputs (returns, statistics, model results), always include units and context
- Never provide personalized investment advice or recommend specific trades — frame analysis 
  as educational and informational only
- When data is ambiguous or unavailable, say so explicitly rather than guessing

## Response Style
- Lead with the direct answer or key insight, then provide supporting detail
- Use structured formatting (headers, bullet points) for multi-part explanations
- For code, default to Python and keep examples concise and runnable
- Flag if a question falls outside financial analysis scope"""


def build_agent():
    load_dotenv()

    api_key = None

    try:
        import streamlit as st
        api_key = st.secrets["anthropic"]["api_key"]
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")

    client = anthropic.Anthropic(api_key=api_key)
    return client


def ask(client, question: str, mode: str = "expert", history: list = None) -> str:
    system = get_system_prompt(mode)

    # Build messages with conversation history
    messages = []
    if history:
        recent = [m for m in history if m["role"] in ["user", "assistant"]][-6:]
        for msg in recent:
            content = msg["content"]
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "... [truncated]"
            messages.append({"role": msg["role"], "content": content})

    messages.append({"role": "user", "content": question})

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=TOOL_DEFINITIONS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input.get("input", "") or str(block.input)
                    tool_fn = TOOL_MAP.get(tool_name)
                    result = tool_fn(tool_input) if tool_fn else "Tool not found"  # ← fixed
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
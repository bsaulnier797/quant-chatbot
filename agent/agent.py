import os
from dotenv import load_dotenv
import anthropic
from agent.tools import stock_performance_tool, stock_comparison_tool, current_price_tool

load_dotenv()

TOOLS = [stock_performance_tool, stock_comparison_tool, current_price_tool]

TOOL_MAP = {
    "stock_performance_tool": stock_performance_tool,
    "stock_comparison_tool": stock_comparison_tool,
    "current_price_tool": current_price_tool,
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
        "name": "stock_comparison_tool",
        "description": "Use when the user wants to compare two stocks. Input format: 'TICKER1 TICKER2 PERIOD' e.g. 'NVDA SPY 6mo'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER1 TICKER2 PERIOD e.g. 'NVDA SPY 6mo'"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "current_price_tool",
        "description": "Use when the user asks for the current price of a stock. Input format: 'TICKER' e.g. 'AAPL'",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "TICKER e.g. 'AAPL'"}
            },
            "required": ["input"]
        }
    }
]


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

# Interaction ask function with memory and tool use handling

def ask(client, question: str, mode: str = "expert", history: list = None) -> str:
    if mode == "learning":
        system = """You are a friendly finance teacher explaining concepts to 
        someone with no finance background. When you show metrics like Sharpe 
        ratio or volatility, always explain what they mean in plain English 
        with a simple analogy before showing the numbers. Use encouraging 
        language and avoid jargon."""
    else:
        system = """You are a quantitative finance analyst. Be concise and 
        precise. Use correct financial terminology."""

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
                    result = tool_fn.invoke(tool_input) if tool_fn else "Tool not found"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

# def ask(client, question: str, mode: str = "expert") -> str:
#     if mode == "learning":
#         system = """You are a friendly finance teacher explaining concepts to 
#         someone with no finance background. When you show metrics like Sharpe 
#         ratio or volatility, always explain what they mean in plain English 
#         with a simple analogy before showing the numbers. Use encouraging 
#         language and avoid jargon."""
#     else:
#         system = """You are a quantitative finance analyst. Be concise and 
#         precise. Use correct financial terminology."""

#     messages = [{"role": "user", "content": question}]

#     while True:
#         response = client.messages.create(
#             model="claude-haiku-4-5-20251001",
#             max_tokens=1024,
#             system=system,
#             tools=TOOL_DEFINITIONS,
#             messages=messages
#         )

#         if response.stop_reason == "end_turn":
#             for block in response.content:
#                 if hasattr(block, "text"):
#                     return block.text

#         if response.stop_reason == "tool_use":
#             messages.append({"role": "assistant", "content": response.content})
#             tool_results = []
#             for block in response.content:
#                 if block.type == "tool_use":
#                     tool_name = block.name
#                     tool_input = block.input.get("input", "")
#                     tool_fn = TOOL_MAP.get(tool_name)
#                     result = tool_fn.invoke(tool_input) if tool_fn else "Tool not found"
#                     tool_results.append({
#                         "type": "tool_result",
#                         "tool_use_id": block.id,
#                         "content": result
#                     })
#             messages.append({"role": "user", "content": tool_results})
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
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client


def ask(client, question: str) -> str:
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOL_DEFINITIONS,
            messages=messages
        )

        # If no tool use, return the text response
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        # Handle tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input.get("input", "")
                    print(f"Calling tool: {tool_name} with input: {tool_input}")

                    tool_fn = TOOL_MAP.get(tool_name)
                    result = tool_fn.invoke(tool_input) if tool_fn else "Tool not found"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})
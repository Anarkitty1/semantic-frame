# Semantic Frame

**The "JSON.stringify()" for the Agentic Age.**

Semantic Frame is a Python library that converts raw numerical data (NumPy, Pandas, Polars) into **token-efficient natural language descriptions**.

It acts as a "Semantic Bridge" between your data and your AI Agents.

## The Problem: "RAG for Numbers"

LLMs are terrible at arithmetic. When you send raw data like `[100, 102, 99, 101, 500]` to GPT-4:

*   **Token Waste**: 10,000 data points = ~20,000 tokens.
*   **Hallucination Risk**: LLMs guess trends instead of calculating them.
*   **Context Overflow**: Large datasets fill the context window.

## The Solution

**Semantic Frame** performs deterministic analysis using NumPy/SciPy, then translates the results into a concise narrative.

```python
from semantic_frame import describe_series
import pandas as pd

data = pd.Series([100, 102, 99, 101, 500, 100, 98])
print(describe_series(data, context="Server Latency"))
```

**Output:**
> "The Server Latency data shows a flat/stationary pattern with stable variability. 1 anomaly detected at index 4 (value: 500.00). Baseline: 100.00."

## Key Features

*   **Universal Dictionary**: Standardized terms for Trend, Volatility, and Seasonality.
*   **Deterministic Math**: No LLM hallucinations. Math is done by NumPy.
*   **Framework Agnostic**: Works with Pandas, Polars, NumPy, and Python lists.
*   **Agent Ready**: Integrates with LangChain, CrewAI, and ElizaOS.
*   **Trading Module**: Specialized tools for trading agents and portfolio analysis.

## Installation

```bash
pip install semantic-frame
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Quick introduction and basic usage |
| [Trading Module](trading-module.md) | Trading-specific analysis tools |
| [MCP Integration](integrations.md) | Model Context Protocol setup |
| [Claude Integration](claude-integration.md) | Using with Anthropic's Claude |
| [API Reference](api.md) | Full API documentation |

[Get Started](getting-started.md){ .md-button .md-button--primary }

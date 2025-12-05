"""Semantic Frame: Token-efficient semantic compression for numerical data.

Semantic Frame converts raw numerical data (NumPy, Pandas, Polars) into
natural language descriptions optimized for LLM consumption. Instead of
sending thousands of data points to an AI agent, send a 50-word semantic summary.

Features:
- 95%+ token reduction for large datasets
- Deterministic math via NumPy/scipy (no hallucination risk)
- Supports Anthropic Advanced Tool Use (beta)
- Framework integrations: Anthropic, LangChain, CrewAI, MCP

Quick Start:
    >>> from semantic_frame import describe_series
    >>> import pandas as pd
    >>>
    >>> data = pd.Series([100, 102, 99, 101, 500, 100, 98])
    >>> print(describe_series(data, context="Server Latency (ms)"))
    "The Server Latency (ms) data shows a flat/stationary pattern..."

For Advanced Tool Use:
    >>> from semantic_frame.integrations.anthropic import get_advanced_tool
    >>> tool = get_advanced_tool()  # All beta features enabled
"""

from semantic_frame.main import compression_stats, describe_dataframe, describe_series

__version__ = "0.2.0"
__all__ = ["describe_series", "describe_dataframe", "compression_stats", "__version__"]

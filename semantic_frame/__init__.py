"""Semantic Frame: Token-efficient semantic compression for numerical data.

Semantic Frame converts raw numerical data (NumPy, Pandas, Polars) into
natural language descriptions optimized for LLM consumption. Instead of
sending thousands of data points to an AI agent, send a 50-word semantic summary.

Features:
- 95%+ token reduction for large datasets
- Deterministic math via NumPy/scipy (no hallucination risk)
- Supports Anthropic Advanced Tool Use (beta)
- Trading module for equity curve and performance analysis
- Framework integrations: Anthropic, LangChain, CrewAI, MCP

Quick Start:
    >>> from semantic_frame import describe_series
    >>> import pandas as pd
    >>>
    >>> data = pd.Series([100, 102, 99, 101, 500, 100, 98])
    >>> print(describe_series(data, context="Server Latency (ms)"))
    "The Server Latency (ms) data shows a flat/stationary pattern..."

Trading Module:
    >>> from semantic_frame.trading import describe_drawdown, describe_trading_performance
    >>> equity = [10000, 10500, 10200, 9800, 9500, 10000, 10800]
    >>> print(describe_drawdown(equity, context="BTC strategy").narrative)

For Advanced Tool Use:
    >>> from semantic_frame.integrations.anthropic import get_advanced_tool
    >>> tool = get_advanced_tool()  # All beta features enabled
"""

from semantic_frame.main import describe_dataframe, describe_series

# Re-export trading module functions for convenience
from semantic_frame.trading import (
    describe_allocation,
    describe_anomalies,
    describe_drawdown,
    describe_rankings,
    describe_regime,
    describe_trading_performance,
    describe_windows,
)

__version__ = "0.4.0"
__all__ = [
    # Core functions
    "describe_series",
    "describe_dataframe",
    # Trading functions (re-exported for convenience)
    "describe_data",
    "describe_anomalies",
    "describe_windows",
    "describe_regime",
    "describe_drawdown",
    "describe_trading_performance",
    "describe_rankings",
    "describe_allocation",
    # Version
    "__version__",
]


def describe_data(data: object, context: str = "Data") -> str:
    """Analyze numerical data and return a semantic description.

    This is a convenience wrapper around describe_series that accepts
    native Python lists, NumPy arrays, or Pandas Series.

    Args:
        data: List of numbers, NumPy array, or Pandas Series
        context: Label for the data (e.g., "Server Latency", "BTC Price")

    Returns:
        Natural language description of the data's patterns

    Example:
        >>> prices = [100, 102, 99, 500, 101, 98]
        >>> describe_data(prices, context="Price")
        "The Price data shows a flat/stationary pattern..."
    """
    return describe_series(data, context=context, output="text")

"""MCP Server integration for semantic analysis.

This module provides a Model Context Protocol (MCP) server that exposes
semantic analysis capabilities to MCP clients (like ElizaOS, Claude Desktop,
or Claude Code).

Features:
- describe_data: Analyze numerical data and return semantic descriptions
- describe_batch: Batch analysis for multiple data series (token efficient)
- wrap_for_semantic_output: Decorator to add semantic compression to any tool

Requires: pip install semantic-frame[mcp]

Usage:
    Run as a standalone server:
    $ mcp run semantic_frame.integrations.mcp:mcp

    Or import in your own MCP server:
    from semantic_frame.integrations.mcp import mcp

Claude Code Setup:
    $ claude mcp add semantic-frame -- uv run --project /path/to/semantic-frame \\
        mcp run /path/to/semantic-frame/semantic_frame/integrations/mcp.py
    $ claude mcp list
    # semantic-frame: ... - ✓ Connected
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "mcp is required for this module. Install with: pip install semantic-frame[mcp]"
    )

# Create the MCP server instance
mcp = FastMCP("semantic-frame")

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def _parse_data_input(data_str: str) -> list[float]:
    """Parse string input to list of floats.

    Supports:
    - JSON array: "[1, 2, 3, 4, 5]"
    - CSV: "1, 2, 3, 4, 5"
    - Newline-separated: "1\\n2\\n3"
    """
    data_str = data_str.strip()

    # Try JSON array first
    if data_str.startswith("["):
        try:
            return [float(x) for x in json.loads(data_str)]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try CSV
    if "," in data_str:
        try:
            return [float(x.strip()) for x in data_str.split(",")]
        except ValueError:
            pass

    # Try newline-separated
    if "\n" in data_str:
        try:
            return [float(x.strip()) for x in data_str.split("\n") if x.strip()]
        except ValueError:
            pass

    raise ValueError(
        f"Could not parse data input. Expected JSON array, CSV, or newline-separated numbers. "
        f"Got: {data_str[:100]}..."
    )


@mcp.tool()  # type: ignore[misc]
def describe_data(data: str, context: str = "Data") -> str:
    """Analyze numerical data and return a semantic description.

    Use this tool when you have a list of numbers (prices, metrics, sensor readings)
    and need to understand the trends, anomalies, and patterns without doing math yourself.

    This tool provides 95%+ token compression - sending 10,000 data points returns
    a ~50 word semantic summary instead of consuming context with raw numbers.

    Args:
        data: A string containing the numbers. Can be a JSON array "[1, 2, 3]",
              CSV "1, 2, 3", or newline-separated values.
        context: A label for the data (e.g., "Server CPU Load", "Daily Sales").
                 This helps the tool generate a more relevant description.

    Returns:
        A natural language paragraph describing the data's behavior including:
        - Trend direction (rising, falling, flat)
        - Volatility level (stable, moderate, extreme)
        - Anomalies detected with positions
        - Baseline statistics

    Examples:
        Input: data="[100, 102, 99, 500, 101]", context="Latency (ms)"
        Output: "The Latency (ms) data shows a flat/stationary pattern with stable
                variability. 1 anomaly detected at index 3 (value: 500.00)."

        Input: data="10, 20, 30, 40, 50", context="Daily Sales"
        Output: "The Daily Sales data shows a rapidly rising pattern..."
    """
    from semantic_frame import describe_series

    try:
        values = _parse_data_input(data)
        result: str = describe_series(values, context=context, output="text")
        return result
    except (ValueError, TypeError) as e:
        return f"Error analyzing data: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_batch(
    datasets: str,
    output_format: str = "text",
) -> str:
    """Analyze multiple data series in a single call.

    Efficient for analyzing DataFrames or multiple metrics at once.
    Each dataset is analyzed independently and results are combined.

    Args:
        datasets: JSON object mapping names to data arrays.
                  Example: '{"cpu": [45, 47, 95, 44], "memory": [60, 61, 60]}'
        output_format: "text" for narratives (default), "json" for structured output.

    Returns:
        Combined analysis for all datasets.

    Example:
        Input: datasets='{"cpu": [45, 47, 95], "mem": [60, 61, 60]}'
        Output: "cpu: The cpu data shows a flat/stationary pattern...
                 mem: The mem data shows a flat/stationary pattern..."
    """
    from semantic_frame import describe_series

    # Validate output_format
    if output_format not in ("text", "json"):
        output_format = "text"

    try:
        data_dict = json.loads(datasets)

        if output_format == "json":
            # Return structured JSON with all results
            json_results: dict[str, Any] = {}
            for name, values in data_dict.items():
                if isinstance(values, str):
                    values = _parse_data_input(values)
                result = describe_series(values, context=name, output="json")
                json_results[name] = result
            return json.dumps(json_results, indent=2)
        else:
            # Return text narratives
            text_results: list[str] = []
            for name, values in data_dict.items():
                if isinstance(values, str):
                    values = _parse_data_input(values)
                text_result = describe_series(values, context=name, output="text")
                text_results.append(f"{name}: {text_result}")
            return "\n\n".join(text_results)

    except json.JSONDecodeError as e:
        return f"Error parsing datasets JSON: {str(e)}"
    except (ValueError, TypeError) as e:
        return f"Error analyzing batch data: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_json(data: str, context: str = "Data") -> str:
    """Analyze numerical data and return structured JSON output.

    Same as describe_data but returns JSON for programmatic use.

    Args:
        data: Numbers as JSON array, CSV, or newline-separated.
        context: Label for the data.

    Returns:
        JSON string with trend, volatility, anomalies, and narrative.
    """
    from semantic_frame import describe_series

    try:
        values = _parse_data_input(data)
        result = describe_series(values, context=context, output="json")
        return json.dumps(result, indent=2)
    except (ValueError, TypeError) as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Trading Analysis Tools
# =============================================================================


@mcp.tool()  # type: ignore[misc]
def describe_drawdown(equity: str, context: str = "Equity") -> str:
    """Analyze drawdowns in an equity curve.

    Use this tool when you have equity/balance data and need to understand
    drawdown risk, recovery patterns, and current drawdown status.

    Args:
        equity: Cumulative equity values as JSON array, CSV, or newline-separated.
               Example: "[10000, 10500, 10200, 9800, 9500, 10000, 10800]"
        context: Label for the strategy (e.g., "BTC strategy", "CLAUDE agent").

    Returns:
        Semantic description of drawdown characteristics including:
        - Maximum drawdown percentage and duration
        - Current drawdown status (at high, recovering, in drawdown)
        - Severity classification (minimal/moderate/significant/severe/catastrophic)
        - Number of drawdown periods and recovery stats

    Example:
        Input: equity="[10000, 10500, 10200, 9800, 9500, 10000, 10800]", context="BTC strategy"
        Output: "The BTC strategy has moderate drawdown risk (max 9.5% over 3 periods).
                 Currently at equity high."
    """
    from semantic_frame.trading import describe_drawdown as _describe_drawdown

    try:
        values = _parse_data_input(equity)
        result = _describe_drawdown(values, context=context)
        return result.narrative
    except (ValueError, TypeError) as e:
        return f"Error analyzing drawdown: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_trading_performance(trades: str, context: str = "Strategy") -> str:
    """Analyze trading performance from a series of trade PnLs.

    Use this tool when you have trade results and need to understand
    win rate, profit factor, risk-adjusted returns, and consistency.

    Args:
        trades: PnL per trade as JSON array, CSV, or newline-separated.
               Positive = profit, negative = loss.
               Example: "[100, -50, 75, -25, 150, -30, 80]"
        context: Label for the strategy (e.g., "CLAUDE agent", "Momentum strategy").

    Returns:
        Semantic description of trading performance including:
        - Win rate and profit factor
        - Risk-adjusted metrics (Sharpe, Sortino if calculable)
        - Performance rating (excellent/good/average/below_average/poor)
        - Risk profile classification
        - Streak analysis and consistency rating

    Example:
        Input: trades="[100, -50, 75, -25, 150, -30, 80]", context="CLAUDE"
        Output: "CLAUDE shows good performance with 57% win rate and 2.86x profit factor.
                 Risk profile: moderate."
    """
    from semantic_frame.trading import describe_trading_performance as _describe_perf

    try:
        values = _parse_data_input(trades)
        result = _describe_perf(values, context=context)
        return result.narrative
    except (ValueError, TypeError) as e:
        return f"Error analyzing trading performance: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_rankings(equity_curves: str, context: str = "agents") -> str:
    """Compare multiple agents/strategies and produce rankings.

    Use this tool when you have equity curves from multiple trading agents
    and need to compare their performance across multiple dimensions.

    Args:
        equity_curves: JSON object mapping names to equity arrays.
                      Example: '{"CLAUDE": [10000, 11000, 12000], "GROK": [10000, 12000, 11000]}'
        context: Label for what's being compared (e.g., "AI agents", "strategies").

    Returns:
        Comparative ranking analysis including:
        - Overall leader (composite score)
        - Best by return, risk-adjusted, volatility, and drawdown
        - Per-agent rankings across all dimensions

    Example:
        Input: equity_curves='{"CLAUDE": [10000, 10500, 11000], "GROK": [10000, 12000, 9000]}'
        Output: "Comparing 2 AI agents: CLAUDE leads overall with 10.0% return.
                 GROK has highest raw return (20.0%). CLAUDE is most stable..."
    """
    from semantic_frame.trading import describe_rankings as _describe_rankings

    try:
        curves_dict = json.loads(equity_curves)
        # Convert any string values to float lists
        parsed_curves = {}
        for name, values in curves_dict.items():
            if isinstance(values, str):
                parsed_curves[name] = _parse_data_input(values)
            else:
                parsed_curves[name] = [float(v) for v in values]

        result = _describe_rankings({k: list(v) for k, v in parsed_curves.items()}, context=context)
        return result.narrative
    except json.JSONDecodeError as e:
        return f"Error parsing equity curves JSON: {str(e)}"
    except (ValueError, TypeError) as e:
        return f"Error analyzing rankings: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_anomalies(
    data: str,
    context: str = "Data",
    is_pnl_data: bool = False,
) -> str:
    """Enhanced anomaly detection with severity and type classification.

    Use this tool when you need detailed analysis of outliers in data,
    including severity levels, anomaly types, and contextual descriptions.

    Args:
        data: Numerical values as JSON array, CSV, or newline-separated.
              Example: "[100, 102, 99, 500, 101, 98, -200]"
        context: Label for the data (e.g., "Trade PnL", "Server Latency").
        is_pnl_data: If True, uses gain/loss terminology instead of spike/drop.

    Returns:
        Enhanced anomaly analysis including:
        - Each anomaly with severity (mild/moderate/severe/extreme)
        - Anomaly type (spike/drop/gain/loss)
        - Contextual descriptions
        - Frequency classification (rare/occasional/frequent/pervasive)

    Example:
        Input: data="[100, 102, 99, 500, 101, -200]", context="Trade PnL", is_pnl_data=True
        Output: "The Trade PnL has occasional anomalies (2 detected in 6 points).
                 Most significant: index 3 (value: 500.00, z-score: 2.3, exceptional profit)."
    """
    from semantic_frame.trading import describe_anomalies as _describe_anomalies

    try:
        values = _parse_data_input(data)
        result = _describe_anomalies(values, context=context, is_pnl_data=is_pnl_data)
        return result.narrative
    except (ValueError, TypeError) as e:
        return f"Error analyzing anomalies: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_windows(
    data: str,
    windows: str = "10,50,200",
    context: str = "Data",
) -> str:
    """Multi-timeframe analysis across different time windows.

    Use this tool when you need to analyze data across multiple timeframes
    to compare short-term vs long-term trends and filter noise from signal.

    Args:
        data: Price/value data as JSON array, CSV, or newline-separated.
              Most recent data at the end.
        windows: Comma-separated window sizes (e.g., "10,50,200" or "1h,4h,1d").
        context: Label for the data (e.g., "BTC/USD", "CPU metrics").

    Returns:
        Multi-timeframe analysis including:
        - Per-window trend and volatility
        - Timeframe alignment (all bullish, all bearish, mixed, diverging)
        - Noise level assessment
        - Suggested positioning

    Example:
        Input: data="[100,102,105,103,108,110,112,109,115,118,120]", windows="5,10"
        Output: "Multi-timeframe analysis: all timeframes bullish.
                 Windows: 5 rising (+4.3%), 10 rising (+20.0%). Noise level: low."
    """
    from semantic_frame.trading import describe_windows as _describe_windows

    try:
        values = _parse_data_input(data)
        # Parse windows - can be comma-separated ints or strings
        window_list: list[int | str] = []
        for w in windows.split(","):
            w = w.strip()
            if w.isdigit():
                window_list.append(int(w))
            else:
                window_list.append(w)

        # Convert to uniform type for type checker
        windows_arg: list[int] | list[str] | None = None
        if all(isinstance(w, int) for w in window_list):
            windows_arg = [int(w) for w in window_list]
        else:
            windows_arg = [str(w) for w in window_list]
        result = _describe_windows(values, windows=windows_arg, context=context)
        return result.narrative
    except (ValueError, TypeError) as e:
        return f"Error analyzing windows: {str(e)}"


# =============================================================================
# MCP Wrapper Utility
# =============================================================================


def wrap_for_semantic_output(
    context_key: str | None = None,
    data_key: str = "data",
) -> Callable[[F], F]:
    """Decorator to add semantic compression to any MCP tool that returns numerical data.

    Use this to wrap existing tools that return raw numbers, automatically
    converting their output to token-efficient semantic descriptions.

    Args:
        context_key: Key in kwargs to use as context label.
                    If None, uses the function name.
        data_key: Key in the return dict containing the data array.
                 Defaults to "data". If the function returns a list directly,
                 this is ignored.

    Returns:
        Decorated function that returns semantic descriptions.

    Example:
        >>> from semantic_frame.integrations.mcp import wrap_for_semantic_output
        >>>
        >>> @mcp.tool()
        >>> @wrap_for_semantic_output(context_key="metric_name")
        >>> def get_cpu_metrics(metric_name: str = "CPU") -> list[float]:
        ...     return [45, 47, 46, 95, 44, 45]  # Raw numbers
        ...     # → Now returns: "The CPU data shows a flat/stationary pattern..."
        >>>
        >>> @mcp.tool()
        >>> @wrap_for_semantic_output()
        >>> def get_sensor_reading() -> dict:
        ...     return {"data": [22.1, 22.3, 22.0, 35.5, 22.2], "unit": "celsius"}
        ...     # → Returns semantic description of the data array
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            from semantic_frame import describe_series

            # Get the original result
            result = func(*args, **kwargs)

            # Determine context
            if context_key and context_key in kwargs:
                context = str(kwargs[context_key])
            else:
                context = func.__name__.replace("_", " ").title()

            # Extract data array
            if isinstance(result, list):
                data = result
            elif isinstance(result, dict) and data_key in result:
                data = result[data_key]
            else:
                # Can't extract data, return original
                return str(result)

            # Convert to semantic description
            try:
                narrative: str = describe_series(data, context=context, output="text")
                return narrative
            except (ValueError, TypeError, RuntimeError) as e:
                return f"Error in semantic conversion: {e}. Original: {result}"

        return wrapper  # type: ignore

    return decorator


def create_semantic_tool(
    name: str,
    data_fetcher: Callable[..., list[float]],
    description: str,
    context: str | None = None,
) -> Callable[..., str]:
    """Create a new MCP tool that fetches data and returns semantic analysis.

    Factory function for creating semantic-aware MCP tools from data sources.

    Args:
        name: Tool name for MCP registration.
        data_fetcher: Function that returns numerical data as a list.
        description: Tool description for MCP.
        context: Context label for the semantic narrative.

    Returns:
        MCP tool function that returns semantic descriptions.

    Example:
        >>> def fetch_btc_prices() -> list[float]:
        ...     return [42000, 43500, 41000, 44000]  # From exchange API
        >>>
        >>> btc_tool = create_semantic_tool(
        ...     name="analyze_btc",
        ...     data_fetcher=fetch_btc_prices,
        ...     description="Get semantic analysis of recent BTC prices",
        ...     context="BTC/USD",
        ... )
        >>> mcp.tool()(btc_tool)  # Register with MCP
    """
    from semantic_frame import describe_series

    def semantic_tool() -> str:
        try:
            data = data_fetcher()
            ctx = context or name.replace("_", " ").title()
            result: str = describe_series(data, context=ctx, output="text")
            return result
        except (ValueError, TypeError) as e:
            return f"Error: {str(e)}"

    semantic_tool.__name__ = name
    semantic_tool.__doc__ = description

    return semantic_tool


# =============================================================================
# Advanced MCP Configuration
# =============================================================================


def get_mcp_tool_config(
    defer_loading: bool = False,
) -> dict[str, Any]:
    """Get MCP server configuration for advanced tool use.

    Returns configuration dict for registering semantic-frame as an MCP
    server with optional deferred loading for large tool libraries.

    Args:
        defer_loading: If True, marks tools for discovery via Tool Search.

    Returns:
        MCP server configuration dict.

    Example:
        >>> # In your MCP server setup
        >>> config = get_mcp_tool_config(defer_loading=True)
        >>> # Register semantic-frame tools with this config
    """
    base_config: dict[str, Any] = {
        "name": "semantic-frame",
        "description": (
            "Token-efficient semantic analysis for numerical data. "
            "Compresses 10,000+ data points into ~50 word descriptions."
        ),
        "tools": ["describe_data", "describe_batch", "describe_json"],
    }

    if defer_loading:
        base_config["default_config"] = {"defer_loading": True}

    return base_config

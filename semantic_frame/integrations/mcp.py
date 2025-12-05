"""MCP Server integration for semantic analysis.

This module provides a Model Context Protocol (MCP) server that exposes
semantic analysis capabilities to MCP clients (like ElizaOS, Claude Desktop,
or Claude Code).

Features:
- describe_data: Analyze numerical data and return semantic descriptions
- describe_batch: Batch analysis for multiple data series (token efficient)
- describe_json: Structured JSON output for programmatic use
- describe_dataframe: Multi-column analysis with correlation detection
- describe_correlations: Cross-column relationship analysis
- compression_stats: Token compression metrics
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()  # type: ignore[misc]
def describe_dataframe(
    datasets: str,
    context: str = "Data",
    correlation_threshold: float = 0.5,
) -> str:
    """Analyze multiple columns with cross-column correlation detection.

    Use this when you have related data columns (like sales + inventory,
    CPU + memory, price + volume) and want to understand both individual
    trends AND relationships between columns.

    This is more powerful than describe_batch because it detects correlations
    like "Sales UP while Inventory DOWN" or "CPU and Memory move together".

    Args:
        datasets: JSON object mapping column names to data arrays.
                  Example: '{"sales": [100, 200, 300], "inventory": [500, 400, 300]}'
        context: Context label for the entire dataset (e.g., "Retail Metrics").
        correlation_threshold: Minimum |r| for reporting correlations (default 0.5).
                              Only correlations with absolute value >= threshold shown.

    Returns:
        Combined analysis including:
        - Per-column semantic descriptions
        - Detected correlations between columns
        - Summary narrative

    Example:
        Input: datasets='{"sales": [100, 200, 300, 400], "inventory": [400, 300, 200, 100]}'
        Output: "Analyzed 2 columns. sales: rapidly rising... inventory: rapidly falling...
                 Correlations: sales and inventory are strongly inverse (r=-1.00)"
    """
    import pandas as pd

    from semantic_frame import describe_dataframe as df_analyze

    try:
        data_dict = json.loads(datasets)

        # Convert to pandas DataFrame
        df = pd.DataFrame(data_dict)

        # Run full DataFrame analysis with correlations
        result = df_analyze(df, context=context, correlation_threshold=correlation_threshold)

        # Format output
        output_parts: list[str] = []

        # Summary
        output_parts.append(result.summary_narrative)
        output_parts.append("")

        # Per-column narratives
        output_parts.append("Column Details:")
        for col_name, col_result in result.columns.items():
            output_parts.append(f"  {col_name}: {col_result.narrative}")

        # Correlations
        if result.correlations:
            output_parts.append("")
            output_parts.append("Correlations:")
            for corr in result.correlations:
                output_parts.append(f"  {corr.narrative}")

        return "\n".join(output_parts)

    except json.JSONDecodeError as e:
        return f"Error parsing datasets JSON: {str(e)}"
    except Exception as e:
        return f"Error analyzing dataframe: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def describe_correlations(
    datasets: str,
    threshold: float = 0.5,
    method: str = "pearson",
) -> str:
    """Analyze correlations between multiple data columns.

    Use this when you specifically want to understand relationships between
    variables, without the full per-column analysis. Faster and more focused
    than describe_dataframe when you only need correlation insights.

    Args:
        datasets: JSON object mapping column names to data arrays.
                  Example: '{"price": [10, 20, 30], "volume": [300, 200, 100]}'
        threshold: Minimum |r| for reporting (default 0.5). Lower values
                  show weaker correlations.
        method: Correlation method - "pearson" (default, linear) or "spearman" (rank-based).
                Use spearman for non-linear monotonic relationships.

    Returns:
        List of significant correlations with strength classifications:
        - Strong positive (r > 0.7): Variables move together strongly
        - Moderate positive (0.4 < r <= 0.7): Variables tend to move together
        - Weak (-0.4 <= r <= 0.4): Little to no relationship
        - Moderate negative (-0.7 <= r < -0.4): Variables tend to move opposite
        - Strong negative (r < -0.7): Variables move opposite strongly

    Example:
        Input: datasets='{"price": [10, 20, 30], "demand": [100, 50, 25]}'
        Output: "Found 1 significant correlation:
                 price and demand are strongly inverse (r=-0.98)"
    """
    import numpy as np

    from semantic_frame.core.correlations import (
        calc_correlation_matrix,
        identify_significant_correlations,
    )
    from semantic_frame.narrators.correlation import generate_correlation_narrative

    try:
        data_dict = json.loads(datasets)

        # Convert to numpy arrays
        values_dict: dict[str, np.ndarray] = {}
        for name, values in data_dict.items():
            if isinstance(values, str):
                values = _parse_data_input(values)
            values_dict[name] = np.array(values, dtype=float)

        if len(values_dict) < 2:
            return "Need at least 2 columns to calculate correlations."

        # Calculate correlations
        corr_matrix = calc_correlation_matrix(values_dict, method=method)
        significant = identify_significant_correlations(corr_matrix, threshold=threshold)

        if not significant:
            return f"No significant correlations found (threshold: |r| >= {threshold})."

        # Format output
        output_parts: list[str] = [
            f"Found {len(significant)} significant correlation(s):",
            "",
        ]

        for col_a, col_b, r, state in significant:
            narrative = generate_correlation_narrative(col_a, col_b, r, state)
            output_parts.append(f"  {narrative}")

        return "\n".join(output_parts)

    except json.JSONDecodeError as e:
        return f"Error parsing datasets JSON: {str(e)}"
    except Exception as e:
        return f"Error calculating correlations: {str(e)}"


@mcp.tool()  # type: ignore[misc]
def compression_stats(data: str, context: str = "Data") -> str:
    """Get token compression statistics for data analysis.

    Use this to understand how much token savings you get from semantic
    compression. Helpful for optimizing LLM context usage and understanding
    the efficiency gains.

    Args:
        data: Numbers as JSON array, CSV, or newline-separated.
        context: Label for the data.

    Returns:
        Compression statistics including:
        - Original data points count
        - Estimated tokens for raw data
        - Tokens in semantic narrative
        - Compression ratio (0.95 = 95% reduction)

    Example:
        Input: data="[1, 2, 3, ..., 1000]" (1000 numbers)
        Output: "Compression Stats:
                 Original: 1000 data points (~2000 tokens)
                 Narrative: 45 tokens
                 Compression ratio: 97.8%"
    """
    from semantic_frame import compression_stats as calc_stats
    from semantic_frame import describe_series

    try:
        values = _parse_data_input(data)

        # Get full result for compression calculation
        result = describe_series(values, context=context, output="full")

        # Calculate stats
        stats = calc_stats(values, result)

        # Format output
        ratio_pct = stats["narrative_compression_ratio"] * 100
        orig_tokens = stats["original_tokens_estimate"]
        output = f"""Compression Stats for {context}:
  Original: {stats['original_data_points']} data points (~{orig_tokens} tokens)
  Narrative: {stats['narrative_tokens']} tokens
  JSON output: {stats['json_tokens']} tokens
  Narrative compression: {ratio_pct:.1f}%
  JSON compression: {stats['json_compression_ratio'] * 100:.1f}%
  Tokenizer: {stats['tokenizer']}

Narrative: {result.narrative}"""

        return output

    except Exception as e:
        return f"Error calculating compression stats: {str(e)}"


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
            except Exception as e:
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
        except Exception as e:
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
        "tools": [
            "describe_data",
            "describe_batch",
            "describe_json",
            "describe_dataframe",
            "describe_correlations",
            "compression_stats",
        ],
    }

    if defer_loading:
        base_config["default_config"] = {"defer_loading": True}

    return base_config

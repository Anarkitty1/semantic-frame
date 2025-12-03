"""MCP Server integration for semantic analysis.

This module provides a Model Context Protocol (MCP) server that exposes
semantic analysis capabilities to MCP clients (like ElizaOS or Claude Desktop).

Requires: pip install semantic-frame[mcp]

Usage:
    Run as a standalone server:
    $ mcp run semantic_frame.integrations.mcp:mcp

    Or import in your own MCP server:
    from semantic_frame.integrations.mcp import mcp
"""

from __future__ import annotations

import json

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "mcp is required for this module. " "Install with: pip install semantic-frame[mcp]"
    )

# Create the MCP server instance
mcp = FastMCP("semantic-frame")


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

    Args:
        data: A string containing the numbers. Can be a JSON array "[1, 2, 3]",
              CSV "1, 2, 3", or newline-separated values.
        context: A label for the data (e.g., "Server CPU Load", "Daily Sales").
                 This helps the tool generate a more relevant description.

    Returns:
        A natural language paragraph describing the data's behavior.
    """
    from semantic_frame import describe_series

    try:
        values = _parse_data_input(data)
        result: str = describe_series(values, context=context, output="text")
        return result
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

"""CrewAI tool integration for semantic analysis.

This module provides a CrewAI-compatible tool for analyzing
numerical data within crew workflows.

Requires: pip install semantic-frame[crewai]

Example:
    >>> from semantic_frame.integrations.crewai import get_crewai_tool
    >>> tool = get_crewai_tool()
    >>> result = tool("[1, 2, 3, 4, 100, 5, 6]")
    >>> print(result)
    'The time series data shows a flat/stationary pattern...'
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# Lazy import check for crewai
_crewai_available: bool | None = None
_tool_decorator: Callable[..., Any] | None = None


def _check_crewai() -> bool:
    """Check if crewai is available and cache the tool decorator."""
    global _crewai_available, _tool_decorator
    if _crewai_available is None:
        try:
            # crewai >= 1.0 moved tool to crewai.tools
            from crewai.tools import tool as crewai_tool

            _tool_decorator = crewai_tool
            _crewai_available = True
        except ImportError:
            try:
                # Fallback for older crewai versions
                from crewai import tool as crewai_tool_legacy  # type: ignore[attr-defined]

                _tool_decorator = crewai_tool_legacy
                _crewai_available = True
            except ImportError:
                _crewai_available = False
    return _crewai_available


def _parse_data_input(data_str: str) -> list[float]:
    """Parse string input to list of floats.

    Supports:
    - JSON array: "[1, 2, 3, 4, 5]"
    - CSV: "1, 2, 3, 4, 5"
    - Newline-separated: "1\\n2\\n3"

    Args:
        data_str: String representation of numerical data.

    Returns:
        List of float values.

    Raises:
        ValueError: If data cannot be parsed.
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

    raise ValueError(f"Could not parse data input: {data_str[:100]}...")


def semantic_analysis(data: str, context: str = "Data") -> str:
    """Analyze numerical data for patterns, trends, and anomalies.

    This function can be used directly or wrapped with @tool decorator.

    Args:
        data: Numerical data as JSON array, CSV, or newline-separated numbers.
        context: Label for the data being analyzed.

    Returns:
        Natural language description of data characteristics.

    Example:
        >>> result = semantic_analysis("[10, 20, 30, 100, 40, 50]", "Sensor Readings")
        >>> print(result)
        'The Sensor Readings data shows a flat/stationary pattern...'
    """
    from semantic_frame import describe_series

    values = _parse_data_input(data)
    result: str = describe_series(values, context=context, output="text")
    return result


def get_crewai_tool() -> Any:
    """Get CrewAI-decorated semantic analysis tool.

    Returns:
        CrewAI tool-decorated function.

    Raises:
        ImportError: If crewai is not installed.

    Example:
        >>> from semantic_frame.integrations.crewai import get_crewai_tool
        >>> from crewai import Agent
        >>> tool = get_crewai_tool()
        >>> agent = Agent(role="Data Analyst", tools=[tool])
    """
    if not _check_crewai() or _tool_decorator is None:
        raise ImportError(
            "crewai is required for get_crewai_tool(). "
            "Install with: pip install semantic-frame[crewai]"
        )

    # Use the cached tool decorator
    tool = _tool_decorator

    @tool("Semantic Data Analysis")  # type: ignore[misc]
    def analyze_data(data: str, context: str = "Data") -> str:
        """Analyze numerical data for trends, anomalies, and patterns.

        Input: JSON array of numbers like [1, 2, 3, 100, 5]
        Output: Natural language insights about the data.

        Args:
            data: Numerical data as JSON array, CSV, or newline-separated numbers.
            context: Label for the data (e.g., "Sales", "Temperature").

        Returns:
            Semantic description of the data characteristics.
        """
        return semantic_analysis(data, context)

    return analyze_data

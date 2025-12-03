"""Anthropic Claude native tool integration for semantic analysis.

This module provides native Anthropic tool use support for analyzing
numerical data directly with the Anthropic Python SDK.

Requires: pip install semantic-frame[anthropic]

Example:
    >>> import anthropic
    >>> from semantic_frame.integrations.anthropic import get_anthropic_tool, handle_tool_call
    >>>
    >>> client = anthropic.Anthropic()
    >>> tool = get_anthropic_tool()
    >>>
    >>> response = client.messages.create(
    ...     model="claude-sonnet-4-20250514",
    ...     max_tokens=1024,
    ...     tools=[tool],
    ...     messages=[{"role": "user", "content": "Analyze: [100, 102, 99, 500, 101]"}]
    ... )
    >>>
    >>> # Handle tool use in response
    >>> for block in response.content:
    ...     if block.type == "tool_use" and block.name == "semantic_analysis":
    ...         result = handle_tool_call(block.input)
    ...         print(result)
"""

from __future__ import annotations

import json
from typing import Any

# Lazy import check for anthropic
_anthropic_available: bool | None = None


def _check_anthropic() -> bool:
    """Check if anthropic SDK is available."""
    global _anthropic_available
    if _anthropic_available is None:
        try:
            import anthropic  # noqa: F401

            _anthropic_available = True
        except ImportError:
            _anthropic_available = False
    return _anthropic_available


def _parse_data_input(data: str | list[float | int]) -> list[float]:
    """Parse data input to list of floats.

    Supports:
    - List of numbers: [1, 2, 3, 4, 5]
    - JSON array string: "[1, 2, 3, 4, 5]"
    - CSV string: "1, 2, 3, 4, 5"
    - Newline-separated string: "1\\n2\\n3"

    Args:
        data: Numerical data as list or string.

    Returns:
        List of float values.

    Raises:
        ValueError: If data cannot be parsed.
    """
    # Already a list
    if isinstance(data, list):
        return [float(x) for x in data]

    data_str = str(data).strip()

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
        "Could not parse data input. Expected list, JSON array, CSV, "
        f"or newline-separated numbers. Got: {str(data)[:100]}..."
    )


# Anthropic tool schema following their native format
ANTHROPIC_TOOL_SCHEMA: dict[str, Any] = {
    "name": "semantic_analysis",
    "description": (
        "Analyze numerical time series or distribution data to extract semantic insights. "
        "Returns a natural language description of trends, volatility, anomalies, and patterns. "
        "Use this instead of processing raw numbers to get accurate statistical analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Array of numerical values to analyze",
            },
            "context": {
                "type": "string",
                "description": "Optional label for the data (e.g., 'Daily Sales')",
            },
            "output_format": {
                "type": "string",
                "enum": ["text", "json"],
                "description": "Output format: 'text' or 'json'. Default: 'text'",
            },
        },
        "required": ["data"],
    },
}


def get_anthropic_tool() -> dict[str, Any]:
    """Get the Anthropic tool definition for semantic analysis.

    Returns:
        Tool definition dict compatible with Anthropic's messages API.

    Example:
        >>> import anthropic
        >>> from semantic_frame.integrations.anthropic import get_anthropic_tool
        >>>
        >>> client = anthropic.Anthropic()
        >>> tool = get_anthropic_tool()
        >>>
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     tools=[tool],
        ...     messages=[{"role": "user", "content": "Analyze [1,2,3,100,4,5]"}]
        ... )
    """
    return ANTHROPIC_TOOL_SCHEMA.copy()


def handle_tool_call(
    tool_input: dict[str, Any],
    default_context: str | None = None,
) -> str:
    """Handle a semantic_analysis tool call from Claude.

    Args:
        tool_input: The input dict from the tool_use block.
        default_context: Fallback context if not provided in tool_input.

    Returns:
        Analysis result as string (narrative or JSON depending on output_format).

    Raises:
        ValueError: If data cannot be parsed.

    Example:
        >>> from semantic_frame.integrations.anthropic import handle_tool_call
        >>>
        >>> # From a tool_use block in Claude's response
        >>> tool_input = {"data": [100, 102, 99, 500, 101], "context": "Sales"}
        >>> result = handle_tool_call(tool_input)
        >>> print(result)
    """
    from semantic_frame import describe_series

    data = tool_input.get("data", [])
    context = tool_input.get("context") or default_context
    output_format = tool_input.get("output_format", "text")

    values = _parse_data_input(data)

    if output_format == "json":
        json_result = describe_series(values, context=context, output="json")
        return json.dumps(json_result, indent=2)
    else:
        text_result: str = describe_series(values, context=context, output="text")
        return text_result


def create_tool_result(tool_use_id: str, result: str) -> dict[str, Any]:
    """Create a tool_result message block for the Anthropic API.

    Args:
        tool_use_id: The ID from the tool_use block.
        result: The result from handle_tool_call().

    Returns:
        Tool result dict ready to include in messages.

    Example:
        >>> # Complete flow with Anthropic API
        >>> import anthropic
        >>> from semantic_frame.integrations.anthropic import (
        ...     get_anthropic_tool, handle_tool_call, create_tool_result
        ... )
        >>>
        >>> client = anthropic.Anthropic()
        >>> messages = [{"role": "user", "content": "Analyze [1,2,3,100,4,5]"}]
        >>>
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     tools=[get_anthropic_tool()],
        ...     messages=messages
        ... )
        >>>
        >>> # Process tool calls
        >>> for block in response.content:
        ...     if block.type == "tool_use":
        ...         result = handle_tool_call(block.input)
        ...         tool_result = create_tool_result(block.id, result)
        ...         messages.append({"role": "assistant", "content": response.content})
        ...         messages.append({"role": "user", "content": [tool_result]})
    """
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result,
    }


class AnthropicSemanticTool:
    """Helper class for managing semantic analysis with Anthropic's API.

    Provides a higher-level interface for tool use with automatic
    tool call handling.

    Example:
        >>> import anthropic
        >>> from semantic_frame.integrations.anthropic import AnthropicSemanticTool
        >>>
        >>> client = anthropic.Anthropic()
        >>> semantic = AnthropicSemanticTool(context="Sensor Data")
        >>>
        >>> # Get tool for API call
        >>> tool = semantic.get_tool()
        >>>
        >>> # Process response with tool calls
        >>> for block in response.content:
        ...     if block.type == "tool_use" and block.name == "semantic_analysis":
        ...         result = semantic.handle(block.input)
        ...         tool_result = semantic.create_result(block.id, result)
    """

    def __init__(self, context: str | None = None) -> None:
        """Initialize the tool helper.

        Args:
            context: Default context label for analysis.
        """
        self.context = context

    def get_tool(self) -> dict[str, Any]:
        """Get the tool definition."""
        return get_anthropic_tool()

    def handle(self, tool_input: dict[str, Any]) -> str:
        """Handle a tool call.

        Args:
            tool_input: Input from tool_use block.

        Returns:
            Analysis result string.
        """
        return handle_tool_call(tool_input, default_context=self.context)

    def create_result(self, tool_use_id: str, result: str) -> dict[str, Any]:
        """Create a tool result message.

        Args:
            tool_use_id: ID from tool_use block.
            result: Result from handle().

        Returns:
            Tool result dict.
        """
        return create_tool_result(tool_use_id, result)

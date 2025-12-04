"""Anthropic Claude native tool integration for semantic analysis.

This module provides native Anthropic tool use support for analyzing
numerical data directly with the Anthropic Python SDK.

Supports Anthropic's Advanced Tool Use features (beta):
- Tool Use Examples: Concrete examples for +18% parameter accuracy
- Deferred Loading: For agents with 1000+ tools
- Programmatic Tool Calling: Batch analysis via code execution

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

Advanced Tool Use Example (Beta):
    >>> # For agents with many tools, use deferred loading
    >>> tool = get_anthropic_tool(
    ...     defer_loading=True,
    ...     allowed_callers=["code_execution"],
    ...     include_examples=True,
    ... )
    >>>
    >>> response = client.beta.messages.create(
    ...     betas=["advanced-tool-use-2025-11-20"],
    ...     model="claude-sonnet-4-5-20250929",
    ...     max_tokens=4096,
    ...     tools=[
    ...         {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
    ...         {"type": "code_execution_20250825", "name": "code_execution"},
    ...         tool,
    ...     ],
    ...     messages=[...]
    ... )
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


# =============================================================================
# Tool Use Examples (Anthropic Advanced Tool Use)
# These concrete examples boost parameter accuracy by ~18% per Anthropic's testing
# =============================================================================

TOOL_USE_EXAMPLES: list[dict[str, Any]] = [
    # Example 1: Anomaly detection (spike in otherwise stable data)
    {
        "input": {
            "data": [100, 102, 99, 101, 500, 100, 98],
            "context": "Server Latency (ms)",
        },
        "expected_output": (
            "The Server Latency (ms) data shows a flat/stationary pattern with "
            "stable variability. 1 anomaly detected at index 4 (value: 500.00). "
            "Baseline: 100.00 (range: 98.00-500.00)."
        ),
    },
    # Example 2: Clear upward trend
    {
        "input": {
            "data": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "context": "Daily Sales ($)",
        },
        "expected_output": (
            "The Daily Sales ($) data shows a rapidly rising pattern with "
            "stable variability. No anomalies detected. "
            "Baseline: 55.00 (range: 10.00-100.00)."
        ),
    },
    # Example 3: JSON output format
    {
        "input": {
            "data": [45, 47, 46, 95, 44, 45, 46],
            "context": "CPU Usage %",
            "output_format": "json",
        },
        "expected_output": (
            '{"narrative": "The CPU Usage % data shows...", '
            '"trend": "flat/stationary", "volatility": "stable", '
            '"anomalies": [{"index": 3, "value": 95.0, "z_score": 4.2}]}'
        ),
    },
    # Example 4: Minimal input (no context)
    {
        "input": {
            "data": [1.5, 1.4, 1.6, 1.5, 1.3, 1.7],
        },
        "expected_output": (
            "The data shows a flat/stationary pattern with stable variability. "
            "No anomalies detected. Baseline: 1.50 (range: 1.30-1.70)."
        ),
    },
    # Example 5: Crypto-style volatile data
    {
        "input": {
            "data": [42000, 43500, 41000, 44000, 39000, 46000, 38000, 47000],
            "context": "BTC/USD Hourly",
        },
        "expected_output": (
            "The BTC/USD Hourly data shows a rising pattern with expanding "
            "variability. No anomalies detected but volatility is high. "
            "Baseline: 42562.50 (range: 38000.00-47000.00)."
        ),
    },
]


# =============================================================================
# Base Tool Schema (Anthropic Native Format)
# =============================================================================

ANTHROPIC_TOOL_SCHEMA: dict[str, Any] = {
    "name": "semantic_analysis",
    "description": (
        "Analyze numerical time series or distribution data to extract semantic insights. "
        "Returns a natural language description of trends, volatility, anomalies, and patterns. "
        "Use this instead of processing raw numbers to get accurate statistical analysis. "
        "Supports arrays of any size - the tool compresses 10,000+ data points into ~50 words."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Array of numerical values to analyze (prices, metrics, readings)",
            },
            "context": {
                "type": "string",
                "description": (
                    "Label for the data that appears in the narrative. "
                    "Examples: 'CPU Usage %', 'Daily Revenue', 'BTC/USD', 'Temperature (C)'"
                ),
            },
            "output_format": {
                "type": "string",
                "enum": ["text", "json"],
                "description": "Output format: 'text' (narrative) or 'json' (structured)",
            },
        },
        "required": ["data"],
    },
}


def get_anthropic_tool(
    *,
    defer_loading: bool = False,
    allowed_callers: list[str] | None = None,
    include_examples: bool = False,
) -> dict[str, Any]:
    """Get the Anthropic tool definition for semantic analysis.

    Supports Anthropic's Advanced Tool Use features (beta) for improved
    accuracy and efficiency in large tool libraries.

    Args:
        defer_loading: If True, tool is discovered via Tool Search Tool rather
            than loaded into context upfront. Use for agents with 50+ tools.
            Requires: betas=["advanced-tool-use-2025-11-20"]
        allowed_callers: List of callers that can invoke this tool.
            Set to ["code_execution"] to enable batch analysis via
            Programmatic Tool Calling. Requires code_execution tool enabled.
        include_examples: If True, includes input_examples for +18% parameter
            accuracy per Anthropic's testing. Default is False for standard API
            compatibility. Set to True only when using the beta API:
            client.beta.messages.create(betas=["advanced-tool-use-2025-11-20"], ...)

    Returns:
        Tool definition dict compatible with Anthropic's messages API.

    Example (Standard):
        >>> tool = get_anthropic_tool()
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     tools=[tool],
        ...     ...
        ... )

    Example (Advanced Tool Use - Beta):
        >>> # For large tool libraries with batch processing
        >>> tool = get_anthropic_tool(
        ...     defer_loading=True,
        ...     allowed_callers=["code_execution"],
        ... )
        >>> response = client.beta.messages.create(
        ...     betas=["advanced-tool-use-2025-11-20"],
        ...     model="claude-sonnet-4-5-20250929",
        ...     tools=[
        ...         {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        ...         {"type": "code_execution_20250825", "name": "code_execution"},
        ...         tool,
        ...     ],
        ...     ...
        ... )
    """
    tool = ANTHROPIC_TOOL_SCHEMA.copy()
    tool["input_schema"] = ANTHROPIC_TOOL_SCHEMA["input_schema"].copy()

    # Advanced Tool Use: Deferred Loading
    if defer_loading:
        tool["defer_loading"] = True

    # Advanced Tool Use: Programmatic Tool Calling
    if allowed_callers:
        tool["allowed_callers"] = allowed_callers

    # Advanced Tool Use: Input Examples (+18% accuracy)
    if include_examples:
        tool["input_examples"] = TOOL_USE_EXAMPLES

    return tool


def get_tool_for_discovery() -> dict[str, Any]:
    """Get tool configured for Tool Search discovery.

    Returns a tool definition optimized for large tool libraries where
    tools are discovered on-demand via Tool Search Tool rather than
    loaded into context upfront.

    Returns:
        Tool definition with defer_loading=True.

    Example:
        >>> tools = [
        ...     {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        ...     get_tool_for_discovery(),
        ...     # ... hundreds more deferred tools
        ... ]
    """
    return get_anthropic_tool(defer_loading=True, include_examples=True)


def get_tool_for_batch_processing() -> dict[str, Any]:
    """Get tool configured for Programmatic Tool Calling.

    Returns a tool definition that can be called from code execution,
    enabling parallel batch analysis of multiple data series.

    Returns:
        Tool definition with allowed_callers=["code_execution"].

    Example:
        >>> # Claude can now call this from code:
        >>> # results = await asyncio.gather(*[
        >>> #     semantic_analysis(col) for col in dataframe.columns
        >>> # ])
    """
    return get_anthropic_tool(
        allowed_callers=["code_execution"],
        include_examples=True,
    )


def get_advanced_tool() -> dict[str, Any]:
    """Get fully-featured tool for Advanced Tool Use.

    Combines all advanced features:
    - defer_loading: Discovered via search, not loaded upfront
    - allowed_callers: Can be called from code for batch processing
    - input_examples: Concrete examples for better accuracy

    Returns:
        Tool definition with all advanced features enabled.

    Example:
        >>> response = client.beta.messages.create(
        ...     betas=["advanced-tool-use-2025-11-20"],
        ...     model="claude-sonnet-4-5-20250929",
        ...     tools=[
        ...         {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        ...         {"type": "code_execution_20250825", "name": "code_execution"},
        ...         get_advanced_tool(),
        ...     ],
        ...     messages=[...]
        ... )
    """
    return get_anthropic_tool(
        defer_loading=True,
        allowed_callers=["code_execution"],
        include_examples=True,
    )


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


def handle_batch_tool_calls(
    tool_inputs: list[dict[str, Any]],
    default_context: str | None = None,
) -> list[str]:
    """Handle multiple semantic_analysis tool calls in batch.

    Useful for Programmatic Tool Calling where Claude invokes the tool
    multiple times from code execution.

    Args:
        tool_inputs: List of input dicts from tool_use blocks.
        default_context: Fallback context prefix for all calls.

    Returns:
        List of analysis results in the same order as inputs.

    Example:
        >>> inputs = [
        ...     {"data": [1, 2, 3], "context": "Series A"},
        ...     {"data": [4, 5, 6], "context": "Series B"},
        ... ]
        >>> results = handle_batch_tool_calls(inputs)
    """
    return [handle_tool_call(inp, default_context) for inp in tool_inputs]


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
    tool call handling. Supports Advanced Tool Use features.

    Example (Standard):
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

    Example (Advanced Tool Use):
        >>> semantic = AnthropicSemanticTool(
        ...     context="Trading Metrics",
        ...     defer_loading=True,
        ...     allowed_callers=["code_execution"],
        ... )
        >>> tool = semantic.get_tool()  # Configured for advanced features
    """

    def __init__(
        self,
        context: str | None = None,
        *,
        defer_loading: bool = False,
        allowed_callers: list[str] | None = None,
        include_examples: bool = True,
    ) -> None:
        """Initialize the tool helper.

        Args:
            context: Default context label for analysis.
            defer_loading: Enable Tool Search discovery.
            allowed_callers: Enable Programmatic Tool Calling.
            include_examples: Include input examples for accuracy.
        """
        self.context = context
        self.defer_loading = defer_loading
        self.allowed_callers = allowed_callers
        self.include_examples = include_examples

    def get_tool(self) -> dict[str, Any]:
        """Get the tool definition with configured options."""
        return get_anthropic_tool(
            defer_loading=self.defer_loading,
            allowed_callers=self.allowed_callers,
            include_examples=self.include_examples,
        )

    def handle(self, tool_input: dict[str, Any]) -> str:
        """Handle a tool call.

        Args:
            tool_input: Input from tool_use block.

        Returns:
            Analysis result string.
        """
        return handle_tool_call(tool_input, default_context=self.context)

    def handle_batch(self, tool_inputs: list[dict[str, Any]]) -> list[str]:
        """Handle multiple tool calls in batch.

        Args:
            tool_inputs: List of inputs from tool_use blocks.

        Returns:
            List of analysis results.
        """
        return handle_batch_tool_calls(tool_inputs, default_context=self.context)

    def create_result(self, tool_use_id: str, result: str) -> dict[str, Any]:
        """Create a tool result message.

        Args:
            tool_use_id: ID from tool_use block.
            result: Result from handle().

        Returns:
            Tool result dict.
        """
        return create_tool_result(tool_use_id, result)

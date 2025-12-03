"""LangChain tool integration for semantic analysis.

This module provides a LangChain-compatible tool for analyzing
numerical data within agent workflows.

Requires: pip install semantic-frame[langchain]

Example:
    >>> from semantic_frame.integrations.langchain import get_semantic_tool
    >>> tool = get_semantic_tool()
    >>> result = tool.run("[1, 2, 3, 4, 100, 5, 6]")
    >>> print(result)
    'The time series data shows a flat/stationary pattern...'
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Lazy import check for langchain
_langchain_available: bool | None = None


def _check_langchain() -> bool:
    """Check if langchain is available."""
    global _langchain_available
    if _langchain_available is None:
        try:
            from langchain.tools import BaseTool  # noqa: F401

            _langchain_available = True
        except ImportError:
            _langchain_available = False
    return _langchain_available


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

    raise ValueError(
        f"Could not parse data input. Expected JSON array, CSV, or newline-separated numbers. "
        f"Got: {data_str[:100]}..."
    )


class SemanticAnalysisTool:
    """LangChain tool wrapper for semantic data analysis.

    This class provides a standalone tool that can analyze numerical data
    and return semantic descriptions. It can be converted to a LangChain
    BaseTool for use in agent workflows.

    Usage:
        >>> from semantic_frame.integrations.langchain import SemanticAnalysisTool
        >>> tool = SemanticAnalysisTool(context="Sensor Data")
        >>> result = tool._run("[1, 2, 3, 4, 100, 5, 6]")
        >>> print(result)

    Attributes:
        name: Tool name for agent identification.
        description: Tool description for agent decision-making.
        context: Default context label for analysis.
    """

    name: str = "semantic_analysis"
    description: str = (
        "Analyze numerical data for patterns, trends, anomalies, and statistical insights. "
        "Input: JSON array, CSV, or newline-separated numbers. "
        "Output: Natural language description of data characteristics."
    )

    def __init__(self, context: str | None = None) -> None:
        """Initialize the tool.

        Args:
            context: Default context label for analysis (e.g., "Temperature Readings").
        """
        self.context = context

    def _run(self, data: str) -> str:
        """Execute analysis on data string.

        Args:
            data: Numerical data as string (JSON array, CSV, or newline-separated).

        Returns:
            Semantic narrative describing the data.

        Raises:
            ValueError: If data cannot be parsed.
        """
        from semantic_frame import describe_series

        values = _parse_data_input(data)
        result: str = describe_series(values, context=self.context, output="text")
        return result

    async def _arun(self, data: str) -> str:
        """Async version - delegates to sync (analysis is CPU-bound).

        Args:
            data: Numerical data as string.

        Returns:
            Semantic narrative describing the data.
        """
        return self._run(data)

    def as_langchain_tool(self) -> Any:
        """Convert to LangChain BaseTool.

        Returns:
            LangChain BaseTool instance.

        Raises:
            ImportError: If langchain is not installed.
        """
        if not _check_langchain():
            raise ImportError(
                "langchain is required for as_langchain_tool(). "
                "Install with: pip install semantic-frame[langchain]"
            )

        from langchain.tools import BaseTool

        outer_self = self

        class _SemanticTool(BaseTool):  # type: ignore[misc]
            name: str = outer_self.name
            description: str = outer_self.description

            def _run(self, data: str) -> str:
                return outer_self._run(data)

            async def _arun(self, data: str) -> str:
                return outer_self._run(data)

        return _SemanticTool()


def get_semantic_tool(context: str | None = None) -> Any:
    """Factory function to create LangChain-compatible semantic analysis tool.

    Args:
        context: Optional default context for analysis.

    Returns:
        LangChain BaseTool instance.

    Raises:
        ImportError: If langchain is not installed.

    Example:
        >>> from semantic_frame.integrations.langchain import get_semantic_tool
        >>> from langchain.agents import create_openai_tools_agent
        >>> tool = get_semantic_tool(context="Sales Data")
        >>> agent = create_openai_tools_agent(llm, [tool], prompt)
    """
    return SemanticAnalysisTool(context=context).as_langchain_tool()

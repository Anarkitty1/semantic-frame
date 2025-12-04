"""Framework integrations for semantic-frame.

This package provides tool wrappers for popular AI agent frameworks,
with support for Anthropic's Advanced Tool Use features (beta).

Available integrations:
- anthropic: Native Anthropic Claude tool use with Advanced Tool Use support
- langchain: LangChain BaseTool wrapper
- crewai: CrewAI @tool decorator wrapper
- mcp: Model Context Protocol server
- mcp_wrapper: Utilities for wrapping MCP tools with semantic output

Advanced Tool Use Features (Anthropic Beta):
- Tool Search: Deferred loading for 1000+ tool libraries
- Programmatic Calling: Batch analysis via code execution
- Input Examples: +18% parameter accuracy

Install optional dependencies:
    pip install semantic-frame[anthropic]  # For Anthropic SDK
    pip install semantic-frame[langchain]  # For LangChain
    pip install semantic-frame[crewai]     # For CrewAI
    pip install semantic-frame[mcp]        # For MCP server
    pip install semantic-frame[all]        # All integrations

Quick Start (Anthropic):
    >>> from semantic_frame.integrations.anthropic import (
    ...     get_anthropic_tool,      # Standard tool
    ...     get_advanced_tool,       # All advanced features
    ...     get_tool_for_discovery,  # For Tool Search
    ...     handle_tool_call,
    ... )

Quick Start (MCP Wrapper):
    >>> from semantic_frame.integrations.mcp_wrapper import (
    ...     wrap_numeric_output,      # Decorator for functions
    ...     transform_to_semantic,    # One-off transformation
    ...     SemanticMCPWrapper,       # Class-based wrapper
    ... )
    >>>
    >>> @wrap_numeric_output(context="CPU Usage %")
    ... def get_cpu_metrics():
    ...     return [45, 47, 46, 95, 44, 45]
    >>>
    >>> # Now returns semantic narrative instead of raw numbers
    >>> print(get_cpu_metrics())
    "The CPU Usage % data shows a flat/stationary pattern..."
"""

# Core wrapper utilities (no optional dependencies)
from semantic_frame.integrations.mcp_wrapper import (
    SemanticMCPWrapper,
    semantic_wrapper,
    transform_to_semantic,
    wrap_numeric_output,
)

__all__ = [
    # MCP Wrapper utilities (core, no optional deps)
    "wrap_numeric_output",
    "transform_to_semantic",
    "SemanticMCPWrapper",
    "semantic_wrapper",
]

"""MCP Wrapper utilities for semantic output transformation.

This module provides utilities for wrapping existing MCP tools or data sources
to automatically transform numerical outputs into semantic narratives.

Use this to make any numerical data source "LLM-friendly" without modifying
the original implementation.

Example:
    >>> from semantic_frame.integrations.mcp_wrapper import wrap_numeric_output
    >>>
    >>> # Wrap a function that returns numerical data
    >>> @wrap_numeric_output(context="CPU Metrics")
    >>> def get_cpu_readings():
    ...     return [45, 47, 46, 95, 44, 45]
    >>>
    >>> # Now returns semantic narrative instead of raw numbers
    >>> print(get_cpu_readings())
    "The CPU Metrics data shows a flat/stationary pattern..."
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

# Type for numerical data that can be wrapped
NumericData = list | np.ndarray | dict[str, Any]


def _extract_numeric_array(data: Any) -> list[float] | None:
    """Extract numeric array from various data formats.

    Args:
        data: Raw data to extract numbers from.

    Returns:
        List of floats if numeric data found, None otherwise.
    """
    # Direct list of numbers
    if isinstance(data, list | tuple):
        try:
            return [float(x) for x in data]
        except (TypeError, ValueError):
            pass

    # NumPy array
    if isinstance(data, np.ndarray):
        try:
            result: list[float] = data.astype(float).tolist()
            return result
        except (TypeError, ValueError):
            pass

    # Dict with 'data' or 'values' key
    if isinstance(data, dict):
        for key in ("data", "values", "readings", "prices", "metrics"):
            if key in data and isinstance(data[key], list | tuple):
                try:
                    return [float(x) for x in data[key]]
                except (TypeError, ValueError):
                    pass

    # JSON string
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return _extract_numeric_array(parsed)
        except json.JSONDecodeError:
            pass

    return None


def wrap_numeric_output(
    context: str | None = None,
    context_key: str | None = None,
    output_format: Literal["text", "json"] = "text",
    passthrough_on_failure: bool = True,
) -> Callable[[F], F]:
    """Decorator to wrap functions that return numerical data.

    Transforms raw numerical outputs into semantic narratives automatically.
    Perfect for MCP tools that return metrics, prices, or sensor readings.

    Args:
        context: Static context label for the data (e.g., "CPU Usage %").
        context_key: Key in return dict to use as dynamic context label.
            If the wrapped function returns a dict with this key, its value
            becomes the context. Takes precedence over static context.
        output_format: "text" for narrative, "json" for structured output.
        passthrough_on_failure: If True, returns original data when
            semantic transformation fails. If False, raises exception.

    Returns:
        Decorated function that returns semantic output.

    Example:
        >>> @wrap_numeric_output(context="Server Latency (ms)")
        ... def get_latencies():
        ...     return [100, 102, 99, 500, 101, 100]
        >>>
        >>> print(get_latencies())
        "The Server Latency (ms) data shows a flat/stationary pattern..."

    Example with dynamic context:
        >>> @wrap_numeric_output(context_key="metric_name")
        ... def get_metrics():
        ...     return {"metric_name": "CPU Load", "values": [45, 47, 46, 95, 44]}
        >>>
        >>> print(get_metrics())
        "The CPU Load data shows..."
    """
    from semantic_frame import describe_series

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Call original function
            result = func(*args, **kwargs)

            # Try to extract numeric data
            numeric_data = _extract_numeric_array(result)

            if numeric_data is None:
                if passthrough_on_failure:
                    return result
                raise ValueError(
                    f"Could not extract numeric data from {type(result).__name__}. "
                    "Expected list, numpy array, or dict with 'data'/'values' key."
                )

            # Determine context
            actual_context = context
            if context_key and isinstance(result, dict) and context_key in result:
                actual_context = str(result[context_key])

            # Generate semantic output
            try:
                if output_format == "json":
                    semantic_result: str | dict[str, Any] = describe_series(
                        numeric_data, context=actual_context, output="json"
                    )
                else:
                    semantic_result = describe_series(
                        numeric_data, context=actual_context, output="text"
                    )

                # If original was dict, optionally preserve structure
                if isinstance(result, dict) and output_format == "text":
                    return {
                        **result,
                        "semantic_narrative": semantic_result,
                        "_original_values": numeric_data,
                    }
                return semantic_result

            except Exception as e:
                if passthrough_on_failure:
                    return result
                raise ValueError(f"Semantic analysis failed: {e}") from e

        return wrapper  # type: ignore

    return decorator


def transform_to_semantic(
    data: NumericData,
    context: str | None = None,
    output_format: Literal["text", "json"] = "text",
) -> str | dict[str, Any]:
    """Transform numerical data to semantic representation.

    Standalone function for one-off transformations without decoration.

    Args:
        data: Numerical data (list, numpy array, or dict with data key).
        context: Context label for the narrative.
        output_format: "text" or "json".

    Returns:
        Semantic narrative or structured result.

    Raises:
        ValueError: If data cannot be converted.

    Example:
        >>> readings = [100, 102, 99, 500, 101]
        >>> narrative = transform_to_semantic(readings, context="Latency (ms)")
        >>> print(narrative)
        "The Latency (ms) data shows..."
    """
    from semantic_frame import describe_series

    numeric_data = _extract_numeric_array(data)
    if numeric_data is None:
        raise ValueError(
            f"Could not extract numeric data from {type(data).__name__}. "
            "Expected list, numpy array, or dict with 'data'/'values' key."
        )

    if output_format == "json":
        return describe_series(numeric_data, context=context, output="json")
    return describe_series(numeric_data, context=context, output="text")


class SemanticMCPWrapper:
    """Wrapper for transforming MCP tool outputs to semantic format.

    Provides a class-based interface for wrapping MCP tools with
    semantic transformation capabilities.

    Example:
        >>> from semantic_frame.integrations.mcp_wrapper import SemanticMCPWrapper
        >>>
        >>> wrapper = SemanticMCPWrapper(default_context="Sensor Readings")
        >>>
        >>> # Transform any numerical response
        >>> raw_data = [100, 102, 99, 500, 101]
        >>> semantic = wrapper.transform(raw_data)
        >>>
        >>> # Use with MCP tool results
        >>> @wrapper.wrap(context="CPU Usage %")
        ... def get_cpu_usage():
        ...     return fetch_cpu_metrics()
    """

    def __init__(
        self,
        default_context: str | None = None,
        default_format: Literal["text", "json"] = "text",
        passthrough_on_failure: bool = True,
    ) -> None:
        """Initialize the wrapper.

        Args:
            default_context: Default context label for transformations.
            default_format: Default output format ("text" or "json").
            passthrough_on_failure: Return original on transformation failure.
        """
        self.default_context = default_context
        self.default_format: Literal["text", "json"] = default_format
        self.passthrough_on_failure = passthrough_on_failure

    def transform(
        self,
        data: NumericData,
        context: str | None = None,
        output_format: Literal["text", "json"] | None = None,
    ) -> str | dict[str, Any]:
        """Transform numerical data to semantic representation.

        Args:
            data: Numerical data to transform.
            context: Context label (overrides default).
            output_format: Output format (overrides default).

        Returns:
            Semantic narrative or structured result.
        """
        return transform_to_semantic(
            data,
            context=context or self.default_context,
            output_format=output_format or self.default_format,
        )

    def wrap(
        self,
        context: str | None = None,
        context_key: str | None = None,
        output_format: Literal["text", "json"] | None = None,
    ) -> Callable[[F], F]:
        """Get a decorator for wrapping functions.

        Args:
            context: Static context (overrides instance default).
            context_key: Dict key for dynamic context.
            output_format: Output format (overrides instance default).

        Returns:
            Decorator function.
        """
        return wrap_numeric_output(
            context=context or self.default_context,
            context_key=context_key,
            output_format=output_format or self.default_format,
            passthrough_on_failure=self.passthrough_on_failure,
        )


# Convenience alias
semantic_wrapper = SemanticMCPWrapper

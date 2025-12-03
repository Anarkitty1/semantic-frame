"""Main entry point for Semantic Frame.

This module provides the primary API for converting numerical data
into semantic descriptions.

Usage:
    >>> import pandas as pd
    >>> from semantic_frame import describe_series
    >>>
    >>> data = pd.Series([100, 102, 99, 101, 500, 100, 98])
    >>> print(describe_series(data, context="Server Latency (ms)"))
    "The Server Latency (ms) data shows a flat/stationary pattern..."
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Union, overload

import numpy as np

from semantic_frame.core.translator import analyze_series
from semantic_frame.interfaces.json_schema import SemanticResult

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)

# Type alias for supported input types
ArrayLike = Union["pd.Series", "np.ndarray", "pl.Series", list]
DataFrameLike = Union["pd.DataFrame", "pl.DataFrame"]

# Valid output formats
VALID_OUTPUT_FORMATS = {"text", "json", "full"}


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert any supported input type to NumPy array.

    Args:
        data: Input data (Pandas Series, NumPy array, Polars Series, or list).

    Returns:
        NumPy array of float64 values with NaN for missing data.

    Raises:
        TypeError: If input type is not supported or contains non-numeric data.
    """
    # Already numpy
    if isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.number) and data.dtype != object:
            raise TypeError(
                f"Expected numeric numpy array, got dtype {data.dtype}. "
                "Convert to numeric type before passing to describe_series."
            )
        try:
            return data.astype(float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Could not convert numpy array to float: {e}. "
                "Ensure array contains only numeric values."
            ) from e

    # Python list
    if isinstance(data, list):
        try:
            return np.array(data, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"List contains non-numeric values: {e}. "
                "Expected a list of numbers."
            ) from e

    # Check for Pandas Series
    type_name = type(data).__name__
    module_name = type(data).__module__

    if "pandas" in module_name or type_name == "Series":
        # Pandas Series - handle carefully
        try:
            return data.to_numpy(dtype=float, na_value=np.nan)  # type: ignore
        except (TypeError, ValueError) as original_error:
            # Fallback for older pandas versions or type issues
            logger.debug(
                "Primary pandas conversion failed (%s), attempting fallback",
                str(original_error),
            )
            try:
                arr = data.to_numpy()  # type: ignore
                return arr.astype(float)
            except (TypeError, ValueError) as fallback_error:
                raise TypeError(
                    f"Could not convert pandas Series to float array: {fallback_error}. "
                    "Ensure Series contains numeric data."
                ) from original_error

    if "polars" in module_name:
        # Polars Series
        try:
            return data.to_numpy()  # type: ignore
        except Exception as e:
            raise TypeError(
                f"Could not convert polars Series to numpy array: {e}. "
                "Ensure the Series contains numeric data."
            ) from e

    # Fallback: try array protocol
    try:
        return np.asarray(data, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__} "
            f"(from module {type(data).__module__}). "
            f"Expected pandas.Series, numpy.ndarray, polars.Series, or list. "
            f"Error: {e}"
        ) from e


@overload
def describe_series(
    data: ArrayLike,
    context: str | None = None,
    output: Literal["text"] = "text",
) -> str: ...


@overload
def describe_series(
    data: ArrayLike,
    context: str | None = None,
    output: Literal["json"] = ...,
) -> dict[str, Any]: ...


@overload
def describe_series(
    data: ArrayLike,
    context: str | None = None,
    output: Literal["full"] = ...,
) -> SemanticResult: ...


def describe_series(
    data: ArrayLike,
    context: str | None = None,
    output: str = "text",
) -> str | dict[str, Any] | SemanticResult:
    """Convert a data series into a semantic description.

    This is the primary API for analyzing single-column data. It converts
    raw numerical data into token-efficient natural language descriptions
    suitable for LLM context.

    Args:
        data: Input data. Supports:
            - pandas.Series
            - numpy.ndarray
            - polars.Series
            - Python list of numbers
        context: Optional context label for the data (e.g., "CPU Usage",
                "Sales Data", "Temperature Readings"). Used in narrative.
        output: Output format:
            - "text": Returns narrative string only (default)
            - "json": Returns dict suitable for JSON serialization
            - "full": Returns complete SemanticResult object

    Returns:
        Semantic description in the requested format.

    Raises:
        TypeError: If data is not a supported type or contains non-numeric values.
        ValueError: If output format is not valid.

    Examples:
        >>> import pandas as pd
        >>> data = pd.Series([100, 102, 99, 101, 500, 100, 98])

        >>> # Get narrative text (default)
        >>> describe_series(data, context="Server Latency (ms)")
        'The Server Latency (ms) data shows a flat/stationary pattern...'

        >>> # Get structured JSON
        >>> describe_series(data, output="json")
        {'narrative': '...', 'trend': 'flat/stationary', ...}

        >>> # Get full result object
        >>> result = describe_series(data, output="full")
        >>> print(result.compression_ratio)
        0.95
    """
    # Validate output format
    if output not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format: {output!r}. "
            f"Expected one of: {VALID_OUTPUT_FORMATS}"
        )

    # Convert to numpy
    values = _to_numpy(data)

    # Run analysis
    result = analyze_series(values, context=context)

    # Return in requested format
    if output == "text":
        return result.narrative
    elif output == "json":
        return result.model_dump(mode="json", by_alias=True)
    else:  # output == "full"
        return result


def describe_dataframe(
    df: DataFrameLike,
    context: str | None = None,
) -> dict[str, SemanticResult]:
    """Analyze all numeric columns in a DataFrame.

    Runs describe_series on each numeric column and returns a dict
    mapping column names to their SemanticResult.

    Args:
        df: Input DataFrame (pandas or polars).
        context: Optional context prefix. Column names will be appended.

    Returns:
        Dict mapping column names to SemanticResult objects.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'cpu': [40, 42, 41, 95, 40],
        ...     'memory': [60, 61, 60, 60, 61],
        ... })
        >>> results = describe_dataframe(df, context="Server Metrics")
        >>> print(results['cpu'].narrative)
        'The Server Metrics - cpu data shows a flat/stationary pattern...'
    """
    results: dict[str, SemanticResult] = {}

    # Detect Polars vs Pandas
    type_name = type(df).__name__
    module_name = type(df).__module__

    if "polars" in module_name:
        # Polars DataFrame
        import polars as pl

        for col_name in df.columns:
            dtype = df[col_name].dtype
            # Check if numeric (int, float types)
            if dtype.is_numeric():
                col_context = f"{context} - {col_name}" if context else col_name
                result = describe_series(
                    df[col_name],
                    context=col_context,
                    output="full",
                )
                results[col_name] = result  # type: ignore
    else:
        # Pandas DataFrame
        numeric_cols = df.select_dtypes(include=[np.number]).columns  # type: ignore
        for col_name in numeric_cols:
            col_context = f"{context} - {col_name}" if context else str(col_name)
            result = describe_series(
                df[col_name],  # type: ignore
                context=col_context,
                output="full",
            )
            results[str(col_name)] = result  # type: ignore

    return results


def compression_stats(original_data: ArrayLike, result: SemanticResult) -> dict:
    """Calculate detailed compression statistics.

    Args:
        original_data: The original input data.
        result: The SemanticResult from describe_series.

    Returns:
        Dict with compression statistics.
    """
    values = _to_numpy(original_data)

    # Estimate original token count (rough: 2 tokens per number)
    original_tokens = len(values) * 2

    # Narrative tokens (rough: 1 token per word)
    narrative_tokens = len(result.narrative.split())

    # JSON output tokens
    json_str = result.to_json_str()
    json_tokens = len(json_str.split())

    return {
        "original_data_points": len(values),
        "original_tokens_estimate": original_tokens,
        "narrative_tokens": narrative_tokens,
        "json_tokens": json_tokens,
        "narrative_compression_ratio": 1 - (narrative_tokens / max(original_tokens, 1)),
        "json_compression_ratio": 1 - (json_tokens / max(original_tokens, 1)),
    }

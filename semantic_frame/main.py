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

from semantic_frame.core.correlations import (
    calc_correlation_matrix,
    identify_significant_correlations,
)
from semantic_frame.core.translator import analyze_series
from semantic_frame.interfaces.json_schema import (
    CorrelationInsight,
    DataFrameResult,
    SemanticResult,
)
from semantic_frame.narrators.correlation import (
    generate_correlation_narrative,
    generate_dataframe_summary,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)

# Type alias for supported input types
ArrayLike = Union["pd.Series", "np.ndarray", "pl.Series", "list[float]"]
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
            arr: np.ndarray = np.array(data, dtype=float)
            return arr
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"List contains non-numeric values: {e}. Expected a list of numbers."
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
        fallback_arr: np.ndarray = np.asarray(data, dtype=float)
        return fallback_arr
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
            f"Invalid output format: {output!r}. Expected one of: {VALID_OUTPUT_FORMATS}"
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
    correlation_threshold: float = 0.5,
) -> DataFrameResult:
    """Analyze all numeric columns in a DataFrame with correlation analysis.

    Runs describe_series on each numeric column and detects cross-column
    correlations to identify relationships like "Sales UP, Inventory DOWN".

    Args:
        df: Input DataFrame (pandas or polars).
        context: Optional context prefix. Column names will be appended.
        correlation_threshold: Minimum |r| for correlation reporting (default 0.5).
            Only correlations with absolute value >= threshold are included.

    Returns:
        DataFrameResult with per-column analysis and correlation insights.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'sales': [100, 200, 300, 400, 500],
        ...     'inventory': [500, 400, 300, 200, 100],
        ... })
        >>> result = describe_dataframe(df, context="Retail Metrics")
        >>> print(result.summary_narrative)
        'Analyzed 2 numeric column(s) in Retail Metrics...'
        >>> for corr in result.correlations:
        ...     print(corr.narrative)
        'inventory and sales are strongly inverse (r=-1.00)'
    """
    column_results: dict[str, SemanticResult] = {}
    values_dict: dict[str, np.ndarray] = {}

    # Detect Polars vs Pandas
    module_name = type(df).__module__

    if "polars" in module_name:
        # Polars DataFrame
        for col_name in df.columns:
            dtype = df[col_name].dtype
            # Check if numeric (int, float types)
            if dtype.is_numeric():  # type: ignore[union-attr]
                col_context = f"{context} - {col_name}" if context else col_name
                values = df[col_name].to_numpy()
                values_dict[col_name] = values.astype(float)
                result = describe_series(
                    df[col_name],
                    context=col_context,
                    output="full",
                )
                column_results[col_name] = result  # type: ignore
    else:
        # Pandas DataFrame
        numeric_cols = df.select_dtypes(include=[np.number]).columns  # type: ignore
        for col_name in numeric_cols:
            col_context = f"{context} - {col_name}" if context else str(col_name)
            values = df[col_name].to_numpy()  # type: ignore
            values_dict[str(col_name)] = values.astype(float)
            result = describe_series(
                df[col_name],  # type: ignore
                context=col_context,
                output="full",
            )
            column_results[str(col_name)] = result  # type: ignore

    # Calculate correlations
    correlation_matrix = calc_correlation_matrix(values_dict)
    significant = identify_significant_correlations(
        correlation_matrix, threshold=correlation_threshold
    )

    # Build CorrelationInsight objects
    correlation_insights: list[CorrelationInsight] = []
    key_insights: list[str] = []

    for col_a, col_b, r, state in significant:
        narrative = generate_correlation_narrative(col_a, col_b, r, state)
        correlation_insights.append(
            CorrelationInsight(
                column_a=col_a,
                column_b=col_b,
                correlation=r,
                state=state,
                narrative=narrative,
            )
        )
        key_insights.append(narrative)

    # Generate summary narrative
    summary = generate_dataframe_summary(
        column_count=len(column_results),
        significant_correlations=len(correlation_insights),
        key_insights=key_insights,
        context=context,
    )

    return DataFrameResult(
        columns=column_results,
        correlations=tuple(correlation_insights),
        summary_narrative=summary,
    )


def compression_stats(
    original_data: ArrayLike,
    result: SemanticResult,
    use_real_tokenizer: bool = False,
) -> dict[str, Any]:
    """Calculate detailed compression statistics.

    Args:
        original_data: The original input data.
        result: The SemanticResult from describe_series.
        use_real_tokenizer: If True, use tiktoken for accurate token counts.
            Requires: pip install semantic-frame[validation]

    Returns:
        Dict with compression statistics including:
        - original_data_points: Number of data points
        - original_tokens_estimate: Estimated/actual tokens for original data
        - narrative_tokens: Tokens in the narrative
        - json_tokens: Tokens in the JSON output
        - narrative_compression_ratio: Compression ratio for narrative
        - json_compression_ratio: Compression ratio for JSON
        - tokenizer: "estimate" or "tiktoken" depending on method used
    """
    values = _to_numpy(original_data)

    if use_real_tokenizer:
        try:
            import json as json_module

            import tiktoken

            encoder = tiktoken.get_encoding("cl100k_base")

            # Count tokens for original data formatted as JSON array
            data_str = json_module.dumps(values.tolist())
            original_tokens = len(encoder.encode(data_str))

            # Count tokens for narrative
            narrative_tokens = len(encoder.encode(result.narrative))

            # Count tokens for JSON output
            json_str = result.to_json_str()
            json_tokens = len(encoder.encode(json_str))

            tokenizer = "tiktoken"

        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "tiktoken not available, falling back to estimate. "
                "Install with: pip install semantic-frame[validation]"
            )
            use_real_tokenizer = False

    if not use_real_tokenizer:
        # Estimate original token count (rough: 2 tokens per number)
        original_tokens = len(values) * 2

        # Narrative tokens (rough: 1 token per word)
        narrative_tokens = len(result.narrative.split())

        # JSON output tokens
        json_str = result.to_json_str()
        json_tokens = len(json_str.split())

        tokenizer = "estimate"

    return {
        "original_data_points": len(values),
        "original_tokens_estimate": original_tokens,
        "narrative_tokens": narrative_tokens,
        "json_tokens": json_tokens,
        "narrative_compression_ratio": 1 - (narrative_tokens / max(original_tokens, 1)),
        "json_compression_ratio": 1 - (json_tokens / max(original_tokens, 1)),
        "tokenizer": tokenizer,
    }

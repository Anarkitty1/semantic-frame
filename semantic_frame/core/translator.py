"""Translation layer: Maps raw analysis to semantic results.

This module orchestrates the full analysis pipeline:
1. Profile the data (basic statistics)
2. Run analyzers (trend, volatility, anomalies, etc.)
3. Generate narrative via narrators
4. Package into SemanticResult
"""

from __future__ import annotations

import logging

import numpy as np

from semantic_frame.core.analyzers import (
    assess_data_quality,
    calc_acceleration,
    calc_distribution_shape,
    calc_linear_slope,
    calc_seasonality,
    calc_volatility,
    classify_acceleration,
    classify_anomaly_state,
    classify_trend,
    detect_anomalies,
    detect_step_changes,
)
from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    StructuralChange,
    TrendState,
    VolatilityState,
)
from semantic_frame.interfaces.json_schema import SemanticResult, SeriesProfile
from semantic_frame.narrators.time_series import generate_time_series_narrative

logger = logging.getLogger(__name__)


def analyze_series(
    values: np.ndarray,
    context: str | None = None,
    is_time_series: bool = True,
) -> SemanticResult:
    """Run full analysis pipeline on a data series.

    Args:
        values: NumPy array of numerical values (may contain NaN or Inf).
        context: Optional context label for the data.
        is_time_series: If True, treat as ordered data (compute trend/seasonality).
                       If False, treat as distribution data.

    Returns:
        Complete SemanticResult with narrative and structured data.
    """
    # Handle edge case of empty array
    if len(values) == 0:
        return _empty_result(context)

    # Filter out NaN and Inf values for analysis
    # NaN and Inf would skew statistical calculations
    clean_mask = ~(np.isnan(values) | np.isinf(values))
    clean_values = values[clean_mask]

    # Log if Inf values were filtered
    inf_count = int(np.sum(np.isinf(values)))
    if inf_count > 0:
        logger.warning(
            "Filtered %d infinite values from input (context: %s)",
            inf_count,
            context or "unspecified",
        )

    if len(clean_values) == 0:
        return _empty_result(context)

    # Build profile
    profile = _build_profile(values, clean_values)

    # Run analyzers
    slope = calc_linear_slope(clean_values) if is_time_series else 0.0
    trend = classify_trend(slope)
    cv, volatility = calc_volatility(clean_values)
    anomalies = detect_anomalies(clean_values)
    anomaly_state = classify_anomaly_state(anomalies)
    _, data_quality = assess_data_quality(values)

    # Acceleration analysis (rate of change in trend)
    acceleration = None
    if is_time_series and len(clean_values) >= 5:
        accel_value = calc_acceleration(clean_values)
        acceleration = classify_acceleration(accel_value)

    # Optional analyses
    seasonality = None
    if is_time_series and len(clean_values) >= 10:
        _, seasonality = calc_seasonality(clean_values)

    distribution = None
    if len(clean_values) >= 4:
        distribution = calc_distribution_shape(clean_values)

    step_change = StructuralChange.NONE
    step_change_idx = None
    if is_time_series and len(clean_values) >= 10:  # Step change detection needs enough data
        step_change, step_change_idx = detect_step_changes(clean_values)

    # Generate narrative
    narrative = generate_time_series_narrative(
        trend=trend,
        volatility=volatility,
        anomaly_state=anomaly_state,
        anomalies=anomalies[:5],  # Limit to top 5
        profile=profile,
        context=context,
        data_quality=data_quality,
        seasonality=seasonality,
        step_change=step_change,
        step_change_index=step_change_idx,
        acceleration=acceleration,
    )

    # Calculate compression ratio
    # Rough estimate: each number = ~2 tokens, narrative = ~1 token per word
    original_tokens = len(values) * 2
    output_tokens = len(narrative.split())
    # Clamp to [0.0, 1.0] - 0.0 means no compression (narrative >= original)
    compression_ratio = max(0.0, min(1.0, 1.0 - (output_tokens / max(original_tokens, 1))))

    return SemanticResult(
        narrative=narrative,
        trend=trend,
        volatility=volatility,
        data_quality=data_quality,
        anomaly_state=anomaly_state,
        anomalies=tuple(anomalies[:5]),  # Convert to tuple for immutable result
        seasonality=seasonality,
        distribution=distribution,
        step_change=step_change,
        step_change_index=step_change_idx,
        acceleration=acceleration,
        profile=profile,
        context=context,
        compression_ratio=compression_ratio,
    )


def _build_profile(values: np.ndarray, clean_values: np.ndarray) -> SeriesProfile:
    """Build statistical profile from data.

    Args:
        values: Original array (may contain NaN/Inf).
        clean_values: Filtered array with only valid numeric values.

    Returns:
        SeriesProfile with computed statistics.
    """
    # Count both NaN and Inf as missing/invalid
    invalid_count: int = int(np.sum(np.isnan(values) | np.isinf(values)))
    missing_pct = float(invalid_count / len(values) * 100)

    return SeriesProfile(
        count=len(values),
        mean=float(np.mean(clean_values)),
        median=float(np.median(clean_values)),
        std=float(np.std(clean_values)),
        min=float(np.min(clean_values)),
        max=float(np.max(clean_values)),
        missing_pct=missing_pct,
    )


def _empty_result(context: str | None) -> SemanticResult:
    """Return a result for empty/all-NaN/all-Inf data."""
    # Import enums locally to avoid circular dependencies if SemanticResult is in the same module
    # and enums are used in its definition.
    # However, in this case, they are already imported at the top level.
    # Keeping this for consistency with the original code's structure.

    profile = SeriesProfile(
        count=0,
        mean=0.0,
        median=0.0,
        std=0.0,
        min=0.0,
        max=0.0,
        missing_pct=100.0,
    )

    ctx_name = context or "data"
    narrative = f"The {ctx_name} contains no valid data points."

    return SemanticResult(
        narrative=narrative,
        trend=TrendState.FLAT,
        volatility=VolatilityState.STABLE,
        data_quality=DataQuality.FRAGMENTED,
        anomaly_state=AnomalyState.NONE,
        anomalies=(),  # Empty tuple for immutable result
        seasonality=None,
        distribution=None,
        profile=profile,
        context=context,
        compression_ratio=1.0,
    )

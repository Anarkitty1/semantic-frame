"""Narrative generation for time-series (ordered) data.

This module generates natural language descriptions for data that has
a meaningful order or temporal component (e.g., stock prices, server logs,
sensor readings over time).
"""

from __future__ import annotations

from semantic_frame.core.enums import (
    AccelerationState,
    AnomalyState,
    DataQuality,
    SeasonalityState,
    StructuralChange,
    TrendState,
    VolatilityState,
)
from semantic_frame.interfaces.json_schema import AnomalyInfo, SeriesProfile

# Template strings for narrative construction
TEMPLATES = {
    "base": "The {context} data shows a {trend} pattern with {volatility} variability.",
    "anomaly_single": " 1 anomaly detected at index {position} (value: {value:.2f}).",
    "anomaly_multi": " {count} anomalies detected at indices {positions}.",
    "seasonality": " {seasonality} detected.",
    "quality_good": "",  # Don't mention if data quality is good
    "quality_bad": " Data quality is {quality} ({missing:.1f}% missing).",
    "stats": " Baseline: {median:.2f} (range: {min:.2f}-{max:.2f}).",
}


def generate_time_series_narrative(
    trend: TrendState,
    volatility: VolatilityState,
    anomaly_state: AnomalyState,
    anomalies: list[AnomalyInfo],
    profile: SeriesProfile,
    context: str | None = None,
    data_quality: DataQuality = DataQuality.PRISTINE,
    seasonality: SeasonalityState | None = None,
    step_change: StructuralChange | None = None,
    step_change_index: int | None = None,
    acceleration: AccelerationState | None = None,
) -> str:
    """Generate a natural language description for time series data.

    Args:
        trend: TrendState enum.
        volatility: VolatilityState enum.
        anomaly_state: AnomalyState enum.
        anomalies: List of detected anomalies.
        profile: Statistical profile.
        context: Optional context label.
        data_quality: DataQuality enum.
        seasonality: Optional SeasonalityState enum.
        step_change: Optional StructuralChange enum.
        step_change_index: Optional index of step change.
        acceleration: Optional AccelerationState enum.

    Returns:
        Semantic narrative string.
    """
    ctx = context or "time series"

    parts: list[str] = []

    # Base description
    parts.append(
        TEMPLATES["base"].format(
            context=ctx,
            trend=trend.value,
            volatility=volatility.value,
        )
    )

    # Anomaly information
    if anomalies:
        if len(anomalies) == 1:
            parts.append(
                TEMPLATES["anomaly_single"].format(
                    position=anomalies[0].index,
                    value=anomalies[0].value,
                )
            )
        else:
            # Show up to first 3 positions
            positions = ", ".join(str(a.index) for a in anomalies[:3])
            if len(anomalies) > 3:
                positions += f" (+{len(anomalies) - 3} more)"
            parts.append(
                TEMPLATES["anomaly_multi"].format(
                    count=len(anomalies),
                    positions=positions,
                )
            )

    # Acceleration (rate of change in trend) - only mention if not steady
    if acceleration and acceleration != AccelerationState.STEADY:
        parts.append(f" The trend is {acceleration.value}.")

    # Seasonality (if detected)
    if seasonality and seasonality != SeasonalityState.NONE:
        parts.append(f"A {seasonality.value} was detected.")

    # Step Change
    if step_change and step_change != StructuralChange.NONE:
        idx_str = f" at index {step_change_index}" if step_change_index is not None else ""
        parts.append(f"A significant {step_change.value} was detected{idx_str}.")

    # Data quality (only mention if poor)
    if data_quality and data_quality not in (DataQuality.PRISTINE, DataQuality.GOOD):
        parts.append(
            TEMPLATES["quality_bad"].format(
                quality=data_quality.value,
                missing=profile.missing_pct,
            )
        )

    # Statistics summary
    parts.append(
        TEMPLATES["stats"].format(
            median=profile.median,
            min=profile.min_val,
            max=profile.max_val,
        )
    )

    return "".join(parts)

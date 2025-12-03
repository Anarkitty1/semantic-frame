"""Narrative generation for time-series (ordered) data.

This module generates natural language descriptions for data that has
a meaningful order or temporal component (e.g., stock prices, server logs,
sensor readings over time).
"""

from __future__ import annotations

from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    SeasonalityState,
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
    data_quality: DataQuality | None = None,
    seasonality: SeasonalityState | None = None,
) -> str:
    """Generate natural language narrative for time-series data.

    Args:
        trend: Classified trend state.
        volatility: Classified volatility state.
        anomaly_state: Classified anomaly severity.
        anomalies: List of detected anomalies.
        profile: Statistical profile of the data.
        context: Optional context label (e.g., "CPU Usage", "Sales").
        data_quality: Optional data quality classification.
        seasonality: Optional seasonality classification.

    Returns:
        Human/LLM-readable narrative string.
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

    # Seasonality (if detected)
    if seasonality and seasonality != SeasonalityState.NONE:
        parts.append(TEMPLATES["seasonality"].format(seasonality=seasonality.value.capitalize()))

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

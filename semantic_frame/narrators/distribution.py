"""Narrative generation for distribution (unordered) data.

This module generates natural language descriptions for data without
meaningful order (e.g., test scores, ages, survey responses).
Focus is on distribution shape rather than trend.
"""

from __future__ import annotations

from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    DistributionShape,
    VolatilityState,
)
from semantic_frame.interfaces.json_schema import AnomalyInfo, SeriesProfile

# Template strings for distribution narrative
TEMPLATES = {
    "base": "The {context} data is {distribution} with {volatility} spread.",
    "center": " Central tendency: mean={mean:.2f}, median={median:.2f}.",
    "range": " Range: {min:.2f} to {max:.2f} (std={std:.2f}).",
    "anomaly_single": " 1 outlier detected (value: {value:.2f}, {zscore:.1f} std from mean).",
    "anomaly_multi": " {count} outliers detected (most extreme: {value:.2f}).",
    "skew_note": " Distribution shows {skew_direction} skew.",
    "quality_bad": " Note: {missing:.1f}% of values are missing.",
}


def generate_distribution_narrative(
    distribution: DistributionShape,
    volatility: VolatilityState,
    anomaly_state: AnomalyState,
    anomalies: list[AnomalyInfo],
    profile: SeriesProfile,
    context: str | None = None,
    data_quality: DataQuality | None = None,
) -> str:
    """Generate natural language narrative for distribution data.

    Args:
        distribution: Classified distribution shape.
        volatility: Classified volatility/spread state.
        anomaly_state: Classified anomaly severity.
        anomalies: List of detected anomalies.
        profile: Statistical profile of the data.
        context: Optional context label (e.g., "Test Scores", "Ages").
        data_quality: Optional data quality classification.

    Returns:
        Human/LLM-readable narrative string.
    """
    ctx = context or "dataset"

    parts: list[str] = []

    # Base distribution description
    parts.append(
        TEMPLATES["base"].format(
            context=ctx,
            distribution=distribution.value,
            volatility=volatility.value,
        )
    )

    # Central tendency
    parts.append(
        TEMPLATES["center"].format(
            mean=profile.mean,
            median=profile.median,
        )
    )

    # Range and spread
    parts.append(
        TEMPLATES["range"].format(
            min=profile.min_val,
            max=profile.max_val,
            std=profile.std,
        )
    )

    # Skewness note for non-normal distributions
    if distribution == DistributionShape.LEFT_SKEWED:
        parts.append(TEMPLATES["skew_note"].format(skew_direction="negative (left)"))
    elif distribution == DistributionShape.RIGHT_SKEWED:
        parts.append(TEMPLATES["skew_note"].format(skew_direction="positive (right)"))

    # Outlier information
    if anomalies:
        if len(anomalies) == 1:
            parts.append(
                TEMPLATES["anomaly_single"].format(
                    value=anomalies[0].value,
                    zscore=anomalies[0].z_score,
                )
            )
        else:
            # Report most extreme outlier
            most_extreme = max(anomalies, key=lambda a: a.z_score)
            parts.append(
                TEMPLATES["anomaly_multi"].format(
                    count=len(anomalies),
                    value=most_extreme.value,
                )
            )

    # Data quality warning
    if data_quality and data_quality not in (DataQuality.PRISTINE, DataQuality.GOOD):
        parts.append(TEMPLATES["quality_bad"].format(missing=profile.missing_pct))

    return "".join(parts)

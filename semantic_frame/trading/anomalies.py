"""Enhanced anomaly analysis for trading data.

This module provides trading-optimized anomaly detection with:
- Severity classification (mild, moderate, extreme)
- Type classification (gain, loss, spike, drop)
- Contextual descriptions for each anomaly
- Frequency assessment

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    """Classification of anomaly severity.

    Thresholds (based on z-score magnitude):
        - MILD: |z| 2.0 - 2.5
        - MODERATE: |z| 2.5 - 3.5
        - SEVERE: |z| 3.5 - 5.0
        - EXTREME: |z| >= 5.0
    """

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class AnomalyType(str, Enum):
    """Classification of anomaly type based on direction and context."""

    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    GAIN = "gain"  # Positive value anomaly (for PnL data)
    LOSS = "loss"  # Negative value anomaly (for PnL data)
    OUTLIER_HIGH = "outlier_high"  # Generic high outlier
    OUTLIER_LOW = "outlier_low"  # Generic low outlier


class AnomalyFrequency(str, Enum):
    """Classification of how often anomalies occur.

    Thresholds (anomalies per 100 data points):
        - RARE: < 1%
        - OCCASIONAL: 1-3%
        - FREQUENT: 3-5%
        - PERVASIVE: > 5%
    """

    RARE = "rare"
    OCCASIONAL = "occasional"
    FREQUENT = "frequent"
    PERVASIVE = "pervasive"


class EnhancedAnomaly(BaseModel):
    """Detailed information about a single anomaly."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Position in the data array")
    value: float = Field(description="The anomalous value")
    z_score: float = Field(description="Z-score (signed, indicates direction)")
    severity: AnomalySeverity = Field(description="Severity classification")
    anomaly_type: AnomalyType = Field(description="Type of anomaly")
    context: str = Field(description="Human-readable context description")
    deviation_multiple: float = Field(description="How many times larger than typical deviation")


class EnhancedAnomalyResult(BaseModel):
    """Complete enhanced anomaly analysis result."""

    model_config = ConfigDict(frozen=True)

    # Individual anomalies
    anomalies: tuple[EnhancedAnomaly, ...] = Field(
        default_factory=tuple, description="Detected anomalies with full context"
    )

    # Summary metrics
    total_anomalies: int = Field(ge=0, description="Total number of anomalies detected")
    frequency: AnomalyFrequency = Field(description="How often anomalies occur")
    max_severity: AnomalySeverity | None = Field(
        default=None, description="Highest severity detected"
    )

    # Statistics
    anomaly_rate_pct: float = Field(
        ge=0.0, le=100.0, description="Percentage of data points that are anomalies"
    )
    avg_z_score: float | None = Field(
        default=None, description="Average absolute z-score of anomalies"
    )

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable summary")

    # Metadata
    data_context: str | None = Field(default=None, description="User-provided context")


def _classify_severity(z_score: float) -> AnomalySeverity:
    """Classify anomaly severity based on z-score magnitude."""
    abs_z = abs(z_score)
    if abs_z >= 5.0:
        return AnomalySeverity.EXTREME
    if abs_z >= 3.5:
        return AnomalySeverity.SEVERE
    if abs_z >= 2.5:
        return AnomalySeverity.MODERATE
    return AnomalySeverity.MILD


def _classify_type(
    value: float,
    z_score: float,
    mean: float,
    is_pnl_data: bool = False,
) -> AnomalyType:
    """Classify anomaly type based on value and context."""
    if is_pnl_data:
        # For profit/loss data, use gain/loss terminology
        if value > 0:
            return AnomalyType.GAIN
        return AnomalyType.LOSS

    # For general data, use spike/drop or outlier terminology
    if z_score > 0:
        # Value is above mean
        if abs(z_score) > 3.0:
            return AnomalyType.SPIKE
        return AnomalyType.OUTLIER_HIGH
    else:
        # Value is below mean
        if abs(z_score) > 3.0:
            return AnomalyType.DROP
        return AnomalyType.OUTLIER_LOW


def _classify_frequency(anomaly_rate_pct: float) -> AnomalyFrequency:
    """Classify anomaly frequency based on rate."""
    if anomaly_rate_pct < 1.0:
        return AnomalyFrequency.RARE
    if anomaly_rate_pct < 3.0:
        return AnomalyFrequency.OCCASIONAL
    if anomaly_rate_pct < 5.0:
        return AnomalyFrequency.FREQUENT
    return AnomalyFrequency.PERVASIVE


def _generate_anomaly_context(
    value: float,
    z_score: float,
    severity: AnomalySeverity,
    anomaly_type: AnomalyType,
    mean: float,
    std: float,
    rank_info: str | None = None,
) -> str:
    """Generate contextual description for an anomaly."""
    parts: list[str] = []

    # Deviation description
    deviation_mult = abs(value - mean) / std if std > 0 else 0
    parts.append(f"{deviation_mult:.1f}x typical deviation")

    # Type-specific context
    if anomaly_type == AnomalyType.GAIN:
        parts.append("exceptional profit")
    elif anomaly_type == AnomalyType.LOSS:
        parts.append("significant loss")
    elif anomaly_type == AnomalyType.SPIKE:
        parts.append("sudden spike")
    elif anomaly_type == AnomalyType.DROP:
        parts.append("sudden drop")

    # Rank info if provided
    if rank_info:
        parts.append(rank_info)

    return ", ".join(parts)


def _generate_narrative(
    anomalies: list[EnhancedAnomaly],
    frequency: AnomalyFrequency,
    max_severity: AnomalySeverity | None,
    data_context: str | None,
    data_len: int,
) -> str:
    """Generate natural language narrative for anomaly analysis."""
    parts: list[str] = []

    prefix = f"The {data_context}" if data_context else "The data"

    if not anomalies:
        parts.append(
            f"{prefix} contains no significant anomalies - values are within normal bounds."
        )
        return " ".join(parts)

    # Count by severity
    severity_counts: dict[AnomalySeverity, int] = {}
    for a in anomalies:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    # Opening statement
    freq_desc = {
        AnomalyFrequency.RARE: "rare",
        AnomalyFrequency.OCCASIONAL: "occasional",
        AnomalyFrequency.FREQUENT: "frequent",
        AnomalyFrequency.PERVASIVE: "pervasive",
    }

    parts.append(
        f"{prefix} has {freq_desc[frequency]} anomalies "
        f"({len(anomalies)} detected in {data_len} points)."
    )

    # Severity breakdown
    if max_severity in [AnomalySeverity.SEVERE, AnomalySeverity.EXTREME]:
        extreme_count = severity_counts.get(AnomalySeverity.EXTREME, 0)
        severe_count = severity_counts.get(AnomalySeverity.SEVERE, 0)
        if extreme_count > 0:
            parts.append(f"{extreme_count} extreme outlier(s) detected - investigate immediately.")
        elif severe_count > 0:
            parts.append(f"{severe_count} severe outlier(s) require attention.")

    # Top anomalies detail
    if anomalies:
        # Sort by severity (extreme first) then by z-score magnitude
        sorted_anomalies = sorted(
            anomalies,
            key=lambda a: (
                -[
                    AnomalySeverity.EXTREME,
                    AnomalySeverity.SEVERE,
                    AnomalySeverity.MODERATE,
                    AnomalySeverity.MILD,
                ].index(a.severity),
                -abs(a.z_score),
            ),
        )
        top = sorted_anomalies[0]
        parts.append(
            f"Most significant: index {top.index} (value: {top.value:.2f}, "
            f"z-score: {top.z_score:.1f}, {top.context})."
        )

    return " ".join(parts)


def describe_anomalies(
    data: np.ndarray | list[float],
    context: str | None = None,
    is_pnl_data: bool = False,
    z_threshold: float = 2.0,
    max_anomalies: int = 20,
) -> EnhancedAnomalyResult:
    """Perform enhanced anomaly detection with severity and type classification.

    Analyzes data for outliers and provides rich context including severity,
    type classification, and human-readable descriptions for each anomaly.

    Args:
        data: Array of numerical values to analyze.
        context: Optional label (e.g., "Trade PnL", "CPU Usage").
        is_pnl_data: If True, uses gain/loss terminology instead of spike/drop.
        z_threshold: Z-score threshold for anomaly detection (default 2.0).
        max_anomalies: Maximum number of anomalies to return (default 20).

    Returns:
        EnhancedAnomalyResult with detailed anomaly information.

    Example:
        >>> data = [100, 102, 99, 101, 500, 100, 98, -200]
        >>> result = describe_anomalies(data, context="Trade Returns", is_pnl_data=True)
        >>> print(result.narrative)
        "The Trade Returns has occasional anomalies (2 detected in 8 points).
         Most significant: index 4 (value: 500.00, z-score: 2.8, exceptional profit)."
    """
    # Convert to numpy array (handles both list and ndarray input)
    arr: np.ndarray = np.asarray(data, dtype=float)

    # Filter out non-finite values (inf, -inf, nan)
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = arr[finite_mask]

    n = len(arr)

    if n < 3:
        return EnhancedAnomalyResult(
            anomalies=(),
            total_anomalies=0,
            frequency=AnomalyFrequency.RARE,
            max_severity=None,
            anomaly_rate_pct=0.0,
            avg_z_score=None,
            narrative="Insufficient data for anomaly analysis.",
            data_context=context,
        )

    # Calculate statistics
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    if std == 0:
        return EnhancedAnomalyResult(
            anomalies=(),
            total_anomalies=0,
            frequency=AnomalyFrequency.RARE,
            max_severity=None,
            anomaly_rate_pct=0.0,
            avg_z_score=None,
            narrative=f"The {context or 'data'} has zero variance - all values are identical.",
            data_context=context,
        )

    # Calculate z-scores
    z_scores = (arr - mean) / std

    # Find anomalies
    anomaly_indices = np.where(np.abs(z_scores) >= z_threshold)[0]

    # Build enhanced anomaly objects
    enhanced_anomalies: list[EnhancedAnomaly] = []

    # Sort indices by z-score magnitude for ranking
    sorted_indices = sorted(anomaly_indices, key=lambda i: abs(z_scores[i]), reverse=True)

    for rank, idx in enumerate(sorted_indices[:max_anomalies]):
        value = float(arr[idx])
        z = float(z_scores[idx])
        severity = _classify_severity(z)
        atype = _classify_type(value, z, mean, is_pnl_data)

        rank_info = None
        if rank == 0:
            rank_info = "largest outlier"
        elif rank == 1:
            rank_info = "2nd largest"
        elif rank == 2:
            rank_info = "3rd largest"

        context_str = _generate_anomaly_context(value, z, severity, atype, mean, std, rank_info)

        enhanced_anomalies.append(
            EnhancedAnomaly(
                index=int(idx),
                value=value,
                z_score=round(z, 2),
                severity=severity,
                anomaly_type=atype,
                context=context_str,
                deviation_multiple=round(abs(value - mean) / std, 2),
            )
        )

    # Calculate summary metrics
    total = len(anomaly_indices)
    anomaly_rate = (total / n) * 100 if n > 0 else 0.0
    frequency = _classify_frequency(anomaly_rate)
    max_severity = max(
        (a.severity for a in enhanced_anomalies),
        key=lambda s: [
            AnomalySeverity.MILD,
            AnomalySeverity.MODERATE,
            AnomalySeverity.SEVERE,
            AnomalySeverity.EXTREME,
        ].index(s),
        default=None,
    )
    avg_z = float(np.mean(np.abs(z_scores[anomaly_indices]))) if total > 0 else None

    # Generate narrative
    narrative = _generate_narrative(enhanced_anomalies, frequency, max_severity, context, n)

    return EnhancedAnomalyResult(
        anomalies=tuple(enhanced_anomalies),
        total_anomalies=total,
        frequency=frequency,
        max_severity=max_severity,
        anomaly_rate_pct=round(anomaly_rate, 2),
        avg_z_score=round(avg_z, 2) if avg_z else None,
        narrative=narrative,
        data_context=context,
    )

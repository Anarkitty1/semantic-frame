"""Mathematical analysis functions for data profiling.

This module contains the core statistical functions that power the semantic
analysis. All calculations are deterministic (NumPy-based) - no LLM involvement.

Key functions:
- calc_linear_slope: Trend direction analysis
- classify_trend: Classify trend based on normalized slope
- calc_volatility: Variability measurement using coefficient of variation
- detect_anomalies: Outlier detection with adaptive IQR/Z-score methods
- classify_anomaly_state: Classify anomaly severity
- assess_data_quality: Assess data completeness
- calc_seasonality: Cyclic pattern detection via autocorrelation
- calc_distribution_shape: Distribution classification via skewness/kurtosis
- calc_acceleration: Rate of change analysis (second derivative)
- classify_acceleration: Classify acceleration based on normalized second derivative
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import kurtosis, pearsonr, skew

from semantic_frame.core.enums import (
    AccelerationState,
    AnomalyState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
    StructuralChange,
    TrendState,
    VolatilityState,
)
from semantic_frame.interfaces.json_schema import AnomalyInfo

logger = logging.getLogger(__name__)


def calc_linear_slope(values: np.ndarray) -> float:
    """Calculate normalized linear regression slope.

    The slope is normalized by data range and length to make it
    scale-independent. This allows consistent threshold comparisons
    across datasets with different magnitudes.

    Args:
        values: NumPy array of numerical values.

    Returns:
        Normalized slope value. Positive = upward trend, negative = downward.
    """
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))
    # Use polyfit for simple linear regression
    slope = np.polyfit(x, values, 1)[0]

    # Normalize by data range to make scale-independent
    data_range = float(np.ptp(values))
    if data_range == 0:
        return 0.0

    return float(slope * len(values) / data_range)


def classify_trend(normalized_slope: float) -> TrendState:
    """Classify trend based on normalized slope value.

    Thresholds:
        - > 0.5: RISING_SHARP
        - > 0.1: RISING_STEADY
        - < -0.5: FALLING_SHARP
        - < -0.1: FALLING_STEADY
        - else: FLAT

    Args:
        normalized_slope: Output from calc_linear_slope.

    Returns:
        TrendState enum value.
    """
    if normalized_slope > 0.5:
        return TrendState.RISING_SHARP
    if normalized_slope > 0.1:
        return TrendState.RISING_STEADY
    if normalized_slope < -0.5:
        return TrendState.FALLING_SHARP
    if normalized_slope < -0.1:
        return TrendState.FALLING_STEADY
    return TrendState.FLAT


def calc_volatility(values: np.ndarray) -> tuple[float, VolatilityState]:
    """Calculate volatility using coefficient of variation.

    CV = std / |mean|, providing a scale-independent measure of variability.

    Thresholds:
        - < 0.05: COMPRESSED
        - < 0.15: STABLE
        - < 0.30: MODERATE
        - < 0.50: EXPANDING
        - >= 0.50: EXTREME

    Args:
        values: NumPy array of numerical values.

    Returns:
        Tuple of (coefficient_of_variation, VolatilityState).
    """
    if len(values) == 0:
        return 0.0, VolatilityState.STABLE

    mean = float(np.mean(values))
    if mean == 0:
        # Handle zero mean - use std relative to data range
        data_range = float(np.ptp(values))
        if data_range == 0:
            return 0.0, VolatilityState.COMPRESSED
        cv = float(np.std(values) / data_range)
    else:
        cv = float(np.std(values) / abs(mean))

    if cv < 0.05:
        return cv, VolatilityState.COMPRESSED
    if cv < 0.15:
        return cv, VolatilityState.STABLE
    if cv < 0.30:
        return cv, VolatilityState.MODERATE
    if cv < 0.50:
        return cv, VolatilityState.EXPANDING
    return cv, VolatilityState.EXTREME


def detect_anomalies(values: np.ndarray, z_threshold: float = 3.0) -> list[AnomalyInfo]:
    """Detect anomalies using adaptive method based on sample size.

    The threshold of 10 samples balances:
    - IQR method: More robust to outliers with small samples, but loses
      precision with few data points for quartile estimation.
    - Z-score method: Requires sufficient samples for stable mean/std
      estimates; unreliable below ~10 samples.

    Args:
        values: NumPy array of numerical values.
        z_threshold: Z-score threshold for large sample detection (default 3.0).
            Must be positive.

    Returns:
        List of AnomalyInfo objects for detected outliers.

    Raises:
        ValueError: If z_threshold is not positive.
    """
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}")

    if len(values) < 3:
        return []

    if len(values) < 10:
        # IQR method for small samples - more robust with limited data
        return _detect_anomalies_iqr(values)

    return _detect_anomalies_zscore(values, z_threshold)


def _detect_anomalies_iqr(values: np.ndarray) -> list[AnomalyInfo]:
    """Detect anomalies using Interquartile Range method."""
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    iqr = q3 - q1
    median = float(np.median(values))

    # When IQR is 0 (most values are identical), use MAD-like approach
    # Detect values that deviate significantly from the median
    if iqr == 0:
        deviations = np.abs(values - median)
        max_dev = float(np.max(deviations))
        if max_dev == 0:
            return []  # All values truly identical

        # When IQR=0, use 50% of max deviation as threshold.
        # This is a heuristic: values in the outer half of the deviation
        # range are considered anomalous in the absence of quartile spread.
        threshold = max_dev * 0.5
        anomalies: list[AnomalyInfo] = []
        for i, val in enumerate(values):
            dev = abs(val - median)
            if dev > threshold:
                # Approximate z-score: divide by (max_dev/3) to roughly map
                # to ~3 sigma equivalent for the most extreme values
                z_approx = dev / (max_dev / 3) if max_dev > 0 else 0.0
                anomalies.append(AnomalyInfo(index=i, value=float(val), z_score=float(z_approx)))
        return sorted(anomalies, key=lambda a: a.z_score, reverse=True)

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    result_anomalies: list[AnomalyInfo] = []
    for i, val in enumerate(values):
        if val < lower_bound or val > upper_bound:
            # For normal distributions, IQR = 1.349 * std (from quartile z-scores).
            # Using this relationship to estimate z-score from IQR-based deviation.
            z_approx = abs(val - median) / (iqr / 1.35) if iqr > 0 else 0.0
            result_anomalies.append(AnomalyInfo(index=i, value=float(val), z_score=float(z_approx)))

    return sorted(result_anomalies, key=lambda a: a.z_score, reverse=True)


def _detect_anomalies_zscore(values: np.ndarray, threshold: float) -> list[AnomalyInfo]:
    """Detect anomalies using Z-score method."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    median = float(np.median(values))

    # When std is very low (near-identical values with outliers),
    # fall back to deviation-based detection
    if std == 0:
        deviations = np.abs(values - median)
        max_dev = float(np.max(deviations))
        if max_dev == 0:
            return []

        # Flag values that deviate significantly
        threshold_dev = max_dev * 0.5
        anomalies: list[AnomalyInfo] = []
        for i, val in enumerate(values):
            dev = abs(val - median)
            if dev > threshold_dev:
                z_approx = dev / (max_dev / 3) if max_dev > 0 else 0.0
                anomalies.append(AnomalyInfo(index=i, value=float(val), z_score=float(z_approx)))
        return sorted(anomalies, key=lambda a: a.z_score, reverse=True)

    result_anomalies: list[AnomalyInfo] = []
    for i, val in enumerate(values):
        z_score = abs(val - mean) / std
        if z_score >= threshold:
            result_anomalies.append(AnomalyInfo(index=i, value=float(val), z_score=float(z_score)))

    return sorted(result_anomalies, key=lambda a: a.z_score, reverse=True)


def classify_anomaly_state(anomalies: list[AnomalyInfo]) -> AnomalyState:
    """Classify anomaly severity based on count and z-scores.

    Args:
        anomalies: List of detected anomalies.

    Returns:
        AnomalyState classification.
    """
    if not anomalies:
        return AnomalyState.NONE

    # Check for extreme z-scores
    max_z = max(a.z_score for a in anomalies)
    if max_z > 5.0:
        return AnomalyState.EXTREME

    count = len(anomalies)
    if count > 5:
        return AnomalyState.EXTREME
    if count >= 3:
        return AnomalyState.SIGNIFICANT
    return AnomalyState.MINOR


def assess_data_quality(values: np.ndarray) -> tuple[float, DataQuality]:
    """Assess data quality based on missing value percentage.

    Args:
        values: NumPy array (may contain NaN values).

    Returns:
        Tuple of (missing_percentage, DataQuality).
    """
    if len(values) == 0:
        return 100.0, DataQuality.FRAGMENTED

    missing_pct = float(np.sum(np.isnan(values)) / len(values) * 100)

    if missing_pct < 1:
        return missing_pct, DataQuality.PRISTINE
    if missing_pct < 5:
        return missing_pct, DataQuality.GOOD
    if missing_pct < 20:
        return missing_pct, DataQuality.SPARSE
    return missing_pct, DataQuality.FRAGMENTED


def calc_distribution_shape(values: np.ndarray) -> DistributionShape:
    """Analyze distribution shape using skewness and kurtosis.

    Thresholds:
        - UNIFORM: kurtosis < -1.2 and |skewness| < 0.3
        - BIMODAL: kurtosis < -1 (flat-topped distribution)
        - NORMAL: |skewness| < 0.5
        - LEFT_SKEWED: skewness < -0.5
        - RIGHT_SKEWED: skewness > 0.5

    Args:
        values: NumPy array of numerical values (no NaN).

    Returns:
        DistributionShape classification.
    """
    if len(values) < 4:
        return DistributionShape.NORMAL

    # Check for constant data (all same values) - skip scipy to avoid warnings
    if np.ptp(values) == 0:
        return DistributionShape.NORMAL

    # Check for near-constant data (numerical precision issues)
    # If the coefficient of variation is extremely low, the values are
    # effectively identical and scipy will produce spurious kurtosis/skewness
    mean_val = abs(float(np.mean(values)))
    std_val = float(np.std(values))
    if mean_val > 0:
        cv = std_val / mean_val
        # CV < 1e-10 means values differ by less than 1 part in 10 billion
        # relative to the mean - effectively constant for statistical purposes
        if cv < 1e-10:
            return DistributionShape.NORMAL

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = float(skew(values))
            k = float(kurtosis(values))
    except (ValueError, FloatingPointError, RuntimeWarning) as e:
        # scipy failed on this data - log and return safe default
        logger.debug(
            "Distribution shape calculation failed for array of length %d: %s",
            len(values),
            str(e),
        )
        return DistributionShape.NORMAL

    # Handle NaN results from near-constant data
    if np.isnan(s) or np.isnan(k):
        logger.debug(
            "Distribution calculation returned NaN (skew=%s, kurtosis=%s), defaulting to NORMAL",
            s,
            k,
        )
        return DistributionShape.NORMAL

    # Check for uniform first (more restrictive: very negative kurtosis AND low skewness)
    # Uniform distributions have kurtosis ≈ -1.2 and near-zero skewness
    if k < -1.2 and abs(s) < 0.3:
        return DistributionShape.UNIFORM

    # Check for bimodality (negative excess kurtosis suggests flatter/bimodal)
    # Note: True bimodality requires different analysis; this detects flat-topped distributions
    if k < -1:
        return DistributionShape.BIMODAL

    # Classify by skewness
    if abs(s) < 0.5:
        return DistributionShape.NORMAL
    if s < -0.5:
        return DistributionShape.LEFT_SKEWED
    return DistributionShape.RIGHT_SKEWED


def calc_seasonality(values: np.ndarray, max_lag: int = 30) -> tuple[float, SeasonalityState]:
    """Detect seasonality using autocorrelation analysis on detrended data.

    Calculates autocorrelation at various lags to detect cyclic patterns.
    Data is first detrended (linear trend removed) to avoid false positives
    from monotonically increasing/decreasing series.

    Thresholds (peak autocorrelation):
        - NONE: < 0.3
        - WEAK: 0.3 - 0.5
        - MODERATE: 0.5 - 0.7
        - STRONG: >= 0.7

    Args:
        values: NumPy array of time-ordered values.
        max_lag: Maximum lag to check (default 30).

    Returns:
        Tuple of (peak_autocorrelation, SeasonalityState).
    """
    n = len(values)

    if n < 4:
        return 0.0, SeasonalityState.NONE

    # Check for constant data - no seasonality possible
    if np.ptp(values) == 0:
        return 0.0, SeasonalityState.NONE

    # Adjust max_lag for short series
    effective_max_lag = min(max_lag, n // 2)
    if effective_max_lag < 2:
        return 0.0, SeasonalityState.NONE

    # Detrend data: remove linear trend to avoid false positives from
    # monotonically increasing/decreasing series (which have high
    # autocorrelation at all lags but no cyclic pattern)
    x = np.arange(n)
    coeffs = np.polyfit(x, values, 1)
    detrended = values - np.polyval(coeffs, x)

    # Check if detrended data has any meaningful variance
    # After removing linear trend, pure linear data will have near-zero residuals
    # (only numerical noise). We use coefficient of variation to check.
    detrended_std = float(np.std(detrended))
    detrended_mean = abs(float(np.mean(values)))  # Use original mean for scale
    if detrended_mean > 0:
        cv = detrended_std / detrended_mean
        if cv < 1e-10:  # Effectively no variance relative to data scale
            return 0.0, SeasonalityState.NONE
    elif detrended_std < 1e-10:  # Near-zero data with near-zero residuals
        return 0.0, SeasonalityState.NONE

    # Calculate autocorrelation at different lags on detrended data
    autocorrs: list[float] = []
    logged_error = False
    for lag in range(1, effective_max_lag):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = pearsonr(detrended[:-lag], detrended[lag:])
            if not np.isnan(corr):
                autocorrs.append(abs(float(corr)))
        except (ValueError, FloatingPointError) as e:
            # Log first failure, then continue silently
            if not logged_error:
                logger.debug(
                    "Seasonality calculation had errors at lag %d: %s "
                    "(continuing with remaining lags)",
                    lag,
                    str(e),
                )
                logged_error = True
            continue

    if not autocorrs:
        return 0.0, SeasonalityState.NONE

    peak_autocorr = max(autocorrs)

    if peak_autocorr < 0.3:
        return peak_autocorr, SeasonalityState.NONE
    if peak_autocorr < 0.5:
        return peak_autocorr, SeasonalityState.WEAK
    if peak_autocorr < 0.7:
        return peak_autocorr, SeasonalityState.MODERATE
    return peak_autocorr, SeasonalityState.STRONG


def detect_step_changes(
    values: np.ndarray, window_size: int = 10, threshold: float = 2.0
) -> tuple[StructuralChange, int | None]:
    """Detect significant step changes in the data baseline.

    Uses a sliding window approach to compare means before and after each point.
    A step change is detected if the difference in means exceeds the threshold
    (measured in standard deviations of the entire series).

    Args:
        values: NumPy array of time-ordered values.
        window_size: Size of the window to compare means (default 10).
        threshold: Z-score threshold for mean difference (default 2.0).

    Returns:
        Tuple of (StructuralChange, index_of_change).
        index_of_change is None if StructuralChange is NONE.
    """
    n = len(values)
    if n < window_size * 2:
        return StructuralChange.NONE, None

    # Calculate global std for thresholding
    global_std = float(np.std(values))
    if global_std == 0:
        return StructuralChange.NONE, None

    max_diff = 0.0
    change_idx = -1
    change_type = StructuralChange.NONE

    # Slide window through the data
    # We need window_size points before and after 'i'
    for i in range(window_size, n - window_size):
        before = values[i - window_size : i]
        after = values[i : i + window_size]

        mean_before = np.mean(before)
        mean_after = np.mean(after)

        diff = mean_after - mean_before
        z_score = abs(diff) / global_std

        if z_score >= threshold and z_score > max_diff:
            max_diff = z_score
            change_idx = i
            if diff > 0:
                change_type = StructuralChange.STEP_UP
            else:
                change_type = StructuralChange.STEP_DOWN

    if change_type != StructuralChange.NONE:
        return change_type, change_idx

    return StructuralChange.NONE, None


def calc_acceleration(values: np.ndarray) -> float:
    """Calculate normalized acceleration (second derivative) of the data.

    Uses second-order polynomial fitting to estimate the rate of change
    of the trend (acceleration/deceleration). This complements trend
    analysis by capturing how the trend itself is changing.

    The acceleration is normalized by data range and length squared to make
    it scale-independent, similar to slope normalization.

    Args:
        values: NumPy array of numerical values.

    Returns:
        Normalized second derivative value.
        Positive = trend is speeding up (accelerating growth or steepening decline)
        Negative = trend is slowing down (decelerating)
    """
    if len(values) < 3:
        return 0.0

    # Check for constant data
    data_range = float(np.ptp(values))
    if data_range == 0:
        return 0.0

    x = np.arange(len(values))

    # Fit second-order polynomial: ax^2 + bx + c
    # The coefficient 'a' represents the curvature (second derivative / 2)
    try:
        coeffs = np.polyfit(x, values, 2)
        # Second derivative of ax^2 + bx + c is 2a
        second_derivative = 2 * coeffs[0]

        # Normalize by data range and length squared for scale independence
        # This ensures similar thresholds work across different data magnitudes
        n = len(values)
        normalized = float(second_derivative * (n**2) / data_range)

        return normalized
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def classify_acceleration(normalized_acceleration: float) -> AccelerationState:
    """Classify acceleration based on normalized second derivative.

    Thresholds (5 states to match TrendState granularity):
        - > 0.3: ACCELERATING_SHARPLY
        - > 0.1: ACCELERATING
        - < -0.3: DECELERATING_SHARPLY
        - < -0.1: DECELERATING
        - else: STEADY

    Args:
        normalized_acceleration: Output from calc_acceleration.

    Returns:
        AccelerationState enum value.
    """
    if normalized_acceleration > 0.3:
        return AccelerationState.ACCELERATING_SHARPLY
    if normalized_acceleration > 0.1:
        return AccelerationState.ACCELERATING
    if normalized_acceleration < -0.3:
        return AccelerationState.DECELERATING_SHARPLY
    if normalized_acceleration < -0.1:
        return AccelerationState.DECELERATING
    return AccelerationState.STEADY

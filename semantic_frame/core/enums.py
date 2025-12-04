"""Semantic vocabulary enums for data classification.

These enums define the "dictionary" that translates mathematical analysis
into natural language descriptions for LLM consumption.
"""

from enum import Enum


class TrendState(str, Enum):
    """Classification of data trend direction and intensity.

    Thresholds (normalized slope):
        - RISING_SHARP: slope > 0.5
        - RISING_STEADY: slope > 0.1
        - FLAT: -0.1 <= slope <= 0.1
        - FALLING_STEADY: slope < -0.1
        - FALLING_SHARP: slope < -0.5
    """

    RISING_SHARP = "rapidly rising"
    RISING_STEADY = "steadily rising"
    FLAT = "flat/stationary"
    FALLING_STEADY = "steadily falling"
    FALLING_SHARP = "rapidly falling"


class VolatilityState(str, Enum):
    """Classification of data variability using coefficient of variation.

    Thresholds:
        - COMPRESSED: CV < 0.05 (very low variance)
        - STABLE: CV 0.05-0.15 (normal variance)
        - MODERATE: CV 0.15-0.30 (elevated variance)
        - EXPANDING: CV 0.30-0.50 (high variance)
        - EXTREME: CV > 0.50 (very high variance)
    """

    COMPRESSED = "compressed"
    STABLE = "stable"
    MODERATE = "moderate"
    EXPANDING = "expanding"
    EXTREME = "extreme"


class DataQuality(str, Enum):
    """Classification of data completeness based on missing values.

    Thresholds:
        - PRISTINE: <1% missing
        - GOOD: 1-5% missing
        - SPARSE: 5-20% missing
        - FRAGMENTED: >20% missing
    """

    PRISTINE = "high quality"
    GOOD = "good quality"
    SPARSE = "sparse"
    FRAGMENTED = "fragmented"


class AnomalyState(str, Enum):
    """Classification of outlier presence and severity.

    Thresholds:
        - NONE: No outliers detected
        - MINOR: 1-2 outliers
        - SIGNIFICANT: 3-5 outliers
        - EXTREME: >5 outliers or any with z-score >5
    """

    NONE = "no anomalies"
    MINOR = "minor outliers"
    SIGNIFICANT = "significant outliers"
    EXTREME = "extreme outliers"


class SeasonalityState(str, Enum):
    """Classification of cyclic patterns via autocorrelation.

    Thresholds (peak autocorrelation):
        - NONE: <0.3
        - WEAK: 0.3-0.5
        - MODERATE: 0.5-0.7
        - STRONG: >0.7
    """

    NONE = "no seasonality"
    WEAK = "weak cyclic pattern"
    MODERATE = "moderate seasonality"
    STRONG = "strong seasonality"


class DistributionShape(str, Enum):
    """Classification of data distribution shape.

    Based on skewness and kurtosis analysis.

    Thresholds:
        - UNIFORM: kurtosis < -1.2 and |skewness| < 0.3
        - BIMODAL: kurtosis < -1 (flat-topped distribution)
        - NORMAL: |skewness| < 0.5
        - LEFT_SKEWED: skewness < -0.5
        - RIGHT_SKEWED: skewness > 0.5

    Note: BIMODAL detection is heuristic based on kurtosis; true bimodality
    requires more sophisticated analysis (e.g., Hartigan's dip test).
    """

    NORMAL = "normally distributed"
    LEFT_SKEWED = "left-skewed"
    RIGHT_SKEWED = "right-skewed"
    BIMODAL = "bimodal"
    UNIFORM = "uniformly distributed"


class CorrelationState(str, Enum):
    """Classification of correlation strength between two variables.

    Thresholds (Pearson r):
        - STRONG_POSITIVE: r > 0.7
        - MODERATE_POSITIVE: 0.4 < r <= 0.7
        - WEAK: |r| <= 0.4
        - MODERATE_NEGATIVE: -0.7 <= r < -0.4
        - STRONG_NEGATIVE: r < -0.7
    """

    STRONG_POSITIVE = "strongly correlated"
    MODERATE_POSITIVE = "moderately correlated"
    WEAK = "weakly related"
    MODERATE_NEGATIVE = "inversely related"
    STRONG_NEGATIVE = "strongly inverse"


class StructuralChange(str, Enum):
    """Classification of structural baseline shifts in time series data.

    Detected using step change analysis.

    Types:
        - NONE: No significant baseline shift
        - STEP_UP: Sudden increase in baseline mean
        - STEP_DOWN: Sudden decrease in baseline mean
    """

    NONE = "no step change"
    STEP_UP = "step up"
    STEP_DOWN = "step down"


class AccelerationState(str, Enum):
    """Classification of rate of change in trend (second derivative).

    Measures whether the trend is speeding up, slowing down, or constant.
    This complements TrendState by capturing not just direction but how
    that direction is changing over time.

    Thresholds (normalized second derivative):
        - ACCELERATING_SHARPLY: second_derivative > 0.3
        - ACCELERATING: 0.1 < second_derivative <= 0.3
        - STEADY: -0.1 <= second_derivative <= 0.1
        - DECELERATING: -0.3 <= second_derivative < -0.1
        - DECELERATING_SHARPLY: second_derivative < -0.3
    """

    ACCELERATING_SHARPLY = "rapidly accelerating"
    ACCELERATING = "accelerating"
    STEADY = "steady rate of change"
    DECELERATING = "decelerating"
    DECELERATING_SHARPLY = "rapidly decelerating"

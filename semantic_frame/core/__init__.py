"""Core analysis components."""

from semantic_frame.core.analyzers import (
    assess_data_quality,
    calc_distribution_shape,
    calc_linear_slope,
    calc_seasonality,
    calc_volatility,
    classify_anomaly_state,
    classify_trend,
    detect_anomalies,
)
from semantic_frame.core.correlations import (
    calc_correlation_matrix,
    classify_correlation,
    identify_significant_correlations,
)
from semantic_frame.core.enums import (
    AnomalyState,
    CorrelationState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
    TrendState,
    VolatilityState,
)
from semantic_frame.core.translator import analyze_series

__all__ = [
    # Enums
    "TrendState",
    "VolatilityState",
    "DataQuality",
    "AnomalyState",
    "SeasonalityState",
    "DistributionShape",
    "CorrelationState",
    # Analyzers
    "calc_linear_slope",
    "classify_trend",
    "calc_volatility",
    "detect_anomalies",
    "classify_anomaly_state",
    "assess_data_quality",
    "calc_distribution_shape",
    "calc_seasonality",
    # Correlation Analyzers
    "classify_correlation",
    "calc_correlation_matrix",
    "identify_significant_correlations",
    # Translator
    "analyze_series",
]

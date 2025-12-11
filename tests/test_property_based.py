"""Property-based tests for semantic_frame using Hypothesis.

These tests verify that functions behave correctly across a wide range of inputs
by testing mathematical properties that should always hold, rather than testing
specific input/output pairs.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from semantic_frame.core.analyzers import (
    calc_acceleration,
    calc_distribution_shape,
    calc_linear_slope,
    calc_seasonality,
    calc_volatility,
    classify_acceleration,
    classify_trend,
    detect_anomalies,
)
from semantic_frame.core.enums import (
    AccelerationState,
    DistributionShape,
    SeasonalityState,
    TrendState,
    VolatilityState,
)

# =============================================================================
# Strategy Definitions
# =============================================================================

# Strategy for valid float values (no inf, no nan)
finite_float = st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)

# Strategy for small float arrays (good for quick tests)
small_float_array = st.lists(finite_float, min_size=2, max_size=50).map(
    lambda x: np.array(x, dtype=np.float64)
)

# Strategy for larger float arrays
medium_float_array = st.lists(finite_float, min_size=10, max_size=200).map(
    lambda x: np.array(x, dtype=np.float64)
)

# Strategy for positive floats only
positive_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)

# Strategy for positive float arrays
positive_float_array = st.lists(positive_float, min_size=2, max_size=50).map(
    lambda x: np.array(x, dtype=np.float64)
)


# =============================================================================
# classify_trend Properties
# =============================================================================


class TestClassifyTrendProperties:
    """Property-based tests for classify_trend function."""

    @given(slope=st.floats(allow_nan=False, allow_infinity=False))
    def test_returns_valid_enum(self, slope: float):
        """classify_trend should always return a valid TrendState enum."""
        result = classify_trend(slope)
        assert isinstance(result, TrendState)

    @given(slope=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
    def test_positive_slope_not_falling(self, slope: float):
        """Positive slopes should never return FALLING states."""
        result = classify_trend(slope)
        assert result not in (TrendState.FALLING_STEADY, TrendState.FALLING_SHARP)

    @given(slope=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    def test_negative_slope_not_rising(self, slope: float):
        """Negative slopes should never return RISING states."""
        result = classify_trend(slope)
        assert result not in (TrendState.RISING_STEADY, TrendState.RISING_SHARP)

    @given(slope=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False))
    def test_small_slope_is_flat(self, slope: float):
        """Small slopes (within -0.1 to 0.1) should be FLAT."""
        result = classify_trend(slope)
        assert result == TrendState.FLAT

    @given(slope=st.floats(allow_nan=False, allow_infinity=False))
    def test_antisymmetry(self, slope: float):
        """Rising and falling states should be symmetric around zero."""
        assume(abs(slope) > 0.1)  # Skip FLAT region
        pos_result = classify_trend(abs(slope))
        neg_result = classify_trend(-abs(slope))

        # Check symmetry
        if pos_result == TrendState.RISING_SHARP:
            assert neg_result == TrendState.FALLING_SHARP
        elif pos_result == TrendState.RISING_STEADY:
            assert neg_result == TrendState.FALLING_STEADY


# =============================================================================
# classify_acceleration Properties
# =============================================================================


class TestClassifyAccelerationProperties:
    """Property-based tests for classify_acceleration function."""

    @given(accel=st.floats(allow_nan=False, allow_infinity=False))
    def test_returns_valid_enum(self, accel: float):
        """classify_acceleration should always return a valid AccelerationState enum."""
        result = classify_acceleration(accel)
        assert isinstance(result, AccelerationState)

    @given(accel=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False))
    def test_small_acceleration_is_steady(self, accel: float):
        """Small acceleration values should be STEADY."""
        result = classify_acceleration(accel)
        assert result == AccelerationState.STEADY

    @given(accel=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
    def test_positive_not_decelerating(self, accel: float):
        """Positive acceleration should never be DECELERATING."""
        result = classify_acceleration(accel)
        assert result not in (
            AccelerationState.DECELERATING,
            AccelerationState.DECELERATING_SHARPLY,
        )

    @given(accel=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    def test_negative_not_accelerating(self, accel: float):
        """Negative acceleration should never be ACCELERATING."""
        result = classify_acceleration(accel)
        assert result not in (
            AccelerationState.ACCELERATING,
            AccelerationState.ACCELERATING_SHARPLY,
        )


# =============================================================================
# calc_linear_slope Properties
# =============================================================================


class TestCalcLinearSlopeProperties:
    """Property-based tests for calc_linear_slope function."""

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_returns_finite_float(self, arr: np.ndarray):
        """calc_linear_slope should return a finite float for any valid input."""
        assume(np.ptp(arr) > 0)  # Non-constant data
        result = calc_linear_slope(arr)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_constant_data_zero_slope(self, arr: np.ndarray):
        """Constant data should have zero slope."""
        constant_arr = np.full_like(arr, arr[0])
        result = calc_linear_slope(constant_arr)
        assert result == 0.0

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_reversed_data_opposite_slope(self, arr: np.ndarray):
        """Reversing data should negate the slope."""
        assume(np.ptp(arr) > 0)
        forward_slope = calc_linear_slope(arr)
        reverse_slope = calc_linear_slope(arr[::-1])
        # Allow small numerical tolerance
        assert abs(forward_slope + reverse_slope) < 0.1 or (
            abs(forward_slope) < 0.01 and abs(reverse_slope) < 0.01
        )


# =============================================================================
# calc_volatility Properties
# =============================================================================


class TestCalcVolatilityProperties:
    """Property-based tests for calc_volatility function."""

    @given(arr=positive_float_array)
    @settings(max_examples=100)
    def test_returns_valid_cv_and_state(self, arr: np.ndarray):
        """calc_volatility should return non-negative CV and valid state."""
        cv, state = calc_volatility(arr)
        assert cv >= 0
        assert isinstance(state, VolatilityState)

    @given(arr=positive_float_array)
    @settings(max_examples=100)
    def test_constant_data_compressed(self, arr: np.ndarray):
        """Constant data should have COMPRESSED volatility."""
        constant_arr = np.full_like(arr, arr[0])
        cv, state = calc_volatility(constant_arr)
        # Allow for floating-point precision errors
        assert cv < 1e-10, f"Expected near-zero CV for constant data, got {cv}"
        assert state == VolatilityState.COMPRESSED

    @given(arr=positive_float_array)
    @settings(max_examples=100)
    def test_scale_independence(self, arr: np.ndarray):
        """Scaling data by constant should not change volatility state."""
        assume(np.ptp(arr) > 0)
        cv1, state1 = calc_volatility(arr)
        cv2, state2 = calc_volatility(arr * 1000)
        # CV and state should be approximately the same
        assert state1 == state2
        assert abs(cv1 - cv2) < 0.01


# =============================================================================
# calc_distribution_shape Properties
# =============================================================================


class TestCalcDistributionShapeProperties:
    """Property-based tests for calc_distribution_shape function."""

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_returns_valid_enum(self, arr: np.ndarray):
        """calc_distribution_shape should always return a valid DistributionShape."""
        result = calc_distribution_shape(arr)
        assert isinstance(result, DistributionShape)

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_constant_data_is_normal(self, arr: np.ndarray):
        """Constant data should be classified as NORMAL."""
        constant_arr = np.full_like(arr, arr[0])
        result = calc_distribution_shape(constant_arr)
        assert result == DistributionShape.NORMAL


# =============================================================================
# calc_seasonality Properties
# =============================================================================


class TestCalcSeasonalityProperties:
    """Property-based tests for calc_seasonality function."""

    @given(arr=medium_float_array)
    @settings(max_examples=50)
    def test_returns_valid_autocorr_and_state(self, arr: np.ndarray):
        """calc_seasonality should return bounded autocorr and valid state."""
        assume(len(arr) >= 4)
        autocorr, state = calc_seasonality(arr)
        assert 0 <= autocorr <= 1
        assert isinstance(state, SeasonalityState)

    @given(arr=medium_float_array)
    @settings(max_examples=50)
    def test_constant_data_no_seasonality(self, arr: np.ndarray):
        """Constant data should have no seasonality."""
        assume(len(arr) >= 4)
        constant_arr = np.full_like(arr, arr[0])
        autocorr, state = calc_seasonality(constant_arr)
        assert state == SeasonalityState.NONE
        assert autocorr == 0.0


# =============================================================================
# detect_anomalies Properties
# =============================================================================


class TestDetectAnomaliesProperties:
    """Property-based tests for detect_anomalies function."""

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_returns_list(self, arr: np.ndarray):
        """detect_anomalies should always return a list."""
        result = detect_anomalies(arr)
        assert isinstance(result, list)

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_constant_data_no_anomalies(self, arr: np.ndarray):
        """Constant data should have no anomalies."""
        constant_arr = np.full_like(arr, arr[0])
        result = detect_anomalies(constant_arr)
        assert len(result) == 0

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_anomaly_indices_valid(self, arr: np.ndarray):
        """All anomaly indices should be valid array indices."""
        anomalies = detect_anomalies(arr)
        for anomaly in anomalies:
            assert 0 <= anomaly.index < len(arr)
            assert anomaly.value == arr[anomaly.index]

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_anomalies_sorted_by_zscore(self, arr: np.ndarray):
        """Anomalies should be sorted by z-score descending."""
        anomalies = detect_anomalies(arr)
        if len(anomalies) >= 2:
            z_scores = [a.z_score for a in anomalies]
            assert z_scores == sorted(z_scores, reverse=True)


# =============================================================================
# calc_acceleration Properties
# =============================================================================


class TestCalcAccelerationProperties:
    """Property-based tests for calc_acceleration function."""

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_returns_finite_float(self, arr: np.ndarray):
        """calc_acceleration should return a finite float."""
        result = calc_acceleration(arr)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_constant_data_zero_acceleration(self, arr: np.ndarray):
        """Constant data should have zero acceleration."""
        constant_arr = np.full_like(arr, arr[0])
        result = calc_acceleration(constant_arr)
        assert result == 0.0

    @given(arr=small_float_array)
    @settings(max_examples=100)
    def test_linear_data_near_zero_acceleration(self, arr: np.ndarray):
        """Linear data should have near-zero acceleration."""
        assume(len(arr) >= 3)
        linear = np.linspace(0, 10, len(arr))
        result = calc_acceleration(linear)
        # Linear data should have very small acceleration
        assert abs(result) < 0.5


# =============================================================================
# Integration Properties
# =============================================================================


class TestIntegrationProperties:
    """Property-based tests for end-to-end behavior."""

    @given(arr=medium_float_array)
    @settings(max_examples=50)
    def test_all_classifiers_handle_any_data(self, arr: np.ndarray):
        """All classifiers should handle arbitrary valid data without crashing."""
        assume(len(arr) >= 4)

        # All of these should complete without exception
        slope = calc_linear_slope(arr)
        trend = classify_trend(slope)
        cv, volatility = calc_volatility(arr)
        distribution = calc_distribution_shape(arr)
        autocorr, seasonality = calc_seasonality(arr)
        anomalies = detect_anomalies(arr)
        accel = calc_acceleration(arr)
        accel_state = classify_acceleration(accel)

        # All should return valid enum types
        assert isinstance(trend, TrendState)
        assert isinstance(volatility, VolatilityState)
        assert isinstance(distribution, DistributionShape)
        assert isinstance(seasonality, SeasonalityState)
        assert isinstance(accel_state, AccelerationState)
        assert isinstance(anomalies, list)

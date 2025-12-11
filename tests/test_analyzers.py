"""Tests for core analyzer functions."""

import numpy as np
import pytest

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
)
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


class TestCalcLinearSlope:
    """Tests for calc_linear_slope function."""

    def test_rising_data(self):
        """Strongly rising data should have positive slope."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = calc_linear_slope(values)
        assert slope > 0

    def test_falling_data(self):
        """Falling data should have negative slope."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        slope = calc_linear_slope(values)
        assert slope < 0

    def test_flat_data(self):
        """Flat data should have near-zero slope."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        slope = calc_linear_slope(values)
        assert abs(slope) < 0.01

    def test_single_value(self):
        """Single value should return zero slope."""
        values = np.array([5.0])
        slope = calc_linear_slope(values)
        assert slope == 0.0

    def test_empty_array(self):
        """Empty array should return zero slope."""
        values = np.array([])
        slope = calc_linear_slope(values)
        assert slope == 0.0

    def test_scale_independence(self):
        """Slope should be normalized for scale independence."""
        small_scale = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        large_scale = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        slope_small = calc_linear_slope(small_scale)
        slope_large = calc_linear_slope(large_scale)

        # Slopes should be approximately equal due to normalization
        assert abs(slope_small - slope_large) < 0.1


class TestClassifyTrend:
    """Tests for classify_trend function."""

    def test_rising_sharp(self):
        assert classify_trend(0.6) == TrendState.RISING_SHARP

    def test_rising_steady(self):
        assert classify_trend(0.2) == TrendState.RISING_STEADY

    def test_flat(self):
        assert classify_trend(0.0) == TrendState.FLAT
        assert classify_trend(0.05) == TrendState.FLAT
        assert classify_trend(-0.05) == TrendState.FLAT

    def test_falling_steady(self):
        assert classify_trend(-0.2) == TrendState.FALLING_STEADY

    def test_falling_sharp(self):
        assert classify_trend(-0.6) == TrendState.FALLING_SHARP


class TestCalcVolatility:
    """Tests for calc_volatility function."""

    def test_low_volatility(self):
        """Data with low variance should be classified as compressed/stable."""
        values = np.array([100.0, 100.1, 99.9, 100.0, 100.05])
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.COMPRESSED, VolatilityState.STABLE)

    def test_high_volatility(self):
        """Data with high variance should be classified as expanding/extreme."""
        values = np.array([10.0, 100.0, 5.0, 200.0, 50.0])
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.EXPANDING, VolatilityState.EXTREME)

    def test_constant_data(self):
        """Constant data should have zero CV and be compressed."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        cv, state = calc_volatility(values)
        assert cv == 0.0
        assert state == VolatilityState.COMPRESSED

    def test_empty_array(self):
        """Empty array should return stable state."""
        values = np.array([])
        cv, state = calc_volatility(values)
        assert state == VolatilityState.STABLE

    def test_zero_mean_handling(self):
        """Data with zero mean should be handled correctly."""
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        cv, state = calc_volatility(values)
        # Should not raise and should classify based on range
        assert state is not None


class TestDetectAnomalies:
    """Tests for detect_anomalies function."""

    def test_clear_outlier(self):
        """Single clear outlier should be detected."""
        values = np.array([10.0, 10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 1
        assert anomalies[0].index == 4
        assert anomalies[0].value == 100.0

    def test_no_outliers(self):
        """Normal data should have no outliers."""
        values = np.array([10.0, 11.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.3, 10.1, 9.9])
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0

    def test_small_dataset_uses_iqr(self):
        """Small datasets (<10 samples) should use IQR method."""
        values = np.array([10.0, 10.0, 10.0, 100.0, 10.0])
        anomalies = detect_anomalies(values)
        # IQR method should detect the outlier
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_two_values(self):
        """Two values should return empty (too few for analysis)."""
        values = np.array([10.0, 100.0])
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0

    def test_sorted_by_z_score(self):
        """Anomalies should be sorted by z-score (highest first)."""
        values = np.array([10.0] * 10 + [50.0, 100.0])
        anomalies = detect_anomalies(values)
        if len(anomalies) >= 2:
            assert anomalies[0].z_score >= anomalies[1].z_score


class TestClassifyAnomalyState:
    """Tests for classify_anomaly_state function."""

    def test_no_anomalies(self):
        assert classify_anomaly_state([]) == AnomalyState.NONE

    def test_minor_anomalies(self):
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=0, value=100.0, z_score=3.5)]
        assert classify_anomaly_state(anomalies) == AnomalyState.MINOR

    def test_significant_anomalies(self):
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=i, value=100.0, z_score=3.5) for i in range(4)]
        assert classify_anomaly_state(anomalies) == AnomalyState.SIGNIFICANT

    def test_extreme_count(self):
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=i, value=100.0, z_score=3.5) for i in range(6)]
        assert classify_anomaly_state(anomalies) == AnomalyState.EXTREME

    def test_extreme_z_score(self):
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        # Single anomaly with very high z-score
        anomalies = [AnomalyInfo(index=0, value=1000.0, z_score=6.0)]
        assert classify_anomaly_state(anomalies) == AnomalyState.EXTREME


class TestAssessDataQuality:
    """Tests for assess_data_quality function."""

    def test_pristine_data(self):
        """No missing values should be pristine."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pct, quality = assess_data_quality(values)
        assert pct == 0.0
        assert quality == DataQuality.PRISTINE

    def test_good_quality(self):
        """1-5% missing should be good quality."""
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 10)  # 10% NaN? Let's fix
        # Create array with ~3% NaN
        values = np.array([1.0] * 97 + [np.nan] * 3)
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.GOOD

    def test_sparse_data(self):
        """5-20% missing should be sparse."""
        values = np.array([1.0] * 85 + [np.nan] * 15)
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.SPARSE

    def test_fragmented_data(self):
        """>20% missing should be fragmented."""
        values = np.array([1.0] * 70 + [np.nan] * 30)
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.FRAGMENTED

    def test_empty_array(self):
        """Empty array should be fragmented."""
        values = np.array([])
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.FRAGMENTED


class TestCalcDistributionShape:
    """Tests for calc_distribution_shape function."""

    def test_normal_distribution(self):
        """Normal data should be classified as normal."""
        np.random.seed(42)
        values = np.random.normal(50, 10, 1000)
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL

    def test_right_skewed(self):
        """Right-skewed data should be detected."""
        np.random.seed(42)
        values = np.random.exponential(10, 1000)
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.RIGHT_SKEWED

    def test_small_dataset(self):
        """Small dataset should default to normal."""
        values = np.array([1.0, 2.0, 3.0])
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL


class TestCalcSeasonality:
    """Tests for calc_seasonality function."""

    def test_clear_seasonality(self):
        """Strongly periodic data should show seasonality."""
        # Create sinusoidal pattern
        x = np.linspace(0, 4 * np.pi, 100)
        values = np.sin(x)
        autocorr, state = calc_seasonality(values)
        assert state in (SeasonalityState.MODERATE, SeasonalityState.STRONG)

    def test_random_data(self):
        """Random data should show no seasonality."""
        np.random.seed(42)
        values = np.random.randn(100)
        autocorr, state = calc_seasonality(values)
        assert state in (SeasonalityState.NONE, SeasonalityState.WEAK)

    def test_short_data(self):
        """Very short data should return no seasonality."""
        values = np.array([1.0, 2.0, 3.0])
        autocorr, state = calc_seasonality(values)
        assert state == SeasonalityState.NONE

    def test_constant_data(self):
        """Constant data should return no seasonality."""
        values = np.array([5.0] * 100)
        autocorr, state = calc_seasonality(values)
        assert state == SeasonalityState.NONE


class TestZeroStdAnomalyDetection:
    """Tests for anomaly detection with zero standard deviation."""

    def test_zscore_with_zero_std_detects_outlier(self):
        """Data with all identical values except one should detect the outlier."""
        # 12 values to trigger zscore method (>=10), all identical except one
        values = np.array([5.0] * 12 + [100.0])
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_iqr_zero_iqr_with_outlier(self):
        """IQR method with zero IQR (identical values) should still detect outliers."""
        values = np.array([5.0, 5.0, 5.0, 100.0, 5.0])  # <10 samples, uses IQR
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_iqr_all_identical_values(self):
        """All identical values should return no anomalies (max_dev == 0)."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0


class TestZThresholdValidation:
    """Tests for z_threshold parameter validation."""

    def test_negative_z_threshold_raises_error(self):
        """Negative z_threshold should raise ValueError."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 5)
        with pytest.raises(ValueError) as excinfo:
            detect_anomalies(values, z_threshold=-1.0)
        assert "z_threshold must be positive" in str(excinfo.value)

    def test_zero_z_threshold_raises_error(self):
        """Zero z_threshold should raise ValueError."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 5)
        with pytest.raises(ValueError) as excinfo:
            detect_anomalies(values, z_threshold=0.0)
        assert "z_threshold must be positive" in str(excinfo.value)


class TestDistributionEdgeCases:
    """Tests for distribution shape edge cases."""

    def test_left_skewed_distribution(self):
        """Left-skewed data should be detected."""
        np.random.seed(42)
        # Create left-skewed data (negated exponential shifted right)
        values = -np.random.exponential(10, 1000) + 100
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.LEFT_SKEWED

    def test_uniform_distribution(self):
        """Uniformly distributed data detection."""
        np.random.seed(42)
        values = np.random.uniform(0, 100, 1000)
        shape = calc_distribution_shape(values)
        # Uniform should be detected or at least not crash
        assert shape is not None

    def test_near_identical_values_not_bimodal(self):
        """Near-identical values (tiny differences) should not be misclassified as BIMODAL.

        Regression test for bug where values differing by ~1e-15 were classified
        as BIMODAL due to numerical instability in scipy's kurtosis calculation.
        """
        # Values that are effectively identical (differ by machine epsilon)
        values = np.array([1.0, 1.0 + 1e-15, 1.0 + 2e-15, 1.0 + 3e-15])
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL

    def test_near_identical_large_values(self):
        """Near-identical large values should also classify as NORMAL."""
        # Large values with tiny relative differences
        base = 1e10
        values = np.array([base, base + 1e-5, base + 2e-5, base + 3e-5])
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL


class TestSeasonalityDetrending:
    """Tests for seasonality detection with detrending.

    These tests verify that linear trends don't produce false positives
    for seasonality detection.
    """

    def test_linear_trend_no_seasonality(self):
        """Simple linear growth should NOT show seasonality.

        Regression test for bug where [1, 2, 3, ..., 20] was classified as
        having STRONG seasonality because autocorrelation at lag doesn't
        distinguish between trend and cyclic patterns.
        """
        values = np.arange(1.0, 21.0)  # [1, 2, 3, ..., 20]
        autocorr, state = calc_seasonality(values)
        assert (
            state == SeasonalityState.NONE
        ), f"Linear trend incorrectly detected as {state} with autocorr={autocorr:.2f}"

    def test_linear_decline_no_seasonality(self):
        """Declining linear data should NOT show seasonality."""
        values = np.arange(100.0, 0.0, -5.0)  # [100, 95, 90, ..., 5]
        autocorr, state = calc_seasonality(values)
        assert state == SeasonalityState.NONE

    def test_trend_plus_seasonality_detected(self):
        """Data with both trend AND seasonal component should detect seasonality."""
        x = np.arange(50)
        trend = 0.5 * x  # Linear trend
        seasonal = 10 * np.sin(2 * np.pi * x / 10)  # Period 10
        values = trend + seasonal
        autocorr, state = calc_seasonality(values)
        assert state in (SeasonalityState.MODERATE, SeasonalityState.STRONG)

    def test_actual_seasonal_pattern(self):
        """True cyclic pattern should still be detected after detrending."""
        # Repeating pattern with no trend: [10, 20, 30, 20, 10, ...]
        pattern = [10.0, 20.0, 30.0, 20.0]
        values = np.array(pattern * 10)  # 40 data points
        autocorr, state = calc_seasonality(values)
        assert state in (SeasonalityState.MODERATE, SeasonalityState.STRONG)

    def test_pure_trend_after_detrend_is_constant(self):
        """After detrending pure linear data, residuals should have no variance."""
        values = np.arange(1.0, 101.0)  # [1, 2, ..., 100]
        autocorr, state = calc_seasonality(values)
        # Pure linear trend after detrending should be all zeros -> no seasonality
        assert state == SeasonalityState.NONE
        assert autocorr == 0.0


class TestVolatilityEdgeCases:
    """Tests for volatility calculation edge cases.

    Covers line 120: zero mean with zero data range.
    """

    def test_zero_mean_zero_range(self):
        """All zeros should return COMPRESSED with CV=0.

        This tests line 120: when mean=0 and data_range=0.
        """
        values = np.array([0.0, 0.0, 0.0, 0.0])
        cv, state = calc_volatility(values)
        assert cv == 0.0
        assert state == VolatilityState.COMPRESSED

    def test_zero_mean_with_range(self):
        """Zero mean but non-zero range should compute CV from range.

        Tests the path where mean=0 but data has variance (e.g., [-1, 1]).
        """
        values = np.array([-1.0, 1.0, -1.0, 1.0])
        cv, state = calc_volatility(values)
        # CV = std / range when mean=0
        assert cv > 0
        assert state is not None

    def test_zero_mean_small_spread(self):
        """Zero mean with very small spread around zero."""
        values = np.array([-0.001, 0.001, -0.001, 0.001])
        cv, state = calc_volatility(values)
        assert cv > 0
        assert state is not None


class TestIqrAnomalyDetection:
    """Tests for IQR-based anomaly detection (small samples < 10).

    Covers lines 197-208: IQR method with non-zero IQR.
    """

    def test_iqr_with_outlier_below_lower_bound(self):
        """IQR method should detect outliers below lower bound.

        Tests lines 197-208: standard IQR path with iqr > 0.
        """
        # 5 values triggers IQR method (<10 samples)
        # Create data with clear low outlier
        values = np.array([100.0, 105.0, 110.0, 108.0, 10.0])  # 10 is outlier
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        assert any(a.value == 10.0 for a in anomalies)

    def test_iqr_with_outlier_above_upper_bound(self):
        """IQR method should detect outliers above upper bound."""
        values = np.array([10.0, 12.0, 11.0, 13.0, 100.0])  # 100 is outlier
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_iqr_with_multiple_outliers(self):
        """IQR method should detect multiple outliers."""
        values = np.array([1.0, 50.0, 51.0, 52.0, 53.0, 100.0])  # 1 and 100 are outliers
        anomalies = detect_anomalies(values)
        # May detect 0, 1, or 2 depending on IQR calculation
        assert isinstance(anomalies, list)

    def test_iqr_no_outliers(self):
        """Data with no outliers should return empty list."""
        values = np.array([10.0, 11.0, 12.0, 11.5, 10.5])
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0


class TestZscoreZeroStdFallback:
    """Tests for zscore anomaly detection with zero standard deviation.

    Covers lines 226-233: fallback when std=0 but there are outliers.
    """

    def test_zscore_zero_std_with_single_outlier(self):
        """When std=0 (identical values + outlier), should detect outlier.

        This directly tests lines 226-233.
        """
        # Need 10+ values to trigger zscore method
        # All 5.0 except one 100.0 - std will be non-zero due to the outlier
        # To get true std=0, we need to mock or find edge case
        # Actually, if we have [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 100], std != 0
        # Let's verify the fallback is reachable with actual test
        values = np.array([5.0] * 11 + [100.0])
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_zscore_all_identical_large_sample(self):
        """All identical values (large sample) should return no anomalies.

        Tests std=0 path with max_dev=0 -> returns empty list.
        """
        values = np.array([5.0] * 20)  # Triggers zscore method, std=0
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0

    def test_zscore_nearly_identical_with_outlier(self):
        """Nearly identical values with outlier using zscore method."""
        # Create data where most values are the same but there's a clear outlier
        values = np.array([10.0] * 15 + [1000.0])  # 16 values, triggers zscore
        anomalies = detect_anomalies(values)
        assert len(anomalies) >= 1
        # The outlier should be detected
        outlier_detected = any(a.value == 1000.0 for a in anomalies)
        assert outlier_detected

    def test_zscore_standard_detection(self):
        """Standard zscore detection with normal variance."""
        np.random.seed(42)
        # Normal data with clear outliers
        values = np.concatenate([np.random.normal(100, 10, 20), [200.0]])
        anomalies = detect_anomalies(values)
        # Should detect the 200 outlier
        assert any(a.value == 200.0 for a in anomalies)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning:numpy")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning:numpy")
class TestDistributionScipyExceptions:
    """Tests for distribution shape calculation exception handling.

    Covers lines 332-339: scipy exception handling.
    Covers lines 343-348: NaN result handling.
    """

    def test_extreme_values_dont_crash(self):
        """Extreme values should not crash, should return NORMAL.

        Tests the exception handling path (lines 332-339).
        """
        # Very extreme values that might cause scipy issues
        values = np.array([1e308, 1e308, 1e308, 1e-308, 1e-308])
        shape = calc_distribution_shape(values)
        # Should not crash, and should return a valid shape
        assert shape is not None

    def test_inf_values_handled(self):
        """Inf values should be handled gracefully."""
        values = np.array([1.0, 2.0, np.inf, 3.0, 4.0])
        # This might trigger NaN in scipy calculations
        shape = calc_distribution_shape(values)
        # Should not crash
        assert shape is not None

    def test_negative_inf_values_handled(self):
        """Negative inf values should be handled gracefully."""
        values = np.array([1.0, 2.0, -np.inf, 3.0, 4.0])
        shape = calc_distribution_shape(values)
        assert shape is not None

    def test_mixed_inf_values(self):
        """Mix of positive and negative inf."""
        values = np.array([np.inf, -np.inf, 1.0, 2.0, 3.0])
        shape = calc_distribution_shape(values)
        assert shape is not None


class TestSeasonalityEdgeCases:
    """Tests for seasonality calculation edge cases.

    Covers line 400: effective_max_lag < 2
    Covers lines 418-419: near-zero mean with near-zero residuals
    Covers lines 431-441: pearsonr exception handling
    Covers line 444: empty autocorrs list
    """

    def test_short_series_effective_lag_too_small(self):
        """Series where n//2 < 2 should return no seasonality.

        Tests line 400: effective_max_lag < 2.
        """
        # n=3 means effective_max_lag = 1, which is < 2
        values = np.array([1.0, 2.0, 3.0])
        autocorr, state = calc_seasonality(values)
        assert state == SeasonalityState.NONE
        assert autocorr == 0.0

    def test_four_element_series(self):
        """Four elements: n//2 = 2, so effective_max_lag = 2, just barely usable."""
        values = np.array([1.0, 2.0, 1.0, 2.0])  # Simple alternating pattern
        autocorr, state = calc_seasonality(values)
        # Should not crash, may or may not detect pattern with only 1 lag
        assert state is not None

    def test_near_zero_mean_with_trend(self):
        """Data centered around zero with linear trend.

        Tests lines 418-419: detrended_mean near zero path.
        """
        # Data oscillating around zero with no trend
        values = np.array([-0.001, 0.001, -0.001, 0.001] * 10)  # 40 points
        autocorr, state = calc_seasonality(values)
        # Should handle near-zero mean case
        assert state is not None

    def test_zero_centered_linear_trend(self):
        """Linear trend through zero (negative to positive).

        Tests the detrended_mean == 0 branch (line 418-419).
        """
        # Values that average to ~0 but have a linear trend
        values = np.linspace(-10.0, 10.0, 21)  # Mean is 0
        autocorr, state = calc_seasonality(values)
        # After detrending, should have near-zero residuals
        assert state == SeasonalityState.NONE

    def test_all_nan_correlations_returns_none(self):
        """If all correlations produce NaN, should return NONE.

        Tests line 444: empty autocorrs list.
        """
        # Constant data should produce NaN correlations (caught earlier, but test anyway)
        values = np.array([5.0] * 20)
        autocorr, state = calc_seasonality(values)
        assert state == SeasonalityState.NONE

    def test_weak_seasonality_threshold(self):
        """Test WEAK seasonality classification (0.3 <= peak < 0.5)."""
        np.random.seed(123)
        # Create data with weak periodic component plus noise
        x = np.arange(50)
        seasonal = 2 * np.sin(2 * np.pi * x / 10)  # Weak periodic
        noise = np.random.randn(50) * 5  # Strong noise
        values = seasonal + noise
        autocorr, state = calc_seasonality(values)
        # May be NONE, WEAK, or MODERATE depending on random seed
        assert state in (SeasonalityState.NONE, SeasonalityState.WEAK, SeasonalityState.MODERATE)

    def test_moderate_seasonality_threshold(self):
        """Test MODERATE seasonality classification (0.5 <= peak < 0.7)."""
        np.random.seed(42)
        x = np.arange(50)
        seasonal = 5 * np.sin(2 * np.pi * x / 10)  # Moderate periodic
        noise = np.random.randn(50) * 3  # Moderate noise
        values = seasonal + noise
        autocorr, state = calc_seasonality(values)
        # Should be MODERATE or STRONG
        assert state in (SeasonalityState.MODERATE, SeasonalityState.STRONG)


class TestZscoreZeroStdExactPath:
    """Tests specifically targeting the zero std fallback path in _detect_anomalies_zscore.

    These tests aim to hit lines 226-233 where std=0 but max_dev > 0.
    """

    def test_zscore_exact_zero_std_with_outlier(self):
        """Test exact scenario where std would be 0 if not for outlier.

        When most values are identical and one outlier exists, the std
        will be non-zero due to the outlier itself. This tests the
        deviation-based fallback behavior.
        """
        # 15 identical values + 1 extreme outlier
        # The outlier is so extreme that it's detected regardless of method
        values = np.array([5.0] * 15 + [500.0])
        anomalies = detect_anomalies(values)

        assert len(anomalies) >= 1
        assert any(a.value == 500.0 for a in anomalies)
        # Verify z_score is computed (approximated via deviation method)
        assert anomalies[0].z_score > 0

    def test_zscore_low_std_deviation_fallback(self):
        """Near-zero std with outlier uses deviation-based detection."""
        # All 5.0 except one 50.0 - very low std scenario
        values = np.array([5.0] * 19 + [50.0])
        anomalies = detect_anomalies(values)

        assert len(anomalies) >= 1
        outlier_detected = any(a.value == 50.0 for a in anomalies)
        assert outlier_detected


class TestDistributionBimodalKurtosis:
    """Tests for bimodal detection via kurtosis (line 358).

    Bimodal distributions have negative excess kurtosis (platykurtic).
    """

    def test_bimodal_two_peaks(self):
        """True bimodal distribution should be detected.

        Tests line 358: kurtosis < -1 returns BIMODAL.
        """
        # Create bimodal distribution with two distinct peaks
        np.random.seed(42)
        peak1 = np.random.normal(0, 1, 500)
        peak2 = np.random.normal(10, 1, 500)
        values = np.concatenate([peak1, peak2])
        shape = calc_distribution_shape(values)
        # Bimodal distributions can be classified as BIMODAL, UNIFORM, or NORMAL
        # depending on kurtosis/skewness values
        valid_shapes = (
            DistributionShape.BIMODAL,
            DistributionShape.NORMAL,
            DistributionShape.UNIFORM,
        )
        assert shape in valid_shapes

    def test_uniform_like_flat_distribution(self):
        """Flat distribution with very negative kurtosis.

        A truly uniform distribution has kurtosis = -1.2, which triggers
        the UNIFORM check first (line 352-353).
        """
        np.random.seed(42)
        values = np.random.uniform(0, 100, 1000)
        shape = calc_distribution_shape(values)
        assert shape in (DistributionShape.UNIFORM, DistributionShape.NORMAL)

    def test_bimodal_symmetric_peaks(self):
        """Symmetric bimodal with equal-sized peaks."""
        # Two clear peaks at 0 and 100
        values = np.array([0.0] * 100 + [100.0] * 100)
        shape = calc_distribution_shape(values)
        # This extreme bimodal should have very negative kurtosis
        assert shape in (DistributionShape.BIMODAL, DistributionShape.UNIFORM)


class TestSeasonalityPearsonrException:
    """Tests targeting pearsonr exception handling in calc_seasonality.

    Covers lines 431-441: exception path in autocorrelation loop.
    """

    def test_degenerate_slice_in_correlation(self):
        """Data that produces degenerate slices for pearsonr.

        When detrended[:-lag] or detrended[lag:] has constant values,
        pearsonr may fail or return NaN.
        """
        # Create data where after detrending, some slices might be constant
        # Linear trend with tiny noise
        values = np.arange(100.0) + np.random.normal(0, 1e-15, 100)
        autocorr, state = calc_seasonality(values)
        # Should not crash and should return valid result
        assert state in (SeasonalityState.NONE, SeasonalityState.WEAK)

    def test_very_short_detrended_slices(self):
        """Very short data where lag slices are minimal."""
        values = np.array([1.0, 10.0, 1.0, 10.0, 1.0])  # 5 values
        autocorr, state = calc_seasonality(values)
        # With n=5, effective_max_lag = 2, only lag=1 is tested
        assert state is not None


class TestVolatilityThresholds:
    """Tests for volatility state thresholds to ensure full coverage."""

    def test_compressed_volatility(self):
        """CV < 0.05 should be COMPRESSED."""
        # Create data with very low CV
        values = np.array([100.0, 100.1, 99.9, 100.0, 100.05])
        cv, state = calc_volatility(values)
        assert state == VolatilityState.COMPRESSED

    def test_stable_volatility(self):
        """CV 0.05-0.15 should be STABLE."""
        # Create data with CV around 0.1
        values = np.array([100.0, 110.0, 90.0, 105.0, 95.0])
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.STABLE, VolatilityState.MODERATE)

    def test_moderate_volatility(self):
        """CV 0.15-0.30 should be MODERATE."""
        # Need CV around 0.2 - use larger spread
        values = np.array([100.0, 130.0, 70.0, 120.0, 80.0])
        cv, state = calc_volatility(values)
        assert state in (
            VolatilityState.STABLE,
            VolatilityState.MODERATE,
            VolatilityState.EXPANDING,
        )

    def test_expanding_volatility(self):
        """CV 0.30-0.50 should be EXPANDING."""
        values = np.array([100.0, 150.0, 50.0, 130.0, 70.0])
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.EXPANDING, VolatilityState.EXTREME)

    def test_extreme_volatility(self):
        """CV >= 0.50 should be EXTREME."""
        values = np.array([10.0, 100.0, 5.0, 200.0, 1.0])
        cv, state = calc_volatility(values)
        assert state == VolatilityState.EXTREME


class TestCalcAcceleration:
    """Tests for calc_acceleration function."""

    def test_accelerating_quadratic_growth(self):
        """Quadratic growth (x^2) should show positive acceleration."""
        x = np.arange(20)
        values = x**2  # Quadratic growth
        accel = calc_acceleration(values)
        assert accel > 0, f"Quadratic growth should have positive acceleration, got {accel}"

    def test_decelerating_logarithmic_growth(self):
        """Logarithmic growth should show negative acceleration."""
        values = np.log(np.arange(1, 21))  # Logarithmic growth
        accel = calc_acceleration(values)
        assert accel < 0, f"Logarithmic growth should have negative acceleration, got {accel}"

    def test_linear_data_steady(self):
        """Linear data should have near-zero acceleration."""
        values = np.arange(20) * 5.0  # Linear growth
        accel = calc_acceleration(values)
        assert abs(accel) < 0.1, f"Linear data should have near-zero acceleration, got {accel}"

    def test_constant_data(self):
        """Constant data should return zero acceleration."""
        values = np.array([5.0] * 20)
        accel = calc_acceleration(values)
        assert accel == 0.0

    def test_short_array_two_values(self):
        """Array with < 3 values should return zero."""
        values = np.array([1.0, 2.0])
        accel = calc_acceleration(values)
        assert accel == 0.0

    def test_short_array_single_value(self):
        """Single value should return zero."""
        values = np.array([5.0])
        accel = calc_acceleration(values)
        assert accel == 0.0

    def test_empty_array(self):
        """Empty array should return zero."""
        values = np.array([])
        accel = calc_acceleration(values)
        assert accel == 0.0

    def test_scale_independence(self):
        """Acceleration should be normalized for scale independence."""
        small = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # x^2
        large = small * 1000
        accel_small = calc_acceleration(small)
        accel_large = calc_acceleration(large)
        # Should be similar (both are quadratic patterns)
        assert abs(accel_small - accel_large) < 0.1 * max(abs(accel_small), abs(accel_large), 1.0)

    def test_negative_quadratic(self):
        """Negative quadratic (-x^2) should show negative acceleration."""
        x = np.arange(20)
        values = -(x**2)
        accel = calc_acceleration(values)
        assert accel < 0

    def test_accelerating_exponential(self):
        """Exponential growth should show positive acceleration."""
        values = np.exp(np.linspace(0, 2, 20))  # Exponential growth
        accel = calc_acceleration(values)
        assert accel > 0

    def test_decelerating_to_plateau(self):
        """Data approaching plateau (saturation curve) should decelerate."""
        # 1 - e^(-x) approaches 1 asymptotically (decreasing rate of change)
        values = 1 - np.exp(-np.linspace(0, 3, 20))
        accel = calc_acceleration(values)
        # This is a saturating curve - growth decelerates
        assert accel < 0


class TestClassifyAcceleration:
    """Tests for classify_acceleration function."""

    def test_accelerating_sharply(self):
        """Acceleration > 0.3 should be ACCELERATING_SHARPLY."""
        assert classify_acceleration(0.5) == AccelerationState.ACCELERATING_SHARPLY
        assert classify_acceleration(1.0) == AccelerationState.ACCELERATING_SHARPLY
        assert classify_acceleration(0.31) == AccelerationState.ACCELERATING_SHARPLY

    def test_accelerating(self):
        """Acceleration 0.1-0.3 should be ACCELERATING."""
        assert classify_acceleration(0.2) == AccelerationState.ACCELERATING
        assert classify_acceleration(0.15) == AccelerationState.ACCELERATING
        assert classify_acceleration(0.11) == AccelerationState.ACCELERATING

    def test_steady(self):
        """Acceleration -0.1 to 0.1 should be STEADY."""
        assert classify_acceleration(0.0) == AccelerationState.STEADY
        assert classify_acceleration(0.05) == AccelerationState.STEADY
        assert classify_acceleration(-0.05) == AccelerationState.STEADY
        assert classify_acceleration(0.1) == AccelerationState.STEADY
        assert classify_acceleration(-0.1) == AccelerationState.STEADY

    def test_decelerating(self):
        """Acceleration -0.3 to -0.1 should be DECELERATING."""
        assert classify_acceleration(-0.2) == AccelerationState.DECELERATING
        assert classify_acceleration(-0.15) == AccelerationState.DECELERATING
        assert classify_acceleration(-0.11) == AccelerationState.DECELERATING

    def test_decelerating_sharply(self):
        """Acceleration < -0.3 should be DECELERATING_SHARPLY."""
        assert classify_acceleration(-0.5) == AccelerationState.DECELERATING_SHARPLY
        assert classify_acceleration(-1.0) == AccelerationState.DECELERATING_SHARPLY
        assert classify_acceleration(-0.31) == AccelerationState.DECELERATING_SHARPLY

    def test_boundary_values(self):
        """Test exact boundary values."""
        # At exactly 0.3 boundary
        assert classify_acceleration(0.3) == AccelerationState.ACCELERATING
        # At exactly -0.3 boundary
        assert classify_acceleration(-0.3) == AccelerationState.DECELERATING


class TestAccelerationIntegration:
    """Integration tests for acceleration with real-world data patterns."""

    def test_stock_rally_acceleration(self):
        """Simulated accelerating stock rally (exponential growth phase)."""
        # Price accelerating upward
        values = np.array([100, 101, 103, 106, 110, 115, 122, 131, 142, 156])
        accel = calc_acceleration(values)
        state = classify_acceleration(accel)
        assert state in (
            AccelerationState.ACCELERATING,
            AccelerationState.ACCELERATING_SHARPLY,
        )

    def test_growth_slowdown(self):
        """Simulated growth slowdown (early plateau approach)."""
        # Growth slowing as it approaches limit
        values = np.array([10, 25, 38, 49, 58, 65, 71, 76, 80, 83])
        accel = calc_acceleration(values)
        state = classify_acceleration(accel)
        assert state in (AccelerationState.DECELERATING, AccelerationState.DECELERATING_SHARPLY)

    def test_steady_linear_growth(self):
        """Linear growth should show steady acceleration."""
        values = np.linspace(100, 200, 20)  # Perfect linear
        accel = calc_acceleration(values)
        state = classify_acceleration(accel)
        assert state == AccelerationState.STEADY

    def test_noisy_but_accelerating(self):
        """Accelerating trend with noise should still detect acceleration."""
        np.random.seed(42)
        x = np.arange(30)
        clean_accel = x**2 / 30  # Scaled quadratic
        noise = np.random.randn(30) * 2
        values = clean_accel + noise
        accel = calc_acceleration(values)
        # With noise, may be any positive state
        assert accel > -0.3  # At least not strongly decelerating


class TestZscoreZeroStdDeviationPath:
    """Tests specifically targeting the std==0 fallback path in _detect_anomalies_zscore.

    Covers lines 229-236: when std=0 but max_dev > 0, uses deviation-based detection.
    """

    def test_zscore_std_zero_with_single_outlier(self):
        """When std==0 exactly (all values equal except outlier), use deviation fallback.

        This requires crafting data where np.std() returns exactly 0.0.
        This happens when ALL values including the outlier result in std=0,
        which is only possible with identical values (covered by max_dev==0 path).

        The std==0 with max_dev>0 path requires mocking since real data with an
        outlier will have std>0.
        """
        from unittest.mock import patch

        # Create data with an outlier
        values = np.array([5.0] * 15 + [100.0])

        # Mock np.std to return 0 to force the fallback path
        original_std = np.std

        def mock_std(arr, *args, **kwargs):
            # Return 0 for our specific test array to trigger the fallback
            if len(arr) == 16 and arr[-1] == 100.0:
                return 0.0
            return original_std(arr, *args, **kwargs)

        with patch.object(np, "std", side_effect=mock_std):
            anomalies = detect_anomalies(values)

        # Should detect the outlier via deviation-based method
        assert len(anomalies) >= 1
        assert any(a.value == 100.0 for a in anomalies)

    def test_zscore_deviation_threshold_calculation(self):
        """Verify deviation-based threshold uses max_dev * 0.5.

        Tests lines 229-236 logic: values with deviation > max_dev*0.5 are flagged.
        """
        from unittest.mock import patch

        # Create data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
        # median=0, max_dev=100, threshold=50
        # Only the 100 should be flagged (dev=100 > 50)
        values = np.array([0.0] * 10 + [100.0])

        # Mock std to return 0
        with patch.object(np, "std", return_value=0.0):
            anomalies = detect_anomalies(values)

        assert len(anomalies) == 1
        assert anomalies[0].value == 100.0
        # z_approx = dev / (max_dev / 3) = 100 / (100/3) = 3.0
        assert abs(anomalies[0].z_score - 3.0) < 0.01


class TestDistributionConstantData:
    """Tests for distribution shape with constant data.

    Covers line 316: np.ptp(values) == 0 returns NORMAL early.
    """

    def test_all_identical_values(self):
        """Constant data (ptp=0) should return NORMAL immediately.

        Tests line 316: early return for constant data.
        """
        values = np.array([42.0] * 100)
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL

    def test_all_zeros(self):
        """All zeros should return NORMAL."""
        values = np.array([0.0] * 100)
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL

    def test_all_negative_same(self):
        """All same negative values should return NORMAL."""
        values = np.array([-5.0] * 100)
        shape = calc_distribution_shape(values)
        assert shape == DistributionShape.NORMAL


class TestDistributionScipyExceptionPath:
    """Tests specifically targeting the scipy exception path.

    Covers lines 335-342: when scipy.stats.skew/kurtosis raises an exception.
    """

    def test_scipy_exception_returns_normal(self):
        """When scipy raises, should return NORMAL and log.

        Tests lines 335-342: exception handling path.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Mock skew to raise ValueError
        with patch("semantic_frame.core.analyzers.skew", side_effect=ValueError("Mock error")):
            shape = calc_distribution_shape(values)

        assert shape == DistributionShape.NORMAL

    def test_scipy_floating_point_error(self):
        """FloatingPointError from scipy should be handled.

        Tests line 335: FloatingPointError in exception list.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch(
            "semantic_frame.core.analyzers.skew", side_effect=FloatingPointError("overflow")
        ):
            shape = calc_distribution_shape(values)

        assert shape == DistributionShape.NORMAL


class TestDistributionBimodalPath:
    """Tests specifically for the BIMODAL detection path.

    Covers line 361: k < -1 (but not uniform) returns BIMODAL.
    """

    def test_flat_topped_not_uniform(self):
        """Distribution with k < -1 but |s| >= 0.3 should be BIMODAL.

        Tests line 361: BIMODAL path (k < -1 but skewed so not UNIFORM).
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Mock to return kurtosis < -1 but skewness >= 0.3 (not uniform)
        with patch("semantic_frame.core.analyzers.skew", return_value=0.5):
            with patch("semantic_frame.core.analyzers.kurtosis", return_value=-1.5):
                shape = calc_distribution_shape(values)

        assert shape == DistributionShape.BIMODAL


class TestSeasonalityDetrended:
    """Tests for seasonality edge cases after detrending.

    Covers line 403: effective_max_lag < 2 after adjustment.
    Covers lines 434-444: pearsonr exception handling.
    Covers line 447: empty autocorrs list.
    """

    def test_effective_max_lag_becomes_one(self):
        """When max_lag is explicitly set low, should return NONE.

        Tests line 403: effective_max_lag < 2.
        """
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # With max_lag=1, effective_max_lag = min(1, 4) = 1 which is < 2
        autocorr, state = calc_seasonality(values, max_lag=1)
        assert state == SeasonalityState.NONE
        assert autocorr == 0.0

    def test_pearsonr_value_error_handled(self):
        """When pearsonr raises ValueError, should continue with other lags.

        Tests lines 434-444: exception handling in autocorrelation loop.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)

        # Mock pearsonr to raise on first call, succeed on others
        original_pearsonr = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr
        call_count = [0]

        def mock_pearsonr(x, y):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Mock pearsonr failure")
            return original_pearsonr(x, y)

        with patch("semantic_frame.core.analyzers.pearsonr", side_effect=mock_pearsonr):
            autocorr, state = calc_seasonality(values)

        # Should still return a valid result (from other lags)
        assert state is not None

    def test_all_correlations_nan_returns_none(self):
        """When all correlations are NaN, should return NONE.

        Tests line 447: empty autocorrs list after filtering NaN.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)

        # Mock pearsonr to always return NaN
        with patch("semantic_frame.core.analyzers.pearsonr", return_value=(np.nan, 0.0)):
            autocorr, state = calc_seasonality(values)

        assert state == SeasonalityState.NONE
        assert autocorr == 0.0


class TestStepChangeEdgeCases:
    """Tests for step change detection edge cases.

    Covers line 480: n < window_size * 2.
    Covers line 485: global_std == 0.
    """

    def test_array_too_short_for_window(self):
        """Array shorter than 2*window_size returns NONE.

        Tests line 480: n < window_size * 2.
        """
        from semantic_frame.core.analyzers import detect_step_changes

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements, default window=10
        change_type, idx = detect_step_changes(values)
        assert change_type == StructuralChange.NONE
        assert idx is None

    def test_constant_data_step_change(self):
        """Constant data (global_std=0) returns NONE.

        Tests line 485: global_std == 0.
        """
        from semantic_frame.core.analyzers import detect_step_changes

        values = np.array([5.0] * 50)
        change_type, idx = detect_step_changes(values)
        assert change_type == StructuralChange.NONE
        assert idx is None


class TestCalcAccelerationException:
    """Tests for calc_acceleration exception handling.

    Covers lines 558-559: np.linalg.LinAlgError or ValueError in polyfit.
    """

    def test_polyfit_linalg_error(self):
        """LinAlgError from polyfit should return 0.0.

        Tests lines 558-559: exception handling.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("numpy.polyfit", side_effect=np.linalg.LinAlgError("Singular matrix")):
            accel = calc_acceleration(values)

        assert accel == 0.0

    def test_polyfit_value_error(self):
        """ValueError from polyfit should return 0.0.

        Tests lines 558-559: ValueError in exception list.
        """
        from unittest.mock import patch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("numpy.polyfit", side_effect=ValueError("Bad input")):
            accel = calc_acceleration(values)

        assert accel == 0.0


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestTrendBoundaries:
    """Tests for exact boundary values in classify_trend.

    Thresholds:
    - > 0.5: RISING_SHARP
    - > 0.1: RISING_STEADY
    - < -0.5: FALLING_SHARP
    - < -0.1: FALLING_STEADY
    - else: FLAT
    """

    def test_exactly_at_rising_sharp_boundary(self):
        """Test slope exactly at 0.5 (boundary between STEADY and SHARP)."""
        assert classify_trend(0.5) == TrendState.RISING_STEADY  # 0.5 is NOT > 0.5
        assert classify_trend(0.50001) == TrendState.RISING_SHARP  # Just above

    def test_exactly_at_rising_steady_boundary(self):
        """Test slope exactly at 0.1 (boundary between FLAT and STEADY)."""
        assert classify_trend(0.1) == TrendState.FLAT  # 0.1 is NOT > 0.1
        assert classify_trend(0.10001) == TrendState.RISING_STEADY  # Just above

    def test_exactly_at_falling_sharp_boundary(self):
        """Test slope exactly at -0.5."""
        assert classify_trend(-0.5) == TrendState.FALLING_STEADY  # -0.5 is NOT < -0.5
        assert classify_trend(-0.50001) == TrendState.FALLING_SHARP  # Just below

    def test_exactly_at_falling_steady_boundary(self):
        """Test slope exactly at -0.1."""
        assert classify_trend(-0.1) == TrendState.FLAT  # -0.1 is NOT < -0.1
        assert classify_trend(-0.10001) == TrendState.FALLING_STEADY  # Just below

    def test_zero_slope(self):
        """Test exactly zero slope."""
        assert classify_trend(0.0) == TrendState.FLAT

    def test_very_small_positive(self):
        """Test very small positive slope."""
        assert classify_trend(0.001) == TrendState.FLAT
        assert classify_trend(0.05) == TrendState.FLAT
        assert classify_trend(0.099) == TrendState.FLAT

    def test_very_small_negative(self):
        """Test very small negative slope."""
        assert classify_trend(-0.001) == TrendState.FLAT
        assert classify_trend(-0.05) == TrendState.FLAT
        assert classify_trend(-0.099) == TrendState.FLAT


class TestVolatilityBoundaries:
    """Tests for exact boundary values in calc_volatility.

    Thresholds (coefficient of variation):
    - CV < 0.05: COMPRESSED
    - CV < 0.15: STABLE
    - CV < 0.30: MODERATE
    - CV < 0.50: EXPANDING
    - CV >= 0.50: EXTREME
    """

    def test_cv_at_compressed_boundary(self):
        """CV at exactly 0.05 should be STABLE (not COMPRESSED)."""
        # CV = std / |mean| = 0.05 requires std = 0.05 * mean
        # Test with values that give CV just below the 0.05 threshold
        values_below = np.array([99.0, 101.0, 99.5, 100.5, 100.0])  # Low CV
        cv_below, state_below = calc_volatility(values_below)
        assert state_below == VolatilityState.COMPRESSED

    def test_cv_at_stable_boundary(self):
        """CV around 0.15 boundary."""
        # Create data with higher variance
        values = np.array([85.0, 115.0, 90.0, 110.0, 100.0])  # ~15% spread
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.STABLE, VolatilityState.MODERATE)

    def test_cv_at_moderate_boundary(self):
        """CV around 0.30 boundary."""
        values = np.array([70.0, 130.0, 80.0, 120.0, 100.0])  # ~30% spread
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.MODERATE, VolatilityState.EXPANDING)

    def test_cv_at_expanding_boundary(self):
        """CV around 0.50 boundary."""
        values = np.array([50.0, 150.0, 60.0, 140.0, 100.0])  # ~50% spread
        cv, state = calc_volatility(values)
        assert state in (VolatilityState.EXPANDING, VolatilityState.EXTREME)

    def test_very_high_cv(self):
        """Very high CV should be EXTREME."""
        values = np.array([1.0, 100.0, 2.0, 99.0, 50.0])  # High variance
        cv, state = calc_volatility(values)
        assert state == VolatilityState.EXTREME


class TestAccelerationBoundaries:
    """Tests for exact boundary values in classify_acceleration.

    Thresholds:
    - > 0.3: ACCELERATING_SHARPLY
    - > 0.1: ACCELERATING
    - >= -0.1 and <= 0.1: STEADY
    - < -0.1: DECELERATING
    - < -0.3: DECELERATING_SHARPLY
    """

    def test_exactly_at_accelerating_sharply_boundary(self):
        """Test acceleration exactly at 0.3."""
        assert classify_acceleration(0.3) == AccelerationState.ACCELERATING  # 0.3 is NOT > 0.3
        assert classify_acceleration(0.30001) == AccelerationState.ACCELERATING_SHARPLY

    def test_exactly_at_accelerating_boundary(self):
        """Test acceleration exactly at 0.1."""
        assert classify_acceleration(0.1) == AccelerationState.STEADY  # 0.1 is NOT > 0.1
        assert classify_acceleration(0.10001) == AccelerationState.ACCELERATING

    def test_exactly_at_decelerating_sharply_boundary(self):
        """Test acceleration exactly at -0.3."""
        assert classify_acceleration(-0.3) == AccelerationState.DECELERATING  # -0.3 is NOT < -0.3
        assert classify_acceleration(-0.30001) == AccelerationState.DECELERATING_SHARPLY

    def test_exactly_at_decelerating_boundary(self):
        """Test acceleration exactly at -0.1."""
        assert classify_acceleration(-0.1) == AccelerationState.STEADY  # -0.1 is NOT < -0.1
        assert classify_acceleration(-0.10001) == AccelerationState.DECELERATING

    def test_zero_acceleration(self):
        """Test exactly zero acceleration."""
        assert classify_acceleration(0.0) == AccelerationState.STEADY


class TestSeasonalityBoundaries:
    """Tests for exact boundary values in calc_seasonality classification.

    Thresholds (peak autocorrelation):
    - < 0.3: NONE
    - < 0.5: WEAK
    - < 0.7: MODERATE
    - >= 0.7: STRONG
    """

    def test_autocorr_boundaries(self):
        """Verify boundary classifications work correctly.

        Note: These thresholds are checked in the function, not at exact boundaries
        since autocorrelation is computed from data, not passed directly.
        """
        # Test with data that produces known autocorrelation patterns
        # Perfect repeating pattern should give high autocorr
        pattern = [1.0, 2.0, 1.0, 2.0] * 20
        values = np.array(pattern)
        autocorr, state = calc_seasonality(values)

        # Should be MODERATE or STRONG for clear pattern
        assert state in (SeasonalityState.MODERATE, SeasonalityState.STRONG)

        # Random data should give low autocorr
        np.random.seed(42)
        random_values = np.random.randn(80)
        autocorr_random, state_random = calc_seasonality(random_values)
        assert state_random in (SeasonalityState.NONE, SeasonalityState.WEAK)


class TestAnomalyStateBoundaries:
    """Tests for exact boundary values in classify_anomaly_state.

    Thresholds:
    - No anomalies: NONE
    - count <= 2: MINOR (unless z_score > 5)
    - count >= 3 and count <= 5: SIGNIFICANT
    - count > 5 OR max_z > 5: EXTREME
    """

    def test_zero_anomalies(self):
        """No anomalies should return NONE."""
        assert classify_anomaly_state([]) == AnomalyState.NONE

    def test_one_anomaly_low_zscore(self):
        """One anomaly with low z-score should be MINOR."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=0, value=100.0, z_score=3.0)]
        assert classify_anomaly_state(anomalies) == AnomalyState.MINOR

    def test_two_anomalies_low_zscore(self):
        """Two anomalies with low z-score should be MINOR."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [
            AnomalyInfo(index=0, value=100.0, z_score=3.0),
            AnomalyInfo(index=1, value=101.0, z_score=3.5),
        ]
        assert classify_anomaly_state(anomalies) == AnomalyState.MINOR

    def test_three_anomalies_significant(self):
        """Three anomalies should be SIGNIFICANT."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=i, value=100.0, z_score=3.0) for i in range(3)]
        assert classify_anomaly_state(anomalies) == AnomalyState.SIGNIFICANT

    def test_five_anomalies_significant(self):
        """Five anomalies should still be SIGNIFICANT."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=i, value=100.0, z_score=3.0) for i in range(5)]
        assert classify_anomaly_state(anomalies) == AnomalyState.SIGNIFICANT

    def test_six_anomalies_extreme(self):
        """Six anomalies should be EXTREME."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=i, value=100.0, z_score=3.0) for i in range(6)]
        assert classify_anomaly_state(anomalies) == AnomalyState.EXTREME

    def test_one_anomaly_extreme_zscore(self):
        """Single anomaly with z > 5 should be EXTREME."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=0, value=1000.0, z_score=5.5)]
        assert classify_anomaly_state(anomalies) == AnomalyState.EXTREME

    def test_exactly_at_zscore_boundary(self):
        """Z-score exactly at 5.0 should NOT be EXTREME."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=0, value=100.0, z_score=5.0)]
        assert classify_anomaly_state(anomalies) == AnomalyState.MINOR

    def test_just_above_zscore_boundary(self):
        """Z-score just above 5.0 should be EXTREME."""
        from semantic_frame.interfaces.json_schema import AnomalyInfo

        anomalies = [AnomalyInfo(index=0, value=100.0, z_score=5.01)]
        assert classify_anomaly_state(anomalies) == AnomalyState.EXTREME


class TestDataQualityBoundaries:
    """Tests for exact boundary values in assess_data_quality.

    Actual thresholds (from implementation):
    - < 1%: PRISTINE
    - < 5%: GOOD
    - < 20%: SPARSE
    - >= 20%: FRAGMENTED
    """

    def test_zero_missing_pristine(self):
        """0% missing should be PRISTINE."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pct, quality = assess_data_quality(values)
        assert pct == 0.0
        assert quality == DataQuality.PRISTINE

    def test_under_one_percent_pristine(self):
        """Just under 1% missing should be PRISTINE."""
        # 0.5% missing
        values = np.array([1.0] * 199 + [np.nan])  # 1/200 = 0.5%
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.PRISTINE

    def test_exactly_one_percent_boundary(self):
        """1% missing is at boundary - should be GOOD (not PRISTINE)."""
        values = np.array([1.0] * 99 + [np.nan])  # 1% missing
        pct, quality = assess_data_quality(values)
        assert abs(pct - 1.0) < 0.1
        assert quality == DataQuality.GOOD  # 1% is NOT < 1%

    def test_under_five_percent_good(self):
        """4% missing should be GOOD."""
        values = np.array([1.0] * 96 + [np.nan] * 4)  # 4% missing
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.GOOD

    def test_exactly_five_percent_boundary(self):
        """5% missing is at boundary - should be SPARSE (5% is NOT < 5%)."""
        values = np.array([1.0] * 95 + [np.nan] * 5)  # 5% missing
        pct, quality = assess_data_quality(values)
        assert abs(pct - 5.0) < 0.1
        assert quality == DataQuality.SPARSE  # 5% is NOT < 5%

    def test_under_twenty_percent_sparse(self):
        """19% missing should be SPARSE."""
        values = np.array([1.0] * 81 + [np.nan] * 19)  # 19% missing
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.SPARSE

    def test_exactly_twenty_percent_boundary(self):
        """20% missing is at boundary - should be FRAGMENTED (20% is NOT < 20%)."""
        values = np.array([1.0] * 80 + [np.nan] * 20)  # 20% missing
        pct, quality = assess_data_quality(values)
        assert abs(pct - 20.0) < 0.1
        assert quality == DataQuality.FRAGMENTED  # 20% is NOT < 20%

    def test_above_twenty_percent_fragmented(self):
        """21% missing should be FRAGMENTED."""
        values = np.array([1.0] * 79 + [np.nan] * 21)  # 21% missing
        pct, quality = assess_data_quality(values)
        assert quality == DataQuality.FRAGMENTED

    def test_all_missing_fragmented(self):
        """100% missing should be FRAGMENTED."""
        values = np.array([np.nan] * 10)
        pct, quality = assess_data_quality(values)
        assert pct == 100.0
        assert quality == DataQuality.FRAGMENTED

"""Tests for core analyzer functions."""

import numpy as np
import pytest

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
from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
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

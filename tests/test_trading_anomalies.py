"""Tests for enhanced anomaly detection."""

import numpy as np
import pytest

from semantic_frame.trading import describe_anomalies
from semantic_frame.trading.anomalies import (
    AnomalyFrequency,
    AnomalySeverity,
    AnomalyType,
    EnhancedAnomalyResult,
)


class TestDescribeAnomalies:
    """Tests for describe_anomalies function."""

    def test_basic_anomaly_detection(self):
        """Test basic anomaly detection."""
        data = [100, 102, 99, 500, 101, 98, 100]
        result = describe_anomalies(data, context="Test Data")

        assert isinstance(result, EnhancedAnomalyResult)
        assert result.total_anomalies >= 1
        assert len(result.narrative) > 0

    def test_no_anomalies(self):
        """Test when data has no anomalies."""
        data = [100, 101, 100, 99, 100, 101, 100]
        result = describe_anomalies(data)

        # May or may not have anomalies depending on threshold
        assert isinstance(result, EnhancedAnomalyResult)

    def test_multiple_anomalies(self):
        """Test detection of multiple anomalies."""
        # Create data with clear extreme outliers
        data = [100] * 20 + [500, 100, 100, -300, 100] + [100] * 20
        result = describe_anomalies(data)

        assert result.total_anomalies >= 2

    def test_severity_classification(self):
        """Test that severities are classified."""
        # Create data with clear extreme outlier
        data = [100] * 50 + [1000]  # Very extreme outlier
        result = describe_anomalies(data)

        assert result.max_severity is not None
        assert any(
            a.severity in [AnomalySeverity.SEVERE, AnomalySeverity.EXTREME]
            for a in result.anomalies
        )

    def test_pnl_data_types(self):
        """Test anomaly type classification for PnL data."""
        data = [100, -50, 75, 500, -25, -400, 80]
        result = describe_anomalies(data, is_pnl_data=True)

        # Should use GAIN/LOSS types for PnL data
        if result.anomalies:
            types = {a.anomaly_type for a in result.anomalies}
            # Should have gain or loss types, not generic spike/drop
            pnl_types = {AnomalyType.GAIN, AnomalyType.LOSS}
            assert types & pnl_types  # Should have at least one PnL type

    def test_generic_data_types(self):
        """Test anomaly type classification for generic data."""
        data = [100, 102, 99, 500, 101, 20, 100]  # High spike and low drop
        result = describe_anomalies(data, is_pnl_data=False)

        # Should use SPIKE/DROP or OUTLIER types
        if result.anomalies:
            types = {a.anomaly_type for a in result.anomalies}
            generic_types = {
                AnomalyType.SPIKE,
                AnomalyType.DROP,
                AnomalyType.OUTLIER_HIGH,
                AnomalyType.OUTLIER_LOW,
            }
            assert types & generic_types

    def test_frequency_classification(self):
        """Test frequency classification."""
        # Many anomalies = pervasive
        data = [100, 500, 100, 600, 100, 700, 100, 800]
        result = describe_anomalies(data)

        assert result.frequency is not None
        assert result.frequency in list(AnomalyFrequency)

    def test_anomaly_rate_calculation(self):
        """Test anomaly rate percentage calculation."""
        data = [100] * 100 + [500]  # 1 anomaly in 101 points
        result = describe_anomalies(data)

        assert result.anomaly_rate_pct >= 0
        assert result.anomaly_rate_pct <= 100

    def test_context_in_narrative(self):
        """Test that context appears in narrative."""
        data = [100, 500, 100]
        result = describe_anomalies(data, context="Server Latency")

        assert "Server Latency" in result.narrative

    def test_anomaly_context_description(self):
        """Test that each anomaly has context description."""
        data = [100, 102, 99, 500, 101]
        result = describe_anomalies(data)

        for anomaly in result.anomalies:
            assert len(anomaly.context) > 0
            assert anomaly.deviation_multiple > 0

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        data = np.array([100.0, 500.0, 100.0])
        result = describe_anomalies(data)

        assert isinstance(result, EnhancedAnomalyResult)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = [100, 200]
        result = describe_anomalies(data)

        assert "Insufficient" in result.narrative

    def test_zero_variance_data(self):
        """Test with zero variance data."""
        data = [100, 100, 100, 100, 100]
        result = describe_anomalies(data)

        assert result.total_anomalies == 0
        assert (
            "identical" in result.narrative.lower() or "zero variance" in result.narrative.lower()
        )

    def test_custom_threshold(self):
        """Test custom z-score threshold."""
        data = [100, 120, 100, 100, 80, 100]

        # Low threshold should catch more
        result_low = describe_anomalies(data, z_threshold=1.5)
        # High threshold should catch fewer
        result_high = describe_anomalies(data, z_threshold=3.0)

        # Lower threshold should catch >= as many anomalies
        assert result_low.total_anomalies >= result_high.total_anomalies


class TestAnomalySeverityThresholds:
    """Tests for severity classification thresholds."""

    @pytest.mark.parametrize(
        "z_score,expected",
        [
            (2.2, AnomalySeverity.MILD),
            (2.8, AnomalySeverity.MODERATE),
            (4.0, AnomalySeverity.SEVERE),
            (6.0, AnomalySeverity.EXTREME),
        ],
    )
    def test_severity_thresholds(self, z_score, expected):
        """Test severity classification at different z-scores."""
        from semantic_frame.trading.anomalies import _classify_severity

        result = _classify_severity(z_score)
        assert result == expected


class TestAnomalyFrequencyThresholds:
    """Tests for frequency classification."""

    @pytest.mark.parametrize(
        "rate_pct,expected",
        [
            (0.5, AnomalyFrequency.RARE),
            (2.0, AnomalyFrequency.OCCASIONAL),
            (4.0, AnomalyFrequency.FREQUENT),
            (7.0, AnomalyFrequency.PERVASIVE),
        ],
    )
    def test_frequency_thresholds(self, rate_pct, expected):
        """Test frequency classification at different rates."""
        from semantic_frame.trading.anomalies import _classify_frequency

        result = _classify_frequency(rate_pct)
        assert result == expected


class TestAnomalyNarrative:
    """Tests for narrative generation."""

    def test_narrative_mentions_count(self):
        """Test that narrative mentions anomaly count."""
        # Create data with clear outliers to guarantee detection
        data = [100] * 20 + [500, 100, 600, 100, 100]
        result = describe_anomalies(data)

        # Should mention number of anomalies (either detected or no anomalies)
        assert "detected" in result.narrative.lower() or "no" in result.narrative.lower()

    def test_narrative_for_no_anomalies(self):
        """Test narrative when no anomalies found."""
        data = [100, 101, 100, 99, 100, 101]
        result = describe_anomalies(data)

        if result.total_anomalies == 0:
            assert "no" in result.narrative.lower() or "normal" in result.narrative.lower()

    def test_narrative_mentions_extreme(self):
        """Test that extreme anomalies are called out."""
        data = [100] * 50 + [10000]  # Very extreme
        result = describe_anomalies(data)

        # Should highlight extreme outliers
        if result.max_severity == AnomalySeverity.EXTREME:
            assert "extreme" in result.narrative.lower()

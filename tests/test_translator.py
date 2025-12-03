"""Tests for the translator/analysis pipeline."""

import numpy as np

from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    TrendState,
)
from semantic_frame.core.translator import analyze_series
from semantic_frame.interfaces.json_schema import SemanticResult


class TestAnalyzeSeries:
    """Tests for analyze_series function."""

    def test_basic_analysis(self):
        """Basic analysis should return SemanticResult."""
        values = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        result = analyze_series(values)

        assert isinstance(result, SemanticResult)
        assert result.narrative is not None
        assert len(result.narrative) > 0

    def test_with_context(self):
        """Context should appear in narrative."""
        values = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        result = analyze_series(values, context="CPU Usage")

        assert "CPU Usage" in result.narrative
        assert result.context == "CPU Usage"

    def test_empty_array(self):
        """Empty array should return valid result."""
        values = np.array([])
        result = analyze_series(values)

        assert result.data_quality == DataQuality.FRAGMENTED
        assert "no valid data" in result.narrative.lower()

    def test_all_nan_array(self):
        """All-NaN array should be handled like empty."""
        values = np.array([np.nan, np.nan, np.nan])
        result = analyze_series(values)

        assert result.data_quality == DataQuality.FRAGMENTED

    def test_trend_detection(self):
        """Rising data should be detected as rising trend."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = analyze_series(values)

        assert result.trend in (TrendState.RISING_SHARP, TrendState.RISING_STEADY)

    def test_anomaly_detection(self):
        """Clear outlier should be detected."""
        values = np.array([10.0] * 10 + [100.0] + [10.0] * 10)
        result = analyze_series(values)

        assert len(result.anomalies) >= 1
        assert result.anomaly_state != AnomalyState.NONE

    def test_compression_ratio(self):
        """Compression ratio should be positive."""
        values = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        result = analyze_series(values)

        assert 0 <= result.compression_ratio <= 1

    def test_profile_statistics(self):
        """Profile should contain correct statistics."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze_series(values)

        assert result.profile.count == 5
        assert result.profile.mean == 3.0
        assert result.profile.median == 3.0
        assert result.profile.min_val == 1.0
        assert result.profile.max_val == 5.0

    def test_nan_handling_in_profile(self):
        """NaN values should be excluded from statistics."""
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = analyze_series(values)

        assert result.profile.count == 5  # Total count includes NaN
        assert result.profile.missing_pct == 20.0
        # Mean should be calculated from clean values only
        assert result.profile.mean == 3.0  # (1+2+4+5)/4 = 3

    def test_seasonality_detection_long_series(self):
        """Long periodic series should detect seasonality."""
        # Create sinusoidal pattern
        x = np.linspace(0, 4 * np.pi, 100)
        values = np.sin(x) * 10 + 50
        result = analyze_series(values)

        # Should detect some level of seasonality
        assert result.seasonality is not None

    def test_distribution_calculated(self):
        """Distribution shape should be calculated for sufficient data."""
        np.random.seed(42)
        values = np.random.normal(50, 10, 100)
        result = analyze_series(values)

        assert result.distribution is not None

    def test_anomalies_limited_to_five(self):
        """Anomalies should be limited to max 5."""
        # Create data with many outliers
        values = np.array([10.0] * 50 + [100.0] * 10)
        result = analyze_series(values)

        assert len(result.anomalies) <= 5


class TestSemanticResultMethods:
    """Tests for SemanticResult methods."""

    def test_to_prompt(self):
        """to_prompt should format for LLM injection."""
        values = np.array([10.0, 12.0, 11.0])
        result = analyze_series(values, context="Test Data")

        prompt = result.to_prompt()
        assert "DATA CONTEXT:" in prompt
        assert result.narrative in prompt

    def test_to_json_str(self):
        """to_json_str should produce valid JSON."""
        import json

        values = np.array([10.0, 12.0, 11.0])
        result = analyze_series(values)

        json_str = result.to_json_str()
        parsed = json.loads(json_str)

        assert "narrative" in parsed
        assert "trend" in parsed
        assert "volatility" in parsed

    def test_model_dump(self):
        """model_dump should work with aliases."""
        values = np.array([10.0, 12.0, 11.0])
        result = analyze_series(values)

        data = result.model_dump(by_alias=True)
        assert "min" in data["profile"]
        assert "max" in data["profile"]

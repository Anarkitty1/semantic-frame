"""Tests for narrator modules."""

import pytest

from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
    TrendState,
    VolatilityState,
)
from semantic_frame.interfaces.json_schema import AnomalyInfo, SeriesProfile
from semantic_frame.narrators.distribution import generate_distribution_narrative
from semantic_frame.narrators.time_series import generate_time_series_narrative


@pytest.fixture
def sample_profile() -> SeriesProfile:
    """Create a sample profile for testing."""
    return SeriesProfile(
        count=100,
        mean=50.0,
        median=49.5,
        std=10.0,
        min=20.0,
        max=80.0,
        missing_pct=0.0,
    )


@pytest.fixture
def sample_anomalies() -> list[AnomalyInfo]:
    """Create sample anomalies for testing."""
    return [
        AnomalyInfo(index=5, value=100.0, z_score=4.5),
        AnomalyInfo(index=42, value=95.0, z_score=3.8),
    ]


class TestTimeSeriesNarrator:
    """Tests for time series narrative generation."""

    def test_basic_narrative(self, sample_profile):
        """Basic narrative should contain key elements."""
        narrative = generate_time_series_narrative(
            trend=TrendState.RISING_STEADY,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "steadily rising" in narrative
        assert "stable" in narrative
        assert "49.5" in narrative or "49.50" in narrative  # Median

    def test_with_context(self, sample_profile):
        """Context should be included in narrative."""
        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
            context="CPU Usage",
        )

        assert "CPU Usage" in narrative

    def test_single_anomaly(self, sample_profile):
        """Single anomaly should be reported correctly."""
        anomalies = [AnomalyInfo(index=5, value=100.0, z_score=4.5)]

        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.MINOR,
            anomalies=anomalies,
            profile=sample_profile,
        )

        assert "1 anomaly" in narrative
        assert "index 5" in narrative

    def test_multiple_anomalies(self, sample_profile, sample_anomalies):
        """Multiple anomalies should be reported with count."""
        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.MINOR,
            anomalies=sample_anomalies,
            profile=sample_profile,
        )

        assert "2 anomalies" in narrative

    def test_seasonality_mentioned(self, sample_profile):
        """Seasonality should be mentioned when present."""
        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
            seasonality=SeasonalityState.STRONG,
        )

        assert "seasonality" in narrative.lower()

    def test_poor_data_quality_mentioned(self):
        """Poor data quality should be mentioned."""
        # Create a profile with 15% missing data
        poor_quality_profile = SeriesProfile(
            count=100,
            mean=50.0,
            median=49.5,
            std=10.0,
            min=20.0,
            max=80.0,
            missing_pct=15.0,
        )

        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=poor_quality_profile,
            data_quality=DataQuality.SPARSE,
        )

        assert "sparse" in narrative.lower() or "missing" in narrative.lower()

    def test_good_quality_not_mentioned(self, sample_profile):
        """Good data quality should not be explicitly mentioned."""
        narrative = generate_time_series_narrative(
            trend=TrendState.FLAT,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
            data_quality=DataQuality.PRISTINE,
        )

        # "quality" might appear in other contexts, but not as a warning
        assert "fragmented" not in narrative.lower()
        assert "sparse" not in narrative.lower()


class TestDistributionNarrator:
    """Tests for distribution narrative generation."""

    def test_basic_narrative(self, sample_profile):
        """Basic narrative should contain distribution info."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.NORMAL,
            volatility=VolatilityState.MODERATE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "normally distributed" in narrative
        assert "moderate" in narrative.lower()

    def test_with_context(self, sample_profile):
        """Context should be included."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.NORMAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
            context="Test Scores",
        )

        assert "Test Scores" in narrative

    def test_skewed_distribution(self, sample_profile):
        """Skewed distribution should mention skew direction."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.RIGHT_SKEWED,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "right" in narrative.lower() and "skew" in narrative.lower()

    def test_includes_statistics(self, sample_profile):
        """Narrative should include key statistics."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.NORMAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "mean" in narrative.lower() or "50.0" in narrative
        assert "median" in narrative.lower() or "49.5" in narrative

    def test_outliers_mentioned(self, sample_profile, sample_anomalies):
        """Outliers should be mentioned with extreme value."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.NORMAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.MINOR,
            anomalies=sample_anomalies,
            profile=sample_profile,
        )

        assert "outlier" in narrative.lower()

    def test_left_skewed_distribution(self, sample_profile):
        """Left-skewed distribution should mention skew direction.

        Tests line 85: LEFT_SKEWED branch in distribution narrator.
        """
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.LEFT_SKEWED,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "left" in narrative.lower() and "skew" in narrative.lower()

    def test_bimodal_distribution(self, sample_profile):
        """Bimodal distribution should be mentioned.

        Tests line 92: BIMODAL branch in distribution narrator.
        """
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.BIMODAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "bimodal" in narrative.lower()

    def test_bimodal_with_anomalies(self, sample_profile, sample_anomalies):
        """Bimodal distribution with anomalies.

        Tests combined paths for bimodal and anomaly detection.
        """
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.BIMODAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.MINOR,
            anomalies=sample_anomalies,
            profile=sample_profile,
        )

        assert "bimodal" in narrative.lower()
        assert "outlier" in narrative.lower()

    def test_sparse_data_quality_warning(self):
        """Sparse data quality should trigger warning.

        Tests line 110: Data quality warning template.
        """
        sparse_profile = SeriesProfile(
            count=100,
            mean=50.0,
            median=49.5,
            std=10.0,
            min=20.0,
            max=80.0,
            missing_pct=15.0,
        )

        narrative = generate_distribution_narrative(
            distribution=DistributionShape.NORMAL,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sparse_profile,
            data_quality=DataQuality.SPARSE,
        )

        assert "sparse" in narrative.lower() or "missing" in narrative.lower()

    def test_uniform_distribution(self, sample_profile):
        """Uniform distribution should be mentioned."""
        narrative = generate_distribution_narrative(
            distribution=DistributionShape.UNIFORM,
            volatility=VolatilityState.STABLE,
            anomaly_state=AnomalyState.NONE,
            anomalies=[],
            profile=sample_profile,
        )

        assert "uniform" in narrative.lower()

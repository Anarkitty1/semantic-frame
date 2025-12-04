"""Tests for semantic vocabulary enums."""

from semantic_frame.core.enums import (
    AccelerationState,
    AnomalyState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
    TrendState,
    VolatilityState,
)


class TestTrendState:
    """Tests for TrendState enum."""

    def test_values_are_strings(self):
        """Enum values should be descriptive strings."""
        assert TrendState.RISING_SHARP.value == "rapidly rising"
        assert TrendState.FALLING_SHARP.value == "rapidly falling"
        assert TrendState.FLAT.value == "flat/stationary"

    def test_string_subclass(self):
        """Enums should be string subclass for easy serialization."""
        assert isinstance(TrendState.FLAT, str)
        assert TrendState.FLAT == "flat/stationary"

    def test_all_members_exist(self):
        """All expected trend states should exist."""
        expected = ["RISING_SHARP", "RISING_STEADY", "FLAT", "FALLING_STEADY", "FALLING_SHARP"]
        actual = [e.name for e in TrendState]
        assert set(expected) == set(actual)


class TestVolatilityState:
    """Tests for VolatilityState enum."""

    def test_values_are_strings(self):
        assert VolatilityState.COMPRESSED.value == "compressed"
        assert VolatilityState.EXTREME.value == "extreme"

    def test_all_members_exist(self):
        expected = ["COMPRESSED", "STABLE", "MODERATE", "EXPANDING", "EXTREME"]
        actual = [e.name for e in VolatilityState]
        assert set(expected) == set(actual)


class TestDataQuality:
    """Tests for DataQuality enum."""

    def test_values_are_strings(self):
        assert DataQuality.PRISTINE.value == "high quality"
        assert DataQuality.FRAGMENTED.value == "fragmented"

    def test_all_members_exist(self):
        expected = ["PRISTINE", "GOOD", "SPARSE", "FRAGMENTED"]
        actual = [e.name for e in DataQuality]
        assert set(expected) == set(actual)


class TestAnomalyState:
    """Tests for AnomalyState enum."""

    def test_values_are_strings(self):
        assert AnomalyState.NONE.value == "no anomalies"
        assert AnomalyState.EXTREME.value == "extreme outliers"

    def test_all_members_exist(self):
        expected = ["NONE", "MINOR", "SIGNIFICANT", "EXTREME"]
        actual = [e.name for e in AnomalyState]
        assert set(expected) == set(actual)


class TestSeasonalityState:
    """Tests for SeasonalityState enum."""

    def test_values_are_strings(self):
        assert SeasonalityState.NONE.value == "no seasonality"
        assert SeasonalityState.STRONG.value == "strong seasonality"

    def test_all_members_exist(self):
        expected = ["NONE", "WEAK", "MODERATE", "STRONG"]
        actual = [e.name for e in SeasonalityState]
        assert set(expected) == set(actual)


class TestDistributionShape:
    """Tests for DistributionShape enum."""

    def test_values_are_strings(self):
        assert DistributionShape.NORMAL.value == "normally distributed"
        assert DistributionShape.BIMODAL.value == "bimodal"

    def test_all_members_exist(self):
        expected = ["NORMAL", "LEFT_SKEWED", "RIGHT_SKEWED", "BIMODAL", "UNIFORM"]
        actual = [e.name for e in DistributionShape]
        assert set(expected) == set(actual)


class TestAccelerationState:
    """Tests for AccelerationState enum."""

    def test_values_are_strings(self):
        """Enum values should be descriptive strings."""
        assert AccelerationState.ACCELERATING_SHARPLY.value == "rapidly accelerating"
        assert AccelerationState.ACCELERATING.value == "accelerating"
        assert AccelerationState.STEADY.value == "steady rate of change"
        assert AccelerationState.DECELERATING.value == "decelerating"
        assert AccelerationState.DECELERATING_SHARPLY.value == "rapidly decelerating"

    def test_string_subclass(self):
        """Enums should be string subclass for easy serialization."""
        assert isinstance(AccelerationState.STEADY, str)
        assert AccelerationState.STEADY == "steady rate of change"

    def test_all_members_exist(self):
        """All expected acceleration states should exist."""
        expected = [
            "ACCELERATING_SHARPLY",
            "ACCELERATING",
            "STEADY",
            "DECELERATING",
            "DECELERATING_SHARPLY",
        ]
        actual = [e.name for e in AccelerationState]
        assert set(expected) == set(actual)

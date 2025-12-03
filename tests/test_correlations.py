"""Tests for correlation analysis functions."""

import numpy as np

from semantic_frame.core.correlations import (
    calc_correlation_matrix,
    classify_correlation,
    identify_significant_correlations,
)
from semantic_frame.core.enums import CorrelationState


class TestClassifyCorrelation:
    """Tests for classify_correlation function."""

    def test_strong_positive(self) -> None:
        """Test strong positive correlation classification."""
        assert classify_correlation(0.8) == CorrelationState.STRONG_POSITIVE
        assert classify_correlation(0.95) == CorrelationState.STRONG_POSITIVE
        assert classify_correlation(1.0) == CorrelationState.STRONG_POSITIVE

    def test_moderate_positive(self) -> None:
        """Test moderate positive correlation classification."""
        assert classify_correlation(0.5) == CorrelationState.MODERATE_POSITIVE
        assert classify_correlation(0.6) == CorrelationState.MODERATE_POSITIVE
        assert classify_correlation(0.7) == CorrelationState.MODERATE_POSITIVE

    def test_weak(self) -> None:
        """Test weak/no correlation classification."""
        assert classify_correlation(0.0) == CorrelationState.WEAK
        assert classify_correlation(0.3) == CorrelationState.WEAK
        assert classify_correlation(0.4) == CorrelationState.WEAK
        assert classify_correlation(-0.3) == CorrelationState.WEAK
        assert classify_correlation(-0.4) == CorrelationState.WEAK

    def test_moderate_negative(self) -> None:
        """Test moderate negative correlation classification."""
        assert classify_correlation(-0.5) == CorrelationState.MODERATE_NEGATIVE
        assert classify_correlation(-0.6) == CorrelationState.MODERATE_NEGATIVE
        assert classify_correlation(-0.7) == CorrelationState.MODERATE_NEGATIVE

    def test_strong_negative(self) -> None:
        """Test strong negative correlation classification."""
        assert classify_correlation(-0.8) == CorrelationState.STRONG_NEGATIVE
        assert classify_correlation(-0.95) == CorrelationState.STRONG_NEGATIVE
        assert classify_correlation(-1.0) == CorrelationState.STRONG_NEGATIVE

    def test_boundary_values(self) -> None:
        """Test boundary values between categories."""
        # Just above 0.7 should be strong positive
        assert classify_correlation(0.71) == CorrelationState.STRONG_POSITIVE
        # Just below -0.7 should be strong negative
        assert classify_correlation(-0.71) == CorrelationState.STRONG_NEGATIVE


class TestCalcCorrelationMatrix:
    """Tests for calc_correlation_matrix function."""

    def test_perfect_positive_correlation(self) -> None:
        """Test detection of perfect positive correlation."""
        values = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        }
        matrix = calc_correlation_matrix(values)
        assert ("a", "b") in matrix
        assert matrix[("a", "b")] > 0.99

    def test_perfect_negative_correlation(self) -> None:
        """Test detection of perfect negative correlation."""
        values = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }
        matrix = calc_correlation_matrix(values)
        assert ("a", "b") in matrix
        assert matrix[("a", "b")] < -0.99

    def test_no_correlation(self) -> None:
        """Test weak correlation between random data."""
        np.random.seed(42)
        values = {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        }
        matrix = calc_correlation_matrix(values)
        # Random data should have weak correlation
        assert abs(matrix[("a", "b")]) < 0.3

    def test_single_column(self) -> None:
        """Test that single column returns empty matrix."""
        values = {"a": np.array([1.0, 2.0, 3.0])}
        matrix = calc_correlation_matrix(values)
        assert matrix == {}

    def test_empty_input(self) -> None:
        """Test that empty input returns empty matrix."""
        matrix = calc_correlation_matrix({})
        assert matrix == {}

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        values = {
            "a": np.array([1.0, np.nan, 3.0, 4.0, 5.0]),
            "b": np.array([2.0, 4.0, np.nan, 8.0, 10.0]),
        }
        matrix = calc_correlation_matrix(values)
        # Should compute correlation on valid pairs
        assert ("a", "b") in matrix

    def test_insufficient_data(self) -> None:
        """Test that insufficient data (< 3 valid pairs) is skipped."""
        values = {
            "a": np.array([1.0, np.nan]),
            "b": np.array([2.0, np.nan]),
        }
        matrix = calc_correlation_matrix(values)
        # Not enough valid pairs - should be empty
        assert ("a", "b") not in matrix

    def test_spearman_method(self) -> None:
        """Test Spearman correlation method."""
        values = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([1.0, 4.0, 9.0, 16.0, 25.0]),  # Monotonic but not linear
        }
        pearson_matrix = calc_correlation_matrix(values, method="pearson")
        spearman_matrix = calc_correlation_matrix(values, method="spearman")

        # Spearman should give perfect correlation for monotonic relationship
        assert spearman_matrix[("a", "b")] > 0.99
        # Pearson will be high but not perfect
        assert pearson_matrix[("a", "b")] > 0.9

    def test_multiple_columns(self) -> None:
        """Test correlation matrix with multiple columns."""
        values = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),  # Perfect corr with a
            "c": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # Perfect negative corr with a
        }
        matrix = calc_correlation_matrix(values)

        # Should have 3 pairs
        assert len(matrix) == 3
        assert ("a", "b") in matrix
        assert ("a", "c") in matrix
        assert ("b", "c") in matrix

        # Check correlations
        assert matrix[("a", "b")] > 0.99  # Strong positive
        assert matrix[("a", "c")] < -0.99  # Strong negative
        assert matrix[("b", "c")] < -0.99  # Strong negative


class TestIdentifySignificantCorrelations:
    """Tests for identify_significant_correlations function."""

    def test_filters_by_threshold(self) -> None:
        """Test that correlations are filtered by threshold."""
        correlations = {
            ("a", "b"): 0.8,
            ("c", "d"): 0.3,
            ("e", "f"): -0.7,
        }
        significant = identify_significant_correlations(correlations, threshold=0.5)

        assert len(significant) == 2
        # Should include strong positive and strong negative
        columns = [(s[0], s[1]) for s in significant]
        assert ("a", "b") in columns
        assert ("e", "f") in columns
        assert ("c", "d") not in columns

    def test_sorted_by_strength(self) -> None:
        """Test that results are sorted by absolute correlation."""
        correlations = {
            ("a", "b"): 0.6,
            ("c", "d"): 0.9,
            ("e", "f"): -0.8,
        }
        significant = identify_significant_correlations(correlations, threshold=0.5)

        # Should be sorted by |r| descending
        assert significant[0][2] == 0.9
        assert abs(significant[1][2]) == 0.8
        assert significant[2][2] == 0.6

    def test_includes_state(self) -> None:
        """Test that CorrelationState is included in results."""
        correlations = {
            ("a", "b"): 0.8,
            ("c", "d"): -0.9,
        }
        significant = identify_significant_correlations(correlations, threshold=0.5)

        # Check states are included
        assert significant[0][3] == CorrelationState.STRONG_NEGATIVE
        assert significant[1][3] == CorrelationState.STRONG_POSITIVE

    def test_empty_input(self) -> None:
        """Test that empty input returns empty list."""
        significant = identify_significant_correlations({}, threshold=0.5)
        assert significant == []

    def test_none_above_threshold(self) -> None:
        """Test when no correlations are above threshold."""
        correlations = {
            ("a", "b"): 0.3,
            ("c", "d"): -0.2,
        }
        significant = identify_significant_correlations(correlations, threshold=0.5)
        assert significant == []

    def test_default_threshold(self) -> None:
        """Test default threshold of 0.5."""
        correlations = {
            ("a", "b"): 0.5,  # Exactly at threshold
            ("c", "d"): 0.49,  # Just below
        }
        significant = identify_significant_correlations(correlations)  # Default 0.5

        assert len(significant) == 1
        assert significant[0][0] == "a"


class TestCorrelationStateEnum:
    """Tests for CorrelationState enum values."""

    def test_all_members_exist(self) -> None:
        """Test that all expected enum members exist."""
        expected = [
            "STRONG_POSITIVE",
            "MODERATE_POSITIVE",
            "WEAK",
            "MODERATE_NEGATIVE",
            "STRONG_NEGATIVE",
        ]
        actual = [e.name for e in CorrelationState]
        assert set(expected) == set(actual)

    def test_string_serialization(self) -> None:
        """Test that enum values are human-readable strings."""
        assert isinstance(CorrelationState.STRONG_POSITIVE, str)
        assert CorrelationState.STRONG_POSITIVE == "strongly correlated"
        assert CorrelationState.STRONG_NEGATIVE == "strongly inverse"

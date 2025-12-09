"""Tests for benchmarks/metrics.py.

Tests token counting, accuracy metrics, classification metrics, and response parsing.
"""

import pytest

from benchmarks.metrics import (
    AccuracyMetrics,
    AggregatedResults,
    AnomalyDetectionMetrics,
    ClassificationMetrics,
    CostMetrics,
    TokenMetrics,
    TrialResult,
    count_tokens,
    detect_hallucination,
    parse_llm_response,
)


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self) -> None:
        """Test empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_short_string(self) -> None:
        """Test short string token count."""
        result = count_tokens("Hello world")
        assert result > 0
        assert result < 10  # Should be around 2-3 tokens

    def test_longer_string(self) -> None:
        """Test longer string has more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello world, how are you today?")
        assert long > short

    def test_numerical_string(self) -> None:
        """Test numerical data string."""
        data = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        result = count_tokens(data)
        assert result > 10  # Each number is roughly a token


class TestTokenMetrics:
    """Tests for TokenMetrics dataclass."""

    def test_compute_compression_ratio(self) -> None:
        """Test compression ratio computation."""
        raw = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
        compressed = "mean: 8"
        metrics = TokenMetrics.compute(raw, compressed)

        assert metrics.raw_tokens > metrics.compressed_tokens
        assert 0 < metrics.compression_ratio < 1

    def test_compute_zero_raw_tokens(self) -> None:
        """Test handling of empty raw data."""
        metrics = TokenMetrics.compute("", "some output")
        assert metrics.compression_ratio == 0

    def test_no_compression(self) -> None:
        """Test when output equals input size."""
        text = "Hello world"
        metrics = TokenMetrics.compute(text, text)
        assert metrics.compression_ratio == 0


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics static methods."""

    def test_numerical_proximity_exact_match(self) -> None:
        """Test exact match returns 1.0."""
        score = AccuracyMetrics.numerical_proximity_score(10.0, 10.0)
        assert score == 1.0

    def test_numerical_proximity_within_tolerance(self) -> None:
        """Test value within tolerance returns 1.0."""
        score = AccuracyMetrics.numerical_proximity_score(10.05, 10.0, tolerance=0.01)
        assert score == 1.0

    def test_numerical_proximity_outside_tolerance(self) -> None:
        """Test value outside tolerance returns < 1.0."""
        score = AccuracyMetrics.numerical_proximity_score(11.0, 10.0, tolerance=0.01)
        assert score < 1.0
        assert score > 0.0

    def test_numerical_proximity_large_error(self) -> None:
        """Test large error returns 0."""
        score = AccuracyMetrics.numerical_proximity_score(100.0, 10.0)
        assert score == 0.0

    def test_numerical_proximity_zero_actual(self) -> None:
        """Test handling of zero actual value."""
        # Within tolerance
        score1 = AccuracyMetrics.numerical_proximity_score(0.005, 0.0, tolerance=0.01)
        assert score1 == 1.0

        # Outside tolerance
        score2 = AccuracyMetrics.numerical_proximity_score(1.0, 0.0, tolerance=0.01)
        assert score2 == 0.0

    def test_check_exact_match_numeric_exact(self) -> None:
        """Test exact numeric match."""
        assert AccuracyMetrics.check_exact_match(10.0, 10.0) is True

    def test_check_exact_match_numeric_within_tolerance(self) -> None:
        """Test numeric match within tolerance."""
        assert AccuracyMetrics.check_exact_match(10.000001, 10.0, tolerance=1e-5) is True

    def test_check_exact_match_numeric_outside_tolerance(self) -> None:
        """Test numeric match outside tolerance."""
        assert AccuracyMetrics.check_exact_match(10.1, 10.0, tolerance=1e-6) is False

    def test_check_exact_match_string(self) -> None:
        """Test string matching."""
        assert AccuracyMetrics.check_exact_match("rising", "rising") is True
        assert AccuracyMetrics.check_exact_match("RISING", "rising") is True
        assert AccuracyMetrics.check_exact_match("  rising  ", "rising") is True
        assert AccuracyMetrics.check_exact_match("falling", "rising") is False


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_perfect_precision(self) -> None:
        """Test precision with no false positives."""
        metrics = ClassificationMetrics(
            true_positives=10, false_positives=0, true_negatives=90, false_negatives=0
        )
        assert metrics.precision == 1.0

    def test_zero_precision(self) -> None:
        """Test precision with no true positives."""
        metrics = ClassificationMetrics(
            true_positives=0, false_positives=10, true_negatives=90, false_negatives=0
        )
        assert metrics.precision == 0.0

    def test_perfect_recall(self) -> None:
        """Test recall with no false negatives."""
        metrics = ClassificationMetrics(
            true_positives=10, false_positives=5, true_negatives=85, false_negatives=0
        )
        assert metrics.recall == 1.0

    def test_zero_recall(self) -> None:
        """Test recall with no true positives."""
        metrics = ClassificationMetrics(
            true_positives=0, false_positives=0, true_negatives=90, false_negatives=10
        )
        assert metrics.recall == 0.0

    def test_f1_score(self) -> None:
        """Test F1 score calculation."""
        metrics = ClassificationMetrics(
            true_positives=80, false_positives=10, true_negatives=5, false_negatives=5
        )
        # Precision = 80/90, Recall = 80/85
        expected_precision = 80 / 90
        expected_recall = 80 / 85
        expected_f1 = (
            2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        )
        assert abs(metrics.f1_score - expected_f1) < 1e-6

    def test_accuracy(self) -> None:
        """Test accuracy calculation."""
        metrics = ClassificationMetrics(
            true_positives=40, false_positives=5, true_negatives=50, false_negatives=5
        )
        # (40 + 50) / 100 = 0.9
        assert metrics.accuracy == 0.9

    def test_empty_metrics(self) -> None:
        """Test metrics with all zeros."""
        metrics = ClassificationMetrics()
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.accuracy == 0.0


class TestAnomalyDetectionMetrics:
    """Tests for AnomalyDetectionMetrics."""

    def test_compute_perfect_detection(self) -> None:
        """Test perfect anomaly detection."""
        predicted = {10, 20, 30}
        actual = {10, 20, 30}
        metrics = AnomalyDetectionMetrics.compute(predicted, actual, series_length=100)

        assert metrics.point_wise_precision == 1.0
        assert metrics.point_wise_recall == 1.0
        assert metrics.point_wise_f1 == 1.0

    def test_compute_no_detection(self) -> None:
        """Test no anomalies detected."""
        predicted = set()
        actual = {10, 20, 30}
        metrics = AnomalyDetectionMetrics.compute(predicted, actual, series_length=100)

        assert metrics.point_wise_precision == 0.0
        assert metrics.point_wise_recall == 0.0
        assert metrics.point_wise_f1 == 0.0

    def test_compute_all_false_positives(self) -> None:
        """Test all false positives."""
        predicted = {40, 50, 60}
        actual = {10, 20, 30}
        metrics = AnomalyDetectionMetrics.compute(predicted, actual, series_length=100)

        assert metrics.point_wise_precision == 0.0
        assert metrics.point_wise_recall == 0.0

    def test_compute_with_delay_tolerance(self) -> None:
        """Test affinity metrics with delay tolerance."""
        # Predicted is off by 2 positions
        predicted = {12, 22, 32}
        actual = {10, 20, 30}
        metrics = AnomalyDetectionMetrics.compute(
            predicted, actual, series_length=100, delay_tolerance=3
        )

        # Point-wise should be 0, but affinity should be higher
        assert metrics.point_wise_precision == 0.0
        assert metrics.affinity_precision == 1.0  # All predictions near actual
        assert metrics.affinity_recall == 1.0  # All actuals near predictions


class TestCostMetrics:
    """Tests for CostMetrics dataclass."""

    def test_compute_basic(self) -> None:
        """Test basic cost computation."""
        metrics = CostMetrics.compute(input_tokens=1000, output_tokens=500)

        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.total_tokens == 1500

    def test_compute_cost_calculation(self) -> None:
        """Test cost calculation."""
        metrics = CostMetrics.compute(input_tokens=1000, output_tokens=1000)

        # Cost = (1000/1000 * 0.003) + (1000/1000 * 0.015) = 0.018
        expected_cost = 0.003 + 0.015
        assert abs(metrics.estimated_cost_usd - expected_cost) < 1e-6

    def test_compute_zero_tokens(self) -> None:
        """Test with zero tokens."""
        metrics = CostMetrics.compute(input_tokens=0, output_tokens=0)
        assert metrics.total_tokens == 0
        assert metrics.estimated_cost_usd == 0.0


class TestParseLLMResponse:
    """Tests for parse_llm_response function."""

    def test_parse_complete_response(self) -> None:
        """Test parsing a complete well-formatted response."""
        response = """- Answer: 42.5
- Confidence: high
- Reasoning: Based on the data analysis."""

        result = parse_llm_response(response)

        assert result["answer"] == 42.5
        assert result["confidence"] == "high"
        assert result["reasoning"] == "Based on the data analysis."
        assert result["raw"] == response

    def test_parse_integer_answer(self) -> None:
        """Test parsing integer answer."""
        response = "- Answer: 100\n- Confidence: medium"
        result = parse_llm_response(response)
        assert result["answer"] == 100.0

    def test_parse_string_answer(self) -> None:
        """Test parsing string answer."""
        response = "- Answer: rising\n- Confidence: high"
        result = parse_llm_response(response)
        assert result["answer"] == "rising"

    def test_parse_formatted_number(self) -> None:
        """Test parsing number with formatting."""
        response = "- Answer: $1,234.56\n- Confidence: low"
        result = parse_llm_response(response)
        assert result["answer"] == 1234.56

    def test_parse_percentage(self) -> None:
        """Test parsing percentage."""
        response = "- Answer: 95%\n- Confidence: high"
        result = parse_llm_response(response)
        assert result["answer"] == 95.0

    def test_parse_missing_fields(self) -> None:
        """Test parsing with missing fields."""
        response = "Some unstructured text without the expected format."
        result = parse_llm_response(response)
        assert result["answer"] is None
        assert result["confidence"] is None
        assert result["reasoning"] is None

    def test_parse_case_insensitive(self) -> None:
        """Test case-insensitive parsing."""
        response = "- ANSWER: 50\n- CONFIDENCE: HIGH"
        result = parse_llm_response(response)
        assert result["answer"] == 50.0
        assert result["confidence"] == "high"


class TestDetectHallucination:
    """Tests for detect_hallucination function."""

    def test_no_hallucination_with_valid_value(self) -> None:
        """Test no hallucination when using valid data values."""
        raw_data = [10.0, 20.0, 30.0, 40.0, 50.0]
        response = "Answer: 30.0"
        result = detect_hallucination(response, raw_data, "")
        assert result is False

    def test_no_hallucination_with_derived_value(self) -> None:
        """Test no hallucination for derived statistics."""
        raw_data = [10.0, 20.0, 30.0, 40.0, 50.0]
        # Mean = 30, so 30 should be valid
        response = "Answer: 30.0"
        result = detect_hallucination(response, raw_data, "")
        assert result is False

    def test_hallucination_detected(self) -> None:
        """Test hallucination detection for invented value."""
        raw_data = [10.0, 20.0, 30.0, 40.0, 50.0]
        # 999 is not in data or derivable
        response = "Answer: 999.5"
        result = detect_hallucination(response, raw_data, "")
        assert result is True

    def test_no_numbers_in_response(self) -> None:
        """Test response with no numbers."""
        raw_data = [10.0, 20.0, 30.0]
        response = "The trend is rising"
        result = detect_hallucination(response, raw_data, "")
        # No Answer: line means we can't detect hallucination
        assert result is False

    def test_small_numbers_ignored(self) -> None:
        """Test small numbers (like indices) are ignored."""
        raw_data = [100.0, 200.0, 300.0]
        # No Answer: line, so detection returns False
        response = "At index 0 and 1"
        result = detect_hallucination(response, raw_data, "")
        assert result is False

    def test_empty_data(self) -> None:
        """Test with empty raw data."""
        raw_data: list[float] = []
        response = "Answer: 50"
        result = detect_hallucination(response, raw_data, "")
        # With empty data, we return False (can't validate)
        assert result is False


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_create_trial_result(self) -> None:
        """Test creating a TrialResult."""
        token_metrics = TokenMetrics(raw_tokens=100, compressed_tokens=20, compression_ratio=0.8)
        cost_metrics = CostMetrics.compute(input_tokens=100, output_tokens=50)

        result = TrialResult(
            task_type="statistical",
            condition="treatment",
            query="mean",
            token_metrics=token_metrics,
            predicted_answer=42.5,
            actual_answer=42.0,
            is_correct=True,
            numerical_proximity=0.99,
            hallucination_detected=False,
            cost_metrics=cost_metrics,
            latency_ms=150.0,
        )

        assert result.task_type == "statistical"
        assert result.condition == "treatment"
        assert result.is_correct is True


class TestAggregatedResults:
    """Tests for AggregatedResults dataclass."""

    def test_from_trials_empty_raises(self) -> None:
        """Test aggregating empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            AggregatedResults.from_trials([])

    def test_from_trials_single(self) -> None:
        """Test aggregating single trial."""
        token_metrics = TokenMetrics(raw_tokens=100, compressed_tokens=20, compression_ratio=0.8)
        cost_metrics = CostMetrics.compute(input_tokens=100, output_tokens=50)

        trial = TrialResult(
            task_type="statistical",
            condition="treatment",
            query="mean",
            token_metrics=token_metrics,
            predicted_answer=42.0,
            actual_answer=42.0,
            is_correct=True,
            numerical_proximity=1.0,
            hallucination_detected=False,
            cost_metrics=cost_metrics,
            latency_ms=150.0,
        )

        result = AggregatedResults.from_trials([trial])

        assert result.n_trials == 1
        assert result.accuracy == 1.0
        assert result.mean_compression_ratio == 0.8
        assert result.hallucination_rate == 0.0

    def test_from_trials_multiple(self) -> None:
        """Test aggregating multiple trials."""
        trials = []
        for i in range(10):
            token_metrics = TokenMetrics(
                raw_tokens=100, compressed_tokens=20, compression_ratio=0.8
            )
            cost_metrics = CostMetrics.compute(input_tokens=100, output_tokens=50)

            trial = TrialResult(
                task_type="statistical",
                condition="baseline",
                query="mean",
                token_metrics=token_metrics,
                predicted_answer=42.0,
                actual_answer=42.0,
                is_correct=(i % 2 == 0),  # 50% accuracy
                numerical_proximity=0.5,
                hallucination_detected=(i == 0),  # 10% hallucination
                cost_metrics=cost_metrics,
                latency_ms=100.0 + i * 10,
            )
            trials.append(trial)

        result = AggregatedResults.from_trials(trials)

        assert result.n_trials == 10
        assert result.accuracy == 0.5
        assert result.hallucination_rate == 0.1
        assert result.mean_compression_ratio == 0.8

    def test_confidence_interval_bounds(self) -> None:
        """Test confidence interval is within [0, 1]."""
        token_metrics = TokenMetrics(raw_tokens=100, compressed_tokens=20, compression_ratio=0.8)
        cost_metrics = CostMetrics.compute(input_tokens=100, output_tokens=50)

        # Create trials with varying correctness
        trials = []
        for i in range(30):
            trial = TrialResult(
                task_type="statistical",
                condition="baseline",
                query="mean",
                token_metrics=token_metrics,
                predicted_answer=42.0,
                actual_answer=42.0,
                is_correct=(i < 20),  # ~67% accuracy
                numerical_proximity=0.5,
                hallucination_detected=False,
                cost_metrics=cost_metrics,
                latency_ms=100.0,
            )
            trials.append(trial)

        result = AggregatedResults.from_trials(trials)

        assert 0 <= result.accuracy_ci_lower <= result.accuracy
        assert result.accuracy <= result.accuracy_ci_upper <= 1

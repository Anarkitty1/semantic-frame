"""
Tests for Composite Metrics (CUE, SAS, EAP, PEI)

Tests the benchmark methodology metrics from Section 2.3.
"""

import pytest

from benchmarks.metrics import (
    AggregatedResults,
    ContextUtilizationMetrics,
    CostMetrics,
    EfficiencyAccuracyProduct,
    ParetoEfficiencyIndex,
    SemanticAlignmentScore,
    TokenMetrics,
    TrialResult,
)


class TestContextUtilizationMetrics:
    """Tests for CUE (Context Utilization Efficiency)."""

    def test_compute_basic(self) -> None:
        """Test basic CUE computation."""
        # Baseline: 70% accuracy, 1000 tokens
        # Treatment: 95% accuracy, 100 tokens
        cue = ContextUtilizationMetrics.compute(
            baseline_accuracy=0.70,
            baseline_tokens=1000,
            treatment_accuracy=0.95,
            treatment_tokens=100,
        )

        # Info densities
        assert cue.baseline_info_density == pytest.approx(0.0007)  # 0.70/1000
        assert cue.treatment_info_density == pytest.approx(0.0095)  # 0.95/100

        # Compression ratio
        assert cue.compression_ratio == pytest.approx(0.90)  # 1 - 100/1000

        # CUE score = (0.0095/0.0007) * 0.90 ≈ 12.21
        assert cue.cue_score > 1.0  # Treatment is more efficient
        assert cue.cue_score == pytest.approx((0.0095 / 0.0007) * 0.90, rel=0.01)

    def test_compute_zero_baseline_tokens(self) -> None:
        """Test CUE with zero baseline tokens (edge case)."""
        cue = ContextUtilizationMetrics.compute(
            baseline_accuracy=0.70,
            baseline_tokens=0,
            treatment_accuracy=0.95,
            treatment_tokens=100,
        )

        assert cue.baseline_info_density == 0.0
        assert cue.compression_ratio == 0.0
        assert cue.cue_score == 0.0

    def test_compute_zero_treatment_tokens(self) -> None:
        """Test CUE with zero treatment tokens (edge case)."""
        cue = ContextUtilizationMetrics.compute(
            baseline_accuracy=0.70,
            baseline_tokens=1000,
            treatment_accuracy=0.95,
            treatment_tokens=0,
        )

        assert cue.treatment_info_density == 0.0
        # Compression ratio is 100% when treatment has 0 tokens
        assert cue.compression_ratio == 1.0

    def test_compute_equal_accuracy(self) -> None:
        """Test CUE when accuracy is same but tokens differ."""
        cue = ContextUtilizationMetrics.compute(
            baseline_accuracy=0.80,
            baseline_tokens=1000,
            treatment_accuracy=0.80,
            treatment_tokens=100,
        )

        # Same accuracy, 10x compression = CUE should be high
        assert cue.cue_score > 1.0

    def test_immutable(self) -> None:
        """Test that CUE metrics are frozen (immutable)."""
        cue = ContextUtilizationMetrics.compute(
            baseline_accuracy=0.70,
            baseline_tokens=1000,
            treatment_accuracy=0.95,
            treatment_tokens=100,
        )

        with pytest.raises(AttributeError):
            cue.cue_score = 999.0  # type: ignore


class TestSemanticAlignmentScore:
    """Tests for SAS (Semantic Alignment Score)."""

    def test_compute_perfect_scores(self) -> None:
        """Test SAS with perfect accuracy in all categories."""
        sas = SemanticAlignmentScore.compute(
            trend_correct=100,
            trend_total=100,
            magnitude_correct=100,
            magnitude_total=100,
            anomaly_correct=100,
            anomaly_total=100,
        )

        assert sas.trend_accuracy == 1.0
        assert sas.magnitude_accuracy == 1.0
        assert sas.anomaly_accuracy == 1.0
        assert sas.sas_score == pytest.approx(1.0)

    def test_compute_weighted_score(self) -> None:
        """Test SAS weighted average calculation."""
        # 80% trend, 60% magnitude, 40% anomaly
        sas = SemanticAlignmentScore.compute(
            trend_correct=80,
            trend_total=100,
            magnitude_correct=60,
            magnitude_total=100,
            anomaly_correct=40,
            anomaly_total=100,
        )

        # Weighted: 0.8*0.4 + 0.6*0.3 + 0.4*0.3 = 0.32 + 0.18 + 0.12 = 0.62
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.3
        assert sas.sas_score == pytest.approx(expected)

    def test_compute_zero_totals(self) -> None:
        """Test SAS with zero total attempts (edge case)."""
        sas = SemanticAlignmentScore.compute(
            trend_correct=0,
            trend_total=0,
            magnitude_correct=0,
            magnitude_total=0,
            anomaly_correct=0,
            anomaly_total=0,
        )

        assert sas.trend_accuracy == 0.0
        assert sas.magnitude_accuracy == 0.0
        assert sas.anomaly_accuracy == 0.0
        assert sas.sas_score == 0.0

    def test_compute_partial_categories(self) -> None:
        """Test SAS when only some categories have data."""
        sas = SemanticAlignmentScore.compute(
            trend_correct=90,
            trend_total=100,
            magnitude_correct=0,
            magnitude_total=0,  # No magnitude tests
            anomaly_correct=80,
            anomaly_total=100,
        )

        # Only trend (0.4) and anomaly (0.3) contribute
        expected = 0.9 * 0.4 + 0.0 * 0.3 + 0.8 * 0.3
        assert sas.sas_score == pytest.approx(expected)

    def test_from_trial_results(self) -> None:
        """Test SAS computation from TrialResult list."""
        # Create mock trial results
        trials = [
            _create_trial("direction", is_correct=True),
            _create_trial("direction", is_correct=True),
            _create_trial("direction", is_correct=False),
            _create_trial("strength", is_correct=True),
            _create_trial("strength", is_correct=False),
            _create_trial("presence", is_correct=True),
            _create_trial("presence", is_correct=True),
        ]

        sas = SemanticAlignmentScore.from_trial_results(trials)

        # direction: 2/3 correct
        assert sas.trend_accuracy == pytest.approx(2 / 3)
        # strength: 1/2 correct
        assert sas.magnitude_accuracy == pytest.approx(1 / 2)
        # presence: 2/2 correct
        assert sas.anomaly_accuracy == pytest.approx(1.0)

    def test_weights_sum_to_one(self) -> None:
        """Verify the SAS weights sum to 1.0."""
        total_weight = (
            SemanticAlignmentScore.TREND_WEIGHT
            + SemanticAlignmentScore.MAGNITUDE_WEIGHT
            + SemanticAlignmentScore.ANOMALY_WEIGHT
        )
        assert total_weight == pytest.approx(1.0)


class TestEfficiencyAccuracyProduct:
    """Tests for EAP (Efficiency-Accuracy Product)."""

    def test_compute_strong_performance(self) -> None:
        """Test EAP above threshold (> 0.90)."""
        eap = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.95,
            accuracy=0.97,
        )

        assert eap.eap_score == pytest.approx(0.95 * 0.97)
        assert eap.meets_threshold is True

    def test_compute_weak_performance(self) -> None:
        """Test EAP below threshold (< 0.90)."""
        eap = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.70,
            accuracy=0.80,
        )

        assert eap.eap_score == pytest.approx(0.70 * 0.80)
        assert eap.meets_threshold is False

    def test_compute_boundary(self) -> None:
        """Test EAP at exactly 0.90 threshold."""
        # Use values that result in exactly 0.90
        eap = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.95,
            accuracy=0.95,  # 0.95 * 0.95 = 0.9025 > 0.90
        )

        assert eap.eap_score >= 0.90
        assert eap.meets_threshold is True

    def test_compute_just_below_threshold(self) -> None:
        """Test EAP just below threshold."""
        eap = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.89,
            accuracy=0.99,  # 0.89 * 0.99 = 0.8811 < 0.90
        )

        assert eap.eap_score < 0.90
        assert eap.meets_threshold is False

    def test_compute_zero_values(self) -> None:
        """Test EAP with zero compression or accuracy."""
        eap_zero_compression = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.0,
            accuracy=0.95,
        )
        assert eap_zero_compression.eap_score == 0.0
        assert eap_zero_compression.meets_threshold is False

        eap_zero_accuracy = EfficiencyAccuracyProduct.compute(
            compression_ratio=0.95,
            accuracy=0.0,
        )
        assert eap_zero_accuracy.eap_score == 0.0
        assert eap_zero_accuracy.meets_threshold is False

    def test_from_aggregated_results(self) -> None:
        """Test EAP computation from AggregatedResults."""
        baseline = _create_aggregated_results(
            condition="baseline",
            accuracy=0.70,
            compression_ratio=0.0,
        )
        treatment = _create_aggregated_results(
            condition="treatment",
            accuracy=0.95,
            compression_ratio=0.90,
        )

        eap = EfficiencyAccuracyProduct.from_aggregated_results(baseline, treatment)

        assert eap.compression_ratio == 0.90
        assert eap.accuracy == 0.95
        assert eap.eap_score == pytest.approx(0.855)


class TestParetoEfficiencyIndex:
    """Tests for PEI (Pareto Efficiency Index)."""

    def test_calculate_area_single_point(self) -> None:
        """Test area calculation with single point returns 0."""
        area = ParetoEfficiencyIndex._calculate_area([(100.0, 0.5)])
        assert area == 0.0

    def test_calculate_area_two_points(self) -> None:
        """Test trapezoidal area calculation with two points."""
        # Two points: (100, 0.5) and (200, 0.7)
        # Area = (200-100) * (0.5+0.7) / 2 = 100 * 1.2 / 2 = 60
        points = [(100.0, 0.5), (200.0, 0.7)]
        area = ParetoEfficiencyIndex._calculate_area(points)
        assert area == pytest.approx(60.0)

    def test_calculate_area_multiple_points(self) -> None:
        """Test trapezoidal area calculation with multiple points."""
        points = [(0.0, 0.0), (100.0, 0.5), (200.0, 1.0)]
        # Segment 1: (100-0) * (0+0.5) / 2 = 25
        # Segment 2: (200-100) * (0.5+1.0) / 2 = 75
        # Total = 100
        area = ParetoEfficiencyIndex._calculate_area(points)
        assert area == pytest.approx(100.0)

    def test_aggregate_points(self) -> None:
        """Test point aggregation by averaging y values."""
        points = [
            (100.0, 0.4),
            (100.0, 0.6),  # Same x, should average y
            (200.0, 0.8),
        ]
        aggregated = ParetoEfficiencyIndex._aggregate_points(points)

        assert len(aggregated) == 2
        assert aggregated[0] == pytest.approx((100.0, 0.5), rel=0.01)
        assert aggregated[1] == pytest.approx((200.0, 0.8), rel=0.01)

    def test_compute_treatment_dominates(self) -> None:
        """Test PEI when treatment dominates baseline."""
        baseline_results = [
            _create_aggregated_results(
                condition="baseline",
                accuracy=0.6,
                total_raw_tokens=1000,
            ),
            _create_aggregated_results(
                condition="baseline",
                accuracy=0.7,
                total_raw_tokens=2000,
            ),
        ]
        treatment_results = [
            _create_aggregated_results(
                condition="treatment",
                accuracy=0.9,
                total_compressed_tokens=100,
            ),
            _create_aggregated_results(
                condition="treatment",
                accuracy=0.95,
                total_compressed_tokens=200,
            ),
        ]

        pei = ParetoEfficiencyIndex.compute(baseline_results, treatment_results)

        # Treatment has higher accuracy with fewer tokens
        # This should result in positive PEI (treatment better)
        # Note: actual calculation depends on area under curves
        assert len(pei.baseline_points) == 2
        assert len(pei.treatment_points) == 2

    def test_compute_empty_results(self) -> None:
        """Test PEI with empty result lists."""
        pei = ParetoEfficiencyIndex.compute([], [])

        assert pei.baseline_area == 0.0
        assert pei.treatment_area == 0.0
        assert pei.pei_score == 0.0

    def test_from_trial_results(self) -> None:
        """Test PEI computation from trial results."""
        baseline_trials = [
            _create_trial("mean", is_correct=True, raw_tokens=1000),
            _create_trial("mean", is_correct=False, raw_tokens=1000),
            _create_trial("median", is_correct=True, raw_tokens=2000),
        ]
        treatment_trials = [
            _create_trial("mean", is_correct=True, compressed_tokens=100),
            _create_trial("mean", is_correct=True, compressed_tokens=100),
            _create_trial("median", is_correct=True, compressed_tokens=200),
        ]

        pei = ParetoEfficiencyIndex.from_trial_results(baseline_trials, treatment_trials)

        # Verify points are aggregated correctly
        assert len(pei.baseline_points) >= 1
        assert len(pei.treatment_points) >= 1


# =============================================================================
# Helper Functions
# =============================================================================


def _create_trial(
    query: str,
    is_correct: bool,
    raw_tokens: int = 1000,
    compressed_tokens: int = 100,
) -> TrialResult:
    """Create a mock TrialResult for testing."""
    return TrialResult(
        task_type="statistical",
        condition="treatment",
        query=query,
        token_metrics=TokenMetrics(
            raw_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - compressed_tokens / raw_tokens,
        ),
        predicted_answer="42",
        actual_answer="42" if is_correct else "0",
        is_correct=is_correct,
        numerical_proximity=1.0 if is_correct else 0.0,
        hallucination_detected=False,
        cost_metrics=CostMetrics.compute(raw_tokens, 100),
        latency_ms=100.0,
    )


def _create_aggregated_results(
    condition: str,
    accuracy: float = 0.8,
    compression_ratio: float = 0.9,
    total_raw_tokens: int = 10000,
    total_compressed_tokens: int = 1000,
) -> AggregatedResults:
    """Create a mock AggregatedResults for testing."""
    return AggregatedResults(
        task_type="statistical",
        condition=condition,  # type: ignore
        n_trials=100,
        mean_compression_ratio=compression_ratio,
        std_compression_ratio=0.05,
        total_raw_tokens=total_raw_tokens,
        total_compressed_tokens=total_compressed_tokens,
        accuracy=accuracy,
        mean_numerical_proximity=0.95,
        std_numerical_proximity=0.05,
        hallucination_rate=0.02,
        mean_cost_usd=0.01,
        total_cost_usd=1.0,
        mean_latency_ms=100.0,
        std_latency_ms=20.0,
        accuracy_ci_lower=accuracy - 0.05,
        accuracy_ci_upper=accuracy + 0.05,
    )

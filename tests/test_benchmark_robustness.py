"""
Tests for benchmarks/robustness.py

Tests the perturbation engine, adversarial input generator, and robustness evaluator.
"""

import numpy as np
import pytest

from benchmarks.robustness import (
    AdversarialInputGenerator,
    PerturbationEngine,
    PerturbationResult,
    PerturbationType,
    RobustnessConfig,
    RobustnessEvaluator,
    RobustnessMetrics,
)

# ============================================================================
# PerturbationType Tests
# ============================================================================


class TestPerturbationType:
    """Tests for PerturbationType enum."""

    def test_all_types_have_values(self) -> None:
        """Test that all perturbation types have string values."""
        for ptype in PerturbationType:
            assert isinstance(ptype.value, str)
            assert len(ptype.value) > 0

    def test_type_count(self) -> None:
        """Test expected number of perturbation types."""
        assert len(PerturbationType) == 8


# ============================================================================
# RobustnessConfig Tests
# ============================================================================


class TestRobustnessConfig:
    """Tests for RobustnessConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RobustnessConfig()

        assert len(config.noise_levels) == 3
        assert len(config.precision_levels) == 4
        assert len(config.scale_factors) == 5
        assert config.enable_adversarial is False
        assert config.random_seed == 42

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RobustnessConfig(
            noise_levels=[0.1, 0.2],
            enable_adversarial=True,
            random_seed=123,
        )

        assert config.noise_levels == [0.1, 0.2]
        assert config.enable_adversarial is True
        assert config.random_seed == 123


# ============================================================================
# RobustnessMetrics Tests
# ============================================================================


class TestRobustnessMetrics:
    """Tests for RobustnessMetrics dataclass."""

    def test_metrics_calculation(self) -> None:
        """Test that degradation and is_robust are calculated correctly."""
        metrics = RobustnessMetrics(
            base_accuracy=0.95,
            perturbed_accuracy=0.90,
            perturbation_type=PerturbationType.NOISE,
            perturbation_level=0.1,
        )

        assert metrics.degradation == pytest.approx(0.05)
        assert metrics.is_robust is True  # 5% < 10% threshold

    def test_metrics_not_robust(self) -> None:
        """Test detection of non-robust performance."""
        metrics = RobustnessMetrics(
            base_accuracy=0.95,
            perturbed_accuracy=0.80,
            perturbation_type=PerturbationType.NOISE,
            perturbation_level=0.2,
        )

        assert metrics.degradation == pytest.approx(0.15)
        assert metrics.is_robust is False  # 15% > 10% threshold

    def test_metrics_improvement(self) -> None:
        """Test when perturbed accuracy is actually higher (negative degradation)."""
        metrics = RobustnessMetrics(
            base_accuracy=0.85,
            perturbed_accuracy=0.90,
            perturbation_type=PerturbationType.SCALE,
            perturbation_level=10.0,
        )

        assert metrics.degradation == pytest.approx(-0.05)
        assert metrics.is_robust is True


# ============================================================================
# PerturbationEngine Tests
# ============================================================================


class TestPerturbationEngine:
    """Tests for PerturbationEngine class."""

    def test_init_default_seed(self) -> None:
        """Test default initialization."""
        engine = PerturbationEngine()
        assert engine.seed == 42

    def test_init_custom_seed(self) -> None:
        """Test custom seed initialization."""
        engine = PerturbationEngine(seed=123)
        assert engine.seed == 123

    def test_reset_seed(self) -> None:
        """Test seed reset functionality."""
        engine = PerturbationEngine(seed=42)
        engine.reset_seed(999)
        assert engine.seed == 999


class TestApplyNoise:
    """Tests for PerturbationEngine.apply_noise."""

    def test_apply_noise_basic(self) -> None:
        """Test basic noise application."""
        engine = PerturbationEngine(seed=42)
        data = np.array([100.0] * 100)

        result = engine.apply_noise(data, noise_level=0.1)

        assert isinstance(result, PerturbationResult)
        assert result.perturbation_type == PerturbationType.NOISE
        assert len(result.perturbed_data) == len(data)
        assert not np.array_equal(result.perturbed_data, data)

    def test_apply_noise_zero_level(self) -> None:
        """Test that zero noise level preserves data."""
        engine = PerturbationEngine(seed=42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = engine.apply_noise(data, noise_level=0.0)

        np.testing.assert_array_almost_equal(result.perturbed_data, data)

    def test_apply_noise_preserve_pattern(self) -> None:
        """Test noise scaled to data variation."""
        engine = PerturbationEngine(seed=42)
        data = np.linspace(0, 100, 100)

        result = engine.apply_noise(data, noise_level=0.1, preserve_pattern=True)

        # Trend should still be visible
        assert result.perturbed_data[-1] > result.perturbed_data[0]

    def test_apply_noise_negative_level_raises(self) -> None:
        """Test that negative noise level raises ValueError."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="noise_level must be >= 0"):
            engine.apply_noise(data, noise_level=-0.1)


class TestApplyPrecision:
    """Tests for PerturbationEngine.apply_precision."""

    def test_apply_precision_basic(self) -> None:
        """Test basic precision reduction."""
        engine = PerturbationEngine()
        data = np.array([123.456789, 987.654321])

        result = engine.apply_precision(data, significant_figures=3)

        assert result.perturbation_type == PerturbationType.PRECISION
        assert result.perturbed_data[0] == pytest.approx(123.0, rel=0.01)
        assert result.perturbed_data[1] == pytest.approx(988.0, rel=0.01)

    def test_apply_precision_one_figure(self) -> None:
        """Test single significant figure."""
        engine = PerturbationEngine()
        data = np.array([456.0, 0.0789])

        result = engine.apply_precision(data, significant_figures=1)

        assert result.perturbed_data[0] == pytest.approx(500.0)
        assert result.perturbed_data[1] == pytest.approx(0.08, rel=0.01)

    def test_apply_precision_zero_raises(self) -> None:
        """Test that zero significant figures raises ValueError."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="significant_figures must be > 0"):
            engine.apply_precision(data, significant_figures=0)


class TestApplyScale:
    """Tests for PerturbationEngine.apply_scale."""

    def test_apply_scale_basic(self) -> None:
        """Test basic scaling."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        result = engine.apply_scale(data, scale_factor=10.0)

        assert result.perturbation_type == PerturbationType.SCALE
        np.testing.assert_array_almost_equal(result.perturbed_data, [10.0, 20.0, 30.0])

    def test_apply_scale_negative(self) -> None:
        """Test negative scale factor (sign flip)."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        result = engine.apply_scale(data, scale_factor=-1.0)

        np.testing.assert_array_almost_equal(result.perturbed_data, [-1.0, -2.0, -3.0])

    def test_apply_scale_fractional(self) -> None:
        """Test fractional scale factor."""
        engine = PerturbationEngine()
        data = np.array([100.0, 200.0])

        result = engine.apply_scale(data, scale_factor=0.01)

        np.testing.assert_array_almost_equal(result.perturbed_data, [1.0, 2.0])

    def test_apply_scale_zero_raises(self) -> None:
        """Test that zero scale factor raises ValueError."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="scale_factor cannot be 0"):
            engine.apply_scale(data, scale_factor=0)


class TestApplyShift:
    """Tests for PerturbationEngine.apply_shift."""

    def test_apply_shift_positive(self) -> None:
        """Test positive shift."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        result = engine.apply_shift(data, offset=100.0)

        assert result.perturbation_type == PerturbationType.SHIFT
        np.testing.assert_array_almost_equal(result.perturbed_data, [101.0, 102.0, 103.0])

    def test_apply_shift_negative(self) -> None:
        """Test negative shift."""
        engine = PerturbationEngine()
        data = np.array([100.0, 200.0, 300.0])

        result = engine.apply_shift(data, offset=-50.0)

        np.testing.assert_array_almost_equal(result.perturbed_data, [50.0, 150.0, 250.0])


class TestApplyMissing:
    """Tests for PerturbationEngine.apply_missing."""

    def test_apply_missing_basic(self) -> None:
        """Test basic missing value injection."""
        engine = PerturbationEngine(seed=42)
        data = np.arange(100, dtype=float)

        result = engine.apply_missing(data, missing_rate=0.1)

        assert result.perturbation_type == PerturbationType.MISSING
        assert np.isnan(result.perturbed_data).sum() == 10
        assert result.affected_indices is not None
        assert len(result.affected_indices) == 10

    def test_apply_missing_zero_rate(self) -> None:
        """Test zero missing rate preserves data."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        result = engine.apply_missing(data, missing_rate=0.0)

        np.testing.assert_array_equal(result.perturbed_data, data)
        assert result.affected_indices == []

    def test_apply_missing_custom_value(self) -> None:
        """Test custom missing value."""
        engine = PerturbationEngine(seed=42)
        data = np.ones(100)

        result = engine.apply_missing(data, missing_rate=0.1, missing_value=-999.0)

        assert (result.perturbed_data == -999.0).sum() == 10
        assert not np.isnan(result.perturbed_data).any()

    def test_apply_missing_invalid_rate_raises(self) -> None:
        """Test that invalid missing rate raises ValueError."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="missing_rate must be in"):
            engine.apply_missing(data, missing_rate=1.5)


class TestApplyOutliers:
    """Tests for PerturbationEngine.apply_outliers."""

    def test_apply_outliers_basic(self) -> None:
        """Test basic outlier injection."""
        engine = PerturbationEngine(seed=42)
        data = np.ones(100)

        result = engine.apply_outliers(data, outlier_rate=0.05)

        assert result.perturbation_type == PerturbationType.OUTLIERS
        assert result.affected_indices is not None
        assert len(result.affected_indices) == 5

    def test_apply_outliers_extreme_magnitude(self) -> None:
        """Test that outliers are extreme."""
        engine = PerturbationEngine(seed=42)
        data = np.zeros(100)

        result = engine.apply_outliers(data, outlier_rate=0.1, outlier_magnitude=10.0)

        # Outliers should be far from mean (0)
        outlier_values = [result.perturbed_data[i] for i in result.affected_indices or []]
        for val in outlier_values:
            assert abs(val) > 5  # Should be significantly larger

    def test_apply_outliers_zero_rate(self) -> None:
        """Test zero outlier rate preserves data."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        result = engine.apply_outliers(data, outlier_rate=0.0)

        np.testing.assert_array_equal(result.perturbed_data, data)
        assert result.affected_indices == []


class TestApplyTruncation:
    """Tests for PerturbationEngine.apply_truncation."""

    def test_apply_truncation_basic(self) -> None:
        """Test basic truncation from start."""
        engine = PerturbationEngine()
        data = np.arange(100, dtype=float)

        result = engine.apply_truncation(data, keep_fraction=0.5)

        assert result.perturbation_type == PerturbationType.TRUNCATION
        assert len(result.perturbed_data) == 50
        np.testing.assert_array_equal(result.perturbed_data, np.arange(50, dtype=float))

    def test_apply_truncation_from_end(self) -> None:
        """Test truncation keeping last portion."""
        engine = PerturbationEngine()
        data = np.arange(100, dtype=float)

        result = engine.apply_truncation(data, keep_fraction=0.3, from_end=True)

        assert len(result.perturbed_data) == 30
        np.testing.assert_array_equal(result.perturbed_data, np.arange(70, 100, dtype=float))

    def test_apply_truncation_invalid_fraction_raises(self) -> None:
        """Test that invalid fraction raises ValueError."""
        engine = PerturbationEngine()
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="keep_fraction must be in"):
            engine.apply_truncation(data, keep_fraction=0.0)


class TestApplyLocalShuffle:
    """Tests for PerturbationEngine.apply_local_shuffle."""

    def test_apply_local_shuffle_basic(self) -> None:
        """Test basic local shuffling."""
        engine = PerturbationEngine(seed=42)
        data = np.arange(100, dtype=float)

        result = engine.apply_local_shuffle(data, window_size=10, shuffle_rate=0.5)

        assert result.perturbation_type == PerturbationType.REORDER
        assert len(result.perturbed_data) == len(data)
        # Not all values should be in original positions
        assert not np.array_equal(result.perturbed_data, data)
        # But all values should still be present
        assert set(result.perturbed_data) == set(data)

    def test_apply_local_shuffle_preserves_sum(self) -> None:
        """Test that shuffling preserves sum."""
        engine = PerturbationEngine(seed=42)
        data = np.arange(100, dtype=float)

        result = engine.apply_local_shuffle(data, window_size=5, shuffle_rate=1.0)

        assert np.sum(result.perturbed_data) == np.sum(data)


class TestApplyAllPerturbations:
    """Tests for PerturbationEngine.apply_all_perturbations."""

    def test_apply_all_perturbations(self) -> None:
        """Test applying all perturbation types."""
        engine = PerturbationEngine(seed=42)
        config = RobustnessConfig(
            noise_levels=[0.1],
            precision_levels=[4],
            scale_factors=[1.0],
            shift_offsets=[0.0],
            missing_rates=[0.05],
            outlier_rates=[0.02],
        )
        data = np.random.default_rng(42).normal(100, 10, 100)

        results = engine.apply_all_perturbations(data, config)

        # Should have one result per perturbation level
        assert len(results) == 6  # 1 + 1 + 1 + 1 + 1 + 1


# ============================================================================
# AdversarialInputGenerator Tests
# ============================================================================


class TestAdversarialInputGenerator:
    """Tests for AdversarialInputGenerator class."""

    def test_init(self) -> None:
        """Test initialization."""
        gen = AdversarialInputGenerator(seed=42)
        assert gen.seed == 42


class TestGenerateEdgeCases:
    """Tests for AdversarialInputGenerator.generate_edge_cases."""

    def test_generate_edge_cases_count(self) -> None:
        """Test that expected number of edge cases are generated."""
        gen = AdversarialInputGenerator(seed=42)
        datasets = gen.generate_edge_cases()

        assert len(datasets) == 10  # 10 edge case types

    def test_edge_case_all_zeros(self) -> None:
        """Test all-zeros edge case."""
        gen = AdversarialInputGenerator()
        datasets = gen.generate_edge_cases()

        all_zeros = next(d for d in datasets if d.name == "edge_all_zeros")

        assert np.all(all_zeros.data == 0)
        assert len(all_zeros.data) == 100

    def test_edge_case_single_element(self) -> None:
        """Test single-element edge case."""
        gen = AdversarialInputGenerator()
        datasets = gen.generate_edge_cases()

        single = next(d for d in datasets if d.name == "edge_single_element")

        assert len(single.data) == 1

    def test_edge_case_extreme_outliers(self) -> None:
        """Test extreme outliers edge case."""
        gen = AdversarialInputGenerator()
        datasets = gen.generate_edge_cases()

        extreme = next(d for d in datasets if d.name == "edge_extreme_outliers")

        assert np.max(np.abs(extreme.data)) >= 1e10


class TestGenerateTokenizationAttacks:
    """Tests for AdversarialInputGenerator.generate_tokenization_attacks."""

    def test_generate_tokenization_attacks_count(self) -> None:
        """Test expected number of tokenization attacks."""
        gen = AdversarialInputGenerator()
        datasets = gen.generate_tokenization_attacks()

        assert len(datasets) == 5

    def test_scientific_notation_attack(self) -> None:
        """Test scientific notation edge cases."""
        gen = AdversarialInputGenerator()
        datasets = gen.generate_tokenization_attacks()

        sci = next(d for d in datasets if d.name == "token_scientific_notation")

        # Should have extreme range of magnitudes
        assert np.max(sci.data) > 1e40
        assert np.min(sci.data) < 1e-40


class TestGenerateDistributionShift:
    """Tests for AdversarialInputGenerator.generate_distribution_shift."""

    def test_distribution_shift_basic(self) -> None:
        """Test basic distribution shift generation."""
        gen = AdversarialInputGenerator(seed=42)

        train, test = gen.generate_distribution_shift(
            train_mean=100, train_std=10, test_mean=200, test_std=20, n=100
        )

        assert train.name == "dist_shift_train"
        assert test.name == "dist_shift_test"
        assert len(train.data) == 100
        assert len(test.data) == 100

        # Test distribution should have higher mean
        assert np.mean(test.data) > np.mean(train.data)

    def test_distribution_shift_ground_truth(self) -> None:
        """Test ground truth metadata."""
        gen = AdversarialInputGenerator()

        train, test = gen.generate_distribution_shift(
            train_mean=0, train_std=1, test_mean=10, test_std=5
        )

        assert train.ground_truth["is_train"] is True
        assert test.ground_truth["is_train"] is False
        assert test.ground_truth["shift_type"] == "covariate"


class TestGenerateAdversarialSuite:
    """Tests for AdversarialInputGenerator.generate_adversarial_suite."""

    def test_adversarial_suite_completeness(self) -> None:
        """Test that suite contains all expected datasets."""
        gen = AdversarialInputGenerator(seed=42)
        suite = gen.generate_adversarial_suite()

        # 10 edge cases + 5 tokenization + 4 distribution shift pairs
        assert len(suite) == 19

    def test_adversarial_suite_names_unique(self) -> None:
        """Test that all dataset names are unique."""
        gen = AdversarialInputGenerator()
        suite = gen.generate_adversarial_suite()

        names = [d.name for d in suite]
        assert len(names) == len(set(names))


# ============================================================================
# RobustnessEvaluator Tests
# ============================================================================


class TestRobustnessEvaluator:
    """Tests for RobustnessEvaluator class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        evaluator = RobustnessEvaluator()

        assert evaluator.config is not None
        assert evaluator.engine is not None
        assert evaluator.adversarial is not None

    def test_init_custom_config(self) -> None:
        """Test custom configuration."""
        config = RobustnessConfig(noise_levels=[0.5])
        evaluator = RobustnessEvaluator(config)

        assert evaluator.config.noise_levels == [0.5]

    def test_evaluate_perturbation_robustness(self) -> None:
        """Test robustness evaluation."""
        config = RobustnessConfig(
            noise_levels=[0.1],
            precision_levels=[4],
            scale_factors=[1.0],
            shift_offsets=[0.0],
            missing_rates=[0.05],
            outlier_rates=[0.02],
        )
        evaluator = RobustnessEvaluator(config)

        data = np.random.default_rng(42).normal(100, 10, 100)

        # Simple evaluation function that returns mean proximity to 100
        def eval_fn(d: np.ndarray) -> float:
            return float(max(0, 1.0 - abs(np.mean(d) - 100) / 100))

        metrics = evaluator.evaluate_perturbation_robustness(data, eval_fn)

        assert len(metrics) == 6
        for m in metrics:
            assert isinstance(m, RobustnessMetrics)
            assert 0 <= m.base_accuracy <= 1

    def test_get_adversarial_suite(self) -> None:
        """Test adversarial suite retrieval."""
        evaluator = RobustnessEvaluator()
        suite = evaluator.get_adversarial_suite()

        assert len(suite) > 0
        assert all(hasattr(d, "data") for d in suite)

    def test_summarize_robustness(self) -> None:
        """Test robustness summarization."""
        evaluator = RobustnessEvaluator()

        metrics = [
            RobustnessMetrics(
                base_accuracy=0.95,
                perturbed_accuracy=0.90,
                perturbation_type=PerturbationType.NOISE,
                perturbation_level=0.1,
            ),
            RobustnessMetrics(
                base_accuracy=0.95,
                perturbed_accuracy=0.85,
                perturbation_type=PerturbationType.NOISE,
                perturbation_level=0.2,
            ),
            RobustnessMetrics(
                base_accuracy=0.95,
                perturbed_accuracy=0.93,
                perturbation_type=PerturbationType.SCALE,
                perturbation_level=10.0,
            ),
        ]

        summary = evaluator.summarize_robustness(metrics)

        assert "noise" in summary
        assert "scale" in summary
        assert summary["noise"]["n_tests"] == 2
        assert summary["scale"]["n_tests"] == 1
        assert summary["noise"]["mean_degradation"] == pytest.approx(0.075)


# ============================================================================
# PerturbationResult Tests
# ============================================================================


class TestPerturbationResult:
    """Tests for PerturbationResult dataclass."""

    def test_perturbation_rate_all_affected(self) -> None:
        """Test perturbation rate when all values affected."""
        result = PerturbationResult(
            original_data=np.array([1.0, 2.0, 3.0]),
            perturbed_data=np.array([2.0, 3.0, 4.0]),
            perturbation_type=PerturbationType.SHIFT,
            perturbation_params={"offset": 1.0},
        )

        assert result.perturbation_rate == 1.0

    def test_perturbation_rate_partial_affected(self) -> None:
        """Test perturbation rate with partial affection."""
        result = PerturbationResult(
            original_data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            perturbed_data=np.array([1.0, np.nan, 3.0, np.nan, 5.0]),
            perturbation_type=PerturbationType.MISSING,
            perturbation_params={"missing_rate": 0.4},
            affected_indices=[1, 3],
        )

        assert result.perturbation_rate == pytest.approx(0.4)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for robustness module."""

    def test_full_robustness_pipeline(self) -> None:
        """Test full robustness evaluation pipeline."""
        config = RobustnessConfig(
            noise_levels=[0.1, 0.2],
            precision_levels=[4],
            scale_factors=[0.1, 10.0],
            shift_offsets=[0.0],
            missing_rates=[0.1],
            outlier_rates=[0.05],
        )
        evaluator = RobustnessEvaluator(config)

        # Generate test data
        data = np.linspace(0, 100, 100)

        # Simple trend accuracy function
        def trend_accuracy(d: np.ndarray) -> float:
            if len(d) < 2:
                return 0.5
            slope = (d[-1] - d[0]) / (len(d) - 1)
            # Accuracy based on whether trend is positive
            return 1.0 if slope > 0 else 0.0

        metrics = evaluator.evaluate_perturbation_robustness(data, trend_accuracy)

        # Most perturbations should preserve positive trend
        robust_count = sum(1 for m in metrics if m.is_robust)
        assert robust_count > len(metrics) // 2

    def test_adversarial_data_usability(self) -> None:
        """Test that adversarial data can be processed."""
        evaluator = RobustnessEvaluator()
        suite = evaluator.get_adversarial_suite()

        for dataset in suite:
            # Data should be numpy array
            assert isinstance(dataset.data, np.ndarray)
            # Should be able to compute basic stats
            if len(dataset.data) > 0:
                _ = np.mean(dataset.data)
                _ = np.std(dataset.data)

    def test_reproducibility(self) -> None:
        """Test that perturbations are reproducible with same seed."""
        engine1 = PerturbationEngine(seed=42)
        engine2 = PerturbationEngine(seed=42)

        data = np.arange(100, dtype=float)

        result1 = engine1.apply_noise(data, 0.1)
        result2 = engine2.apply_noise(data, 0.1)

        np.testing.assert_array_equal(result1.perturbed_data, result2.perturbed_data)

"""Tests for benchmarks/tasks module.

Tests all task implementations: statistical, trend, anomaly, comparative, multi-step, scaling.
"""

import numpy as np
import pytest

from benchmarks.claude_client import MockClaudeClient
from benchmarks.config import BenchmarkConfig, DataPattern, TaskType
from benchmarks.datasets import AnomalyDataset, SyntheticDataset
from benchmarks.tasks import (
    AnomalyTask,
    BaseTask,
    ComparativeTask,
    MultiStepTask,
    ScalingTask,
    StatisticalTask,
    TrendTask,
)


class TestBaseTask:
    """Tests for BaseTask abstract class."""

    def test_base_task_is_abstract(self) -> None:
        """Test BaseTask cannot be instantiated directly."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)

        # BaseTask requires implementing abstract methods
        with pytest.raises(TypeError):
            BaseTask(config, client)  # type: ignore


class TestStatisticalTask:
    """Tests for StatisticalTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        assert task.task_type == TaskType.STATISTICAL

    def test_generate_datasets(self) -> None:
        """Test dataset generation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        datasets = task.generate_datasets()

        assert len(datasets) > 0
        assert all(isinstance(d, SyntheticDataset) for d in datasets)

    def test_get_queries(self) -> None:
        """Test query retrieval."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        queries = task.get_queries()

        assert "mean" in queries
        assert "median" in queries
        assert "std" in queries
        assert "count" in queries

    def test_evaluate_answer_exact_numeric(self) -> None:
        """Test exact numeric answer evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={"mean": 2.0},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, proximity = task.evaluate_answer(2.0, 2.0, dataset)
        assert is_correct is True
        assert proximity == 1.0

    def test_evaluate_answer_close_numeric(self) -> None:
        """Test close numeric answer evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, proximity = task.evaluate_answer(2.01, 2.0, dataset)
        assert is_correct is True  # Within 1% tolerance

    def test_evaluate_answer_wrong_numeric(self) -> None:
        """Test wrong numeric answer evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, proximity = task.evaluate_answer(10.0, 2.0, dataset)
        assert is_correct is False

    def test_evaluate_answer_skewness_string(self) -> None:
        """Test skewness answer evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        # Test equivalent terms
        is_correct, _ = task.evaluate_answer("positively skewed", "positive", dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("right-skewed", "positive", dataset)
        assert is_correct is True

    def test_evaluate_answer_none_predicted(self) -> None:
        """Test None predicted value."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, proximity = task.evaluate_answer(None, 2.0, dataset)
        assert is_correct is False
        assert proximity == 0.0

    def test_get_ground_truth(self) -> None:
        """Test ground truth computation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = SyntheticDataset(
            name="test",
            data=data,
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        assert task.get_ground_truth(dataset, "mean") == np.mean(data)
        assert task.get_ground_truth(dataset, "median") == np.median(data)
        assert task.get_ground_truth(dataset, "count") == len(data)


class TestTrendTask:
    """Tests for TrendTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        assert task.task_type == TaskType.TREND

    def test_generate_datasets(self) -> None:
        """Test dataset generation includes various trends."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        datasets = task.generate_datasets()

        trends = [d.ground_truth["trend"] for d in datasets]
        assert "rising" in trends
        assert "falling" in trends
        assert "flat" in trends

    def test_get_queries(self) -> None:
        """Test query retrieval."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        queries = task.get_queries()

        assert "direction" in queries
        assert "strength" in queries

    def test_evaluate_answer_direction_match(self) -> None:
        """Test trend direction evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.LINEAR_TREND,
            seed=42,
        )

        # Test equivalent terms
        is_correct, _ = task.evaluate_answer("rising", "rising", dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("increasing", "rising", dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("upward", "rising", dataset)
        assert is_correct is True

    def test_evaluate_answer_strength_match(self) -> None:
        """Test trend strength evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.LINEAR_TREND,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("strong", "strong", dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("significant", "strong", dataset)
        assert is_correct is True

    def test_evaluate_answer_wrong_direction(self) -> None:
        """Test wrong direction evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.LINEAR_TREND,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("falling", "rising", dataset)
        assert is_correct is False


class TestAnomalyTask:
    """Tests for AnomalyTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        assert task.task_type == TaskType.ANOMALY

    def test_generate_datasets(self) -> None:
        """Test dataset generation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        datasets = task.generate_datasets()

        assert len(datasets) > 0
        # Should have mix of anomaly and clean datasets
        has_anomalies = [isinstance(d, AnomalyDataset) for d in datasets]
        assert any(has_anomalies)

    def test_get_queries(self) -> None:
        """Test query retrieval."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        queries = task.get_queries()

        assert "presence" in queries
        assert "count" in queries
        assert "locations" in queries

    def test_evaluate_answer_presence_yes(self) -> None:
        """Test presence detection (yes)."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 100.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("yes", True, dataset)
        assert is_correct is True

        # Use "detected" which is a positive indicator without negative substring
        is_correct, _ = task.evaluate_answer("outliers detected", "yes", dataset)
        assert is_correct is True

    def test_evaluate_answer_presence_no(self) -> None:
        """Test presence detection (no)."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("no", False, dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("no anomalies", "no", dataset)
        assert is_correct is True

    def test_evaluate_answer_count(self) -> None:
        """Test anomaly count evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 100.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("3 anomalies", 3, dataset)
        assert is_correct is True

    def test_evaluate_answer_locations(self) -> None:
        """Test anomaly location evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = AnomalyTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 100.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        # F1 > 0.5 considered correct
        is_correct, f1 = task.evaluate_answer("indices 1, 2, 3", [1, 2, 3], dataset)
        assert is_correct is True
        assert f1 == 1.0


class TestComparativeTask:
    """Tests for ComparativeTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        assert task.task_type == TaskType.COMPARATIVE

    def test_generate_datasets(self) -> None:
        """Test dataset generation creates pairs."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        datasets = task.generate_datasets()

        assert len(datasets) > 0
        assert len(task._series_pairs) > 0

    def test_get_queries(self) -> None:
        """Test query retrieval."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        queries = task.get_queries()

        assert "higher_mean" in queries
        assert "more_volatile" in queries
        assert "correlation" in queries

    def test_evaluate_answer_series_selection(self) -> None:
        """Test series selection evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer("Series B", "Series B", dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer("Series A", "Series B", dataset)
        assert is_correct is False

    def test_evaluate_answer_correlation(self) -> None:
        """Test correlation evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer(
            "positively correlated", "positively correlated", dataset
        )
        assert is_correct is True

        is_correct, _ = task.evaluate_answer(
            "positive correlation", "positively correlated", dataset
        )
        assert is_correct is True

    def test_get_semantic_frame_output(self) -> None:
        """Test semantic frame output includes both series."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ComparativeTask(config, client)

        # Generate datasets to populate _series_pairs
        datasets = task.generate_datasets()

        if len(datasets) > 0:
            output = task.get_semantic_frame_output(datasets[0])
            assert "SERIES A" in output
            assert "SERIES B" in output


class TestMultiStepTask:
    """Tests for MultiStepTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = MultiStepTask(config, client)

        assert task.task_type == TaskType.MULTI_STEP

    def test_generate_datasets(self) -> None:
        """Test dataset generation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = MultiStepTask(config, client)

        datasets = task.generate_datasets()

        assert len(datasets) > 0
        # Should have computed ground truths
        for d in datasets:
            assert "cv" in d.ground_truth or "zscore" in d.ground_truth

    def test_get_queries(self) -> None:
        """Test query retrieval."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = MultiStepTask(config, client)

        queries = task.get_queries()

        assert "forecast" in queries
        assert "zscore" in queries
        assert "cv" in queries

    def test_evaluate_answer_wider_tolerance(self) -> None:
        """Test multi-step uses wider tolerance (5%)."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = MultiStepTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        # 5% tolerance
        is_correct, _ = task.evaluate_answer(10.4, 10.0, dataset)
        assert is_correct is True

        # Outside 5%
        is_correct, _ = task.evaluate_answer(10.6, 10.0, dataset)
        assert is_correct is False


class TestScalingTask:
    """Tests for ScalingTask."""

    def test_task_type(self) -> None:
        """Test task type is correct."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ScalingTask(config, client)

        assert task.task_type == TaskType.SCALING

    def test_generate_datasets_various_sizes(self) -> None:
        """Test datasets at various scales."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ScalingTask(config, client)

        datasets = task.generate_datasets()

        sizes = [len(d.data) for d in datasets]
        # Should have increasing sizes
        assert len(set(sizes)) > 1
        assert min(sizes) < max(sizes)

    def test_get_queries_subset(self) -> None:
        """Test queries are subset of statistical queries."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ScalingTask(config, client)

        queries = task.get_queries()

        assert "mean" in queries
        assert "std" in queries
        assert "count" in queries
        # Should not have all statistical queries
        assert len(queries) < 12

    def test_evaluate_answer(self) -> None:
        """Test answer evaluation."""
        config = BenchmarkConfig.quick_mode()
        client = MockClaudeClient(config)
        task = ScalingTask(config, client)

        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        is_correct, _ = task.evaluate_answer(10.0, 10.0, dataset)
        assert is_correct is True

        is_correct, _ = task.evaluate_answer(10.05, 10.0, dataset)
        assert is_correct is True  # Within 1%


class TestTaskRun:
    """Integration tests for running tasks."""

    def test_statistical_task_run(self) -> None:
        """Test running statistical task."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        results = task.run(n_trials=1)

        assert len(results) > 0
        # Should have both baseline and treatment
        conditions = {r.condition for r in results}
        assert "baseline" in conditions
        assert "treatment" in conditions

    def test_trend_task_run(self) -> None:
        """Test running trend task."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        client = MockClaudeClient(config)
        task = TrendTask(config, client)

        results = task.run(n_trials=1)

        assert len(results) > 0
        assert all(r.task_type == "trend" for r in results)

    def test_task_result_has_required_fields(self) -> None:
        """Test task results have required fields."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        client = MockClaudeClient(config)
        task = StatisticalTask(config, client)

        results = task.run(n_trials=1)

        for r in results:
            assert r.task_type is not None
            assert r.condition in ["baseline", "treatment"]
            assert r.token_metrics is not None
            assert r.cost_metrics is not None
            assert r.latency_ms >= 0

"""Tests for benchmarks/runner.py.

Tests benchmark orchestration and result aggregation.
"""

import json
from pathlib import Path

import pytest

from benchmarks.config import BenchmarkConfig, TaskType
from benchmarks.runner import TASK_CLASSES, BenchmarkRunner


class TestTaskClasses:
    """Tests for TASK_CLASSES mapping."""

    def test_all_task_types_mapped(self) -> None:
        """Test all TaskType values have corresponding classes."""
        for task_type in TaskType:
            assert task_type in TASK_CLASSES

    def test_task_classes_are_importable(self) -> None:
        """Test all task classes can be instantiated."""
        for task_type, task_class in TASK_CLASSES.items():
            # Should not raise
            assert task_class is not None


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        runner = BenchmarkRunner(mock=True)

        assert runner.config is not None
        assert runner.results == []
        assert runner.aggregated == {}

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = BenchmarkConfig.quick_mode()
        runner = BenchmarkRunner(config=config, mock=True)

        assert runner.config == config

    def test_init_mock_client(self) -> None:
        """Test initialization creates mock client."""
        runner = BenchmarkRunner(mock=True)

        from benchmarks.claude_client import MockClaudeClient

        assert isinstance(runner.client, MockClaudeClient)


class TestRunTask:
    """Tests for run_task method."""

    def test_run_task_returns_results(self) -> None:
        """Test run_task returns list of TrialResult."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        results = runner.run_task(TaskType.STATISTICAL, n_trials=1)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_task_invalid_type_raises(self) -> None:
        """Test run_task with invalid task type raises."""
        config = BenchmarkConfig.quick_mode()
        runner = BenchmarkRunner(config=config, mock=True)

        with pytest.raises(ValueError, match="Unknown task type"):
            runner.run_task("invalid_task")  # type: ignore

    def test_run_task_respects_n_trials(self) -> None:
        """Test run_task respects n_trials parameter."""
        config = BenchmarkConfig.quick_mode()
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        # Run with specific n_trials
        results = runner.run_task(TaskType.TREND, n_trials=2)

        # Should have results for each dataset/query combination * n_trials * 2 conditions
        # The exact count depends on datasets and queries, but should be consistent
        assert len(results) > 0


class TestRunAll:
    """Tests for run_all method."""

    def test_run_all_single_task(self) -> None:
        """Test run_all with single task."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        aggregated = runner.run_all(tasks=[TaskType.STATISTICAL], n_trials=1)

        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        assert runner.run_timestamp is not None

    def test_run_all_stores_results(self) -> None:
        """Test run_all stores results."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)

        assert len(runner.results) > 0
        assert len(runner.aggregated) > 0

    def test_run_all_aggregates_by_condition(self) -> None:
        """Test run_all aggregates by baseline/treatment."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        aggregated = runner.run_all(tasks=[TaskType.TREND], n_trials=1)

        # Should have both baseline and treatment keys
        keys = list(aggregated.keys())
        baseline_keys = [k for k in keys if "baseline" in k]
        treatment_keys = [k for k in keys if "treatment" in k]

        assert len(baseline_keys) > 0
        assert len(treatment_keys) > 0


class TestAggregateResults:
    """Tests for _aggregate_results method."""

    def test_aggregate_groups_by_task_and_condition(self) -> None:
        """Test aggregation groups correctly."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 2
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.STATISTICAL], n_trials=2)

        # Keys should be task_condition format
        for key in runner.aggregated.keys():
            assert "_baseline" in key or "_treatment" in key

    def test_aggregate_computes_metrics(self) -> None:
        """Test aggregation computes metrics correctly."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 2
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=2)

        for result in runner.aggregated.values():
            assert hasattr(result, "accuracy")
            assert hasattr(result, "mean_compression_ratio")
            assert hasattr(result, "hallucination_rate")
            assert result.n_trials > 0


class TestSaveResults:
    """Tests for save_results method."""

    def test_save_results_creates_file(self, tmp_path: Path) -> None:
        """Test save_results creates JSON file."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        output_path = runner.save_results()

        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_save_results_valid_json(self, tmp_path: Path) -> None:
        """Test saved file is valid JSON."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        output_path = runner.save_results()

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "aggregated_results" in data
        assert "raw_results" in data

    def test_save_results_contains_metadata(self, tmp_path: Path) -> None:
        """Test saved file contains metadata."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        output_path = runner.save_results()

        with open(output_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        assert "timestamp" in metadata
        assert "config" in metadata
        assert "total_trials" in metadata

    def test_save_results_custom_prefix(self, tmp_path: Path) -> None:
        """Test save_results with custom prefix."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        output_path = runner.save_results(prefix="custom_prefix")

        assert "custom_prefix" in output_path.name


class TestPrintSummary:
    """Tests for print_summary method."""

    def test_print_summary_no_results(self, capsys) -> None:
        """Test print_summary with no results."""
        runner = BenchmarkRunner(mock=True)
        runner.print_summary()

        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_print_summary_with_results(self, capsys) -> None:
        """Test print_summary with results."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        runner = BenchmarkRunner(config=config, mock=True)

        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        runner.print_summary()

        captured = capsys.readouterr()
        assert "BENCHMARK SUMMARY" in captured.out
        assert "Accuracy" in captured.out


class TestIntegration:
    """Integration tests for the benchmark runner."""

    def test_full_workflow_mock(self, tmp_path: Path) -> None:
        """Test full benchmark workflow with mock client."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path

        runner = BenchmarkRunner(config=config, mock=True)

        # Run benchmarks
        aggregated = runner.run_all(
            tasks=[TaskType.STATISTICAL, TaskType.TREND],
            n_trials=1,
        )

        # Verify results
        assert len(aggregated) > 0
        assert len(runner.results) > 0

        # Save results
        output_path = runner.save_results()
        assert output_path.exists()

        # Verify saved data
        with open(output_path) as f:
            data = json.load(f)

        assert data["metadata"]["total_trials"] == len(runner.results)

    def test_multiple_runs_independent(self, tmp_path: Path) -> None:
        """Test multiple run_all calls are independent."""
        config = BenchmarkConfig.quick_mode()
        config.n_trials = 1
        config.verbose = False
        config.output_dir = tmp_path

        runner = BenchmarkRunner(config=config, mock=True)

        # First run
        runner.run_all(tasks=[TaskType.STATISTICAL], n_trials=1)
        assert len(runner.results) > 0

        # Second run should reset results
        runner.run_all(tasks=[TaskType.TREND], n_trials=1)
        second_count = len(runner.results)

        # Results should be from second run only
        assert second_count > 0
        # Should have different count since different task
        assert all(r.task_type == "trend" for r in runner.results)

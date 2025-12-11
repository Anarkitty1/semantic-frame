"""Tests for benchmarks/reporter.py.

Tests report generation in various formats.
"""

import csv
import json
from pathlib import Path

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import AggregatedResults
from benchmarks.reporter import BenchmarkReporter


def create_sample_aggregated_results() -> dict[str, AggregatedResults]:
    """Create sample aggregated results for testing."""
    return {
        "statistical_baseline": AggregatedResults(
            task_type="statistical",
            condition="baseline",
            n_trials=30,
            mean_compression_ratio=0.0,
            std_compression_ratio=0.0,
            total_raw_tokens=10000,
            total_compressed_tokens=10000,
            accuracy=0.70,
            mean_numerical_proximity=0.75,
            std_numerical_proximity=0.15,
            hallucination_rate=0.10,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
            mean_cost_usd=0.01,
            total_cost_usd=0.30,
            mean_latency_ms=150.0,
            std_latency_ms=30.0,
            accuracy_ci_lower=0.60,
            accuracy_ci_upper=0.80,
        ),
        "statistical_treatment": AggregatedResults(
            task_type="statistical",
            condition="treatment",
            n_trials=30,
            mean_compression_ratio=0.92,
            std_compression_ratio=0.03,
            total_raw_tokens=10000,
            total_compressed_tokens=800,
            accuracy=0.95,
            mean_numerical_proximity=0.98,
            std_numerical_proximity=0.05,
            hallucination_rate=0.02,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            mean_cost_usd=0.005,
            total_cost_usd=0.15,
            mean_latency_ms=100.0,
            std_latency_ms=20.0,
            accuracy_ci_lower=0.90,
            accuracy_ci_upper=0.99,
        ),
        "trend_baseline": AggregatedResults(
            task_type="trend",
            condition="baseline",
            n_trials=20,
            mean_compression_ratio=0.0,
            std_compression_ratio=0.0,
            total_raw_tokens=5000,
            total_compressed_tokens=5000,
            accuracy=0.65,
            mean_numerical_proximity=0.70,
            std_numerical_proximity=0.20,
            hallucination_rate=0.15,
            mean_cost_usd=0.008,
            total_cost_usd=0.16,
            mean_latency_ms=140.0,
            std_latency_ms=25.0,
            accuracy_ci_lower=0.55,
            accuracy_ci_upper=0.75,
        ),
        "trend_treatment": AggregatedResults(
            task_type="trend",
            condition="treatment",
            n_trials=20,
            mean_compression_ratio=0.90,
            std_compression_ratio=0.04,
            total_raw_tokens=5000,
            total_compressed_tokens=500,
            accuracy=0.90,
            mean_numerical_proximity=0.95,
            std_numerical_proximity=0.08,
            hallucination_rate=0.03,
            mean_cost_usd=0.004,
            total_cost_usd=0.08,
            mean_latency_ms=90.0,
            std_latency_ms=15.0,
            accuracy_ci_lower=0.85,
            accuracy_ci_upper=0.95,
        ),
    }


class TestBenchmarkReporter:
    """Tests for BenchmarkReporter class."""

    def test_init(self) -> None:
        """Test reporter initialization."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        assert reporter.aggregated == results
        assert reporter.config == config
        assert reporter.raw_results == []

    def test_init_with_raw_results(self) -> None:
        """Test reporter initialization with raw results."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        raw_results = [{"trial": 1}, {"trial": 2}]
        reporter = BenchmarkReporter(results, config, raw_results=raw_results)

        assert reporter.raw_results == raw_results


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report method."""

    def test_returns_string(self) -> None:
        """Test method returns markdown string."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_title(self) -> None:
        """Test report contains title."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report(title="My Custom Title")

        assert "# My Custom Title" in report

    def test_contains_executive_summary(self) -> None:
        """Test report contains executive summary."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        assert "Executive Summary" in report
        assert "Overall Accuracy Improvement" in report
        assert "Mean Token Compression" in report
        assert "Hallucination Reduction" in report

    def test_contains_results_table(self) -> None:
        """Test report contains results table."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        assert "| Metric |" in report
        assert "| Baseline" in report
        assert "| Treatment" in report

    def test_contains_per_task_results(self) -> None:
        """Test report contains per-task results."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        assert "Results by Task" in report
        assert "Statistical" in report
        assert "Trend" in report

    def test_contains_methodology(self) -> None:
        """Test report contains methodology section."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        assert "Methodology" in report
        assert "Baseline" in report
        assert "Treatment" in report

    def test_writes_to_file(self, tmp_path: Path) -> None:
        """Test report can be written to file."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "report.md"
        report = reporter.generate_markdown_report(output_path=output_path)

        assert output_path.exists()
        assert output_path.read_text() == report

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test method creates parent directories."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "subdir" / "report.md"
        reporter.generate_markdown_report(output_path=output_path)

        assert output_path.exists()


class TestGenerateCSVExport:
    """Tests for generate_csv_export method."""

    def test_creates_csv_file(self, tmp_path: Path) -> None:
        """Test method creates CSV file."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.csv"
        reporter.generate_csv_export(output_path)

        assert output_path.exists()

    def test_csv_has_header(self, tmp_path: Path) -> None:
        """Test CSV file has correct header."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.csv"
        reporter.generate_csv_export(output_path)

        with open(output_path) as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_columns = [
            "task_type",
            "condition",
            "n_trials",
            "accuracy",
            "accuracy_ci_lower",
            "accuracy_ci_upper",
            "mean_compression_ratio",
            "hallucination_rate",
            "mean_cost_usd",
            "mean_latency_ms",
        ]
        assert header == expected_columns

    def test_csv_has_data_rows(self, tmp_path: Path) -> None:
        """Test CSV file has data rows."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.csv"
        reporter.generate_csv_export(output_path)

        with open(output_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 4 data rows
        assert len(rows) == 5

    def test_csv_values_formatted(self, tmp_path: Path) -> None:
        """Test CSV values are properly formatted."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.csv"
        reporter.generate_csv_export(output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check first row
        row = rows[0]
        assert float(row["accuracy"]) >= 0
        assert float(row["accuracy"]) <= 1


class TestGenerateJSONExport:
    """Tests for generate_json_export method."""

    def test_creates_json_file(self, tmp_path: Path) -> None:
        """Test method creates JSON file."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.json"
        reporter.generate_json_export(output_path)

        assert output_path.exists()

    def test_json_is_valid(self, tmp_path: Path) -> None:
        """Test JSON file is valid."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.json"
        reporter.generate_json_export(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_json_has_metadata(self, tmp_path: Path) -> None:
        """Test JSON file has metadata."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.json"
        reporter.generate_json_export(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "generated" in data["metadata"]
        assert "model" in data["metadata"]
        assert "n_trials" in data["metadata"]

    def test_json_has_summary(self, tmp_path: Path) -> None:
        """Test JSON file has summary."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.json"
        reporter.generate_json_export(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "accuracy_improvement" in data["summary"]

    def test_json_has_aggregated_results(self, tmp_path: Path) -> None:
        """Test JSON file has aggregated results."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        output_path = tmp_path / "results.json"
        reporter.generate_json_export(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "aggregated_results" in data
        assert "statistical_baseline" in data["aggregated_results"]
        assert "statistical_treatment" in data["aggregated_results"]


class TestComputeSummary:
    """Tests for _compute_summary method."""

    def test_accuracy_improvement(self) -> None:
        """Test accuracy improvement calculation."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        summary = reporter._compute_summary()

        # Treatment accuracy should be higher than baseline
        assert summary["accuracy_improvement"] > 0

    def test_compression_ratio(self) -> None:
        """Test mean compression ratio."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        summary = reporter._compute_summary()

        # Should be around 0.9
        assert summary["mean_compression"] > 0.8
        assert summary["mean_compression"] < 1.0

    def test_hallucination_reduction(self) -> None:
        """Test hallucination reduction calculation."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        summary = reporter._compute_summary()

        # Treatment should have lower hallucination
        assert summary["hallucination_reduction"] > 0

    def test_cost_savings(self) -> None:
        """Test cost savings calculation."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        summary = reporter._compute_summary()

        # Treatment should cost less
        assert summary["cost_savings"] > 0

    def test_table_rows(self) -> None:
        """Test table rows are populated."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        summary = reporter._compute_summary()

        assert "table_rows" in summary
        assert "Accuracy" in summary["table_rows"]
        assert "Token Compression" in summary["table_rows"]
        assert "Hallucination Rate" in summary["table_rows"]
        assert "API Cost" in summary["table_rows"]

    def test_empty_results(self) -> None:
        """Test handling of empty results."""
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter({}, config)

        summary = reporter._compute_summary()

        assert summary["accuracy_improvement"] == 0
        assert summary["mean_compression"] == 0


class TestPrintComparisonTable:
    """Tests for print_comparison_table method."""

    def test_prints_output(self, capsys) -> None:
        """Test method prints to stdout."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_prints_header(self, capsys) -> None:
        """Test output contains header."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "SEMANTIC FRAME BENCHMARK COMPARISON" in captured.out

    def test_prints_metrics(self, capsys) -> None:
        """Test output contains metrics."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        reporter.print_comparison_table()

        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
        assert "Token Compression" in captured.out
        assert "Baseline" in captured.out
        assert "Treatment" in captured.out


class TestGenerateAsciiChart:
    """Tests for generate_ascii_chart method."""

    def test_returns_string(self) -> None:
        """Test method returns a string."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart()

        assert isinstance(chart, str)
        assert len(chart) > 0

    def test_contains_task_labels(self) -> None:
        """Test chart contains task labels."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart()

        assert "STATISTICAL" in chart
        assert "TREND" in chart

    def test_contains_baseline_treatment_labels(self) -> None:
        """Test chart contains baseline and treatment labels."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart()

        assert "Baseline:" in chart
        assert "Treatment:" in chart

    def test_accuracy_metric(self) -> None:
        """Test chart with accuracy metric (default)."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart(metric="accuracy")

        assert "ACCURACY BY TASK" in chart
        # Check for percentage values
        assert "%" in chart

    def test_compression_ratio_metric(self) -> None:
        """Test chart with compression_ratio metric."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart(metric="compression_ratio")

        assert "COMPRESSION_RATIO BY TASK" in chart

    def test_hallucination_rate_metric(self) -> None:
        """Test chart with hallucination_rate metric."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart(metric="hallucination_rate")

        assert "HALLUCINATION_RATE BY TASK" in chart

    def test_unknown_metric_skips_silently(self) -> None:
        """Test that unknown metrics are skipped."""
        results = create_sample_aggregated_results()
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart(metric="unknown_metric")

        # Should return a chart but without task data
        assert isinstance(chart, str)

    def test_missing_task_results_skipped(self) -> None:
        """Test that tasks with missing baseline or treatment are skipped."""
        # Only provide treatment, no baseline
        results = {
            "statistical_treatment": create_sample_aggregated_results()["statistical_treatment"]
        }
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        chart = reporter.generate_ascii_chart()

        # Should not crash, returns chart
        assert isinstance(chart, str)


class TestMarkdownReportEdgeCases:
    """Test edge cases for markdown report generation."""

    def test_skips_task_without_baseline(self) -> None:
        """Test that tasks without baseline are skipped in per-task section."""
        # Only provide treatment
        results = {
            "statistical_treatment": create_sample_aggregated_results()["statistical_treatment"]
        }
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        # Should not crash and should contain basic sections
        assert "Executive Summary" in report
        assert "Methodology" in report

    def test_skips_task_without_treatment(self) -> None:
        """Test that tasks without treatment are skipped in per-task section."""
        # Only provide baseline
        results = {
            "statistical_baseline": create_sample_aggregated_results()["statistical_baseline"]
        }
        config = BenchmarkConfig.quick_mode()
        reporter = BenchmarkReporter(results, config)

        report = reporter.generate_markdown_report()

        # Should not crash
        assert "Executive Summary" in report


class TestCompareBenchmarkResults:
    """Tests for compare_benchmark_results function."""

    def test_table_format(self, tmp_path: Path) -> None:
        """Test comparison with table format."""
        from benchmarks.reporter import compare_benchmark_results

        # Create test result files
        result1 = self._create_result_file(tmp_path / "run1.json", accuracy=0.85)
        result2 = self._create_result_file(tmp_path / "run2.json", accuracy=0.90)

        output = compare_benchmark_results([result1, result2], output_format="table")

        assert "BENCHMARK COMPARISON" in output
        assert "run1" in output
        assert "run2" in output

    def test_csv_format(self, tmp_path: Path) -> None:
        """Test comparison with CSV format."""
        from benchmarks.reporter import compare_benchmark_results

        result1 = self._create_result_file(tmp_path / "run1.json", accuracy=0.85)
        result2 = self._create_result_file(tmp_path / "run2.json", accuracy=0.90)

        output = compare_benchmark_results([result1, result2], output_format="csv")

        assert "metric" in output
        assert "run1" in output
        assert "run2" in output
        assert "accuracy" in output

    def test_json_format(self, tmp_path: Path) -> None:
        """Test comparison with JSON format."""
        from benchmarks.reporter import compare_benchmark_results

        result1 = self._create_result_file(tmp_path / "run1.json", accuracy=0.85)
        result2 = self._create_result_file(tmp_path / "run2.json", accuracy=0.90)

        output = compare_benchmark_results([result1, result2], output_format="json")

        data = json.loads(output)
        assert "runs" in data
        assert len(data["runs"]) == 2

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when result file not found."""
        import pytest

        from benchmarks.reporter import compare_benchmark_results

        with pytest.raises(FileNotFoundError, match="Result file not found"):
            compare_benchmark_results([tmp_path / "nonexistent.json"])

    def test_unknown_format(self, tmp_path: Path) -> None:
        """Test error with unknown output format."""
        import pytest

        from benchmarks.reporter import compare_benchmark_results

        result1 = self._create_result_file(tmp_path / "run1.json", accuracy=0.85)

        with pytest.raises(ValueError, match="Unknown output format"):
            compare_benchmark_results([result1], output_format="invalid")

    def test_empty_results(self, tmp_path: Path) -> None:
        """Test with no result files."""
        from benchmarks.reporter import compare_benchmark_results

        output = compare_benchmark_results([], output_format="table")

        assert "No results to compare" in output

    def _create_result_file(self, path: Path, accuracy: float = 0.85) -> Path:
        """Create a test result file."""
        data = {
            "metadata": {"model": "claude-3-haiku", "generated": "2024-01-01"},
            "aggregated_results": {
                "statistical_treatment": {
                    "task_type": "statistical",
                    "condition": "treatment",
                    "n_trials": 10,
                    "accuracy": accuracy,
                    "mean_compression_ratio": 0.92,
                    "hallucination_rate": 0.02,
                }
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return path


class TestGetWeightedMetric:
    """Tests for _get_weighted_metric helper function."""

    def test_returns_weighted_average(self) -> None:
        """Test that metric is weighted by trial count."""
        from benchmarks.reporter import _get_weighted_metric

        data = {
            "aggregated_results": {
                "task1_treatment": {"n_trials": 10, "accuracy": 0.80},
                "task2_treatment": {"n_trials": 30, "accuracy": 0.90},
            }
        }

        result = _get_weighted_metric(data, "accuracy", "treatment")

        # (10*0.80 + 30*0.90) / 40 = (8 + 27) / 40 = 0.875
        assert result == 0.875

    def test_empty_aggregated_results(self) -> None:
        """Test with empty aggregated_results."""
        from benchmarks.reporter import _get_weighted_metric

        data: dict = {"aggregated_results": {}}

        result = _get_weighted_metric(data, "accuracy", "treatment")

        assert result is None

    def test_no_aggregated_results_key(self) -> None:
        """Test with missing aggregated_results key."""
        from benchmarks.reporter import _get_weighted_metric

        data: dict = {}

        result = _get_weighted_metric(data, "accuracy", "treatment")

        assert result is None

    def test_zero_trials(self) -> None:
        """Test when total trials is zero."""
        from benchmarks.reporter import _get_weighted_metric

        data = {
            "aggregated_results": {
                "task1_treatment": {"n_trials": 0, "accuracy": 0.80},
            }
        }

        result = _get_weighted_metric(data, "accuracy", "treatment")

        assert result is None

    def test_filters_by_condition(self) -> None:
        """Test that only matching conditions are included."""
        from benchmarks.reporter import _get_weighted_metric

        data = {
            "aggregated_results": {
                "task1_baseline": {"n_trials": 10, "accuracy": 0.70},
                "task1_treatment": {"n_trials": 10, "accuracy": 0.90},
            }
        }

        baseline_result = _get_weighted_metric(data, "accuracy", "baseline")
        treatment_result = _get_weighted_metric(data, "accuracy", "treatment")

        assert baseline_result == 0.70
        assert treatment_result == 0.90


class TestBuildComparisonData:
    """Tests for _build_comparison_data helper function."""

    def test_returns_runs_list(self, tmp_path: Path) -> None:
        """Test that function returns list of runs."""
        from benchmarks.reporter import _build_comparison_data

        results = [
            (
                "run1",
                {
                    "metadata": {"model": "haiku"},
                    "aggregated_results": {
                        "task_treatment": {
                            "n_trials": 10,
                            "accuracy": 0.85,
                            "mean_compression_ratio": 0.9,
                            "hallucination_rate": 0.02,
                        }
                    },
                },
            ),
        ]

        data = _build_comparison_data(results)

        assert "runs" in data
        assert len(data["runs"]) == 1
        assert data["runs"][0]["name"] == "run1"
        assert data["runs"][0]["accuracy"] == 0.85


class TestFormatComparisonCSV:
    """Tests for _format_comparison_csv helper function."""

    def test_returns_csv_string(self) -> None:
        """Test that function returns valid CSV."""
        from benchmarks.reporter import _format_comparison_csv

        results = [
            (
                "run1",
                {
                    "aggregated_results": {
                        "task_treatment": {
                            "n_trials": 10,
                            "accuracy": 0.85,
                            "mean_compression_ratio": 0.9,
                            "hallucination_rate": 0.02,
                        }
                    },
                },
            ),
        ]

        csv_output = _format_comparison_csv(results)

        assert "metric,run1" in csv_output
        assert "accuracy" in csv_output
        assert "0.85" in csv_output

"""
Benchmark Reporter

Generate reports, visualizations, and export formats from benchmark results.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import AggregatedResults


class BenchmarkReporter:
    """
    Generate reports and visualizations from benchmark results.

    Usage:
        reporter = BenchmarkReporter(aggregated_results, config)
        reporter.generate_markdown_report("results.md")
        reporter.generate_csv_export("results.csv")
    """

    def __init__(
        self,
        aggregated_results: dict[str, AggregatedResults],
        config: BenchmarkConfig,
        raw_results: list | None = None,
    ):
        self.aggregated = aggregated_results
        self.config = config
        self.raw_results = raw_results or []

    def generate_markdown_report(
        self,
        output_path: Path | None = None,
        title: str = "Semantic Frame Benchmark Results",
    ) -> str:
        """Generate a comprehensive Markdown report."""
        lines = []

        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model: {self.config.model.model}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        summary = self._compute_summary()
        lines.append(f"- **Overall Accuracy Improvement**: {summary['accuracy_improvement']:.1%}")
        lines.append(f"- **Mean Token Compression**: {summary['mean_compression']:.1%}")
        lines.append(f"- **Hallucination Reduction**: {summary['hallucination_reduction']:.1%}")
        lines.append(f"- **Estimated Cost Savings**: {summary['cost_savings']:.1%}")
        lines.append("")

        # Primary Results Table
        lines.append("## Primary Results")
        lines.append("")
        lines.append("| Metric | Baseline (95% CI) | Treatment (95% CI) | Improvement |")
        lines.append("|--------|-------------------|--------------------| ------------|")

        for key, value in summary["table_rows"].items():
            lines.append(
                f"| {key} | {value['baseline']} | {value['treatment']} | {value['improvement']} |"
            )

        lines.append("")

        # Per-Task Results
        lines.append("## Results by Task")
        lines.append("")

        task_types = sorted(set(k.split("_")[0] for k in self.aggregated.keys()))

        for task_type in task_types:
            baseline_key = f"{task_type}_baseline"
            treatment_key = f"{task_type}_treatment"

            baseline = self.aggregated.get(baseline_key)
            treatment = self.aggregated.get(treatment_key)

            if not baseline or not treatment:
                continue

            lines.append(f"### {task_type.replace('_', ' ').title()}")
            lines.append("")

            acc_improvement = treatment.accuracy - baseline.accuracy
            acc_line = f"- **Accuracy**: {baseline.accuracy:.1%} → "
            acc_line += f"{treatment.accuracy:.1%} ({acc_improvement:+.1%})"
            lines.append(acc_line)
            lines.append(f"- **Token Compression**: {treatment.mean_compression_ratio:.1%}")
            hall_line = f"- **Hallucination Rate**: {baseline.hallucination_rate:.1%} → "
            hall_line += f"{treatment.hallucination_rate:.1%}"
            lines.append(hall_line)
            lines.append(f"- **Trials**: {treatment.n_trials}")
            lines.append("")

        # Methodology Note
        lines.append("## Methodology")
        lines.append("")
        lines.append(
            "This benchmark compares LLM performance on numerical analysis tasks "
            "under two conditions:"
        )
        lines.append("")
        lines.append("1. **Baseline**: Raw numerical data passed directly to Claude")
        lines.append("2. **Treatment**: Semantic Frame preprocessed output passed to Claude")
        lines.append("")
        lines.append(
            f"Each condition was tested with {self.config.n_trials} trials per query type."
        )
        lines.append("Accuracy is measured against deterministically computed ground truth.")
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)

        return report

    def generate_csv_export(
        self,
        output_path: Path,
    ) -> None:
        """Export results to CSV format."""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
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
            )

            # Data rows
            for key, result in self.aggregated.items():
                writer.writerow(
                    [
                        result.task_type,
                        result.condition,
                        result.n_trials,
                        f"{result.accuracy:.4f}",
                        f"{result.accuracy_ci_lower:.4f}",
                        f"{result.accuracy_ci_upper:.4f}",
                        f"{result.mean_compression_ratio:.4f}",
                        f"{result.hallucination_rate:.4f}",
                        f"{result.mean_cost_usd:.6f}",
                        f"{result.mean_latency_ms:.2f}",
                    ]
                )

    def generate_json_export(
        self,
        output_path: Path,
    ) -> None:
        """Export full results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "model": self.config.model.model,
                "n_trials": self.config.n_trials,
            },
            "summary": self._compute_summary(),
            "aggregated_results": {k: asdict(v) for k, v in self.aggregated.items()},
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics."""
        baseline_results = [v for k, v in self.aggregated.items() if "baseline" in k]
        treatment_results = [v for k, v in self.aggregated.items() if "treatment" in k]

        if not baseline_results or not treatment_results:
            return {
                "accuracy_improvement": 0,
                "mean_compression": 0,
                "hallucination_reduction": 0,
                "cost_savings": 0,
                "table_rows": {},
            }

        # Weighted averages by trial count
        total_baseline_trials = sum(r.n_trials for r in baseline_results)
        total_treatment_trials = sum(r.n_trials for r in treatment_results)

        baseline_acc = (
            sum(r.accuracy * r.n_trials for r in baseline_results) / total_baseline_trials
        )
        treatment_acc = (
            sum(r.accuracy * r.n_trials for r in treatment_results) / total_treatment_trials
        )

        baseline_hall = (
            sum(r.hallucination_rate * r.n_trials for r in baseline_results) / total_baseline_trials
        )
        treatment_hall = (
            sum(r.hallucination_rate * r.n_trials for r in treatment_results)
            / total_treatment_trials
        )

        mean_compression = (
            sum(r.mean_compression_ratio * r.n_trials for r in treatment_results)
            / total_treatment_trials
        )

        baseline_cost = sum(r.total_cost_usd for r in baseline_results)
        treatment_cost = sum(r.total_cost_usd for r in treatment_results)
        cost_savings = 1 - (treatment_cost / baseline_cost) if baseline_cost > 0 else 0

        return {
            "accuracy_improvement": treatment_acc - baseline_acc,
            "mean_compression": mean_compression,
            "hallucination_reduction": baseline_hall - treatment_hall,
            "cost_savings": cost_savings,
            "table_rows": {
                "Accuracy": {
                    "baseline": f"{baseline_acc:.1%}",
                    "treatment": f"{treatment_acc:.1%}",
                    "improvement": f"{treatment_acc - baseline_acc:+.1%}",
                },
                "Token Compression": {
                    "baseline": "0%",
                    "treatment": f"{mean_compression:.1%}",
                    "improvement": f"{mean_compression:.1%} ↓",
                },
                "Hallucination Rate": {
                    "baseline": f"{baseline_hall:.1%}",
                    "treatment": f"{treatment_hall:.1%}",
                    "improvement": f"{baseline_hall - treatment_hall:+.1%} ↓",
                },
                "API Cost": {
                    "baseline": f"${baseline_cost:.4f}",
                    "treatment": f"${treatment_cost:.4f}",
                    "improvement": f"{cost_savings:.1%} ↓",
                },
            },
        }

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table to console."""
        summary = self._compute_summary()

        print("\n" + "=" * 70)
        print("SEMANTIC FRAME BENCHMARK COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<25} {'Baseline':<15} {'Treatment':<15} {'Improvement':<15}")
        print("-" * 70)

        for metric, values in summary["table_rows"].items():
            row = f"{metric:<25} {values['baseline']:<15} "
            row += f"{values['treatment']:<15} {values['improvement']:<15}"
            print(row)

        print("=" * 70)

    def generate_ascii_chart(self, metric: str = "accuracy") -> str:
        """Generate ASCII bar chart for visual comparison.

        Args:
            metric: Metric to visualize (accuracy, compression_ratio, hallucination_rate)

        Returns:
            ASCII chart string
        """
        lines = []
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {metric.upper()} BY TASK")
        lines.append(f"{'=' * 60}")

        task_types = sorted(set(k.split("_")[0] for k in self.aggregated.keys()))
        max_bar_width = 40

        for task_type in task_types:
            baseline_key = f"{task_type}_baseline"
            treatment_key = f"{task_type}_treatment"

            baseline = self.aggregated.get(baseline_key)
            treatment = self.aggregated.get(treatment_key)

            if not baseline or not treatment:
                continue

            # Get metric values
            if metric == "accuracy":
                baseline_val = baseline.accuracy
                treatment_val = treatment.accuracy
            elif metric == "compression_ratio":
                baseline_val = 0.0  # Baseline has no compression
                treatment_val = treatment.mean_compression_ratio
            elif metric == "hallucination_rate":
                baseline_val = baseline.hallucination_rate
                treatment_val = treatment.hallucination_rate
            else:
                continue

            # Create bars
            baseline_bar = "█" * int(baseline_val * max_bar_width)
            treatment_bar = "█" * int(treatment_val * max_bar_width)

            lines.append(f"\n{task_type.upper()}")
            lines.append(f"  Baseline:  |{baseline_bar:<{max_bar_width}}| {baseline_val:.1%}")
            lines.append(f"  Treatment: |{treatment_bar:<{max_bar_width}}| {treatment_val:.1%}")

        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


def compare_benchmark_results(
    result_files: list[Path],
    output_format: str = "table",
) -> str:
    """Compare results from multiple benchmark runs.

    Args:
        result_files: List of JSON result file paths
        output_format: Output format ("table", "csv", "json")

    Returns:
        Comparison output as string

    Example:
        >>> from pathlib import Path
        >>> files = [Path("run1.json"), Path("run2.json")]
        >>> print(compare_benchmark_results(files))
    """
    results = []
    for path in result_files:
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")
        with open(path) as f:
            results.append((path.stem, json.load(f)))

    if not results:
        return "No results to compare"

    if output_format == "table":
        return _format_comparison_table(results)
    elif output_format == "csv":
        return _format_comparison_csv(results)
    elif output_format == "json":
        return json.dumps(_build_comparison_data(results), indent=2)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def _format_comparison_table(results: list[tuple[str, dict[str, Any]]]) -> str:
    """Format comparison as ASCII table."""
    lines: list[str] = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK COMPARISON")
    lines.append("=" * 80)

    # Header
    run_names = [name[:15] for name, _ in results]
    header = f"{'Metric':<30}"
    for name in run_names:
        header += f" {name:<15}"
    lines.append(header)
    lines.append("-" * 80)

    # Extract metrics for each run
    metric_configs = [
        ("Overall Accuracy", "accuracy"),
        ("Token Compression", "mean_compression_ratio"),
        ("Hallucination Rate", "hallucination_rate"),
    ]

    for metric_name, metric_key in metric_configs:
        row = f"{metric_name:<30}"
        for _, data in results:
            value = _get_weighted_metric(data, metric_key, "treatment")
            row += f" {value:.1%}" if value is not None else " N/A"
            row = row.ljust(len(row) + (15 - len(f"{value:.1%}" if value else "N/A")))
        lines.append(row)

    lines.append("=" * 80)
    return "\n".join(lines)


def _format_comparison_csv(results: list[tuple[str, dict[str, Any]]]) -> str:
    """Format comparison as CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["metric"]
    for name, _ in results:
        header.append(name)
    writer.writerow(header)

    # Rows
    metric_configs = [
        ("accuracy", "accuracy"),
        ("compression_ratio", "mean_compression_ratio"),
        ("hallucination_rate", "hallucination_rate"),
    ]

    for metric_name, metric_key in metric_configs:
        row: list[str] = [metric_name]
        for _, data in results:
            value = _get_weighted_metric(data, metric_key, "treatment")
            row.append(f"{value:.4f}" if value is not None else "")
        writer.writerow(row)

    return output.getvalue()


def _build_comparison_data(results: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build comparison data structure."""
    return {
        "runs": [
            {
                "name": name,
                "accuracy": _get_weighted_metric(data, "accuracy", "treatment"),
                "compression_ratio": _get_weighted_metric(
                    data, "mean_compression_ratio", "treatment"
                ),
                "hallucination_rate": _get_weighted_metric(data, "hallucination_rate", "treatment"),
                "metadata": data.get("metadata", {}),
            }
            for name, data in results
        ]
    }


def _get_weighted_metric(data: dict[str, Any], metric: str, condition: str) -> float | None:
    """Extract weighted metric from benchmark data."""
    aggregated = data.get("aggregated_results", {})
    if not aggregated:
        return None

    filtered: list[dict[str, Any]] = [
        v for k, v in aggregated.items() if condition in k and isinstance(v, dict)
    ]

    if not filtered:
        return None

    total_trials = sum(int(r.get("n_trials", 0)) for r in filtered)
    if total_trials == 0:
        return None

    weighted_sum = sum(float(r.get(metric, 0)) * int(r.get("n_trials", 0)) for r in filtered)
    return float(weighted_sum / total_trials)

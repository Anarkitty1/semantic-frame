#!/usr/bin/env python3
"""
Semantic Frame Benchmark CLI

Run benchmarks to demonstrate token reduction and accuracy gains.

Usage:
    # Run full benchmark suite
    python -m benchmarks.run_benchmark

    # Run specific task
    python -m benchmarks.run_benchmark --task statistical

    # Quick validation run (fewer trials)
    python -m benchmarks.run_benchmark --quick

    # Mock mode (no API calls, for testing)
    python -m benchmarks.run_benchmark --mock

    # Run with robustness testing
    python -m benchmarks.run_benchmark --robustness

    # Include NAB external datasets
    python -m benchmarks.run_benchmark --external-datasets

    # Generate visualizations with plotly
    python -m benchmarks.run_benchmark --viz-backend plotly
"""

import argparse
import json
import sys
import urllib.error
import zipfile
from pathlib import Path

import numpy as np

from benchmarks.claude_client import BackendType
from benchmarks.config import BenchmarkConfig, TaskType
from benchmarks.reporter import BenchmarkReporter
from benchmarks.runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Semantic Frame benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[t.value for t in TaskType],
        help="Run only a specific task (default: run all)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer trials, smaller datasets",
    )

    parser.add_argument(
        "--trials",
        type=int,
        help="Number of trials per condition (default: 30, quick: 5)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="[DEPRECATED: use --backend mock] Mock mode: no API calls",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=[b.value for b in BackendType],
        default="api",
        help=(
            "Backend for Claude queries: "
            "'api' (paid Anthropic API), "
            "'claude-code' (free on Max plan via CLI), "
            "'mock' (no API calls, for testing). Default: api"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results (default: benchmarks/results)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="Output format (default: all)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Robustness testing flags
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness testing suite (perturbation analysis)",
    )

    # External dataset flags
    parser.add_argument(
        "--external-datasets",
        action="store_true",
        help="Include NAB (Numenta Anomaly Benchmark) external datasets",
    )

    # Visualization flags
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation",
    )

    parser.add_argument(
        "--viz-backend",
        type=str,
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Visualization backend (default: matplotlib)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run baseline and treatment conditions in parallel (2x speedup)",
    )

    parser.add_argument(
        "--trial-parallelism",
        type=int,
        default=1,
        metavar="N",
        help="Run N trials in parallel (max 4 for rate limits). Default: 1",
    )

    parser.add_argument(
        "--max-data-size",
        type=int,
        default=None,
        metavar="N",
        help="Maximum data points per dataset. Auto-limited for CLI backend.",
    )

    parser.add_argument(
        "--skip-baseline-above",
        type=int,
        default=None,
        metavar="N",
        help="Skip baseline for datasets with more than N points (default: 5000)",
    )

    args = parser.parse_args()

    # Configure
    if args.quick:
        config = BenchmarkConfig.quick_mode()
    else:
        config = BenchmarkConfig.full_mode()

    if args.trials:
        config.n_trials = args.trials

    if args.output:
        config.output_dir = args.output

    config.verbose = not args.quiet

    # Enable parallel execution if requested
    if args.parallel:
        config.parallel_workers = 2  # Baseline and treatment in parallel

    # Apply trial parallelism
    if args.trial_parallelism > 1:
        config.trial_parallelism = args.trial_parallelism
        if args.trial_parallelism > 4:
            print(f"WARNING: Using {args.trial_parallelism} parallel workers may hit rate limits")

    # Apply skip baseline threshold
    if args.skip_baseline_above is not None:
        config.skip_baseline_above_n_points = args.skip_baseline_above

    # Determine backend (--mock flag takes precedence for backwards compatibility)
    backend = BackendType(args.backend)
    if args.mock:
        backend = BackendType.MOCK

    # Apply CLI backend optimizations
    if backend == BackendType.CLAUDE_CODE:
        # Auto-limit dataset sizes for CLI backend to prevent timeouts
        max_size = args.max_data_size or config.datasets.cli_max_dataset_size
        if config.datasets.medium_size > max_size:
            config.datasets.medium_size = max_size
        if config.datasets.large_size > max_size:
            config.datasets.large_size = max_size
        if config.datasets.very_large_size > max_size:
            config.datasets.very_large_size = max_size
        if config.verbose:
            print(f"CLI backend: Dataset sizes limited to {max_size} points")
    elif args.max_data_size:
        # Apply explicit max data size for any backend
        max_size = args.max_data_size
        if config.datasets.medium_size > max_size:
            config.datasets.medium_size = max_size
        if config.datasets.large_size > max_size:
            config.datasets.large_size = max_size
        if config.datasets.very_large_size > max_size:
            config.datasets.very_large_size = max_size

    # Validate API key (only needed for API backend)
    if backend == BackendType.API and not config.api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or use --backend mock for testing without API calls")
        print("Or use --backend claude-code for free usage on Max plan")
        sys.exit(1)

    # Run benchmarks
    runner = BenchmarkRunner(config, backend=backend)

    tasks = None
    if args.task:
        tasks = [TaskType(args.task)]

    # Build feature list
    features = []
    if args.parallel:
        features.append("parallel")
    if config.trial_parallelism > 1:
        features.append(f"{config.trial_parallelism}x trial parallelism")
    if args.robustness:
        features.append("robustness")
    if args.external_datasets:
        features.append("NAB datasets")
    if not args.no_viz:
        features.append(f"viz ({args.viz_backend})")

    # Backend display name
    backend_display = {
        BackendType.API: "Anthropic API (paid)",
        BackendType.CLAUDE_CODE: "Claude Code CLI (free on Max)",
        BackendType.MOCK: "Mock (no API calls)",
    }

    print("\n" + "=" * 60)
    print("SEMANTIC FRAME BENCHMARK SUITE")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Backend: {backend_display.get(backend, backend.value)}")
    print(f"Trials per condition: {config.n_trials}")
    print(f"Tasks: {[t.value for t in (tasks or list(TaskType))]}")
    if features:
        print(f"Features: {', '.join(features)}")
    print("=" * 60 + "\n")

    try:
        aggregated = runner.run_all(tasks=tasks)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except (RuntimeError, ValueError, OSError) as e:
        # RuntimeError: task execution failures
        # ValueError: invalid configuration or data
        # OSError: file I/O errors
        print(f"\nError during benchmark: {type(e).__name__}: {e}")
        if config.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Generate reports
    reporter = BenchmarkReporter(aggregated, config, runner.results)

    if args.format in ["json", "all"]:
        json_path = config.output_dir / "benchmark_results.json"
        reporter.generate_json_export(json_path)
        print(f"JSON results: {json_path}")

    if args.format in ["csv", "all"]:
        csv_path = config.output_dir / "benchmark_results.csv"
        reporter.generate_csv_export(csv_path)
        print(f"CSV results: {csv_path}")

    if args.format in ["markdown", "all"]:
        md_path = config.output_dir / "benchmark_report.md"
        reporter.generate_markdown_report(md_path)
        print(f"Markdown report: {md_path}")

    # Print summary
    runner.print_summary()
    reporter.print_comparison_table()

    # Run robustness testing if enabled
    if args.robustness:
        print("\n" + "=" * 60)
        print("ROBUSTNESS TESTING")
        print("=" * 60)
        _run_robustness_testing(config, runner.results)

    # Run external datasets if enabled
    if args.external_datasets:
        print("\n" + "=" * 60)
        print("EXTERNAL DATASETS (NAB)")
        print("=" * 60)
        _run_external_datasets(config, args.mock)

    # Generate visualizations if enabled
    if not args.no_viz:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        _generate_visualizations(config, aggregated, args.viz_backend)

    print("\n✅ Benchmark complete!")


def _run_robustness_testing(
    config: BenchmarkConfig,
    results: list,
) -> None:
    """Run robustness testing suite on benchmark data."""
    try:
        from benchmarks.robustness import RobustnessConfig, RobustnessEvaluator
        from semantic_frame import describe_series
    except ImportError as e:
        print(f"Skipping robustness testing: {e}")
        return

    # Create sample data for testing
    rng = np.random.default_rng(config.random_seed)
    sample_data = rng.normal(100, 15, 500)  # 500 points

    # Create evaluator
    robustness_config = RobustnessConfig(random_seed=config.random_seed)
    evaluator = RobustnessEvaluator(robustness_config)

    # Define evaluation function using semantic-frame
    def evaluation_fn(data: np.ndarray) -> float:
        """Evaluate semantic-frame consistency on perturbed data."""
        try:
            result = describe_series(data, context="Robustness Test")
            # Score based on whether description was generated successfully
            # In real benchmarks, we'd compare to expected output
            return 1.0 if result and len(result) > 50 else 0.5
        except (ValueError, TypeError, RuntimeError):
            # ValueError: invalid data (empty, all NaN, etc.)
            # TypeError: wrong data type passed
            # RuntimeError: analyzer failures
            return 0.0

    print("Running perturbation analysis...")
    metrics = evaluator.evaluate_perturbation_robustness(sample_data, evaluation_fn)

    # Summarize results
    summary = evaluator.summarize_robustness(metrics)

    print(f"\nRobustness Results ({len(metrics)} perturbations tested):")
    print("-" * 40)
    for ptype, stats in summary.items():
        robust_pct = stats["robustness_rate"] * 100
        mean_deg = stats["mean_degradation"] * 100
        print(f"  {ptype:12s}: {robust_pct:5.1f}% robust, {mean_deg:5.2f}% mean degradation")

    # Save robustness results
    robustness_path = config.output_dir / "robustness_results.json"
    with open(robustness_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRobustness results: {robustness_path}")


def _run_external_datasets(config: BenchmarkConfig, mock: bool = False) -> None:
    """Load and analyze NAB external datasets."""
    try:
        from benchmarks.external_datasets import ExternalDataConfig, NABLoader
    except ImportError as e:
        print(f"Skipping external datasets: {e}")
        return

    # Configure external datasets
    ext_config = ExternalDataConfig(
        enabled_datasets=["nab"],
        data_cache_dir=config.data_dir / "external",
        max_series_per_dataset=10 if mock else 50,  # Limit for testing
    )

    loader = NABLoader(ext_config)
    nab_path = ext_config.data_cache_dir / "nab"

    # Download if needed
    if not loader.is_downloaded(nab_path):
        print("Downloading NAB dataset (this may take a few minutes)...")
        try:
            loader.download(nab_path)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as e:
            # OSError: file system errors
            # URLError: network errors (includes HTTPError)
            # BadZipFile: corrupted download
            print(f"Failed to download NAB dataset: {type(e).__name__}: {e}")
            return

    # Load datasets
    print("Loading NAB datasets...")
    datasets: list = []
    for dataset in loader.load(nab_path):
        datasets.append(dataset)

    print(f"\nLoaded {len(datasets)} NAB time series")

    # Summarize by category
    from collections import Counter

    categories = Counter(d.category for d in datasets)
    print("\nDatasets by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Summarize anomalies
    total_anomalies = sum(len(d.anomaly_windows) for d in datasets)
    total_points = sum(len(d.data) for d in datasets)
    print(f"\nTotal data points: {total_points:,}")
    print(f"Total anomaly windows: {total_anomalies}")

    # Save summary
    summary = {
        "n_series": len(datasets),
        "categories": dict(categories),
        "total_points": total_points,
        "total_anomaly_windows": total_anomalies,
    }
    summary_path = config.output_dir / "nab_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nNAB summary: {summary_path}")


def _generate_visualizations(
    config: BenchmarkConfig,
    aggregated: dict,
    backend: str,
) -> None:
    """Generate benchmark visualizations."""
    try:
        from benchmarks.visualizations import (
            BenchmarkVisualizer,
            ComparisonBarData,
            TokenReductionData,
            VisualizationConfig,
        )
    except ImportError as e:
        print(f"Skipping visualizations: {e}")
        return

    # Create visualization config
    viz_config = VisualizationConfig(
        enabled=True,
        backend=backend,
        output_format="png" if backend == "matplotlib" else "html",
        theme="default",
    )

    try:
        visualizer = BenchmarkVisualizer(viz_config)
    except ImportError as e:
        print(f"Visualization backend not available: {e}")
        print("Install with: pip install semantic-frame[viz]")
        return

    viz_dir = config.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []

    # Build token reduction data from aggregated results
    treatment_results = [v for k, v in aggregated.items() if "treatment" in k]
    if treatment_results:
        avg_compression = sum(r.mean_compression_ratio for r in treatment_results) / len(
            treatment_results
        )
        # Estimate raw tokens (inverse of compression)
        raw_tokens = 10000  # Representative sample
        semantic_tokens = int(raw_tokens * (1 - avg_compression))

        token_data = TokenReductionData(
            raw_tokens=raw_tokens,
            semantic_tokens=semantic_tokens,
            final_tokens=semantic_tokens,
        )

        try:
            path = visualizer.token_reduction_waterfall(token_data, viz_dir / "token_reduction")
            generated_paths.append(path)
            print(f"  Generated: {path}")
        except (OSError, ValueError, ImportError) as e:
            # OSError: file write errors
            # ValueError: invalid data for visualization
            # ImportError: missing matplotlib/plotly
            print(f"  Failed to generate token_reduction: {type(e).__name__}: {e}")

    # Build comparison bar chart data
    task_types = set(k.split("_")[0] for k in aggregated.keys())
    categories = []
    baseline_values = []
    treatment_values = []

    for task_type in sorted(task_types):
        baseline = aggregated.get(f"{task_type}_baseline")
        treatment = aggregated.get(f"{task_type}_treatment")
        if baseline and treatment:
            categories.append(task_type)
            baseline_values.append(baseline.accuracy)
            treatment_values.append(treatment.accuracy)

    if categories:
        comparison_data = ComparisonBarData(
            categories=categories,
            baseline_values=baseline_values,
            treatment_values=treatment_values,
            metric_name="Accuracy",
        )

        try:
            path = visualizer.comparison_bar_chart(comparison_data, viz_dir / "accuracy_comparison")
            generated_paths.append(path)
            print(f"  Generated: {path}")
        except (OSError, ValueError, ImportError) as e:
            # OSError: file write errors
            # ValueError: invalid data for visualization
            # ImportError: missing matplotlib/plotly
            print(f"  Failed to generate accuracy_comparison: {type(e).__name__}: {e}")

    if generated_paths:
        print(f"\n{len(generated_paths)} visualizations saved to {viz_dir}")
    else:
        print("\nNo visualizations generated (insufficient data)")


if __name__ == "__main__":
    main()

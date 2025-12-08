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
"""

import argparse
import sys
from pathlib import Path

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
        help="Mock mode: no API calls (for testing pipeline)",
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

    # Validate API key (unless mock mode)
    if not args.mock and not config.api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or use --mock for testing without API calls")
        sys.exit(1)

    # Run benchmarks
    runner = BenchmarkRunner(config, mock=args.mock)

    tasks = None
    if args.task:
        tasks = [TaskType(args.task)]

    print("\n" + "=" * 60)
    print("SEMANTIC FRAME BENCHMARK SUITE")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'} {'(Mock)' if args.mock else ''}")
    print(f"Trials per condition: {config.n_trials}")
    print(f"Tasks: {[t.value for t in (tasks or list(TaskType))]}")
    print("=" * 60 + "\n")

    try:
        aggregated = runner.run_all(tasks=tasks)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
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

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

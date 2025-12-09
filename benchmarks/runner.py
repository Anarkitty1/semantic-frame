"""
Benchmark Runner

Main orchestration for running benchmark suites.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmarks.claude_client import BackendType, ClaudeCodeClient, ClientType, get_client
from benchmarks.config import BenchmarkConfig, TaskType
from benchmarks.logging_config import get_logger
from benchmarks.metrics import AggregatedResults, TrialResult
from benchmarks.tasks import (
    AnomalyTask,
    ComparativeTask,
    MultiStepTask,
    ScalingTask,
    StatisticalTask,
    TrendTask,
)

# Module logger
_log = get_logger("runner")

TASK_CLASSES = {
    TaskType.STATISTICAL: StatisticalTask,
    TaskType.TREND: TrendTask,
    TaskType.ANOMALY: AnomalyTask,
    TaskType.COMPARATIVE: ComparativeTask,
    TaskType.MULTI_STEP: MultiStepTask,
    TaskType.SCALING: ScalingTask,
}


class BenchmarkRunner:
    """
    Main runner for executing benchmark suites.

    Usage:
        config = BenchmarkConfig.quick_mode()
        runner = BenchmarkRunner(config)
        results = runner.run_all()
    """

    client: ClientType
    config: BenchmarkConfig
    results: list[TrialResult]
    aggregated: dict[str, AggregatedResults]
    run_timestamp: datetime | None

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        mock: bool = False,
        backend: BackendType | str | None = None,
    ):
        self.config = config or BenchmarkConfig()

        # Determine backend: explicit backend param > mock flag > default API
        if backend is not None:
            self.client = get_client(self.config, backend=backend)
        elif mock:
            # Backwards compatibility: mock=True uses mock backend
            self.client = get_client(self.config, backend=BackendType.MOCK)
        else:
            self.client = get_client(self.config, backend=BackendType.API)

        self.results = []
        self.aggregated: dict[str, AggregatedResults] = {}
        self.run_timestamp: datetime | None = None

    def run_task(
        self,
        task_type: TaskType,
        n_trials: int | None = None,
    ) -> list[TrialResult]:
        """Run a specific task benchmark."""
        if task_type not in TASK_CLASSES:
            raise ValueError(f"Unknown task type: {task_type}")

        task_class = TASK_CLASSES[task_type]
        task = task_class(self.config, self.client)

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"Running {task_type.value} benchmark")
            print(f"{'=' * 60}")

        start_time = time.perf_counter()
        results = task.run(n_trials=n_trials)
        elapsed = time.perf_counter() - start_time

        if self.config.verbose:
            print(f"Completed {len(results)} trials in {elapsed:.1f}s")

        return list(results)

    def run_all(
        self,
        tasks: list[TaskType] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, AggregatedResults]:
        """
        Run all specified tasks (or all tasks if none specified).

        Returns aggregated results for each task and condition.
        """
        self.run_timestamp = datetime.now()

        # Warmup CLI backend to trigger cache creation before benchmarks
        if isinstance(self.client, ClaudeCodeClient):
            if self.config.verbose:
                print("Warming up Claude Code CLI (first call triggers cache creation)...")
            if self.client.warmup():
                if self.config.verbose:
                    print("CLI cache warmed up successfully\n")
            else:
                print("WARNING: CLI warmup failed, first queries may be slow\n")

        if tasks is None:
            tasks = list(TaskType)

        self.results = []

        for task_type in tasks:
            task_results = self.run_task(task_type, n_trials=n_trials)
            self.results.extend(task_results)

        # Aggregate results
        self.aggregated = self._aggregate_results()

        return self.aggregated

    def _aggregate_results(self) -> dict[str, AggregatedResults]:
        """Aggregate results by task type and condition."""
        from collections import defaultdict

        grouped = defaultdict(list)

        for result in self.results:
            key = f"{result.task_type}_{result.condition}"
            grouped[key].append(result)

        aggregated = {}
        failed_aggregations = []
        for key, trials in grouped.items():
            try:
                aggregated[key] = AggregatedResults.from_trials(trials)
            except (ValueError, TypeError, AttributeError) as e:
                # ValueError: empty trials, invalid data
                # TypeError: wrong types in trial data
                # AttributeError: missing fields in trial results
                error_msg = (
                    f"Could not aggregate {key} ({len(trials)} trials): {type(e).__name__}: {e}"
                )
                print(f"ERROR: {error_msg}", flush=True)
                failed_aggregations.append(key)

        if failed_aggregations:
            print(
                f"\nWARNING: {len(failed_aggregations)} aggregations failed: {failed_aggregations}"
            )

        return aggregated

    def save_results(
        self,
        output_dir: Path | None = None,
        prefix: str = "benchmark",
    ) -> Path:
        """Save results to JSON file."""
        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.run_timestamp or datetime.now()
        filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = output_dir / filename

        # Prepare data for serialization
        data = {
            "metadata": {
                "timestamp": timestamp.isoformat(),
                "config": {
                    "model": self.config.model.model,
                    "n_trials": self.config.n_trials,
                    "random_seed": self.config.random_seed,
                },
                "total_trials": len(self.results),
            },
            "aggregated_results": {k: asdict(v) for k, v in self.aggregated.items()},
            "raw_results": [
                {
                    "task_type": r.task_type,
                    "condition": r.condition,
                    "query": r.query,
                    "is_correct": r.is_correct,
                    "numerical_proximity": r.numerical_proximity,
                    "hallucination_detected": r.hallucination_detected,
                    "compression_ratio": r.token_metrics.compression_ratio,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            error_type = type(e).__name__
            print(f"ERROR: Failed to save results to {output_path}: {error_type}: {e}")
            raise
        except (TypeError, ValueError) as e:
            error_type = type(e).__name__
            print(f"ERROR: Failed to serialize results: {error_type}: {e}")
            raise

        if self.config.verbose:
            print(f"\nResults saved to: {output_path}")

        return output_path

    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        if not self.aggregated:
            print("No results to summarize. Run benchmarks first.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by task type
        task_types = set(k.split("_")[0] for k in self.aggregated.keys())

        for task_type in sorted(task_types):
            baseline_key = f"{task_type}_baseline"
            treatment_key = f"{task_type}_treatment"

            baseline = self.aggregated.get(baseline_key)
            treatment = self.aggregated.get(treatment_key)

            if not baseline or not treatment:
                continue

            print(f"\n{task_type.upper()}")
            print("-" * 40)

            # Accuracy comparison
            acc_improvement = treatment.accuracy - baseline.accuracy
            print(
                f"Accuracy:     {baseline.accuracy:.1%} → {treatment.accuracy:.1%} "
                f"({acc_improvement:+.1%})"
            )

            # Token compression
            print(f"Compression:  {treatment.mean_compression_ratio:.1%} token reduction")

            # Hallucination rate
            hall_base = baseline.hallucination_rate
            hall_treat = treatment.hallucination_rate
            print(f"Hallucination: {hall_base:.1%} → {hall_treat:.1%}")

            # Cost
            if baseline.total_cost_usd > 0:
                cost_reduction = 1 - (treatment.total_cost_usd / baseline.total_cost_usd)
                cost_base = baseline.total_cost_usd
                cost_treat = treatment.total_cost_usd
                print(f"Cost:         ${cost_base:.4f} → ${cost_treat:.4f} ", end="")
                print(f"({cost_reduction:.1%} savings)")

        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL")
        print("=" * 80)

        all_baseline = [r for r in self.results if r.condition == "baseline"]
        all_treatment = [r for r in self.results if r.condition == "treatment"]

        if all_baseline and all_treatment:
            baseline_acc = sum(1 for r in all_baseline if r.is_correct) / len(all_baseline)
            treatment_acc = sum(1 for r in all_treatment if r.is_correct) / len(all_treatment)

            baseline_hall = sum(1 for r in all_baseline if r.hallucination_detected) / len(
                all_baseline
            )
            treatment_hall = sum(1 for r in all_treatment if r.hallucination_detected) / len(
                all_treatment
            )

            compressions = [r.token_metrics.compression_ratio for r in all_treatment]
            mean_compression = sum(compressions) / len(compressions)

            print(f"Total trials:        {len(self.results)}")
            print(f"Overall accuracy:    {baseline_acc:.1%} → {treatment_acc:.1%}")
            print(f"Mean compression:    {mean_compression:.1%}")
            print(f"Hallucination rate:  {baseline_hall:.1%} → {treatment_hall:.1%}")

        print("=" * 80)

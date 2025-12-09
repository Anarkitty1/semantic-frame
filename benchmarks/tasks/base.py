"""
Base Task Class

Abstract base class for all benchmark tasks.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from benchmarks.claude_client import ClaudeClient, ClaudeResponse
from benchmarks.config import BenchmarkConfig, TaskType
from benchmarks.datasets import SyntheticDataset
from benchmarks.metrics import (
    Condition,
    CostMetrics,
    TokenMetrics,
    TrialResult,
    detect_hallucination,
)


@dataclass
class TaskResult:
    """Result from a single task evaluation."""

    task_type: TaskType
    dataset_name: str
    query_name: str

    # Baseline results
    baseline_response: ClaudeResponse
    baseline_answer: Any
    baseline_correct: bool
    baseline_tokens: TokenMetrics

    # Treatment results
    treatment_response: ClaudeResponse
    treatment_answer: Any
    treatment_correct: bool
    treatment_tokens: TokenMetrics

    # Ground truth
    expected_answer: Any

    # Semantic Frame output (for reference)
    semantic_frame_output: str

    # Raw data for hallucination detection
    raw_data: list[float] | None = None


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    task_type: TaskType  # Must be set by subclasses

    def __init__(self, config: BenchmarkConfig, client: ClaudeClient):
        self.config = config
        self.client = client

    @abstractmethod
    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate or load datasets for this task."""
        pass

    @abstractmethod
    def get_queries(self) -> dict[str, str]:
        """Get query templates for this task."""
        pass

    @abstractmethod
    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate if predicted answer is correct.

        Returns:
            Tuple of (is_correct, numerical_proximity_score)
        """
        pass

    def get_semantic_frame_output(self, dataset: SyntheticDataset) -> str:
        """Get Semantic Frame description of the data."""
        from semantic_frame import describe_series

        return describe_series(dataset.data, context=dataset.name)

    def run_single_trial(
        self,
        dataset: SyntheticDataset,
        query_name: str,
        query: str,
        expected_answer: Any,
    ) -> TaskResult:
        """
        Run a single trial comparing baseline vs treatment.

        If parallel_workers > 1, runs baseline and treatment concurrently.
        If dataset exceeds skip_baseline_above_n_points, skips baseline query.
        """
        # Prepare data representations
        raw_data_json = dataset.to_json()
        semantic_output = self.get_semantic_frame_output(dataset)

        # Calculate token metrics
        baseline_tokens = TokenMetrics.compute(raw_data_json, raw_data_json)
        treatment_tokens = TokenMetrics.compute(raw_data_json, semantic_output)

        # Check if we should skip baseline for large datasets
        n_points = len(dataset.data)
        skip_baseline = n_points > self.config.skip_baseline_above_n_points

        if skip_baseline:
            # Skip baseline for very large datasets (would timeout or use too many tokens)
            baseline_response = ClaudeResponse(
                content="[SKIPPED: dataset too large for baseline]",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                model=self.config.model.model,
                parsed={},
                error="baseline_skipped_large_dataset",
            )
            # Always run treatment
            treatment_response = self.client.query_treatment(semantic_output, query)
        elif self.config.parallel_workers > 1:
            # Run baseline and treatment in parallel
            baseline_response, treatment_response = self._run_conditions_parallel(
                raw_data_json, semantic_output, query
            )
        else:
            # Sequential execution
            baseline_response = self.client.query_baseline(raw_data_json, query)
            treatment_response = self.client.query_treatment(semantic_output, query)

        # Process baseline results
        if baseline_response.error and baseline_response.error != "baseline_skipped_large_dataset":
            print(
                f"  WARNING: Baseline API error (answer will be marked incorrect): "
                f"{baseline_response.error}"
            )
        baseline_answer = baseline_response.parsed.get("answer")
        baseline_correct, baseline_proximity = (
            (False, 0.0)
            if baseline_response.error
            else self.evaluate_answer(baseline_answer, expected_answer, dataset)
        )

        # Process treatment results
        if treatment_response.error:
            print(
                f"  WARNING: Treatment API error (answer will be marked incorrect): "
                f"{treatment_response.error}"
            )
        treatment_answer = treatment_response.parsed.get("answer")
        treatment_correct, treatment_proximity = (
            (False, 0.0)
            if treatment_response.error
            else self.evaluate_answer(treatment_answer, expected_answer, dataset)
        )

        return TaskResult(
            task_type=self.task_type,
            dataset_name=dataset.name,
            query_name=query_name,
            baseline_response=baseline_response,
            baseline_answer=baseline_answer,
            baseline_correct=baseline_correct,
            baseline_tokens=baseline_tokens,
            treatment_response=treatment_response,
            treatment_answer=treatment_answer,
            treatment_correct=treatment_correct,
            treatment_tokens=treatment_tokens,
            expected_answer=expected_answer,
            semantic_frame_output=semantic_output,
            raw_data=dataset.data.tolist(),
        )

    def _run_conditions_parallel(
        self,
        raw_data_json: str,
        semantic_output: str,
        query: str,
    ) -> tuple[ClaudeResponse, ClaudeResponse]:
        """Run baseline and treatment conditions in parallel."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            baseline_future = executor.submit(self.client.query_baseline, raw_data_json, query)
            treatment_future = executor.submit(self.client.query_treatment, semantic_output, query)

            baseline_response = baseline_future.result()
            treatment_response = treatment_future.result()

        return baseline_response, treatment_response

    def convert_to_trial_result(
        self,
        task_result: TaskResult,
        condition: Condition,
    ) -> TrialResult:
        """Convert TaskResult to TrialResult for aggregation."""
        if condition == "baseline":
            response = task_result.baseline_response
            answer = task_result.baseline_answer
            correct = task_result.baseline_correct
            tokens = task_result.baseline_tokens
        else:
            response = task_result.treatment_response
            answer = task_result.treatment_answer
            correct = task_result.treatment_correct
            tokens = task_result.treatment_tokens

        # Detect hallucinations using raw data
        hallucination = False
        if task_result.raw_data:
            hallucination = detect_hallucination(
                response.content,
                task_result.raw_data,
                task_result.semantic_frame_output,
            )

        return TrialResult(
            task_type=self.task_type.value,
            condition=condition,
            query=task_result.query_name,
            token_metrics=tokens,
            predicted_answer=answer,
            actual_answer=task_result.expected_answer,
            is_correct=correct,
            numerical_proximity=1.0 if correct else 0.0,  # Simplified
            hallucination_detected=hallucination,
            cost_metrics=CostMetrics.compute(
                response.input_tokens,
                response.output_tokens,
            ),
            latency_ms=response.latency_ms,
            raw_response=response.content if self.config.save_raw_responses else None,
            error=response.error,
        )

    def _run_trial_wrapper(
        self,
        dataset: SyntheticDataset,
        query_name: str,
        query_template: str,
        expected: Any,
        trial_num: int,
    ) -> tuple[TaskResult, int]:
        """Wrapper for parallel trial execution. Returns (result, trial_num)."""
        task_result = self.run_single_trial(
            dataset=dataset,
            query_name=query_name,
            query=query_template,
            expected_answer=expected,
        )
        return task_result, trial_num

    def _is_actual_api_error(self, error: str | None) -> bool:
        """Check if error is an actual API failure vs expected skip."""
        if error is None:
            return False
        # Expected skips are not API errors
        expected_skips = {"baseline_skipped_large_dataset"}
        return error not in expected_skips

    def run(self, n_trials: int | None = None) -> list[TrialResult]:
        """
        Run the full task benchmark.

        Returns list of TrialResults for both baseline and treatment conditions.

        If trial_parallelism > 1, runs multiple trials concurrently (max 4).

        Raises:
            RuntimeError: If max_consecutive_errors threshold is reached.
        """
        if n_trials is None:
            n_trials = self.config.n_trials

        datasets = self.generate_datasets()
        queries = self.get_queries()

        # Collect all trial configurations
        trial_configs: list[tuple[SyntheticDataset, str, str, Any, int]] = []
        for dataset in datasets:
            for query_name, query_template in queries.items():
                expected = dataset.ground_truth.get(query_name)
                if expected is None:
                    continue
                for trial_num in range(n_trials):
                    trial_configs.append((dataset, query_name, query_template, expected, trial_num))

        results: list[TrialResult] = []
        total_errors = 0

        # Use parallel execution if configured
        parallelism = self.config.trial_parallelism

        if parallelism > 1:
            # Parallel trial execution
            if self.config.verbose:
                print(f"  Running {len(trial_configs)} trials with {parallelism} workers...")

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = {}
                for config in trial_configs:
                    dataset, query_name, query_template, expected, trial_num = config
                    future = executor.submit(
                        self._run_trial_wrapper,
                        dataset,
                        query_name,
                        query_template,
                        expected,
                        trial_num,
                    )
                    futures[future] = (dataset.name, query_name, trial_num)

                for future in as_completed(futures):
                    dataset_name, query_name, trial_num = futures[future]
                    try:
                        task_result, _ = future.result()

                        # Check for actual API errors (not expected skips)
                        has_api_error = self._is_actual_api_error(
                            task_result.baseline_response.error
                        ) or self._is_actual_api_error(task_result.treatment_response.error)
                        if has_api_error:
                            total_errors += 1

                        # Convert to TrialResults
                        results.append(self.convert_to_trial_result(task_result, "baseline"))
                        results.append(self.convert_to_trial_result(task_result, "treatment"))

                        if self.config.verbose:
                            status = "✓" if not has_api_error else "✗"
                            print(f"  {status} {dataset_name} - {query_name}")

                    except Exception as e:
                        total_errors += 1
                        print(f"  ERROR: Trial failed ({dataset_name} - {query_name}): {e}")

        else:
            # Sequential execution (original behavior)
            consecutive_errors = 0
            max_consecutive = self.config.max_consecutive_errors

            for dataset, query_name, query_template, expected, trial_num in trial_configs:
                if self.config.verbose:
                    print(f"  Trial {trial_num + 1}/{n_trials}: {dataset.name} - {query_name}")

                task_result = self.run_single_trial(
                    dataset=dataset,
                    query_name=query_name,
                    query=query_template,
                    expected_answer=expected,
                )

                # Track consecutive API errors (not expected skips)
                has_api_error = self._is_actual_api_error(
                    task_result.baseline_response.error
                ) or self._is_actual_api_error(task_result.treatment_response.error)
                if has_api_error:
                    consecutive_errors += 1
                    total_errors += 1
                    if consecutive_errors >= max_consecutive:
                        last_error = (
                            task_result.baseline_response.error
                            or task_result.treatment_response.error
                        )
                        error_msg = (
                            f"Aborting: {consecutive_errors} consecutive API errors. "
                            f"Total errors: {total_errors}. "
                            f"Last error: {last_error}"
                        )
                        print(f"ERROR: {error_msg}", flush=True)
                        raise RuntimeError(error_msg)
                else:
                    consecutive_errors = 0  # Reset on success

                # Convert to TrialResults for both conditions
                results.append(self.convert_to_trial_result(task_result, "baseline"))
                results.append(self.convert_to_trial_result(task_result, "treatment"))

        # Log error summary if any occurred
        if total_errors > 0:
            print(
                f"\nWARNING: {total_errors} API error(s) occurred during benchmark. "
                f"These trials are marked as incorrect in results."
            )

        return results

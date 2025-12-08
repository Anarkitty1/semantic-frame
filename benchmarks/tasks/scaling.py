"""
Scaling Task (T6)

Large-scale data handling benchmark.
"""

import re
from typing import Any

import numpy as np

from benchmarks.config import STATISTICAL_QUERIES, DataPattern, TaskType
from benchmarks.datasets import SyntheticDataset
from benchmarks.metrics import TrialResult
from benchmarks.tasks.base import BaseTask


class ScalingTask(BaseTask):
    """
    Task T6: Large-Scale Data Handling

    Tests performance at varying data scales to demonstrate how
    Semantic Frame maintains stable accuracy and token footprint
    even as raw data size increases.
    """

    task_type = TaskType.SCALING

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate datasets at various scales."""
        # Define scale levels
        scales = [
            50,  # Tiny
            100,  # Small
            500,  # Medium-small
            1000,  # Medium
            5000,  # Large
            10000,  # Very large
        ]

        # In quick mode, reduce scales
        if self.config.n_trials < 10:
            scales = [50, 100, 500, 1000]

        datasets = []
        for size in scales:
            rng = np.random.default_rng(self.config.datasets.default_seed)
            data = rng.normal(50, 15, size)

            datasets.append(
                SyntheticDataset(
                    name=f"scale_{size}",
                    data=data,
                    ground_truth={
                        "mean": np.mean(data),
                        "median": np.median(data),
                        "std": np.std(data),
                        "min": np.min(data),
                        "max": np.max(data),
                        "count": size,
                        "scale": size,
                    },
                    pattern=DataPattern.RANDOM,
                    seed=self.config.datasets.default_seed,
                )
            )

        return datasets

    def get_queries(self) -> dict[str, str]:
        """Get scaling query templates - subset of statistical queries."""
        # Use only key statistical queries for scaling tests
        return {
            "mean": STATISTICAL_QUERIES["mean"],
            "std": STATISTICAL_QUERIES["std"],
            "count": STATISTICAL_QUERIES["count"],
        }

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate answer accuracy.
        """
        if predicted is None:
            return False, 0.0

        try:
            pred_num = float(predicted)
        except (ValueError, TypeError):
            numbers = re.findall(r"[-+]?\d*\.?\d+", str(predicted))
            if not numbers:
                return False, 0.0
            pred_num = float(numbers[0])

        exp_num = float(expected)

        # Calculate relative error
        if exp_num != 0:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
        else:
            relative_error = abs(pred_num)

        # Tolerance based on value type
        tolerance = 0.01  # 1% for most values

        is_correct = relative_error <= tolerance
        proximity = max(0, 1 - relative_error)

        return is_correct, proximity

    def analyze_scaling_behavior(
        self,
        results: list,
    ) -> dict:
        """
        Analyze how metrics scale with data size.

        Returns analysis of:
        - Token compression at each scale
        - Accuracy at each scale
        - Latency at each scale
        """
        from collections import defaultdict

        by_scale: dict[int, dict[str, list[TrialResult]]] = defaultdict(
            lambda: {"baseline": [], "treatment": []}
        )

        for result in results:
            # Extract scale from dataset name
            scale = int(result.query.split("_")[1]) if "_" in result.query else 100
            by_scale[scale][result.condition].append(result)

        analysis = {}
        for scale, conditions in sorted(by_scale.items()):
            baseline_acc = (
                sum(1 for r in conditions["baseline"] if r.is_correct) / len(conditions["baseline"])
                if conditions["baseline"]
                else 0
            )
            treatment_acc = (
                sum(1 for r in conditions["treatment"] if r.is_correct)
                / len(conditions["treatment"])
                if conditions["treatment"]
                else 0
            )

            baseline_tokens = sum(r.token_metrics.raw_tokens for r in conditions["baseline"])
            treatment_tokens = sum(
                r.token_metrics.compressed_tokens for r in conditions["treatment"]
            )

            analysis[scale] = {
                "baseline_accuracy": baseline_acc,
                "treatment_accuracy": treatment_acc,
                "accuracy_improvement": treatment_acc - baseline_acc,
                "baseline_tokens": baseline_tokens,
                "treatment_tokens": treatment_tokens,
                "token_reduction": 1 - (treatment_tokens / baseline_tokens)
                if baseline_tokens > 0
                else 0,
            }

        return analysis

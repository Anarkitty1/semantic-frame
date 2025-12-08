"""
Comparative Task (T4)

Comparative analysis benchmark for multi-series comparison.
"""

from typing import Any

import numpy as np

from benchmarks.config import COMPARATIVE_QUERIES, DataPattern, TaskType
from benchmarks.datasets import SyntheticDataset
from benchmarks.tasks.base import BaseTask


class ComparativeTask(BaseTask):
    """
    Task T4: Comparative Analysis

    Tests the ability to compare multiple datasets or variables.
    Semantic Frame should excel here by providing pre-computed
    comparative statistics.
    """

    task_type = TaskType.COMPARATIVE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._series_pairs: list[tuple[SyntheticDataset, SyntheticDataset]] = []

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate dataset pairs for comparison testing."""
        n = min(100, self.config.datasets.small_size)
        datasets = []

        # Pair 1: Clear mean difference
        rng = np.random.default_rng(42)
        series_a = rng.normal(50, 10, n)
        series_b = rng.normal(70, 10, n)  # Higher mean

        pair1_a = SyntheticDataset(
            name="pair1_series_A",
            data=series_a,
            ground_truth={
                "mean": np.mean(series_a),
                "std": np.std(series_a),
                "higher_mean": "Series B",  # B has higher mean
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        pair1_b = SyntheticDataset(
            name="pair1_series_B",
            data=series_b,
            ground_truth={
                "mean": np.mean(series_b),
                "std": np.std(series_b),
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        self._series_pairs.append((pair1_a, pair1_b))

        # Pair 2: Clear volatility difference
        series_a = rng.normal(50, 5, n)  # Low volatility
        series_b = rng.normal(50, 20, n)  # High volatility

        pair2_a = SyntheticDataset(
            name="pair2_series_A",
            data=series_a,
            ground_truth={
                "mean": np.mean(series_a),
                "std": np.std(series_a),
                "more_volatile": "Series B",  # B is more volatile
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        pair2_b = SyntheticDataset(
            name="pair2_series_B",
            data=series_b,
            ground_truth={
                "mean": np.mean(series_b),
                "std": np.std(series_b),
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        self._series_pairs.append((pair2_a, pair2_b))

        # Pair 3: Correlated series
        base = rng.normal(50, 10, n)
        series_a = base + rng.normal(0, 2, n)
        series_b = base + rng.normal(0, 2, n)
        corr = np.corrcoef(series_a, series_b)[0, 1]

        pair3_a = SyntheticDataset(
            name="pair3_series_A",
            data=series_a,
            ground_truth={
                "mean": np.mean(series_a),
                "std": np.std(series_a),
                "correlation": "positively correlated" if corr > 0.5 else "uncorrelated",
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        pair3_b = SyntheticDataset(
            name="pair3_series_B",
            data=series_b,
            ground_truth={
                "mean": np.mean(series_b),
                "std": np.std(series_b),
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        self._series_pairs.append((pair3_a, pair3_b))

        # Pair 4: Different trend strengths
        x = np.arange(n, dtype=np.float64)
        series_a = 0.5 * x + rng.normal(0, 5, n)  # Weak trend
        series_b = 2.0 * x + rng.normal(0, 2, n)  # Strong trend

        pair4_a = SyntheticDataset(
            name="pair4_series_A",
            data=series_a,
            ground_truth={
                "mean": np.mean(series_a),
                "std": np.std(series_a),
                "stronger_trend": "Series B",  # B has stronger trend
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        pair4_b = SyntheticDataset(
            name="pair4_series_B",
            data=series_b,
            ground_truth={
                "mean": np.mean(series_b),
                "std": np.std(series_b),
            },
            pattern=DataPattern.MIXED,
            seed=42,
        )
        self._series_pairs.append((pair4_a, pair4_b))

        # Return first series of each pair (contains ground truth for comparison)
        datasets = [pair[0] for pair in self._series_pairs]
        return datasets

    def get_queries(self) -> dict[str, str]:
        """Get comparative query templates."""
        return COMPARATIVE_QUERIES

    def get_semantic_frame_output(self, dataset: SyntheticDataset) -> str:
        """Get Semantic Frame description of both series in the pair."""
        from semantic_frame import describe_series

        # Find the corresponding pair
        for pair in self._series_pairs:
            if pair[0].name == dataset.name:
                output_a = describe_series(pair[0].data, context="Series A")
                output_b = describe_series(pair[1].data, context="Series B")
                return f"SERIES A ANALYSIS:\n{output_a}\n\nSERIES B ANALYSIS:\n{output_b}"

        # Fallback
        return describe_series(dataset.data, context=dataset.name)

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate comparative answer accuracy.
        """
        if predicted is None:
            return False, 0.0

        pred_str = str(predicted).lower().strip()
        exp_str = str(expected).lower().strip()

        # Check for series selection (A vs B)
        if "series a" in exp_str or "series b" in exp_str:
            # Extract which series was selected
            if "series a" in exp_str:
                correct_series = "a"
            else:
                correct_series = "b"

            # Check prediction
            if "series a" in pred_str or "a" in pred_str.split():
                pred_series = "a"
            elif "series b" in pred_str or "b" in pred_str.split():
                pred_series = "b"
            else:
                # Ambiguous
                return False, 0.0

            is_correct = pred_series == correct_series
            return is_correct, 1.0 if is_correct else 0.0

        # Check for correlation type
        correlation_equivalences = {
            "positively correlated": [
                "positively correlated",
                "positive correlation",
                "positively",
                "direct",
            ],
            "negatively correlated": [
                "negatively correlated",
                "negative correlation",
                "negatively",
                "inverse",
            ],
            "uncorrelated": ["uncorrelated", "no correlation", "independent", "not correlated"],
        }

        for key, variants in correlation_equivalences.items():
            if exp_str in variants or key in exp_str:
                if any(v in pred_str for v in variants):
                    return True, 1.0

        # Generic string matching
        if exp_str in pred_str:
            return True, 0.8

        return False, 0.0

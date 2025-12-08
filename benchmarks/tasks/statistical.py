"""
Statistical Task (T1)

Single-value extraction benchmark for statistical queries.
"""

from typing import Any

import numpy as np

from benchmarks.config import STATISTICAL_QUERIES, TaskType
from benchmarks.datasets import DatasetGenerator, SyntheticDataset
from benchmarks.metrics import AccuracyMetrics
from benchmarks.tasks.base import BaseTask


class StatisticalTask(BaseTask):
    """
    Task T1: Single-Value Extraction

    Tests the ability to extract specific statistical measures from data.
    This is where Semantic Frame should show the most dramatic improvement,
    as LLMs struggle with precise arithmetic on raw numbers.
    """

    task_type = TaskType.STATISTICAL

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate datasets for statistical testing."""
        generator = DatasetGenerator(seed=self.config.datasets.default_seed)

        sizes = [
            self.config.datasets.small_size,
            self.config.datasets.medium_size,
        ]

        # Don't include large sizes for statistical tests (too many tokens)
        # unless in quick mode where large is reduced
        if self.config.datasets.large_size <= 1000:
            sizes.append(self.config.datasets.large_size)

        return generator.generate_statistical_suite(sizes=sizes)

    def get_queries(self) -> dict[str, str]:
        """Get statistical query templates."""
        return STATISTICAL_QUERIES

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate statistical answer accuracy.

        Uses numerical proximity scoring with tolerance based on value magnitude.
        """
        # Handle categorical answers (like skewness)
        if isinstance(expected, str):
            if predicted is None:
                return False, 0.0
            pred_str = str(predicted).lower().strip()
            exp_str = expected.lower().strip()

            # Check for semantic equivalence
            equivalences = {
                "positive": ["positive", "positively skewed", "right-skewed", "right skewed"],
                "negative": ["negative", "negatively skewed", "left-skewed", "left skewed"],
                "none": ["none", "no", "not skewed", "symmetric", "normal"],
            }

            for key, variants in equivalences.items():
                if exp_str in variants:
                    if any(v in pred_str for v in variants):
                        return True, 1.0

            # Direct match
            is_correct = exp_str in pred_str or pred_str in exp_str
            return is_correct, 1.0 if is_correct else 0.0

        # Handle numerical answers
        if predicted is None:
            return False, 0.0

        try:
            pred_num = float(predicted)
        except (ValueError, TypeError):
            # Try to extract number from string
            import re

            numbers = re.findall(r"[-+]?\d*\.?\d+", str(predicted))
            if not numbers:
                return False, 0.0
            pred_num = float(numbers[0])

        exp_num = float(expected)

        # Calculate numerical proximity
        proximity = AccuracyMetrics.numerical_proximity_score(pred_num, exp_num)

        # Use 1% tolerance for exact match
        is_exact = AccuracyMetrics.check_exact_match(pred_num, exp_num, tolerance=0.01)

        return is_exact, proximity

    def get_ground_truth(self, dataset: SyntheticDataset, query_name: str) -> Any:
        """Get ground truth for a specific query."""
        data = dataset.data

        computed = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "range": np.max(data) - np.min(data),
            "p25": np.percentile(data, 25),
            "p75": np.percentile(data, 75),
            "p95": np.percentile(data, 95),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
            "count": len(data),
        }

        if query_name in computed:
            return computed[query_name]

        # Use stored ground truth for non-computable values
        return dataset.ground_truth.get(query_name)

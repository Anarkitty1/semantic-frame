"""
Multi-Step Task (T5)

Multi-step reasoning benchmark for chained numerical analysis.
"""

import re
from typing import Any

import numpy as np

from benchmarks.config import DataPattern, TaskType
from benchmarks.datasets import SyntheticDataset
from benchmarks.tasks.base import BaseTask

MULTI_STEP_QUERIES = {
    "forecast": (
        "If the current trend continues linearly, what will the value be after 10 more time steps?"
    ),
    "zscore": "What is the z-score of the maximum value in this dataset?",
    "cv": "What is the coefficient of variation (std/mean) of this dataset?",
    "range_pct": "What percentage of the range (max-min) is the interquartile range (IQR)?",
    "normalized_max": "What is the max value normalized to [0,1] range (using min-max scaling)?",
}


class MultiStepTask(BaseTask):
    """
    Task T5: Multi-Step Reasoning

    Tests the ability to perform chained numerical calculations.
    This is where error compounding in LLMs becomes critical -
    a mistake in step 1 propagates through all subsequent steps.

    Semantic Frame provides reliable intermediate values,
    preventing error propagation.
    """

    task_type = TaskType.MULTI_STEP

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate datasets for multi-step testing."""
        # Use smaller sizes for multi-step (complexity is in computation, not data size)
        size = min(100, self.config.datasets.small_size)

        datasets = []
        rng = np.random.default_rng(self.config.datasets.default_seed)

        # Dataset with clear linear trend (for forecasting)
        x = np.arange(size, dtype=np.float64)
        slope = 0.5
        data = slope * x + 10 + rng.normal(0, 2, size)

        # Compute multi-step ground truths
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        iqr = np.percentile(data, 75) - np.percentile(data, 25)

        # Forecast: extrapolate 10 steps
        last_val = data[-1]
        forecast_val = last_val + slope * 10

        # Z-score of max
        zscore_max = (max_val - mean) / std if std > 0 else 0

        # Coefficient of variation
        cv = std / mean if mean != 0 else 0

        # Range percentage
        range_pct = (iqr / range_val * 100) if range_val > 0 else 0

        # Normalized max
        normalized_max = (max_val - min_val) / range_val if range_val > 0 else 1.0

        datasets.append(
            SyntheticDataset(
                name="linear_multistep",
                data=data,
                ground_truth={
                    "mean": mean,
                    "std": std,
                    "min": min_val,
                    "max": max_val,
                    "range": range_val,
                    "iqr": iqr,
                    "slope": slope,
                    "forecast": forecast_val,
                    "zscore": zscore_max,
                    "cv": cv,
                    "range_pct": range_pct,
                    "normalized_max": normalized_max,
                },
                pattern=DataPattern.LINEAR_TREND,
                seed=self.config.datasets.default_seed,
            )
        )

        # Dataset with higher variance (more challenging)
        data2 = rng.normal(50, 15, size)
        mean2 = np.mean(data2)
        std2 = np.std(data2)
        min_val2 = np.min(data2)
        max_val2 = np.max(data2)
        range_val2 = max_val2 - min_val2
        iqr2 = np.percentile(data2, 75) - np.percentile(data2, 25)

        datasets.append(
            SyntheticDataset(
                name="variable_multistep",
                data=data2,
                ground_truth={
                    "mean": mean2,
                    "std": std2,
                    "min": min_val2,
                    "max": max_val2,
                    "range": range_val2,
                    "iqr": iqr2,
                    "forecast": data2[-1],  # No clear trend, forecast = last value
                    "zscore": (max_val2 - mean2) / std2 if std2 > 0 else 0,
                    "cv": std2 / mean2 if mean2 != 0 else 0,
                    "range_pct": (iqr2 / range_val2 * 100) if range_val2 > 0 else 0,
                    "normalized_max": 1.0,  # max normalized is always 1
                },
                pattern=DataPattern.RANDOM,
                seed=self.config.datasets.default_seed,
            )
        )

        return datasets

    def get_queries(self) -> dict[str, str]:
        """Get multi-step query templates."""
        return MULTI_STEP_QUERIES

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate multi-step answer accuracy.

        Uses wider tolerance for multi-step calculations due to
        accumulated approximation errors.
        """
        if predicted is None:
            return False, 0.0

        # Extract number from prediction
        try:
            pred_num = float(predicted)
        except (ValueError, TypeError):
            numbers = re.findall(r"[-+]?\d*\.?\d+", str(predicted))
            if not numbers:
                return False, 0.0
            pred_num = float(numbers[0])

        try:
            exp_num = float(expected)
        except (ValueError, TypeError):
            return False, 0.0

        # Calculate relative error
        if exp_num != 0:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
        else:
            relative_error = abs(pred_num)

        # Use 5% tolerance for multi-step calculations
        # (more lenient due to accumulated rounding)
        tolerance = 0.05

        is_correct = relative_error <= tolerance
        proximity = max(0, 1 - relative_error)

        return is_correct, proximity

    def get_ground_truth(self, dataset: SyntheticDataset, query_name: str) -> Any:
        """Get ground truth for a specific query."""
        return dataset.ground_truth.get(query_name)

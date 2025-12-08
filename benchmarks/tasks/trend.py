"""
Trend Task (T2)

Trend classification benchmark for time series analysis.
"""

from typing import Any

from benchmarks.config import TREND_QUERIES, TaskType
from benchmarks.datasets import DatasetGenerator, SyntheticDataset
from benchmarks.tasks.base import BaseTask

# Type alias for ground truth values (float, str, int, bool, or list)
GroundTruthValue = float | str | int | bool | list[int] | None


class TrendTask(BaseTask):
    """
    Task T2: Trend Classification

    Tests the ability to identify directional trends in time series.
    Semantic Frame should improve accuracy by providing explicit trend
    quantification rather than relying on LLM pattern recognition.
    """

    task_type = TaskType.TREND

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate datasets for trend testing."""
        generator = DatasetGenerator(seed=self.config.datasets.default_seed)

        # Use medium size for trend detection
        size = self.config.datasets.medium_size
        if size > 500:
            size = 500  # Cap at 500 for token efficiency

        return generator.generate_trend_suite(size=size)

    def get_queries(self) -> dict[str, str]:
        """Get trend query templates."""
        return TREND_QUERIES

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate trend classification accuracy.
        """
        if predicted is None:
            return False, 0.0

        pred_str = str(predicted).lower().strip()
        exp_str = str(expected).lower().strip()

        # Handle direction classification
        direction_equivalences = {
            "rising": ["rising", "increasing", "upward", "up", "positive", "growing", "ascending"],
            "falling": [
                "falling",
                "decreasing",
                "downward",
                "down",
                "negative",
                "declining",
                "descending",
            ],
            "flat": ["flat", "stable", "constant", "no trend", "stationary", "horizontal", "none"],
            "cyclical": ["cyclical", "seasonal", "periodic", "oscillating", "wave", "sinusoidal"],
        }

        # Handle strength classification
        strength_equivalences = {
            "strong": ["strong", "significant", "clear", "pronounced", "definite", "marked"],
            "moderate": ["moderate", "medium", "some", "noticeable", "visible"],
            "weak": ["weak", "slight", "minor", "subtle", "marginal", "faint"],
            "none": ["none", "no", "absent", "missing", "negligible"],
        }

        # Try direction matching first
        for key, variants in direction_equivalences.items():
            if exp_str in variants or exp_str == key:
                if any(v in pred_str for v in variants):
                    return True, 1.0

        # Try strength matching
        for key, variants in strength_equivalences.items():
            if exp_str in variants or exp_str == key:
                if any(v in pred_str for v in variants):
                    return True, 1.0

        # Handle numerical slope comparison
        if exp_str.replace(".", "").replace("-", "").isdigit():
            try:
                exp_num = float(exp_str)
                # Try to extract number from prediction
                import re

                numbers = re.findall(r"[-+]?\d*\.?\d+", pred_str)
                if numbers:
                    pred_num = float(numbers[0])
                    # Allow 20% tolerance for slope
                    if exp_num != 0:
                        relative_error = abs(pred_num - exp_num) / abs(exp_num)
                        if relative_error <= 0.2:
                            return True, 1.0 - relative_error
                    elif abs(pred_num) < 0.1:
                        return True, 1.0
            except ValueError:
                pass

        # Partial match scoring
        if exp_str in pred_str or pred_str in exp_str:
            return True, 0.8

        return False, 0.0

    def get_ground_truth(self, dataset: SyntheticDataset, query_name: str) -> GroundTruthValue:
        """Get ground truth for a specific query."""
        value: GroundTruthValue = dataset.ground_truth.get(query_name)
        return value

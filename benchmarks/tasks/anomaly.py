"""
Anomaly Task (T3)

Anomaly detection benchmark for time series analysis.
"""

import re
from typing import Any

from benchmarks.config import ANOMALY_QUERIES, TaskType
from benchmarks.datasets import DatasetGenerator, SyntheticDataset
from benchmarks.metrics import AnomalyDetectionMetrics
from benchmarks.tasks.base import BaseTask


class AnomalyTask(BaseTask):
    """
    Task T3: Anomaly Detection

    Tests the ability to detect anomalous points in time series.
    Semantic Frame pre-computes statistical thresholds and flags anomalies,
    which should significantly improve detection accuracy.
    """

    task_type = TaskType.ANOMALY

    def generate_datasets(self) -> list[SyntheticDataset]:
        """Generate datasets for anomaly testing."""
        generator = DatasetGenerator(seed=self.config.datasets.default_seed)

        size = min(200, self.config.datasets.medium_size)
        anomaly_rate = self.config.datasets.anomaly_rates[0]  # Use lowest rate

        # Return type is list[AnomalyDataset] but conforms to list[SyntheticDataset]
        return list(generator.generate_anomaly_suite(size=size, anomaly_rate=anomaly_rate))

    def get_queries(self) -> dict[str, str]:
        """Get anomaly query templates."""
        return ANOMALY_QUERIES

    def evaluate_answer(
        self,
        predicted: Any,
        expected: Any,
        dataset: SyntheticDataset,
    ) -> tuple[bool, float]:
        """
        Evaluate anomaly detection accuracy.
        """
        if predicted is None:
            return False, 0.0

        pred_str = str(predicted).lower().strip()

        # Handle presence detection (yes/no)
        if isinstance(expected, bool) or str(expected).lower() in ["yes", "no", "true", "false"]:
            exp_bool = str(expected).lower() in ["yes", "true"] or expected is True

            # Check predicted
            positive_indicators = ["yes", "true", "are", "detected", "found", "present", "exist"]
            negative_indicators = ["no", "false", "none", "no anomalies", "not detected", "absent"]

            pred_positive = any(ind in pred_str for ind in positive_indicators)
            pred_negative = any(ind in pred_str for ind in negative_indicators)

            # If both or neither, look at the overall sentiment
            if pred_positive and not pred_negative:
                pred_bool = True
            elif pred_negative and not pred_positive:
                pred_bool = False
            else:
                # Ambiguous - check for numbers (likely anomalies detected)
                pred_bool = bool(re.findall(r"\d+", pred_str))

            is_correct = pred_bool == exp_bool
            return is_correct, 1.0 if is_correct else 0.0

        # Handle count
        if isinstance(expected, int):
            numbers = re.findall(r"\d+", pred_str)
            if numbers:
                pred_count = int(numbers[0])
                is_exact = pred_count == expected
                # Allow some tolerance for count
                proximity = max(0, 1 - abs(pred_count - expected) / max(expected, 1))
                return is_exact, proximity
            return False, 0.0

        # Handle index list
        if isinstance(expected, list):
            # Extract numbers from prediction
            pred_indices = set(int(n) for n in re.findall(r"\d+", pred_str))
            exp_indices = set(expected)

            if not exp_indices:
                # No anomalies expected
                is_correct = len(pred_indices) == 0
                return is_correct, 1.0 if is_correct else 0.0

            if not pred_indices:
                return False, 0.0

            # Calculate set overlap metrics
            tp = len(pred_indices & exp_indices)
            fp = len(pred_indices - exp_indices)
            fn = len(exp_indices - pred_indices)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Consider correct if F1 > 0.5
            return f1 > 0.5, f1

        # Handle anomaly type classification
        if isinstance(expected, str):
            exp_lower = expected.lower()

            type_equivalences = {
                "spike": ["spike", "peak", "sudden increase", "sharp rise", "point anomaly"],
                "drop": ["drop", "dip", "sudden decrease", "sharp fall", "point anomaly"],
                "level_shift": ["level shift", "step change", "baseline shift", "mean shift"],
                "trend_change": ["trend change", "slope change", "direction change"],
            }

            for key, variants in type_equivalences.items():
                if exp_lower in variants or key in exp_lower:
                    if any(v in pred_str for v in variants):
                        return True, 1.0

            # Partial match
            if exp_lower in pred_str:
                return True, 0.8

        return False, 0.0

    def compute_detailed_metrics(
        self,
        predicted_indices: set[int],
        actual_indices: set[int],
        series_length: int,
    ) -> AnomalyDetectionMetrics:
        """Compute detailed anomaly detection metrics."""
        return AnomalyDetectionMetrics.compute(
            predicted_indices=predicted_indices,
            actual_indices=actual_indices,
            series_length=series_length,
            delay_tolerance=3,
        )

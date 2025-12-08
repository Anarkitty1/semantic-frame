"""
Benchmark Metrics

All evaluation metrics for measuring token reduction and accuracy gains.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Token counting - use tiktoken if available, fallback to approximation
try:
    import tiktoken

    _encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (Claude-compatible)."""
        return len(_encoding.encode(text))
except ImportError:

    def count_tokens(text: str) -> int:
        """Approximate token count (fallback when tiktoken not available)."""
        # Rough approximation: ~4 characters per token for English
        return len(text) // 4


@dataclass
class TokenMetrics:
    """Token efficiency metrics."""

    raw_tokens: int
    compressed_tokens: int
    compression_ratio: float

    @classmethod
    def compute(cls, raw_data: str, compressed_output: str) -> "TokenMetrics":
        """Compute token metrics from raw data and compressed output."""
        raw_tokens = count_tokens(raw_data)
        compressed_tokens = count_tokens(compressed_output)
        compression_ratio = 1 - (compressed_tokens / raw_tokens) if raw_tokens > 0 else 0
        return cls(
            raw_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
        )


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for numerical analysis tasks."""

    exact_match: bool
    numerical_proximity: float  # 0-1, 1 = perfect
    semantic_alignment: float  # 0-1, 1 = perfect
    hallucination_detected: bool

    @staticmethod
    def numerical_proximity_score(
        predicted: float, actual: float, tolerance: float = 0.01
    ) -> float:
        """
        Calculate numerical proximity score.

        Returns 1.0 for exact match (within tolerance), decreasing for larger errors.
        """
        if actual == 0:
            return 1.0 if abs(predicted) < tolerance else 0.0

        relative_error = abs(predicted - actual) / abs(actual)

        # Perfect if within tolerance
        if relative_error <= tolerance:
            return 1.0

        # Smooth decay for larger errors
        return max(0.0, 1.0 - relative_error)

    @staticmethod
    def check_exact_match(predicted: Any, actual: Any, tolerance: float = 1e-6) -> bool:
        """Check if predicted value exactly matches actual (within tolerance)."""
        if isinstance(actual, int | float) and isinstance(predicted, int | float):
            return (
                abs(predicted - actual) <= abs(actual) * tolerance
                if actual != 0
                else abs(predicted) <= tolerance
            )
        return str(predicted).lower().strip() == str(actual).lower().strip()


@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks (trend, anomaly type, etc.)."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0


@dataclass
class AnomalyDetectionMetrics:
    """Specialized metrics for anomaly detection tasks."""

    point_wise_precision: float = 0.0
    point_wise_recall: float = 0.0
    point_wise_f1: float = 0.0

    # Affinity metrics (segment-aware)
    affinity_precision: float = 0.0
    affinity_recall: float = 0.0
    affinity_f1: float = 0.0

    # Delayed F1 (practical detection with lag tolerance)
    delayed_f1: float = 0.0

    @classmethod
    def compute(
        cls,
        predicted_indices: set[int],
        actual_indices: set[int],
        series_length: int,
        delay_tolerance: int = 3,
    ) -> "AnomalyDetectionMetrics":
        """Compute all anomaly detection metrics."""
        # Point-wise metrics
        tp = len(predicted_indices & actual_indices)
        fp = len(predicted_indices - actual_indices)
        fn = len(actual_indices - predicted_indices)

        pw_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pw_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pw_f1 = (
            2 * pw_precision * pw_recall / (pw_precision + pw_recall)
            if (pw_precision + pw_recall) > 0
            else 0.0
        )

        # Affinity metrics (allow for nearby detection)
        def expand_indices(indices: set[int], tolerance: int, max_len: int) -> set[int]:
            expanded = set()
            for idx in indices:
                for offset in range(-tolerance, tolerance + 1):
                    new_idx = idx + offset
                    if 0 <= new_idx < max_len:
                        expanded.add(new_idx)
            return expanded

        expanded_actual = expand_indices(actual_indices, delay_tolerance, series_length)
        expanded_predicted = expand_indices(predicted_indices, delay_tolerance, series_length)

        aff_tp_pred = len(predicted_indices & expanded_actual)
        aff_tp_actual = len(actual_indices & expanded_predicted)

        aff_precision = aff_tp_pred / len(predicted_indices) if len(predicted_indices) > 0 else 0.0
        aff_recall = aff_tp_actual / len(actual_indices) if len(actual_indices) > 0 else 0.0
        aff_f1 = (
            2 * aff_precision * aff_recall / (aff_precision + aff_recall)
            if (aff_precision + aff_recall) > 0
            else 0.0
        )

        return cls(
            point_wise_precision=pw_precision,
            point_wise_recall=pw_recall,
            point_wise_f1=pw_f1,
            affinity_precision=aff_precision,
            affinity_recall=aff_recall,
            affinity_f1=aff_f1,
            delayed_f1=aff_f1,  # Simplified: use affinity F1 as delayed F1
        )


@dataclass
class CostMetrics:
    """API cost metrics."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    # Anthropic pricing (as of late 2025, adjust as needed)
    INPUT_COST_PER_1K: float = 0.003  # Sonnet input
    OUTPUT_COST_PER_1K: float = 0.015  # Sonnet output

    @classmethod
    def compute(cls, input_tokens: int, output_tokens: int) -> "CostMetrics":
        """Compute cost metrics."""
        total = input_tokens + output_tokens
        cost = (input_tokens / 1000 * cls.INPUT_COST_PER_1K) + (
            output_tokens / 1000 * cls.OUTPUT_COST_PER_1K
        )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
        )


@dataclass
class TrialResult:
    """Results from a single benchmark trial."""

    task_type: str
    condition: str  # "baseline" or "treatment"
    query: str

    # Token metrics
    token_metrics: TokenMetrics

    # Accuracy
    predicted_answer: Any
    actual_answer: Any
    is_correct: bool
    numerical_proximity: float
    hallucination_detected: bool

    # Cost
    cost_metrics: CostMetrics

    # Timing
    latency_ms: float

    # Raw data for debugging
    raw_response: str | None = None
    error: str | None = None


@dataclass
class AggregatedResults:
    """Aggregated results across multiple trials."""

    task_type: str
    condition: str
    n_trials: int

    # Token metrics (aggregated)
    mean_compression_ratio: float
    std_compression_ratio: float
    total_raw_tokens: int
    total_compressed_tokens: int

    # Accuracy metrics (aggregated)
    accuracy: float  # Proportion correct
    mean_numerical_proximity: float
    std_numerical_proximity: float
    hallucination_rate: float

    # Classification metrics (if applicable)
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    # Cost metrics (aggregated)
    mean_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    # Timing
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0

    # Confidence interval
    accuracy_ci_lower: float = 0.0
    accuracy_ci_upper: float = 0.0

    @classmethod
    def from_trials(cls, trials: list[TrialResult]) -> "AggregatedResults":
        """Aggregate results from multiple trials."""
        if not trials:
            raise ValueError("Cannot aggregate empty trial list")

        task_type = trials[0].task_type
        condition = trials[0].condition
        n = len(trials)

        # Token metrics
        compression_ratios = [t.token_metrics.compression_ratio for t in trials]
        total_raw = sum(t.token_metrics.raw_tokens for t in trials)
        total_compressed = sum(t.token_metrics.compressed_tokens for t in trials)

        # Accuracy
        correct_count = sum(1 for t in trials if t.is_correct)
        accuracy = correct_count / n

        proximity_scores = [t.numerical_proximity for t in trials]
        hallucination_count = sum(1 for t in trials if t.hallucination_detected)

        # Cost
        costs = [t.cost_metrics.estimated_cost_usd for t in trials]

        # Latency
        latencies = [t.latency_ms for t in trials]

        # Confidence interval (Wilson score interval for proportions)
        z = 1.96  # 95% CI
        p = accuracy
        denominator = 1 + z**2 / n
        centre = p + z**2 / (2 * n)
        adjustment = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        ci_lower = max(0, (centre - adjustment) / denominator)
        ci_upper = min(1, (centre + adjustment) / denominator)

        return cls(
            task_type=task_type,
            condition=condition,
            n_trials=n,
            mean_compression_ratio=float(np.mean(compression_ratios)),
            std_compression_ratio=float(np.std(compression_ratios)),
            total_raw_tokens=total_raw,
            total_compressed_tokens=total_compressed,
            accuracy=accuracy,
            mean_numerical_proximity=float(np.mean(proximity_scores)),
            std_numerical_proximity=float(np.std(proximity_scores)),
            hallucination_rate=hallucination_count / n,
            mean_cost_usd=float(np.mean(costs)),
            total_cost_usd=sum(costs),
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            accuracy_ci_lower=ci_lower,
            accuracy_ci_upper=ci_upper,
        )


def parse_llm_response(response: str) -> dict[str, Any]:
    """
    Parse structured response from LLM.

    Expected format:
    - Answer: [value]
    - Confidence: [high/medium/low]
    - Reasoning: [text]
    """
    result: dict[str, Any] = {
        "answer": None,
        "confidence": None,
        "reasoning": None,
        "raw": response,
    }

    # Extract answer
    answer_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if answer_match:
        answer_str = answer_match.group(1).strip()
        # Try to parse as number
        try:
            # Handle various number formats
            cleaned = answer_str.replace(",", "").replace("$", "").replace("%", "")
            result["answer"] = float(cleaned)
        except ValueError:
            result["answer"] = answer_str

    # Extract confidence
    conf_match = re.search(r"Confidence:\s*(high|medium|low)", response, re.IGNORECASE)
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    # Extract reasoning
    reason_match = re.search(
        r"Reasoning:\s*(.+?)(?:\n-|\n\n|$)", response, re.IGNORECASE | re.DOTALL
    )
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    return result


def detect_hallucination(
    response: str,
    raw_data: list[float],
    semantic_frame_output: str,
    threshold: float = 0.1,
) -> bool:
    """
    Detect if the LLM hallucinated numerical values.

    A hallucination is a numerical claim that cannot be derived from the input data.
    """
    # Extract all numbers from response
    numbers_in_response = re.findall(r"[-+]?\d*\.?\d+", response)

    if not numbers_in_response:
        return False

    # Get valid numbers from input
    data_set = set(raw_data)

    # Also compute common derived values
    if raw_data:
        derived_values = {
            np.mean(raw_data),
            np.median(raw_data),
            np.std(raw_data),
            np.min(raw_data),
            np.max(raw_data),
            np.max(raw_data) - np.min(raw_data),  # range
            np.percentile(raw_data, 25),
            np.percentile(raw_data, 75),
            np.percentile(raw_data, 95),
            len(raw_data),
        }
        valid_numbers = data_set | derived_values
    else:
        valid_numbers = data_set

    # Check each number in response
    for num_str in numbers_in_response:
        try:
            num = float(num_str)
            # Check if this number is close to any valid number
            is_valid = any(
                abs(num - valid) <= abs(valid) * threshold if valid != 0 else abs(num) <= threshold
                for valid in valid_numbers
            )
            if not is_valid and abs(num) > 1:  # Ignore small numbers like indices
                return True
        except ValueError:
            continue

    return False

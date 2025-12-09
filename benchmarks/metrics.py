"""
Benchmark Metrics

All evaluation metrics for measuring token reduction and accuracy gains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    pass

# Type alias for condition values
Condition = Literal["baseline", "treatment"]

# Token counting - use tiktoken if available, fallback to approximation
# Note: cl100k_base is the encoding used by GPT-4 and similar models.
# While Anthropic has not published exact tokenizer details, cl100k_base
# provides a reasonable approximation for token counting purposes.
# Source: https://github.com/openai/tiktoken
try:
    import tiktoken

    _encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding.

        This uses OpenAI's tiktoken library with cl100k_base encoding,
        which provides a reasonable approximation for Claude models.
        While exact token counts may differ from Claude's internal tokenizer,
        this is sufficient for benchmark comparisons and cost estimation.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens in the text.

        References:
            - tiktoken: https://github.com/openai/tiktoken
            - cl100k_base: Encoding used by GPT-4, text-embedding-ada-002
        """
        return len(_encoding.encode(text))
except ImportError:

    def count_tokens(text: str) -> int:
        """Approximate token count (fallback when tiktoken not available).

        Uses a simple heuristic of ~4 characters per token, which is
        a reasonable approximation for English text with modern tokenizers.

        Args:
            text: The text to tokenize.

        Returns:
            Approximate number of tokens in the text.

        Note:
            For accurate token counting, install tiktoken:
            pip install tiktoken
        """
        # Rough approximation: ~4 characters per token for English
        return len(text) // 4


@dataclass(frozen=True)
class TokenMetrics:
    """Token efficiency metrics.

    This dataclass is frozen (immutable) for thread safety and hashability.
    """

    raw_tokens: int
    compressed_tokens: int
    compression_ratio: float

    @classmethod
    def compute(cls, raw_data: str, compressed_output: str) -> TokenMetrics:
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
    """Metrics for classification tasks (trend, anomaly type, etc.).

    Standard binary classification metrics based on the confusion matrix.

    References:
        - Precision, Recall, F1: Powers, D.M.W. (2011). "Evaluation: From Precision,
          Recall and F-measure to ROC, Informedness, Markedness and Correlation."
          Journal of Machine Learning Technologies. ISSN: 2229-3981
        - https://en.wikipedia.org/wiki/Precision_and_recall
    """

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value).

        Precision = TP / (TP + FP)
        """
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity, true positive rate).

        Recall = TP / (TP + FN)
        """
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall).

        F1 = 2 * (precision * recall) / (precision + recall)
        """
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy (proportion of correct predictions).

        Accuracy = (TP + TN) / (TP + FP + TN + FN)
        """
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0


@dataclass
class AnomalyDetectionMetrics:
    """Specialized metrics for anomaly detection tasks.

    Includes point-wise metrics (exact position matching) and affinity metrics
    (segment-aware with delay tolerance) for practical anomaly detection evaluation.

    References:
        - Affinity metrics: Tatbul et al. (2018). "Precision and Recall for
          Time Series." NeurIPS 2018. https://arxiv.org/abs/1803.03639
        - Delayed detection: Lavin & Ahmad (2015). "Evaluating Real-Time Anomaly
          Detection Algorithms." https://arxiv.org/abs/1510.03336
    """

    point_wise_precision: float = 0.0
    point_wise_recall: float = 0.0
    point_wise_f1: float = 0.0

    # Affinity metrics (segment-aware) - from Tatbul et al. 2018
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
    ) -> AnomalyDetectionMetrics:
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


@dataclass(frozen=True)
class CostMetrics:
    """API cost metrics.

    This dataclass is frozen (immutable) for thread safety and hashability.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    # Anthropic pricing (as of late 2025, adjust as needed)
    INPUT_COST_PER_1K: float = 0.003  # Sonnet input
    OUTPUT_COST_PER_1K: float = 0.015  # Sonnet output

    @classmethod
    def compute(cls, input_tokens: int, output_tokens: int) -> CostMetrics:
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


# =============================================================================
# Composite Metrics (Section 2.3 of Benchmark Methodology)
# =============================================================================


@dataclass(frozen=True)
class ContextUtilizationMetrics:
    """Context Utilization Efficiency (CUE) metrics.

    Measures how efficiently tokens encode answerable information, combining
    compression ratio with information preservation.

    Formula:
        CUE = (info_density_treatment / info_density_baseline) × compression_ratio

    Where:
        info_density = accuracy / tokens_used

    A CUE > 1.0 indicates treatment is more efficient than baseline.
    Target: CUE > 1.5 (treatment is 1.5x more efficient)

    This dataclass is frozen (immutable) for thread safety and hashability.

    References:
        - Semantic Frame Benchmark Methodology, Section 2.1, lines 48-52
    """

    baseline_info_density: float
    treatment_info_density: float
    compression_ratio: float
    cue_score: float

    @classmethod
    def compute(
        cls,
        baseline_accuracy: float,
        baseline_tokens: int,
        treatment_accuracy: float,
        treatment_tokens: int,
    ) -> ContextUtilizationMetrics:
        """Compute CUE metrics from baseline and treatment results.

        Args:
            baseline_accuracy: Accuracy rate for baseline condition (0-1)
            baseline_tokens: Total tokens used in baseline condition
            treatment_accuracy: Accuracy rate for treatment condition (0-1)
            treatment_tokens: Total tokens used in treatment condition

        Returns:
            ContextUtilizationMetrics with computed CUE score
        """
        # Calculate information density (accuracy per token)
        baseline_info_density = baseline_accuracy / baseline_tokens if baseline_tokens > 0 else 0.0
        treatment_info_density = (
            treatment_accuracy / treatment_tokens if treatment_tokens > 0 else 0.0
        )

        # Calculate compression ratio
        compression_ratio = 1 - (treatment_tokens / baseline_tokens) if baseline_tokens > 0 else 0.0

        # Calculate CUE score
        # CUE = (treatment_density / baseline_density) × compression_ratio
        density_ratio = (
            treatment_info_density / baseline_info_density if baseline_info_density > 0 else 0.0
        )
        cue_score = density_ratio * compression_ratio if compression_ratio > 0 else 0.0

        return cls(
            baseline_info_density=baseline_info_density,
            treatment_info_density=treatment_info_density,
            compression_ratio=compression_ratio,
            cue_score=cue_score,
        )


@dataclass(frozen=True)
class SemanticAlignmentScore:
    """Semantic Alignment Score (SAS) for multi-faceted evaluation.

    For qualitative descriptions (trend, volatility, pattern recognition),
    this metric provides a weighted average of different accuracy dimensions.

    Formula:
        SAS = weighted_average(
            trend_direction_correct × 0.4,
            magnitude_category_correct × 0.3,
            anomaly_detection_correct × 0.3
        )

    This dataclass is frozen (immutable) for thread safety and hashability.

    References:
        - Semantic Frame Benchmark Methodology, Section 2.2, lines 77-85
    """

    trend_accuracy: float
    magnitude_accuracy: float
    anomaly_accuracy: float
    sas_score: float

    # Configurable weights (class-level constants)
    TREND_WEIGHT: float = 0.4
    MAGNITUDE_WEIGHT: float = 0.3
    ANOMALY_WEIGHT: float = 0.3

    @classmethod
    def compute(
        cls,
        trend_correct: int,
        trend_total: int,
        magnitude_correct: int,
        magnitude_total: int,
        anomaly_correct: int,
        anomaly_total: int,
    ) -> SemanticAlignmentScore:
        """Compute SAS from task-specific accuracy counts.

        Args:
            trend_correct: Number of correct trend classifications
            trend_total: Total trend classification attempts
            magnitude_correct: Number of correct magnitude classifications
            magnitude_total: Total magnitude classification attempts
            anomaly_correct: Number of correct anomaly detections
            anomaly_total: Total anomaly detection attempts

        Returns:
            SemanticAlignmentScore with computed SAS score
        """
        # Calculate individual accuracies
        trend_accuracy = trend_correct / trend_total if trend_total > 0 else 0.0
        magnitude_accuracy = magnitude_correct / magnitude_total if magnitude_total > 0 else 0.0
        anomaly_accuracy = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0

        # Calculate weighted SAS score
        sas_score = (
            cls.TREND_WEIGHT * trend_accuracy
            + cls.MAGNITUDE_WEIGHT * magnitude_accuracy
            + cls.ANOMALY_WEIGHT * anomaly_accuracy
        )

        return cls(
            trend_accuracy=trend_accuracy,
            magnitude_accuracy=magnitude_accuracy,
            anomaly_accuracy=anomaly_accuracy,
            sas_score=sas_score,
        )

    @classmethod
    def from_trial_results(
        cls,
        trials: list[TrialResult],
        trend_queries: set[str] | None = None,
        magnitude_queries: set[str] | None = None,
        anomaly_queries: set[str] | None = None,
    ) -> SemanticAlignmentScore:
        """Compute SAS from a list of TrialResults.

        Args:
            trials: List of TrialResult objects
            trend_queries: Query names considered trend-related (default: direction, slope)
            magnitude_queries: Query names for magnitude (default: strength, std)
            anomaly_queries: Query names for anomaly (default: presence, count, locations)

        Returns:
            SemanticAlignmentScore computed from trial results
        """
        # Default query categorizations
        if trend_queries is None:
            trend_queries = {"direction", "slope", "trend"}
        if magnitude_queries is None:
            magnitude_queries = {"strength", "std", "volatility", "magnitude"}
        if anomaly_queries is None:
            anomaly_queries = {"presence", "count", "locations", "anomaly", "anomalies"}

        # Count correct/total for each category
        trend_correct = trend_total = 0
        magnitude_correct = magnitude_total = 0
        anomaly_correct = anomaly_total = 0

        for trial in trials:
            query_lower = trial.query.lower()

            if any(tq in query_lower for tq in trend_queries):
                trend_total += 1
                if trial.is_correct:
                    trend_correct += 1
            elif any(mq in query_lower for mq in magnitude_queries):
                magnitude_total += 1
                if trial.is_correct:
                    magnitude_correct += 1
            elif any(aq in query_lower for aq in anomaly_queries):
                anomaly_total += 1
                if trial.is_correct:
                    anomaly_correct += 1

        return cls.compute(
            trend_correct=trend_correct,
            trend_total=trend_total,
            magnitude_correct=magnitude_correct,
            magnitude_total=magnitude_total,
            anomaly_correct=anomaly_correct,
            anomaly_total=anomaly_total,
        )


@dataclass(frozen=True)
class EfficiencyAccuracyProduct:
    """Efficiency-Accuracy Product (EAP) metric.

    A single composite metric balancing token compression vs accuracy.
    Values > 0.90 indicate strong overall performance.

    Formula:
        EAP = TCR × EMA

    Where:
        TCR = Token Compression Ratio (1 - treatment_tokens/baseline_tokens)
        EMA = Exact Match Accuracy (correct_responses / total_responses)

    This dataclass is frozen (immutable) for thread safety and hashability.

    References:
        - Semantic Frame Benchmark Methodology, Section 2.3, lines 98-101
    """

    compression_ratio: float
    accuracy: float
    eap_score: float
    meets_threshold: bool

    # Performance threshold
    THRESHOLD: float = 0.90

    @classmethod
    def compute(
        cls,
        compression_ratio: float,
        accuracy: float,
    ) -> EfficiencyAccuracyProduct:
        """Compute EAP from compression ratio and accuracy.

        Args:
            compression_ratio: Token compression ratio (0-1, higher is better)
            accuracy: Exact match accuracy (0-1, higher is better)

        Returns:
            EfficiencyAccuracyProduct with computed EAP score
        """
        eap_score = compression_ratio * accuracy
        meets_threshold = eap_score >= cls.THRESHOLD

        return cls(
            compression_ratio=compression_ratio,
            accuracy=accuracy,
            eap_score=eap_score,
            meets_threshold=meets_threshold,
        )

    @classmethod
    def from_aggregated_results(
        cls,
        baseline: AggregatedResults,
        treatment: AggregatedResults,
    ) -> EfficiencyAccuracyProduct:
        """Compute EAP from aggregated baseline and treatment results.

        Args:
            baseline: Aggregated results from baseline condition
            treatment: Aggregated results from treatment condition

        Returns:
            EfficiencyAccuracyProduct computed from treatment metrics
        """
        return cls.compute(
            compression_ratio=treatment.mean_compression_ratio,
            accuracy=treatment.accuracy,
        )


@dataclass(frozen=True)
class ParetoEfficiencyIndex:
    """Pareto Efficiency Index (PEI) for accuracy vs token efficiency tradeoff.

    Measures the area between baseline and treatment Pareto frontiers,
    normalized to the baseline. A positive PEI indicates treatment dominates.

    Methodology:
        1. Plot accuracy vs. token count for baseline condition
        2. Plot accuracy vs. token count for treatment condition
        3. Calculate area between curves using trapezoidal rule
        4. Normalize to baseline curve area

    This dataclass is frozen (immutable) for thread safety and hashability.

    References:
        - Semantic Frame Benchmark Methodology, Section 2.3, lines 103-105
    """

    baseline_points: tuple[tuple[float, float], ...]  # (tokens, accuracy) tuples
    treatment_points: tuple[tuple[float, float], ...]
    baseline_area: float
    treatment_area: float
    pei_score: float

    @classmethod
    def compute(
        cls,
        baseline_results: list[AggregatedResults],
        treatment_results: list[AggregatedResults],
    ) -> ParetoEfficiencyIndex:
        """Compute PEI from lists of aggregated results.

        Args:
            baseline_results: List of AggregatedResults for baseline condition
            treatment_results: List of AggregatedResults for treatment condition

        Returns:
            ParetoEfficiencyIndex with computed PEI score
        """
        # Extract (tokens, accuracy) points
        baseline_points = [(float(r.total_raw_tokens), r.accuracy) for r in baseline_results]
        treatment_points = [
            (float(r.total_compressed_tokens), r.accuracy) for r in treatment_results
        ]

        # Sort by tokens (x-axis)
        baseline_points = sorted(baseline_points, key=lambda p: p[0])
        treatment_points = sorted(treatment_points, key=lambda p: p[0])

        # Calculate areas under curves using trapezoidal rule
        baseline_area = cls._calculate_area(baseline_points)
        treatment_area = cls._calculate_area(treatment_points)

        # PEI = normalized area difference
        # Positive means treatment is better (more area = better accuracy per token)
        if baseline_area > 0:
            pei_score = (treatment_area - baseline_area) / baseline_area
        else:
            pei_score = 0.0 if treatment_area == 0 else float("inf")

        return cls(
            baseline_points=tuple(baseline_points),
            treatment_points=tuple(treatment_points),
            baseline_area=baseline_area,
            treatment_area=treatment_area,
            pei_score=pei_score,
        )

    @staticmethod
    def _calculate_area(points: list[tuple[float, float]]) -> float:
        """Calculate area under curve using trapezoidal rule.

        Args:
            points: List of (x, y) tuples, sorted by x

        Returns:
            Area under the curve
        """
        if len(points) < 2:
            return 0.0

        area = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            # Trapezoidal rule: area = (x2-x1) * (y1+y2) / 2
            area += (x2 - x1) * (y1 + y2) / 2

        return area

    @classmethod
    def from_trial_results(
        cls,
        baseline_trials: list[TrialResult],
        treatment_trials: list[TrialResult],
    ) -> ParetoEfficiencyIndex:
        """Compute PEI directly from trial results.

        Groups trials by task type and computes Pareto efficiency.

        Args:
            baseline_trials: List of TrialResults from baseline condition
            treatment_trials: List of TrialResults from treatment condition

        Returns:
            ParetoEfficiencyIndex computed from trial data
        """
        # Extract individual trial points
        baseline_points = [
            (float(t.token_metrics.raw_tokens), 1.0 if t.is_correct else 0.0)
            for t in baseline_trials
        ]
        treatment_points = [
            (float(t.token_metrics.compressed_tokens), 1.0 if t.is_correct else 0.0)
            for t in treatment_trials
        ]

        # Sort and deduplicate by averaging accuracy at same token count
        baseline_points = cls._aggregate_points(baseline_points)
        treatment_points = cls._aggregate_points(treatment_points)

        baseline_area = cls._calculate_area(baseline_points)
        treatment_area = cls._calculate_area(treatment_points)

        if baseline_area > 0:
            pei_score = (treatment_area - baseline_area) / baseline_area
        else:
            pei_score = 0.0 if treatment_area == 0 else float("inf")

        return cls(
            baseline_points=tuple(baseline_points),
            treatment_points=tuple(treatment_points),
            baseline_area=baseline_area,
            treatment_area=treatment_area,
            pei_score=pei_score,
        )

    @staticmethod
    def _aggregate_points(
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Aggregate points with same x value by averaging y values.

        Args:
            points: List of (x, y) tuples

        Returns:
            Deduplicated list sorted by x, with averaged y values
        """
        from collections import defaultdict

        x_to_ys: defaultdict[float, list[float]] = defaultdict(list)
        for x, y in points:
            x_to_ys[x].append(y)

        aggregated = [(x, sum(ys) / len(ys)) for x, ys in x_to_ys.items()]
        return sorted(aggregated, key=lambda p: p[0])


@dataclass
class TrialResult:
    """Results from a single benchmark trial."""

    task_type: str  # TaskType.value string for serialization compatibility
    condition: Condition  # "baseline" or "treatment"
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

    task_type: str  # TaskType.value string for serialization compatibility
    condition: Condition
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
    def from_trials(cls, trials: list[TrialResult]) -> AggregatedResults:
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
    threshold: float = 0.15,
) -> bool:
    """
    Detect if the LLM hallucinated numerical values.

    A hallucination is a numerical claim that cannot be derived from the input data.
    Uses a conservative approach to avoid false positives from:
    - Step numbers in reasoning (1, 2, 3, etc.)
    - Intermediate calculations (sums, products)
    - Formatting artifacts (comma-separated thousands)

    Args:
        response: The LLM's response text.
        raw_data: The original numerical data.
        semantic_frame_output: The semantic frame description (unused but kept for API).
        threshold: Relative tolerance for matching (default 0.15 = 15%).

    Returns:
        True if a likely hallucination is detected, False otherwise.
    """
    if not raw_data:
        return False

    # Extract the Answer line specifically - this is what we care about for hallucination
    answer_match = re.search(r"Answer:\s*([-+]?\d[\d,]*\.?\d*)", response, re.IGNORECASE)
    if not answer_match:
        # No structured answer found - can't reliably detect hallucination
        return False

    # Parse the answer value (handle comma-separated thousands)
    answer_str = answer_match.group(1).replace(",", "")
    try:
        answer_num = float(answer_str)
    except ValueError:
        return False

    # Build set of valid numbers the answer could legitimately be
    data_array = np.array(raw_data)
    valid_numbers = set()

    # Raw data values
    valid_numbers.update(raw_data)

    # Common derived statistics
    valid_numbers.add(float(np.mean(data_array)))
    valid_numbers.add(float(np.median(data_array)))
    valid_numbers.add(float(np.std(data_array)))
    valid_numbers.add(float(np.var(data_array)))
    valid_numbers.add(float(np.min(data_array)))
    valid_numbers.add(float(np.max(data_array)))
    valid_numbers.add(float(np.max(data_array) - np.min(data_array)))  # range
    valid_numbers.add(float(np.sum(data_array)))  # sum (for intermediate calcs)
    valid_numbers.add(float(len(data_array)))  # count

    # Percentiles
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        valid_numbers.add(float(np.percentile(data_array, p)))

    # IQR
    valid_numbers.add(float(np.percentile(data_array, 75) - np.percentile(data_array, 25)))

    # Check if answer is close to any valid number
    for valid in valid_numbers:
        if valid == 0:
            if abs(answer_num) <= threshold:
                return False
        else:
            relative_error = abs(answer_num - valid) / abs(valid)
            if relative_error <= threshold:
                return False

    # Answer doesn't match any valid derived value - likely hallucination
    return True

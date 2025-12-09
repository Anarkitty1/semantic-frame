"""
Benchmark Configuration

Central configuration for all benchmark parameters, thresholds, and settings.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    """Available benchmark task types."""

    STATISTICAL = "statistical"  # T1: Single-value extraction
    TREND = "trend"  # T2: Trend classification
    ANOMALY = "anomaly"  # T3: Anomaly detection
    COMPARATIVE = "comparative"  # T4: Comparative analysis
    MULTI_STEP = "multi_step"  # T5: Multi-step reasoning
    SCALING = "scaling"  # T6: Large-scale data handling


class DataPattern(Enum):
    """Data generation patterns."""

    RANDOM = "random"
    LINEAR_TREND = "linear_trend"
    EXPONENTIAL_TREND = "exponential_trend"
    POLYNOMIAL_TREND = "polynomial_trend"
    SEASONAL = "seasonal"
    RANDOM_WALK = "random_walk"
    STEP_FUNCTION = "step_function"
    MIXED = "mixed"


class AnomalyType(Enum):
    """Types of anomalies for injection."""

    POINT_SPIKE = "point_spike"
    POINT_DROP = "point_drop"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    LEVEL_SHIFT = "level_shift"
    TREND_CHANGE = "trend_change"


@dataclass
class ModelConfig:
    """Claude model configuration."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 1000
    timeout: float = 180.0  # Base timeout in seconds
    # Timeout scaling for large prompts (CLI backend)
    timeout_per_1k_chars: float = 10.0  # Extra seconds per 1K chars of prompt
    max_timeout: float = 600.0  # Maximum timeout (10 minutes)


@dataclass
class DatasetConfig:
    """Dataset generation configuration."""

    # Synthetic dataset sizes
    small_size: int = 100
    medium_size: int = 1_000
    large_size: int = 10_000
    very_large_size: int = 100_000

    # CLI backend limits (large prompts cause timeouts)
    cli_max_dataset_size: int = 1_000  # Max points for CLI backend

    # Default parameters
    default_seed: int = 42
    noise_levels: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5])
    anomaly_rates: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])

    # Multivariate settings
    min_variables: int = 2
    max_variables: int = 10

    def __post_init__(self) -> None:
        """Validate dataset configuration."""
        if not (self.small_size < self.medium_size < self.large_size < self.very_large_size):
            raise ValueError(
                "Dataset sizes must be in ascending order: "
                f"small_size ({self.small_size}) < medium_size ({self.medium_size}) "
                f"< large_size ({self.large_size}) < very_large_size ({self.very_large_size})"
            )
        if self.min_variables > self.max_variables:
            raise ValueError(
                f"min_variables ({self.min_variables}) must be <= "
                f"max_variables ({self.max_variables})"
            )


@dataclass
class MetricThresholds:
    """Expected performance thresholds."""

    # Token compression
    min_compression_ratio: float = 0.90  # At least 90% reduction
    target_compression_ratio: float = 0.95  # Target 95% reduction

    # Accuracy baselines (expected baseline LLM performance)
    baseline_statistical_accuracy: float = 0.70
    baseline_trend_accuracy: float = 0.65
    baseline_anomaly_f1: float = 0.55
    baseline_comparative_accuracy: float = 0.60
    baseline_multi_step_accuracy: float = 0.50

    # Treatment targets (expected with Semantic Frame)
    target_statistical_accuracy: float = 0.95
    target_trend_accuracy: float = 0.90
    target_anomaly_f1: float = 0.80
    target_comparative_accuracy: float = 0.90
    target_multi_step_accuracy: float = 0.85

    # Hallucination
    max_hallucination_rate: float = 0.02  # Max 2% hallucination

    def __post_init__(self) -> None:
        """Validate metric thresholds are in valid ranges."""
        rate_fields = [
            ("min_compression_ratio", self.min_compression_ratio),
            ("target_compression_ratio", self.target_compression_ratio),
            ("baseline_statistical_accuracy", self.baseline_statistical_accuracy),
            ("baseline_trend_accuracy", self.baseline_trend_accuracy),
            ("baseline_anomaly_f1", self.baseline_anomaly_f1),
            ("baseline_comparative_accuracy", self.baseline_comparative_accuracy),
            ("baseline_multi_step_accuracy", self.baseline_multi_step_accuracy),
            ("target_statistical_accuracy", self.target_statistical_accuracy),
            ("target_trend_accuracy", self.target_trend_accuracy),
            ("target_anomaly_f1", self.target_anomaly_f1),
            ("target_comparative_accuracy", self.target_comparative_accuracy),
            ("target_multi_step_accuracy", self.target_multi_step_accuracy),
            ("max_hallucination_rate", self.max_hallucination_rate),
        ]
        for name, value in rate_fields:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")

    # Model settings
    model: ModelConfig = field(default_factory=ModelConfig)

    # Dataset settings
    datasets: DatasetConfig = field(default_factory=DatasetConfig)

    # Metric thresholds
    thresholds: MetricThresholds = field(default_factory=MetricThresholds)

    # Execution settings
    n_trials: int = 30  # Minimum trials per condition for statistical power
    quick_mode_trials: int = 5  # Reduced trials for quick validation
    random_seed: int = 42
    parallel_workers: int = 1  # Baseline vs treatment parallelism (within trial)
    trial_parallelism: int = 1  # Multiple trials in parallel (max 4 for rate limits)

    # Large dataset handling
    skip_baseline_above_n_points: int = 5_000  # Skip baseline for very large datasets

    # API settings
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_consecutive_errors: int = 5  # Abort after N consecutive API failures

    # Logging
    verbose: bool = True
    save_raw_responses: bool = True

    def __post_init__(self) -> None:
        """Validate configuration and ensure directories exist."""
        # Validate execution settings
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        if self.retry_attempts <= 0:
            raise ValueError(f"retry_attempts must be > 0, got {self.retry_attempts}")
        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {self.retry_delay}")
        if self.max_consecutive_errors <= 0:
            raise ValueError(
                f"max_consecutive_errors must be > 0, got {self.max_consecutive_errors}"
            )

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def quick_mode(cls) -> "BenchmarkConfig":
        """Return configuration for quick validation runs."""
        config = cls()
        config.n_trials = config.quick_mode_trials
        config.datasets.large_size = 1_000
        config.datasets.very_large_size = 5_000
        return config

    @classmethod
    def full_mode(cls) -> "BenchmarkConfig":
        """Return configuration for full benchmark runs."""
        return cls()


# Prompt templates
BASELINE_PROMPT_TEMPLATE = """Analyze the following numerical data and answer the query.

DATA:
{data}

QUERY: {query}

Provide your answer in the following format:
- Answer: [your numerical or categorical answer]
- Confidence: [high/medium/low]
- Reasoning: [brief explanation of how you arrived at the answer]
"""

TREATMENT_PROMPT_TEMPLATE = """Analyze the following data summary and answer the query.

DATA ANALYSIS:
{semantic_frame_output}

QUERY: {query}

Provide your answer in the following format:
- Answer: [your numerical or categorical answer]
- Confidence: [high/medium/low]
- Reasoning: [brief explanation of how you arrived at the answer]
"""

# Task-specific query templates
STATISTICAL_QUERIES = {
    "mean": "What is the mean (average) of this dataset?",
    "median": "What is the median of this dataset?",
    "std": "What is the standard deviation of this dataset?",
    "min": "What is the minimum value in this dataset?",
    "max": "What is the maximum value in this dataset?",
    "range": "What is the range (max - min) of this dataset?",
    "p25": "What is the 25th percentile of this dataset?",
    "p75": "What is the 75th percentile of this dataset?",
    "p95": "What is the 95th percentile of this dataset?",
    "iqr": "What is the interquartile range (IQR) of this dataset?",
    "skewness": "Is this dataset skewed? If so, is it positively or negatively skewed?",
    "count": "How many data points are in this dataset?",
}

TREND_QUERIES = {
    "direction": (
        "What is the overall trend direction of this time series? (rising/falling/flat/cyclical)"
    ),
    "strength": "How strong is the trend? (strong/moderate/weak/none)",
    "slope": "Approximately what is the slope of the trend (change per unit time)?",
}

ANOMALY_QUERIES = {
    "presence": "Are there any anomalies in this data? (yes/no)",
    "count": "How many anomalies are present in this dataset?",
    "locations": "At which positions (indices) are the anomalies located?",
    "types": "What types of anomalies are present? (spike/drop/level_shift/trend_change)",
}

COMPARATIVE_QUERIES = {
    "higher_mean": "Which series has the higher mean: Series A or Series B?",
    "more_volatile": "Which series is more volatile: Series A or Series B?",
    "correlation": (
        "Are Series A and Series B positively correlated, negatively correlated, or uncorrelated?"
    ),
    "stronger_trend": "Which series has a stronger trend: Series A or Series B?",
}

"""Tests for benchmarks/config.py.

Tests configuration dataclasses, enums, and prompt templates.
"""

import os
from pathlib import Path
from unittest import mock

from benchmarks.config import (
    ANOMALY_QUERIES,
    BASELINE_PROMPT_TEMPLATE,
    COMPARATIVE_QUERIES,
    STATISTICAL_QUERIES,
    TREATMENT_PROMPT_TEMPLATE,
    TREND_QUERIES,
    AnomalyType,
    BenchmarkConfig,
    DataPattern,
    DatasetConfig,
    MetricThresholds,
    ModelConfig,
    TaskType,
)


class TestTaskTypeEnum:
    """Tests for TaskType enum."""

    def test_all_task_types_defined(self) -> None:
        """Verify all expected task types exist."""
        expected = {
            "STATISTICAL",
            "TREND",
            "ANOMALY",
            "COMPARATIVE",
            "MULTI_STEP",
            "SCALING",
        }
        actual = {t.name for t in TaskType}
        assert actual == expected

    def test_task_type_values(self) -> None:
        """Verify task type values are lowercase strings."""
        for task_type in TaskType:
            assert task_type.value == task_type.name.lower()


class TestDataPatternEnum:
    """Tests for DataPattern enum."""

    def test_all_patterns_defined(self) -> None:
        """Verify all expected data patterns exist."""
        expected = {
            "RANDOM",
            "LINEAR_TREND",
            "EXPONENTIAL_TREND",
            "POLYNOMIAL_TREND",
            "SEASONAL",
            "RANDOM_WALK",
            "STEP_FUNCTION",
            "MIXED",
        }
        actual = {p.name for p in DataPattern}
        assert actual == expected

    def test_pattern_values_snake_case(self) -> None:
        """Verify pattern values are snake_case strings."""
        for pattern in DataPattern:
            assert pattern.value == pattern.name.lower()


class TestAnomalyTypeEnum:
    """Tests for AnomalyType enum."""

    def test_all_anomaly_types_defined(self) -> None:
        """Verify all expected anomaly types exist."""
        expected = {
            "POINT_SPIKE",
            "POINT_DROP",
            "CONTEXTUAL",
            "COLLECTIVE",
            "LEVEL_SHIFT",
            "TREND_CHANGE",
        }
        actual = {a.name for a in AnomalyType}
        assert actual == expected


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.timeout == 180.0  # Base timeout for CLI backend
        assert config.timeout_per_1k_chars == 10.0  # Scaled timeout for large prompts
        assert config.max_timeout == 600.0  # Maximum timeout (10 minutes)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(
            model="claude-opus-4-20250514",
            temperature=0.5,
            max_tokens=2000,
            timeout=120.0,
        )
        assert config.model == "claude-opus-4-20250514"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 120.0


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.small_size == 100
        assert config.medium_size == 1_000
        assert config.large_size == 10_000
        assert config.very_large_size == 100_000
        assert config.default_seed == 42
        assert config.min_variables == 2
        assert config.max_variables == 10

    def test_default_noise_levels(self) -> None:
        """Test default noise levels list."""
        config = DatasetConfig()
        assert config.noise_levels == [0.0, 0.1, 0.25, 0.5]

    def test_default_anomaly_rates(self) -> None:
        """Test default anomaly rates list."""
        config = DatasetConfig()
        assert config.anomaly_rates == [0.01, 0.02, 0.05]

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DatasetConfig(
            small_size=50,
            medium_size=500,
            noise_levels=[0.1, 0.2],
        )
        assert config.small_size == 50
        assert config.medium_size == 500
        assert config.noise_levels == [0.1, 0.2]


class TestMetricThresholds:
    """Tests for MetricThresholds dataclass."""

    def test_compression_thresholds(self) -> None:
        """Test compression ratio thresholds."""
        thresholds = MetricThresholds()
        assert thresholds.min_compression_ratio == 0.90
        assert thresholds.target_compression_ratio == 0.95

    def test_baseline_accuracy_thresholds(self) -> None:
        """Test baseline accuracy thresholds."""
        thresholds = MetricThresholds()
        assert thresholds.baseline_statistical_accuracy == 0.70
        assert thresholds.baseline_trend_accuracy == 0.65
        assert thresholds.baseline_anomaly_f1 == 0.55
        assert thresholds.baseline_comparative_accuracy == 0.60
        assert thresholds.baseline_multi_step_accuracy == 0.50

    def test_target_accuracy_thresholds(self) -> None:
        """Test target accuracy thresholds."""
        thresholds = MetricThresholds()
        assert thresholds.target_statistical_accuracy == 0.95
        assert thresholds.target_trend_accuracy == 0.90
        assert thresholds.target_anomaly_f1 == 0.80
        assert thresholds.target_comparative_accuracy == 0.90
        assert thresholds.target_multi_step_accuracy == 0.85

    def test_hallucination_threshold(self) -> None:
        """Test hallucination rate threshold."""
        thresholds = MetricThresholds()
        assert thresholds.max_hallucination_rate == 0.02


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.n_trials == 30
        assert config.quick_mode_trials == 5
        assert config.random_seed == 42
        assert config.parallel_workers == 1
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.verbose is True
        assert config.save_raw_responses is True

    def test_nested_configs_created(self) -> None:
        """Test nested config objects are created."""
        config = BenchmarkConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.datasets, DatasetConfig)
        assert isinstance(config.thresholds, MetricThresholds)

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Test output directory is created on init."""
        output_dir = tmp_path / "results"
        _ = BenchmarkConfig(output_dir=output_dir)
        assert output_dir.exists()

    def test_data_dir_created(self, tmp_path: Path) -> None:
        """Test data directory is created on init."""
        data_dir = tmp_path / "data"
        _ = BenchmarkConfig(data_dir=data_dir)
        assert data_dir.exists()

    def test_api_key_from_env(self) -> None:
        """Test API key is read from environment."""
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            config = BenchmarkConfig()
            assert config.api_key == "test-key-123"

    def test_quick_mode(self) -> None:
        """Test quick_mode class method."""
        config = BenchmarkConfig.quick_mode()
        assert config.n_trials == config.quick_mode_trials
        assert config.datasets.large_size == 1_000
        assert config.datasets.very_large_size == 5_000

    def test_full_mode(self) -> None:
        """Test full_mode class method."""
        config = BenchmarkConfig.full_mode()
        assert config.n_trials == 30
        assert config.datasets.large_size == 10_000


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_baseline_prompt_has_placeholders(self) -> None:
        """Test baseline prompt template has required placeholders."""
        assert "{data}" in BASELINE_PROMPT_TEMPLATE
        assert "{query}" in BASELINE_PROMPT_TEMPLATE

    def test_baseline_prompt_format(self) -> None:
        """Test baseline prompt template formats correctly."""
        result = BASELINE_PROMPT_TEMPLATE.format(
            data="[1, 2, 3, 4, 5]",
            query="What is the mean?",
        )
        assert "[1, 2, 3, 4, 5]" in result
        assert "What is the mean?" in result
        # Check case-insensitive
        result_lower = result.lower()
        assert "answer:" in result_lower
        assert "confidence:" in result_lower
        assert "reasoning:" in result_lower

    def test_treatment_prompt_has_placeholders(self) -> None:
        """Test treatment prompt template has required placeholders."""
        assert "{semantic_frame_output}" in TREATMENT_PROMPT_TEMPLATE
        assert "{query}" in TREATMENT_PROMPT_TEMPLATE

    def test_treatment_prompt_format(self) -> None:
        """Test treatment prompt template formats correctly."""
        result = TREATMENT_PROMPT_TEMPLATE.format(
            semantic_frame_output="Mean: 3.0, Trend: rising",
            query="What is the mean?",
        )
        assert "Mean: 3.0, Trend: rising" in result
        assert "What is the mean?" in result


class TestQueryTemplates:
    """Tests for query templates."""

    def test_statistical_queries_defined(self) -> None:
        """Test all expected statistical queries are defined."""
        expected_keys = {
            "mean",
            "median",
            "std",
            "min",
            "max",
            "range",
            "p25",
            "p75",
            "p95",
            "iqr",
            "skewness",
            "count",
        }
        assert set(STATISTICAL_QUERIES.keys()) == expected_keys

    def test_statistical_queries_are_questions(self) -> None:
        """Test statistical queries end with question marks."""
        for key, query in STATISTICAL_QUERIES.items():
            assert query.endswith("?"), f"Query '{key}' should end with ?"

    def test_trend_queries_defined(self) -> None:
        """Test all expected trend queries are defined."""
        expected_keys = {"direction", "strength", "slope"}
        assert set(TREND_QUERIES.keys()) == expected_keys

    def test_anomaly_queries_defined(self) -> None:
        """Test all expected anomaly queries are defined."""
        expected_keys = {"presence", "count", "locations", "types"}
        assert set(ANOMALY_QUERIES.keys()) == expected_keys

    def test_comparative_queries_defined(self) -> None:
        """Test all expected comparative queries are defined."""
        expected_keys = {"higher_mean", "more_volatile", "correlation", "stronger_trend"}
        assert set(COMPARATIVE_QUERIES.keys()) == expected_keys

    def test_comparative_queries_mention_series(self) -> None:
        """Test comparative queries mention Series A and B."""
        for key, query in COMPARATIVE_QUERIES.items():
            assert "Series A" in query or "series A" in query.lower()
            assert "Series B" in query or "series B" in query.lower()

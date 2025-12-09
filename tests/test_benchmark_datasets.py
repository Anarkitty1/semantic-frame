"""Tests for benchmarks/datasets.py.

Tests synthetic dataset generation, anomaly injection, and data serialization.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.config import AnomalyType, DataPattern
from benchmarks.datasets import (
    AnomalyDataset,
    DatasetGenerator,
    SyntheticDataset,
    load_dataset,
    save_dataset,
)


class TestSyntheticDataset:
    """Tests for SyntheticDataset dataclass."""

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = SyntheticDataset(
            name="test",
            data=data,
            ground_truth={"mean": 3.0},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        json_str = dataset.to_json()
        parsed = json.loads(json_str)

        assert parsed == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_to_csv_string(self) -> None:
        """Test CSV serialization."""
        data = np.array([10.0, 20.0, 30.0])
        dataset = SyntheticDataset(
            name="test",
            data=data,
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        csv_str = dataset.to_csv_string()
        lines = csv_str.split("\n")

        assert lines[0] == "index,value"
        assert lines[1] == "0,10.0"
        assert lines[2] == "1,20.0"
        assert lines[3] == "2,30.0"


class TestAnomalyDataset:
    """Tests for AnomalyDataset dataclass."""

    def test_post_init_defaults(self) -> None:
        """Test default values are set in post_init."""
        dataset = AnomalyDataset(
            name="test",
            data=np.array([1.0, 2.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        assert dataset.anomaly_indices == []
        assert dataset.anomaly_types == []

    def test_with_anomaly_data(self) -> None:
        """Test with explicit anomaly data."""
        dataset = AnomalyDataset(
            name="test",
            data=np.array([1.0, 100.0, 3.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
            anomaly_indices=[1],
            anomaly_types=[AnomalyType.POINT_SPIKE],
        )

        assert dataset.anomaly_indices == [1]
        assert dataset.anomaly_types == [AnomalyType.POINT_SPIKE]


class TestDatasetGenerator:
    """Tests for DatasetGenerator class."""

    def test_init_with_seed(self) -> None:
        """Test initialization with seed."""
        gen = DatasetGenerator(seed=123)
        assert gen.seed == 123

    def test_reset_seed(self) -> None:
        """Test seed reset produces same results."""
        gen = DatasetGenerator(seed=42)
        data1 = gen.generate_random(100)

        gen.reset_seed(42)
        data2 = gen.generate_random(100)

        np.testing.assert_array_equal(data1.data, data2.data)

    def test_reset_seed_different_seed(self) -> None:
        """Test reset with different seed produces different results."""
        gen = DatasetGenerator(seed=42)
        data1 = gen.generate_random(100)

        gen.reset_seed(99)
        data2 = gen.generate_random(100)

        assert not np.array_equal(data1.data, data2.data)


class TestGenerateRandom:
    """Tests for generate_random method."""

    def test_random_size(self) -> None:
        """Test random data has correct size."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random(100)

        assert len(dataset.data) == 100

    def test_random_range(self) -> None:
        """Test random data is within specified range."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random(1000, low=10.0, high=20.0)

        assert dataset.data.min() >= 10.0
        assert dataset.data.max() <= 20.0

    def test_random_ground_truth(self) -> None:
        """Test random data ground truth."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random(100)

        assert "mean" in dataset.ground_truth
        assert "median" in dataset.ground_truth
        assert "std" in dataset.ground_truth
        assert "min" in dataset.ground_truth
        assert "max" in dataset.ground_truth
        assert dataset.ground_truth["count"] == 100
        assert dataset.ground_truth["trend"] == "none"

    def test_random_metadata(self) -> None:
        """Test random data metadata."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random(100, name="my_random")

        assert dataset.name == "my_random"
        assert dataset.pattern == DataPattern.RANDOM
        assert dataset.seed == 42


class TestGenerateLinearTrend:
    """Tests for generate_linear_trend method."""

    def test_linear_trend_size(self) -> None:
        """Test linear trend data has correct size."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_linear_trend(100)

        assert len(dataset.data) == 100

    def test_linear_trend_rising(self) -> None:
        """Test rising linear trend."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_linear_trend(100, slope=2.0, noise_std=0.1)

        assert dataset.ground_truth["trend"] == "rising"
        assert dataset.ground_truth["slope"] == 2.0
        # Data should generally increase
        assert dataset.data[-1] > dataset.data[0]

    def test_linear_trend_falling(self) -> None:
        """Test falling linear trend."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_linear_trend(100, slope=-2.0, noise_std=0.1)

        assert dataset.ground_truth["trend"] == "falling"
        assert dataset.ground_truth["slope"] == -2.0
        # Data should generally decrease
        assert dataset.data[-1] < dataset.data[0]

    def test_linear_trend_flat(self) -> None:
        """Test flat (no trend)."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_linear_trend(100, slope=0.0, noise_std=1.0)

        assert dataset.ground_truth["trend"] == "flat"
        assert dataset.ground_truth["slope"] == 0.0

    def test_linear_trend_strength(self) -> None:
        """Test trend strength classification.

        NOTE: Strength is based on normalized slope (slope * n / data_range),
        which aligns with semantic-frame's thresholds. Most trends with
        any meaningful slope over 100 points normalize to "strong" because
        the trend dominates the data range.
        """
        gen = DatasetGenerator(seed=42)

        # Clear rising trend normalizes to strong
        strong = gen.generate_linear_trend(100, slope=1.0)
        assert strong.ground_truth["trend_strength"] == "strong"

        # Even moderate raw slopes normalize to strong with low noise
        moderate_raw = gen.generate_linear_trend(100, slope=0.3)
        assert moderate_raw.ground_truth["trend_strength"] == "strong"

        # Only with high noise can we get weaker normalized trends
        # High noise increases data_range, reducing normalized slope
        weak_due_to_noise = gen.generate_linear_trend(100, slope=0.01, noise_std=10.0)
        assert weak_due_to_noise.ground_truth["trend_strength"] in ["weak", "moderate"]

        # Flat trend (zero slope) is always weak
        flat = gen.generate_linear_trend(100, slope=0.0, noise_std=1.0)
        assert flat.ground_truth["trend_strength"] == "weak"


class TestGenerateExponentialTrend:
    """Tests for generate_exponential_trend method."""

    def test_exponential_trend_growth(self) -> None:
        """Test exponential growth."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_exponential_trend(100, growth_rate=0.05, noise_std=0.1)

        assert dataset.ground_truth["trend"] == "rising"
        assert dataset.ground_truth["growth_rate"] == 0.05
        # End should be much larger than start
        assert dataset.data[-1] > dataset.data[0] * 2

    def test_exponential_trend_decay(self) -> None:
        """Test exponential decay."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_exponential_trend(100, growth_rate=-0.05, noise_std=0.1)

        assert dataset.ground_truth["trend"] == "falling"

    def test_exponential_pattern(self) -> None:
        """Test exponential pattern metadata."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_exponential_trend(100)

        assert dataset.pattern == DataPattern.EXPONENTIAL_TREND


class TestGenerateSeasonal:
    """Tests for generate_seasonal method."""

    def test_seasonal_size(self) -> None:
        """Test seasonal data has correct size."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_seasonal(100)

        assert len(dataset.data) == 100

    def test_seasonal_ground_truth(self) -> None:
        """Test seasonal ground truth."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_seasonal(100, period=20, amplitude=15.0)

        assert dataset.ground_truth["trend"] == "cyclical"
        assert dataset.ground_truth["period"] == 20
        assert dataset.ground_truth["amplitude"] == 15.0

    def test_seasonal_range(self) -> None:
        """Test seasonal data oscillates around baseline."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_seasonal(200, baseline=50.0, amplitude=10.0, noise_std=0.1)

        # Data should oscillate around 50
        assert abs(dataset.data.mean() - 50.0) < 5.0
        # Range should be roughly 2*amplitude
        data_range = dataset.data.max() - dataset.data.min()
        assert data_range > 15.0  # At least 1.5 * amplitude


class TestGenerateRandomWalk:
    """Tests for generate_random_walk method."""

    def test_random_walk_size(self) -> None:
        """Test random walk data has correct size."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random_walk(100)

        assert len(dataset.data) == 100

    def test_random_walk_starts_at_start(self) -> None:
        """Test random walk starts at specified value."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random_walk(100, start=100.0, step_std=0.1)

        # First value should be close to start
        assert abs(dataset.data[0] - 100.0) < 1.0

    def test_random_walk_ground_truth(self) -> None:
        """Test random walk ground truth."""
        gen = DatasetGenerator(seed=42)
        dataset = gen.generate_random_walk(100)

        assert "trend" in dataset.ground_truth
        assert "volatility" in dataset.ground_truth
        assert dataset.pattern == DataPattern.RANDOM_WALK


class TestInjectAnomalies:
    """Tests for inject_anomalies method."""

    def test_inject_anomalies_count(self) -> None:
        """Test correct number of anomalies injected."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(100)
        anomaly_ds = gen.inject_anomalies(base, anomaly_rate=0.1)

        # Should have ~10 anomalies
        assert len(anomaly_ds.anomaly_indices) == 10

    def test_inject_anomalies_minimum(self) -> None:
        """Test at least one anomaly is injected."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(10)
        anomaly_ds = gen.inject_anomalies(base, anomaly_rate=0.01)

        assert len(anomaly_ds.anomaly_indices) >= 1

    def test_inject_anomalies_spike(self) -> None:
        """Test spike anomaly injection."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(100, low=40, high=60)
        anomaly_ds = gen.inject_anomalies(
            base,
            anomaly_rate=0.05,
            anomaly_types=[AnomalyType.POINT_SPIKE],
        )

        # Spike values should be above normal range
        for idx in anomaly_ds.anomaly_indices:
            # Spikes should be significantly above mean
            assert anomaly_ds.data[idx] > 60

    def test_inject_anomalies_drop(self) -> None:
        """Test drop anomaly injection."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(100, low=40, high=60)
        anomaly_ds = gen.inject_anomalies(
            base,
            anomaly_rate=0.05,
            anomaly_types=[AnomalyType.POINT_DROP],
        )

        # Drop values should be below normal range
        for idx in anomaly_ds.anomaly_indices:
            assert anomaly_ds.data[idx] < 40

    def test_inject_anomalies_ground_truth(self) -> None:
        """Test anomaly ground truth is updated."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(100)
        anomaly_ds = gen.inject_anomalies(base, anomaly_rate=0.05)

        assert anomaly_ds.ground_truth["has_anomalies"] is True
        assert anomaly_ds.ground_truth["n_anomalies"] == 5
        assert len(anomaly_ds.ground_truth["anomaly_indices"]) == 5
        assert len(anomaly_ds.ground_truth["anomaly_types"]) == 5

    def test_inject_anomalies_preserves_base(self) -> None:
        """Test original dataset is not modified."""
        gen = DatasetGenerator(seed=42)
        base = gen.generate_random(100)
        original_data = base.data.copy()

        gen.inject_anomalies(base, anomaly_rate=0.1)

        np.testing.assert_array_equal(base.data, original_data)


class TestGenerateCorrelatedSeries:
    """Tests for generate_correlated_series method."""

    def test_correlated_series_count(self) -> None:
        """Test correct number of series generated."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_correlated_series(100, n_series=5)

        assert len(datasets) == 5
        assert "series_A" in datasets
        assert "series_E" in datasets

    def test_correlated_series_correlation(self) -> None:
        """Test series are actually correlated."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_correlated_series(1000, n_series=3, correlation_strength=0.9)

        corr_ab = np.corrcoef(datasets["series_A"].data, datasets["series_B"].data)[0, 1]
        # Should be highly correlated
        assert corr_ab > 0.7

    def test_correlated_series_weak_correlation(self) -> None:
        """Test weak correlation setting."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_correlated_series(1000, n_series=2, correlation_strength=0.1)

        corr_ab = np.corrcoef(datasets["series_A"].data, datasets["series_B"].data)[0, 1]
        # Should be weakly correlated
        assert abs(corr_ab) < 0.5


class TestGenerateStatisticalSuite:
    """Tests for generate_statistical_suite method."""

    def test_statistical_suite_sizes(self) -> None:
        """Test suite generates correct sizes."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_statistical_suite(sizes=[50, 100, 200])

        sizes = [len(d.data) for d in datasets]
        # Each size has normal and skewed variant
        assert sizes.count(50) == 2
        assert sizes.count(100) == 2
        assert sizes.count(200) == 2

    def test_statistical_suite_patterns(self) -> None:
        """Test suite contains normal and skewed distributions."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_statistical_suite(sizes=[100])

        names = [d.name for d in datasets]
        assert any("normal" in name for name in names)
        assert any("skewed" in name for name in names)

    def test_statistical_suite_ground_truth(self) -> None:
        """Test ground truth includes required statistics."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_statistical_suite(sizes=[100])

        for dataset in datasets:
            gt = dataset.ground_truth
            assert "mean" in gt
            assert "median" in gt
            assert "std" in gt
            assert "p25" in gt
            assert "p75" in gt
            assert "iqr" in gt
            assert "count" in gt


class TestGenerateTrendSuite:
    """Tests for generate_trend_suite method."""

    def test_trend_suite_types(self) -> None:
        """Test suite contains different trend types."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_trend_suite(size=100)

        trends = [d.ground_truth["trend"] for d in datasets]
        assert "rising" in trends
        assert "falling" in trends
        assert "flat" in trends
        assert "cyclical" in trends

    def test_trend_suite_strengths(self) -> None:
        """Test suite contains different trend strengths."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_trend_suite(size=100)

        names = [d.name for d in datasets]
        assert any("strong" in name for name in names)
        assert any("moderate" in name for name in names)
        assert any("weak" in name for name in names)


class TestGenerateAnomalySuite:
    """Tests for generate_anomaly_suite method."""

    def test_anomaly_suite_contains_clean(self) -> None:
        """Test suite contains clean (no anomaly) datasets."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_anomaly_suite(size=100)

        clean_datasets = [d for d in datasets if "clean" in d.name]
        assert len(clean_datasets) > 0

        for d in clean_datasets:
            assert d.anomaly_indices == []

    def test_anomaly_suite_contains_anomalies(self) -> None:
        """Test suite contains datasets with anomalies."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_anomaly_suite(size=100, anomaly_rate=0.05)

        anomaly_datasets = [d for d in datasets if "clean" not in d.name]
        assert len(anomaly_datasets) > 0

        for d in anomaly_datasets:
            assert len(d.anomaly_indices) > 0

    def test_anomaly_suite_different_types(self) -> None:
        """Test suite contains different anomaly types."""
        gen = DatasetGenerator(seed=42)
        datasets = gen.generate_anomaly_suite(size=100)

        names = [d.name for d in datasets]
        assert any("spike" in name.lower() for name in names)
        assert any("drop" in name.lower() for name in names)
        assert any("level_shift" in name.lower() for name in names)


class TestSaveLoadDataset:
    """Tests for save_dataset and load_dataset functions."""

    def test_save_load_synthetic_dataset(self, tmp_path: Path) -> None:
        """Test saving and loading a SyntheticDataset."""
        dataset = SyntheticDataset(
            name="test_save",
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            ground_truth={"mean": 3.0, "std": 1.41},
            pattern=DataPattern.LINEAR_TREND,
            seed=42,
        )

        path = tmp_path / "test_dataset.json"
        save_dataset(dataset, path)
        loaded = load_dataset(path)

        assert loaded.name == dataset.name
        np.testing.assert_array_equal(loaded.data, dataset.data)
        assert loaded.ground_truth == dataset.ground_truth
        assert loaded.pattern == dataset.pattern
        assert loaded.seed == dataset.seed

    def test_save_load_anomaly_dataset(self, tmp_path: Path) -> None:
        """Test saving and loading an AnomalyDataset."""
        dataset = AnomalyDataset(
            name="test_anomaly",
            data=np.array([1.0, 100.0, 3.0, 4.0, 5.0]),
            ground_truth={"has_anomalies": True},
            pattern=DataPattern.RANDOM,
            seed=42,
            anomaly_indices=[1],
            anomaly_types=[AnomalyType.POINT_SPIKE],
        )

        path = tmp_path / "test_anomaly.json"
        save_dataset(dataset, path)
        loaded = load_dataset(path)

        assert isinstance(loaded, AnomalyDataset)
        assert loaded.anomaly_indices == [1]
        assert loaded.anomaly_types == [AnomalyType.POINT_SPIKE]

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """Test saved file is valid JSON."""
        dataset = SyntheticDataset(
            name="test",
            data=np.array([1.0, 2.0]),
            ground_truth={},
            pattern=DataPattern.RANDOM,
            seed=42,
        )

        path = tmp_path / "test.json"
        save_dataset(dataset, path)

        with open(path) as f:
            data = json.load(f)

        assert data["name"] == "test"
        assert data["pattern"] == "random"
        assert data["seed"] == 42


class TestInputValidation:
    """Tests for input validation in dataset generators."""

    def test_generate_random_invalid_n(self) -> None:
        """Test generate_random rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_random(0)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_random(-5)

    def test_generate_random_invalid_range(self) -> None:
        """Test generate_random rejects low >= high."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="low must be < high"):
            gen.generate_random(100, low=50.0, high=50.0)
        with pytest.raises(ValueError, match="low must be < high"):
            gen.generate_random(100, low=100.0, high=50.0)

    def test_generate_linear_trend_invalid_n(self) -> None:
        """Test generate_linear_trend rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_linear_trend(0)

    def test_generate_exponential_trend_invalid_n(self) -> None:
        """Test generate_exponential_trend rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_exponential_trend(0)

    def test_generate_seasonal_invalid_n(self) -> None:
        """Test generate_seasonal rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_seasonal(0)

    def test_generate_seasonal_invalid_period(self) -> None:
        """Test generate_seasonal rejects period <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="period must be > 0"):
            gen.generate_seasonal(100, period=0)
        with pytest.raises(ValueError, match="period must be > 0"):
            gen.generate_seasonal(100, period=-10)

    def test_generate_random_walk_invalid_n(self) -> None:
        """Test generate_random_walk rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_random_walk(0)

    def test_generate_correlated_series_invalid_n(self) -> None:
        """Test generate_correlated_series rejects n <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_correlated_series(0)

    def test_generate_correlated_series_invalid_n_series(self) -> None:
        """Test generate_correlated_series rejects n_series <= 0."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="n_series must be > 0"):
            gen.generate_correlated_series(100, n_series=0)

    def test_generate_correlated_series_invalid_correlation(self) -> None:
        """Test generate_correlated_series rejects correlation outside [0, 1]."""
        gen = DatasetGenerator(seed=42)
        with pytest.raises(ValueError, match="correlation_strength must be in"):
            gen.generate_correlated_series(100, correlation_strength=1.5)
        with pytest.raises(ValueError, match="correlation_strength must be in"):
            gen.generate_correlated_series(100, correlation_strength=-0.1)

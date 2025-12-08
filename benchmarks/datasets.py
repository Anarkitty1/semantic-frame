"""
Dataset Generation

Synthetic and real-world dataset generation for benchmarking.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.config import AnomalyType, DataPattern


@dataclass
class SyntheticDataset:
    """A synthetic dataset with known ground truth."""

    name: str
    data: NDArray[np.float64]
    ground_truth: dict[str, Any]
    pattern: DataPattern
    seed: int

    def to_json(self) -> str:
        """Convert data to JSON string for LLM input."""
        return json.dumps(self.data.tolist())

    def to_csv_string(self) -> str:
        """Convert to CSV format string."""
        lines = ["index,value"]
        for i, v in enumerate(self.data):
            lines.append(f"{i},{v}")
        return "\n".join(lines)


@dataclass
class AnomalyDataset(SyntheticDataset):
    """Dataset with injected anomalies."""

    anomaly_indices: list[int] = field(default_factory=list)
    anomaly_types: list[AnomalyType] = field(default_factory=list)


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset random number generator."""
        self.seed = seed if seed is not None else self.seed
        self.rng = np.random.default_rng(self.seed)

    # -------------------------------------------------------------------------
    # Basic Pattern Generators
    # -------------------------------------------------------------------------

    def generate_random(
        self,
        n: int,
        low: float = 0.0,
        high: float = 100.0,
        name: str = "random",
    ) -> SyntheticDataset:
        """Generate uniform random data."""
        data = self.rng.uniform(low, high, n)
        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": "none",
        }
        return SyntheticDataset(
            name=name,
            data=data,
            ground_truth=ground_truth,
            pattern=DataPattern.RANDOM,
            seed=self.seed,
        )

    def generate_linear_trend(
        self,
        n: int,
        slope: float = 1.0,
        intercept: float = 0.0,
        noise_std: float = 1.0,
        name: str = "linear_trend",
    ) -> SyntheticDataset:
        """Generate data with linear trend."""
        x = np.arange(n, dtype=np.float64)
        noise = self.rng.normal(0, noise_std, n)
        data = slope * x + intercept + noise

        trend_direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": trend_direction,
            "slope": slope,
            "trend_strength": "strong"
            if abs(slope) > 0.5
            else "moderate"
            if abs(slope) > 0.1
            else "weak",
        }
        return SyntheticDataset(
            name=name,
            data=data,
            ground_truth=ground_truth,
            pattern=DataPattern.LINEAR_TREND,
            seed=self.seed,
        )

    def generate_exponential_trend(
        self,
        n: int,
        growth_rate: float = 0.05,
        initial_value: float = 10.0,
        noise_std: float = 1.0,
        name: str = "exponential_trend",
    ) -> SyntheticDataset:
        """Generate data with exponential trend."""
        x = np.arange(n, dtype=np.float64)
        base = initial_value * np.exp(growth_rate * x)
        noise = self.rng.normal(0, noise_std, n)
        data = base + noise

        trend_direction = "rising" if growth_rate > 0 else "falling" if growth_rate < 0 else "flat"
        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": trend_direction,
            "growth_rate": growth_rate,
            "trend_strength": "strong",
        }
        return SyntheticDataset(
            name=name,
            data=data,
            ground_truth=ground_truth,
            pattern=DataPattern.EXPONENTIAL_TREND,
            seed=self.seed,
        )

    def generate_seasonal(
        self,
        n: int,
        period: int = 50,
        amplitude: float = 10.0,
        baseline: float = 50.0,
        noise_std: float = 1.0,
        name: str = "seasonal",
    ) -> SyntheticDataset:
        """Generate data with seasonal pattern."""
        x = np.arange(n, dtype=np.float64)
        seasonal = amplitude * np.sin(2 * np.pi * x / period)
        noise = self.rng.normal(0, noise_std, n)
        data = baseline + seasonal + noise

        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": "cyclical",
            "period": period,
            "amplitude": amplitude,
        }
        return SyntheticDataset(
            name=name,
            data=data,
            ground_truth=ground_truth,
            pattern=DataPattern.SEASONAL,
            seed=self.seed,
        )

    def generate_random_walk(
        self,
        n: int,
        start: float = 50.0,
        step_std: float = 1.0,
        name: str = "random_walk",
    ) -> SyntheticDataset:
        """Generate random walk data."""
        steps = self.rng.normal(0, step_std, n)
        data = np.cumsum(steps) + start

        # Determine overall trend from start to end
        overall_change = data[-1] - data[0]
        if abs(overall_change) < step_std * np.sqrt(n) * 0.5:
            trend = "flat"
        else:
            trend = "rising" if overall_change > 0 else "falling"

        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": trend,
            "volatility": "high" if step_std > 2 else "moderate" if step_std > 0.5 else "low",
        }
        return SyntheticDataset(
            name=name,
            data=data,
            ground_truth=ground_truth,
            pattern=DataPattern.RANDOM_WALK,
            seed=self.seed,
        )

    # -------------------------------------------------------------------------
    # Anomaly Injection
    # -------------------------------------------------------------------------

    def inject_anomalies(
        self,
        dataset: SyntheticDataset,
        anomaly_rate: float = 0.02,
        anomaly_types: list[AnomalyType] | None = None,
        name: str | None = None,
    ) -> AnomalyDataset:
        """Inject anomalies into a dataset."""
        if anomaly_types is None:
            anomaly_types = [AnomalyType.POINT_SPIKE, AnomalyType.POINT_DROP]

        data = dataset.data.copy()
        n = len(data)
        n_anomalies = max(1, int(n * anomaly_rate))

        # Select anomaly positions
        anomaly_indices = sorted(self.rng.choice(n, size=n_anomalies, replace=False).tolist())

        # Calculate data statistics for anomaly magnitude
        data_mean = np.mean(data)
        data_std = np.std(data)

        # Inject anomalies
        injected_types: list[AnomalyType] = []
        for idx in anomaly_indices:
            atype = AnomalyType(self.rng.choice([t.value for t in anomaly_types]))
            injected_types.append(atype)

            if atype == AnomalyType.POINT_SPIKE:
                data[idx] = data_mean + self.rng.uniform(3, 5) * data_std
            elif atype == AnomalyType.POINT_DROP:
                data[idx] = data_mean - self.rng.uniform(3, 5) * data_std
            elif atype == AnomalyType.CONTEXTUAL:
                # Value that's unusual for this position but not extreme globally
                local_mean = np.mean(data[max(0, idx - 5) : min(n, idx + 5)])
                data[idx] = local_mean + self.rng.choice([-1, 1]) * 2.5 * data_std
            elif atype == AnomalyType.LEVEL_SHIFT:
                # Shift remaining data
                shift = self.rng.choice([-1, 1]) * 2 * data_std
                data[idx:] += shift

        # Update ground truth
        ground_truth = dataset.ground_truth.copy()
        ground_truth.update(
            {
                "has_anomalies": True,
                "n_anomalies": n_anomalies,
                "anomaly_indices": anomaly_indices,
                "anomaly_types": [t.value for t in injected_types],
            }
        )

        return AnomalyDataset(
            name=name or f"{dataset.name}_with_anomalies",
            data=data,
            ground_truth=ground_truth,
            pattern=dataset.pattern,
            seed=self.seed,
            anomaly_indices=anomaly_indices,
            anomaly_types=injected_types,
        )

    # -------------------------------------------------------------------------
    # Multivariate Data
    # -------------------------------------------------------------------------

    def generate_correlated_series(
        self,
        n: int,
        n_series: int = 3,
        correlation_strength: float = 0.8,
        name: str = "correlated_series",
    ) -> dict[str, SyntheticDataset]:
        """Generate multiple correlated time series."""
        # Generate base series
        base = self.rng.normal(50, 10, n)

        datasets = {}
        correlations = {}

        for i in range(n_series):
            if i == 0:
                data = base.copy()
            else:
                # Mix base with independent noise
                noise = self.rng.normal(0, 10, n)
                data = correlation_strength * base + (1 - correlation_strength) * noise

            series_name = f"series_{chr(65 + i)}"  # series_A, series_B, etc.

            datasets[series_name] = SyntheticDataset(
                name=series_name,
                data=data,
                ground_truth={
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "min": np.min(data),
                    "max": np.max(data),
                },
                pattern=DataPattern.MIXED,
                seed=self.seed,
            )

            if i > 0:
                correlations[f"series_A_{series_name}"] = np.corrcoef(base, data)[0, 1]

        # Add correlation info to first series ground truth
        datasets["series_A"].ground_truth["correlations"] = correlations

        return datasets

    # -------------------------------------------------------------------------
    # Task-Specific Dataset Collections
    # -------------------------------------------------------------------------

    def generate_statistical_suite(
        self,
        sizes: list[int] = [100, 1000, 10000],
    ) -> list[SyntheticDataset]:
        """Generate suite of datasets for statistical query testing."""
        datasets = []

        for size in sizes:
            # Normal distribution
            data = self.rng.normal(50, 10, size)
            datasets.append(
                SyntheticDataset(
                    name=f"normal_{size}",
                    data=data,
                    ground_truth={
                        "mean": np.mean(data),
                        "median": np.median(data),
                        "std": np.std(data),
                        "min": np.min(data),
                        "max": np.max(data),
                        "p25": np.percentile(data, 25),
                        "p75": np.percentile(data, 75),
                        "p95": np.percentile(data, 95),
                        "iqr": np.percentile(data, 75) - np.percentile(data, 25),
                        "count": size,
                        "skewness": "none",
                    },
                    pattern=DataPattern.RANDOM,
                    seed=self.seed,
                )
            )

            # Skewed distribution
            data = self.rng.exponential(10, size)
            datasets.append(
                SyntheticDataset(
                    name=f"skewed_{size}",
                    data=data,
                    ground_truth={
                        "mean": np.mean(data),
                        "median": np.median(data),
                        "std": np.std(data),
                        "min": np.min(data),
                        "max": np.max(data),
                        "p25": np.percentile(data, 25),
                        "p75": np.percentile(data, 75),
                        "p95": np.percentile(data, 95),
                        "iqr": np.percentile(data, 75) - np.percentile(data, 25),
                        "count": size,
                        "skewness": "positive",
                    },
                    pattern=DataPattern.RANDOM,
                    seed=self.seed,
                )
            )

        return datasets

    def generate_trend_suite(
        self,
        size: int = 100,
    ) -> list[SyntheticDataset]:
        """Generate suite of datasets for trend detection testing."""
        datasets = []

        # Strong rising trend
        datasets.append(
            self.generate_linear_trend(size, slope=2.0, noise_std=1.0, name="strong_rising")
        )

        # Moderate rising trend
        datasets.append(
            self.generate_linear_trend(size, slope=0.5, noise_std=2.0, name="moderate_rising")
        )

        # Weak rising trend
        datasets.append(
            self.generate_linear_trend(size, slope=0.1, noise_std=3.0, name="weak_rising")
        )

        # Strong falling trend
        datasets.append(
            self.generate_linear_trend(size, slope=-2.0, noise_std=1.0, name="strong_falling")
        )

        # Flat (no trend)
        datasets.append(self.generate_linear_trend(size, slope=0.0, noise_std=5.0, name="flat"))

        # Cyclical
        datasets.append(self.generate_seasonal(size, period=20, amplitude=15.0, name="cyclical"))

        # Exponential
        datasets.append(
            self.generate_exponential_trend(size, growth_rate=0.03, name="exponential_rising")
        )

        return datasets

    def generate_anomaly_suite(
        self,
        size: int = 200,
        anomaly_rate: float = 0.02,
    ) -> list[AnomalyDataset]:
        """Generate suite of datasets for anomaly detection testing."""
        datasets = []

        # Base patterns with anomalies
        base_patterns = [
            self.generate_random(size, name="base_random"),
            self.generate_linear_trend(size, slope=0.5, name="base_trend"),
            self.generate_seasonal(size, period=40, name="base_seasonal"),
        ]

        anomaly_type_sets = [
            [AnomalyType.POINT_SPIKE],
            [AnomalyType.POINT_DROP],
            [AnomalyType.POINT_SPIKE, AnomalyType.POINT_DROP],
            [AnomalyType.LEVEL_SHIFT],
        ]

        for base in base_patterns:
            for atype_set in anomaly_type_sets:
                name = f"{base.name}_{atype_set[0].value}"
                datasets.append(
                    self.inject_anomalies(
                        base,
                        anomaly_rate=anomaly_rate,
                        anomaly_types=atype_set,
                        name=name,
                    )
                )

        # Also include clean datasets (no anomalies) for false positive testing
        for base in base_patterns:
            clean = AnomalyDataset(
                name=f"{base.name}_clean",
                data=base.data,
                ground_truth={
                    **base.ground_truth,
                    "has_anomalies": False,
                    "n_anomalies": 0,
                    "anomaly_indices": [],
                },
                pattern=base.pattern,
                seed=self.seed,
                anomaly_indices=[],
                anomaly_types=[],
            )
            datasets.append(clean)

        return datasets


def save_dataset(dataset: SyntheticDataset, path: Path) -> None:
    """Save dataset to file."""
    data = {
        "name": dataset.name,
        "data": dataset.data.tolist(),
        "ground_truth": dataset.ground_truth,
        "pattern": dataset.pattern.value,
        "seed": dataset.seed,
    }

    if isinstance(dataset, AnomalyDataset):
        data["anomaly_indices"] = dataset.anomaly_indices
        data["anomaly_types"] = [t.value for t in dataset.anomaly_types]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: Path) -> SyntheticDataset:
    """Load dataset from file."""
    with open(path) as f:
        data = json.load(f)

    if "anomaly_indices" in data:
        return AnomalyDataset(
            name=data["name"],
            data=np.array(data["data"]),
            ground_truth=data["ground_truth"],
            pattern=DataPattern(data["pattern"]),
            seed=data["seed"],
            anomaly_indices=data["anomaly_indices"],
            anomaly_types=[AnomalyType(t) for t in data.get("anomaly_types", [])],
        )

    return SyntheticDataset(
        name=data["name"],
        data=np.array(data["data"]),
        ground_truth=data["ground_truth"],
        pattern=DataPattern(data["pattern"]),
        seed=data["seed"],
    )

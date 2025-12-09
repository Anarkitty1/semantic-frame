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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if low >= high:
            raise ValueError(f"low must be < high, got low={low}, high={high}")
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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        x = np.arange(n, dtype=np.float64)
        noise = self.rng.normal(0, noise_std, n)
        data = slope * x + intercept + noise

        trend_direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"

        # Calculate normalized slope (same as semantic-frame) for strength classification
        # This ensures ground truth matches what semantic-frame will produce
        data_range = float(np.max(data) - np.min(data))
        if data_range > 0:
            normalized_slope = abs(slope * n / data_range)
        else:
            normalized_slope = 0.0

        # Use semantic-frame's thresholds: >0.5 = strong, >0.1 = moderate, else weak
        trend_strength = (
            "strong" if normalized_slope > 0.5 else "moderate" if normalized_slope > 0.1 else "weak"
        )
        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": trend_direction,
            "direction": trend_direction,  # Alias for TREND_QUERIES compatibility
            "slope": slope,
            "trend_strength": trend_strength,
            "strength": trend_strength,  # Alias for TREND_QUERIES compatibility
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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
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
            "direction": trend_direction,  # Alias for TREND_QUERIES compatibility
            "growth_rate": growth_rate,
            "trend_strength": "strong",
            "strength": "strong",  # Alias for TREND_QUERIES compatibility
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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
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
            "direction": "cyclical",  # Alias for TREND_QUERIES compatibility
            "period": period,
            "amplitude": amplitude,
            "strength": "strong",  # Cyclical patterns have clear oscillation
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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        steps = self.rng.normal(0, step_std, n)
        data = np.cumsum(steps) + start

        # Determine overall trend from start to end
        overall_change = data[-1] - data[0]
        if abs(overall_change) < step_std * np.sqrt(n) * 0.5:
            trend = "flat"
        else:
            trend = "rising" if overall_change > 0 else "falling"

        # Random walks have weak/moderate strength since they're stochastic
        trend_strength = "weak" if trend == "flat" else "moderate"
        ground_truth = {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": n,
            "trend": trend,
            "direction": trend,  # Alias for TREND_QUERIES compatibility
            "volatility": "high" if step_std > 2 else "moderate" if step_std > 0.5 else "low",
            "strength": trend_strength,  # Alias for TREND_QUERIES compatibility
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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if n_series <= 0:
            raise ValueError(f"n_series must be > 0, got {n_series}")
        if not 0.0 <= correlation_strength <= 1.0:
            raise ValueError(
                f"correlation_strength must be in [0.0, 1.0], got {correlation_strength}"
            )
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
    """Save dataset to file.

    Raises:
        OSError: If file cannot be written
        TypeError/ValueError: If data cannot be serialized to JSON
    """
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

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise OSError(f"Failed to save dataset to {path}: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize dataset '{dataset.name}': {e}") from e


def load_dataset(path: Path) -> SyntheticDataset:
    """Load dataset from file.

    Raises:
        FileNotFoundError: If file does not exist
        OSError: If file cannot be read
        ValueError: If file contains invalid JSON or missing required fields
        KeyError: If required fields are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as e:
        raise OSError(f"Failed to read dataset from {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file {path}: {e}") from e

    # Validate required fields
    required_fields = {"name", "data", "ground_truth", "pattern", "seed"}
    missing = required_fields - set(data.keys())
    if missing:
        raise KeyError(f"Dataset file {path} missing required fields: {missing}")

    try:
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
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid data in dataset file {path}: {e}") from e


# =============================================================================
# Domain-Specific Data Structures
# =============================================================================


@dataclass
class OHLCVDataset:
    """OHLCV (Open, High, Low, Close, Volume) financial dataset.

    Standard financial time series format used for stock prices,
    cryptocurrency, futures, etc.
    """

    name: str
    timestamps: NDArray[np.int64]  # Unix timestamps or indices
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    ground_truth: dict[str, Any]
    seed: int

    def to_json(self) -> str:
        """Convert to JSON string for LLM input."""
        return json.dumps(
            {
                "timestamps": self.timestamps.tolist(),
                "open": self.open.tolist(),
                "high": self.high.tolist(),
                "low": self.low.tolist(),
                "close": self.close.tolist(),
                "volume": self.volume.tolist(),
            }
        )

    def to_close_series(self) -> SyntheticDataset:
        """Extract close prices as a SyntheticDataset."""
        return SyntheticDataset(
            name=f"{self.name}_close",
            data=self.close,
            ground_truth={
                "mean": float(np.mean(self.close)),
                "std": float(np.std(self.close)),
                "min": float(np.min(self.close)),
                "max": float(np.max(self.close)),
                "return": float((self.close[-1] / self.close[0] - 1) * 100),
            },
            pattern=DataPattern.RANDOM_WALK,
            seed=self.seed,
        )


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bid/ask levels.

    Represents market depth at a point in time.
    """

    name: str
    timestamp: int
    bid_prices: NDArray[np.float64]  # Descending order
    bid_sizes: NDArray[np.float64]
    ask_prices: NDArray[np.float64]  # Ascending order
    ask_sizes: NDArray[np.float64]
    ground_truth: dict[str, Any]
    seed: int

    def to_json(self) -> str:
        """Convert to JSON string for LLM input."""
        return json.dumps(
            {
                "timestamp": int(self.timestamp),
                "bids": [
                    {"price": float(p), "size": float(s)}
                    for p, s in zip(self.bid_prices, self.bid_sizes)
                ],
                "asks": [
                    {"price": float(p), "size": float(s)}
                    for p, s in zip(self.ask_prices, self.ask_sizes)
                ],
            }
        )


@dataclass
class SensorDataset:
    """IoT sensor dataset with readings and metadata.

    Supports various sensor types with configurable failure modes.
    """

    name: str
    timestamps: NDArray[np.int64]
    readings: NDArray[np.float64]
    sensor_type: str  # temperature, pressure, vibration, etc.
    unit: str
    ground_truth: dict[str, Any]
    seed: int
    failure_mode: str | None = None  # drift, spike, flatline, etc.
    failure_start_idx: int | None = None

    def to_json(self) -> str:
        """Convert to JSON string for LLM input."""
        return json.dumps(
            {
                "sensor_type": self.sensor_type,
                "unit": self.unit,
                "timestamps": self.timestamps.tolist(),
                "readings": self.readings.tolist(),
            }
        )

    def to_synthetic_dataset(self) -> SyntheticDataset:
        """Convert to SyntheticDataset for analysis."""
        return SyntheticDataset(
            name=self.name,
            data=self.readings,
            ground_truth=self.ground_truth,
            pattern=DataPattern.MIXED,
            seed=self.seed,
        )


# =============================================================================
# Financial Data Generator
# =============================================================================


class FinancialDataGenerator:
    """Generator for realistic financial market data.

    Generates:
    - OHLCV price data with configurable volatility and drift
    - Order book snapshots with realistic depth
    - Portfolio returns with correlation structure

    References:
        - Geometric Brownian Motion for price simulation
        - GARCH-like volatility clustering
        - Realistic market microstructure patterns
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset random number generator."""
        self.seed = seed if seed is not None else self.seed
        self.rng = np.random.default_rng(self.seed)

    def generate_ohlcv(
        self,
        n: int,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        volume_mean: float = 1_000_000,
        volume_std: float = 500_000,
        name: str = "ohlcv_data",
    ) -> OHLCVDataset:
        """Generate OHLCV price data using Geometric Brownian Motion.

        Args:
            n: Number of periods (e.g., days, hours)
            initial_price: Starting price
            volatility: Daily/period volatility (std of returns)
            drift: Expected return per period (mu)
            volume_mean: Mean trading volume
            volume_std: Std of trading volume
            name: Dataset name

        Returns:
            OHLCVDataset with synthetic price data
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if initial_price <= 0:
            raise ValueError(f"initial_price must be > 0, got {initial_price}")
        if volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {volatility}")

        # Generate returns using GBM: dS = mu*S*dt + sigma*S*dW
        returns = self.rng.normal(drift, volatility, n)

        # Convert to prices
        close = np.zeros(n)
        close[0] = initial_price
        for i in range(1, n):
            close[i] = close[i - 1] * (1 + returns[i])

        # Generate OHLC from close prices
        # Intraperiod range is proportional to volatility
        intraperiod_range = np.abs(self.rng.normal(0, volatility, n)) * close

        open_prices = np.zeros(n)
        high = np.zeros(n)
        low = np.zeros(n)

        open_prices[0] = initial_price
        for i in range(1, n):
            open_prices[i] = close[i - 1] * (1 + self.rng.normal(0, volatility * 0.3))

        # High is max of open and close, plus some random extension
        high = np.maximum(open_prices, close) + intraperiod_range * self.rng.uniform(0.3, 0.7, n)
        # Low is min of open and close, minus some random extension
        low = np.minimum(open_prices, close) - intraperiod_range * self.rng.uniform(0.3, 0.7, n)

        # Ensure low <= open, close <= high
        low = np.minimum(low, np.minimum(open_prices, close))
        high = np.maximum(high, np.maximum(open_prices, close))

        # Generate volume with some clustering (higher volume on big move days)
        abs_returns = np.abs(returns)
        volume_multiplier = 1 + 2 * (abs_returns / (volatility + 1e-6))
        volume = np.maximum(
            1000,
            self.rng.normal(volume_mean, volume_std, n) * volume_multiplier,
        )

        timestamps = np.arange(n, dtype=np.int64)

        # Calculate ground truth metrics
        total_return = (close[-1] / close[0] - 1) * 100
        realized_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        max_drawdown = self._calculate_max_drawdown(close)
        sharpe_ratio = (
            (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
            if np.std(returns) > 0
            else 0
        )

        ground_truth = {
            "total_return_pct": float(total_return),
            "realized_volatility": float(realized_volatility),
            "max_drawdown_pct": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "avg_volume": float(np.mean(volume)),
            "price_range": float(np.max(high) - np.min(low)),
            "trend": "rising" if total_return > 5 else "falling" if total_return < -5 else "flat",
            "n_periods": n,
        }

        return OHLCVDataset(
            name=name,
            timestamps=timestamps,
            open=open_prices,
            high=high,
            low=low,
            close=close,
            volume=volume,
            ground_truth=ground_truth,
            seed=self.seed,
        )

    def _calculate_max_drawdown(self, prices: NDArray[np.float64]) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def generate_order_book(
        self,
        mid_price: float = 100.0,
        spread_bps: float = 10.0,
        n_levels: int = 10,
        level_size_mean: float = 10000,
        level_size_decay: float = 0.8,
        name: str = "order_book",
    ) -> OrderBookSnapshot:
        """Generate realistic order book snapshot.

        Args:
            mid_price: Middle price between best bid and ask
            spread_bps: Bid-ask spread in basis points
            n_levels: Number of price levels on each side
            level_size_mean: Mean size at best bid/ask
            level_size_decay: Size decay factor per level (deeper = smaller)
            name: Dataset name

        Returns:
            OrderBookSnapshot with bid/ask depth
        """
        if mid_price <= 0:
            raise ValueError(f"mid_price must be > 0, got {mid_price}")
        if spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {spread_bps}")
        if n_levels <= 0:
            raise ValueError(f"n_levels must be > 0, got {n_levels}")

        spread = mid_price * spread_bps / 10000
        tick_size = spread / 2  # Minimum price increment

        # Generate bid levels (descending from best bid)
        best_bid = mid_price - spread / 2
        bid_prices = np.array(
            [best_bid - i * tick_size * self.rng.uniform(1, 2) for i in range(n_levels)]
        )
        # Ensure bids are sorted descending (highest price first)
        bid_prices = np.sort(bid_prices)[::-1]

        # Generate ask levels (ascending from best ask)
        best_ask = mid_price + spread / 2
        ask_prices = np.array(
            [best_ask + i * tick_size * self.rng.uniform(1, 2) for i in range(n_levels)]
        )
        # Ensure asks are sorted ascending (lowest price first)
        ask_prices = np.sort(ask_prices)

        # Generate sizes with decay and randomness
        bid_sizes = np.array(
            [
                max(100, level_size_mean * (level_size_decay**i) * self.rng.uniform(0.5, 1.5))
                for i in range(n_levels)
            ]
        )
        ask_sizes = np.array(
            [
                max(100, level_size_mean * (level_size_decay**i) * self.rng.uniform(0.5, 1.5))
                for i in range(n_levels)
            ]
        )

        # Calculate ground truth
        total_bid_value = np.sum(bid_prices * bid_sizes)
        total_ask_value = np.sum(ask_prices * ask_sizes)
        imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)

        ground_truth = {
            "mid_price": float(mid_price),
            "spread_bps": float(spread_bps),
            "best_bid": float(bid_prices[0]),
            "best_ask": float(ask_prices[0]),
            "total_bid_size": float(np.sum(bid_sizes)),
            "total_ask_size": float(np.sum(ask_sizes)),
            "imbalance": float(imbalance),  # Positive = more bids, negative = more asks
            "depth_levels": n_levels,
        }

        return OrderBookSnapshot(
            name=name,
            timestamp=0,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            ground_truth=ground_truth,
            seed=self.seed,
        )

    def generate_portfolio_returns(
        self,
        n: int,
        n_assets: int = 5,
        correlation_strength: float = 0.5,
        asset_volatilities: list[float] | None = None,
        asset_drifts: list[float] | None = None,
        name: str = "portfolio_returns",
    ) -> dict[str, SyntheticDataset]:
        """Generate correlated asset returns for portfolio analysis.

        Args:
            n: Number of periods
            n_assets: Number of assets in portfolio
            correlation_strength: Average pairwise correlation
            asset_volatilities: Volatility for each asset (default: random 0.01-0.05)
            asset_drifts: Drift for each asset (default: random -0.001 to 0.002)
            name: Base name for datasets

        Returns:
            Dict mapping asset names to SyntheticDataset of returns
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if n_assets <= 0:
            raise ValueError(f"n_assets must be > 0, got {n_assets}")
        if not 0.0 <= correlation_strength <= 1.0:
            raise ValueError(f"correlation_strength must be in [0, 1], got {correlation_strength}")

        # Generate correlation matrix
        corr_matrix = np.full((n_assets, n_assets), correlation_strength)
        np.fill_diagonal(corr_matrix, 1.0)

        # Add some randomness to off-diagonal elements
        noise = self.rng.uniform(-0.1, 0.1, (n_assets, n_assets))
        noise = (noise + noise.T) / 2  # Make symmetric
        np.fill_diagonal(noise, 0)
        corr_matrix = np.clip(corr_matrix + noise, -0.99, 0.99)
        np.fill_diagonal(corr_matrix, 1.0)

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.01)
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Set volatilities and drifts
        if asset_volatilities is None:
            asset_volatilities = self.rng.uniform(0.01, 0.05, n_assets).tolist()
        if asset_drifts is None:
            asset_drifts = self.rng.uniform(-0.001, 0.002, n_assets).tolist()

        # Generate correlated returns using Cholesky decomposition
        cholesky = np.linalg.cholesky(corr_matrix)
        uncorrelated = self.rng.standard_normal((n, n_assets))
        correlated = uncorrelated @ cholesky.T

        # Scale by volatilities and add drifts
        returns = np.zeros((n, n_assets))
        for i in range(n_assets):
            returns[:, i] = asset_drifts[i] + asset_volatilities[i] * correlated[:, i]

        # Create datasets
        datasets = {}
        asset_names = [f"asset_{chr(65 + i)}" for i in range(n_assets)]

        actual_corr = np.corrcoef(returns.T)

        for i, asset_name in enumerate(asset_names):
            asset_returns = returns[:, i]
            cumulative_return = np.prod(1 + asset_returns) - 1

            ground_truth = {
                "mean_return": float(np.mean(asset_returns)),
                "volatility": float(np.std(asset_returns)),
                "cumulative_return": float(cumulative_return),
                "sharpe": (
                    float(np.mean(asset_returns) / np.std(asset_returns) * np.sqrt(252))
                    if np.std(asset_returns) > 0
                    else 0.0
                ),
                "max_return": float(np.max(asset_returns)),
                "min_return": float(np.min(asset_returns)),
                "correlations": {
                    asset_names[j]: float(actual_corr[i, j]) for j in range(n_assets) if j != i
                },
            }

            datasets[asset_name] = SyntheticDataset(
                name=f"{name}_{asset_name}",
                data=asset_returns,
                ground_truth=ground_truth,
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )

        return datasets


# =============================================================================
# IoT Data Generator
# =============================================================================


class IoTDataGenerator:
    """Generator for IoT sensor and device metrics data.

    Generates:
    - Sensor readings with various failure modes
    - Device metrics with degradation patterns
    - Multi-sensor systems with correlated failures

    Failure Modes:
    - drift: Gradual sensor drift over time
    - spike: Random anomalous spikes
    - flatline: Sensor stuck at constant value
    - noise_increase: Growing measurement noise
    - intermittent: Periodic sensor failures
    """

    # Sensor type configurations
    SENSOR_CONFIGS: dict[str, dict[str, Any]] = {
        "temperature": {
            "unit": "°C",
            "baseline_mean": 25.0,
            "baseline_std": 2.0,
            "min_valid": -40.0,
            "max_valid": 85.0,
            "seasonal_amplitude": 5.0,
        },
        "pressure": {
            "unit": "kPa",
            "baseline_mean": 101.3,
            "baseline_std": 0.5,
            "min_valid": 80.0,
            "max_valid": 120.0,
            "seasonal_amplitude": 1.0,
        },
        "humidity": {
            "unit": "%",
            "baseline_mean": 50.0,
            "baseline_std": 10.0,
            "min_valid": 0.0,
            "max_valid": 100.0,
            "seasonal_amplitude": 15.0,
        },
        "vibration": {
            "unit": "mm/s",
            "baseline_mean": 2.0,
            "baseline_std": 0.5,
            "min_valid": 0.0,
            "max_valid": 50.0,
            "seasonal_amplitude": 0.2,
        },
        "current": {
            "unit": "A",
            "baseline_mean": 5.0,
            "baseline_std": 0.3,
            "min_valid": 0.0,
            "max_valid": 20.0,
            "seasonal_amplitude": 0.5,
        },
        "flow_rate": {
            "unit": "L/min",
            "baseline_mean": 100.0,
            "baseline_std": 5.0,
            "min_valid": 0.0,
            "max_valid": 500.0,
            "seasonal_amplitude": 10.0,
        },
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset random number generator."""
        self.seed = seed if seed is not None else self.seed
        self.rng = np.random.default_rng(self.seed)

    def generate_sensor_readings(
        self,
        n: int,
        sensor_type: str = "temperature",
        failure_mode: str | None = None,
        failure_start_pct: float = 0.7,
        include_seasonal: bool = True,
        sampling_interval_sec: int = 60,
        name: str | None = None,
    ) -> SensorDataset:
        """Generate sensor readings with optional failure mode.

        Args:
            n: Number of readings
            sensor_type: Type of sensor (temperature, pressure, etc.)
            failure_mode: Type of failure to inject (None, drift, spike, flatline, etc.)
            failure_start_pct: When failure starts (0.7 = 70% through data)
            include_seasonal: Include daily/periodic patterns
            sampling_interval_sec: Time between samples in seconds
            name: Dataset name

        Returns:
            SensorDataset with readings and metadata
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if sensor_type not in self.SENSOR_CONFIGS:
            raise ValueError(
                f"Unknown sensor_type '{sensor_type}'. "
                f"Valid types: {list(self.SENSOR_CONFIGS.keys())}"
            )
        if failure_mode and failure_mode not in [
            "drift",
            "spike",
            "flatline",
            "noise_increase",
            "intermittent",
        ]:
            raise ValueError(f"Unknown failure_mode: {failure_mode}")

        config = self.SENSOR_CONFIGS[sensor_type]
        name = name or f"{sensor_type}_sensor"

        # Generate timestamps
        timestamps = np.arange(n, dtype=np.int64) * sampling_interval_sec

        # Generate baseline readings
        readings = self.rng.normal(config["baseline_mean"], config["baseline_std"], n)

        # Add seasonal pattern if requested
        if include_seasonal:
            # Daily pattern (assuming sampling_interval allows it)
            daily_period = 24 * 3600 / sampling_interval_sec
            if daily_period > 1:
                seasonal = config["seasonal_amplitude"] * np.sin(
                    2 * np.pi * np.arange(n) / daily_period
                )
                readings += seasonal

        # Calculate failure start index
        failure_start_idx = int(n * failure_start_pct) if failure_mode else None

        # Inject failure mode
        if failure_mode and failure_start_idx:
            readings = self._inject_failure(readings, failure_mode, failure_start_idx, config)

        # Clip to valid range
        readings = np.clip(readings, config["min_valid"], config["max_valid"])

        # Calculate ground truth
        has_failure = failure_mode is not None
        ground_truth = {
            "mean": float(np.mean(readings)),
            "std": float(np.std(readings)),
            "min": float(np.min(readings)),
            "max": float(np.max(readings)),
            "sensor_type": sensor_type,
            "has_anomaly": has_failure,
            "failure_mode": failure_mode,
            "failure_start_idx": failure_start_idx,
            "n_readings": n,
            "normal_range": [
                config["baseline_mean"] - 2 * config["baseline_std"],
                config["baseline_mean"] + 2 * config["baseline_std"],
            ],
        }

        # Add failure-specific metrics
        if failure_mode == "drift":
            drift_amount = readings[-1] - readings[failure_start_idx]
            ground_truth["drift_amount"] = float(drift_amount)
        elif failure_mode == "spike":
            spike_indices = np.where(
                np.abs(readings - config["baseline_mean"]) > 3 * config["baseline_std"]
            )[0]
            ground_truth["n_spikes"] = len(spike_indices)
            ground_truth["spike_indices"] = spike_indices.tolist()

        return SensorDataset(
            name=name,
            timestamps=timestamps,
            readings=readings,
            sensor_type=sensor_type,
            unit=config["unit"],
            ground_truth=ground_truth,
            seed=self.seed,
            failure_mode=failure_mode,
            failure_start_idx=failure_start_idx,
        )

    def _inject_failure(
        self,
        readings: NDArray[np.float64],
        failure_mode: str,
        start_idx: int,
        config: dict[str, Any],
    ) -> NDArray[np.float64]:
        """Inject failure pattern into readings."""
        n = len(readings)
        failure_length = n - start_idx

        if failure_mode == "drift":
            # Gradual linear drift
            drift_rate = config["baseline_std"] * 0.05  # Drift per sample
            drift = np.linspace(0, drift_rate * failure_length, failure_length)
            readings[start_idx:] += drift * self.rng.choice([-1, 1])

        elif failure_mode == "spike":
            # Random spikes
            n_spikes = max(1, int(failure_length * 0.05))
            spike_positions = self.rng.choice(range(start_idx, n), size=n_spikes, replace=False)
            for pos in spike_positions:
                spike_magnitude = self.rng.uniform(4, 8) * config["baseline_std"]
                readings[pos] += spike_magnitude * self.rng.choice([-1, 1])

        elif failure_mode == "flatline":
            # Sensor stuck at last value before failure
            readings[start_idx:] = readings[start_idx - 1]

        elif failure_mode == "noise_increase":
            # Increasing measurement noise
            noise_multiplier = np.linspace(1, 5, failure_length)
            extra_noise = self.rng.normal(0, config["baseline_std"], failure_length)
            readings[start_idx:] += extra_noise * noise_multiplier

        elif failure_mode == "intermittent":
            # Periodic dropouts (NaN or extreme values)
            dropout_period = max(5, failure_length // 10)
            for i in range(start_idx, n, dropout_period):
                if i < n:
                    readings[i] = config["max_valid"] * 10  # Out of range

        return readings

    def generate_device_metrics(
        self,
        n: int,
        metric_types: list[str] | None = None,
        include_degradation: bool = True,
        degradation_start_pct: float = 0.5,
        name: str = "device_metrics",
    ) -> dict[str, SyntheticDataset]:
        """Generate multiple correlated device metrics.

        Args:
            n: Number of time points
            metric_types: List of metrics to generate (default: cpu, memory, disk_io)
            include_degradation: Whether to simulate performance degradation
            degradation_start_pct: When degradation starts
            name: Base name for datasets

        Returns:
            Dict mapping metric names to SyntheticDataset
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        if metric_types is None:
            metric_types = ["cpu_percent", "memory_percent", "disk_io_rate", "network_bytes"]

        # Metric configurations
        metric_configs: dict[str, dict[str, Any]] = {
            "cpu_percent": {
                "baseline_mean": 30.0,
                "baseline_std": 10.0,
                "min": 0.0,
                "max": 100.0,
                "degradation_effect": 20.0,
            },
            "memory_percent": {
                "baseline_mean": 45.0,
                "baseline_std": 5.0,
                "min": 0.0,
                "max": 100.0,
                "degradation_effect": 30.0,  # Memory leak
            },
            "disk_io_rate": {
                "baseline_mean": 50.0,
                "baseline_std": 20.0,
                "min": 0.0,
                "max": 500.0,
                "degradation_effect": 50.0,
            },
            "network_bytes": {
                "baseline_mean": 1000000,
                "baseline_std": 500000,
                "min": 0.0,
                "max": 10000000,
                "degradation_effect": -500000,  # Network issues reduce throughput
            },
            "response_time_ms": {
                "baseline_mean": 100.0,
                "baseline_std": 30.0,
                "min": 0.0,
                "max": 10000.0,
                "degradation_effect": 200.0,
            },
            "error_rate": {
                "baseline_mean": 0.01,
                "baseline_std": 0.005,
                "min": 0.0,
                "max": 1.0,
                "degradation_effect": 0.1,
            },
        }

        datasets = {}
        degradation_start_idx = int(n * degradation_start_pct) if include_degradation else n

        # Generate correlated base signal for realistic correlation
        base_signal = self.rng.standard_normal(n)

        for metric_type in metric_types:
            if metric_type not in metric_configs:
                raise ValueError(f"Unknown metric_type: {metric_type}")

            config = metric_configs[metric_type]

            # Generate metric with some correlation to base signal
            correlation = self.rng.uniform(0.3, 0.7)
            independent = self.rng.standard_normal(n)
            combined = correlation * base_signal + np.sqrt(1 - correlation**2) * independent

            readings = config["baseline_mean"] + config["baseline_std"] * combined

            # Add degradation if enabled
            if include_degradation:
                degradation_length = n - degradation_start_idx
                if degradation_length > 0:
                    # Gradual degradation
                    degradation = np.linspace(0, config["degradation_effect"], degradation_length)
                    readings[degradation_start_idx:] += degradation

            # Clip to valid range
            readings = np.clip(readings, config["min"], config["max"])

            ground_truth = {
                "mean": float(np.mean(readings)),
                "std": float(np.std(readings)),
                "min": float(np.min(readings)),
                "max": float(np.max(readings)),
                "metric_type": metric_type,
                "has_degradation": include_degradation,
                "degradation_start_idx": degradation_start_idx if include_degradation else None,
                "trend": "rising"
                if include_degradation and config["degradation_effect"] > 0
                else "falling"
                if include_degradation and config["degradation_effect"] < 0
                else "stable",
            }

            datasets[metric_type] = SyntheticDataset(
                name=f"{name}_{metric_type}",
                data=readings,
                ground_truth=ground_truth,
                pattern=DataPattern.MIXED,
                seed=self.seed,
            )

        return datasets

    def generate_multi_sensor_system(
        self,
        n: int,
        sensor_types: list[str] | None = None,
        failure_propagation: bool = True,
        name: str = "multi_sensor",
    ) -> dict[str, SensorDataset]:
        """Generate correlated multi-sensor system data.

        Simulates a system where sensor failures can propagate
        (e.g., temperature spike causes pressure change).

        Args:
            n: Number of readings
            sensor_types: Types of sensors (default: temperature, pressure, vibration)
            failure_propagation: Whether failures propagate between sensors
            name: Base name for datasets

        Returns:
            Dict mapping sensor names to SensorDataset
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        if sensor_types is None:
            sensor_types = ["temperature", "pressure", "vibration"]

        datasets = {}

        # Primary sensor (may have failure)
        primary_has_failure = self.rng.random() > 0.5
        primary_failure_mode = "spike" if primary_has_failure else None

        primary_sensor = self.generate_sensor_readings(
            n=n,
            sensor_type=sensor_types[0],
            failure_mode=primary_failure_mode,
            name=f"{name}_{sensor_types[0]}",
        )
        datasets[sensor_types[0]] = primary_sensor

        # Secondary sensors (may be affected by primary)
        for sensor_type in sensor_types[1:]:
            # Secondary sensor failure depends on primary
            if failure_propagation and primary_has_failure:
                # Delayed propagation
                secondary_failure = self.rng.random() > 0.3
                failure_mode = "drift" if secondary_failure else None
            else:
                failure_mode = None

            sensor = self.generate_sensor_readings(
                n=n,
                sensor_type=sensor_type,
                failure_mode=failure_mode,
                name=f"{name}_{sensor_type}",
            )
            datasets[sensor_type] = sensor

        return datasets

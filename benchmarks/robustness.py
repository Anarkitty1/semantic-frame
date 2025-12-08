"""
Robustness Testing Suite

Perturbation engine and adversarial input generation for testing
semantic-frame's resilience to noisy, edge-case, and adversarial inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from benchmarks.config import DataPattern
from benchmarks.datasets import SyntheticDataset

if TYPE_CHECKING:
    from collections.abc import Callable


class PerturbationType(Enum):
    """Types of data perturbations for robustness testing."""

    NOISE = "noise"  # Add Gaussian noise
    PRECISION = "precision"  # Reduce numeric precision
    SCALE = "scale"  # Scale values by factor
    SHIFT = "shift"  # Add constant offset
    MISSING = "missing"  # Introduce missing values
    OUTLIERS = "outliers"  # Inject extreme values
    TRUNCATION = "truncation"  # Truncate data length
    REORDER = "reorder"  # Shuffle local windows


@dataclass(frozen=True)
class PerturbationResult:
    """Result of applying a perturbation."""

    original_data: NDArray[np.float64]
    perturbed_data: NDArray[np.float64]
    perturbation_type: PerturbationType
    perturbation_params: dict[str, float | int | bool]
    affected_indices: list[int] | None = None

    @property
    def perturbation_rate(self) -> float:
        """Calculate what fraction of data was perturbed."""
        if self.affected_indices is not None:
            return len(self.affected_indices) / len(self.original_data)
        # For noise, scale, shift - all values affected
        return 1.0


@dataclass
class RobustnessMetrics:
    """Metrics for evaluating robustness of semantic-frame output."""

    base_accuracy: float
    perturbed_accuracy: float
    perturbation_type: PerturbationType
    perturbation_level: float
    degradation: float = field(init=False)
    is_robust: bool = field(init=False)
    max_degradation_threshold: float = 0.10  # 10% max acceptable degradation

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        self.degradation = self.base_accuracy - self.perturbed_accuracy
        self.is_robust = self.degradation <= self.max_degradation_threshold


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""

    noise_levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    precision_levels: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    scale_factors: list[float] = field(default_factory=lambda: [0.001, 0.01, 1, 100, 1000])
    shift_offsets: list[float] = field(default_factory=lambda: [-1000, -100, 0, 100, 1000])
    missing_rates: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    outlier_rates: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    enable_adversarial: bool = False
    random_seed: int = 42


class PerturbationEngine:
    """
    Engine for applying various perturbations to test data robustness.

    Perturbations are designed to test how well semantic-frame handles:
    - Noise: Sensor noise, measurement error
    - Precision: Floating point artifacts, rounding
    - Scale: Different units (mm vs m, cents vs dollars)
    - Shift: Baseline offsets, calibration drift
    - Missing: Data gaps, transmission errors
    - Outliers: Equipment failures, data corruption
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset random number generator."""
        self.seed = seed if seed is not None else self.seed
        self.rng = np.random.default_rng(self.seed)

    def apply_noise(
        self,
        data: NDArray[np.float64],
        noise_level: float,
        preserve_pattern: bool = True,
    ) -> PerturbationResult:
        """
        Add Gaussian noise to data.

        Args:
            data: Input data array
            noise_level: Standard deviation of noise as fraction of data std
            preserve_pattern: If True, add noise relative to local variation

        Returns:
            PerturbationResult with perturbed data
        """
        if noise_level < 0:
            raise ValueError(f"noise_level must be >= 0, got {noise_level}")

        data = np.asarray(data, dtype=np.float64)

        if preserve_pattern:
            # Noise relative to data's standard deviation
            data_std = float(np.std(data))
            if data_std == 0:
                data_std = 1.0
            noise = self.rng.normal(0, noise_level * data_std, len(data))
        else:
            # Absolute noise level
            noise = self.rng.normal(0, noise_level, len(data))

        perturbed = data + noise

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.NOISE,
            perturbation_params={"noise_level": noise_level, "preserve_pattern": preserve_pattern},
        )

    def apply_precision(
        self,
        data: NDArray[np.float64],
        significant_figures: int,
    ) -> PerturbationResult:
        """
        Reduce numeric precision to test handling of rounding artifacts.

        Args:
            data: Input data array
            significant_figures: Number of significant figures to keep

        Returns:
            PerturbationResult with reduced precision data
        """
        if significant_figures <= 0:
            raise ValueError(f"significant_figures must be > 0, got {significant_figures}")

        data = np.asarray(data, dtype=np.float64)

        # Round to significant figures
        perturbed = np.zeros_like(data)
        for i, val in enumerate(data):
            if val == 0:
                perturbed[i] = 0
            else:
                magnitude = int(np.floor(np.log10(abs(val))))
                perturbed[i] = round(val, significant_figures - 1 - magnitude)

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.PRECISION,
            perturbation_params={"significant_figures": significant_figures},
        )

    def apply_scale(
        self,
        data: NDArray[np.float64],
        scale_factor: float,
    ) -> PerturbationResult:
        """
        Scale data by a constant factor (test unit invariance).

        Args:
            data: Input data array
            scale_factor: Multiplicative scale factor

        Returns:
            PerturbationResult with scaled data
        """
        if scale_factor == 0:
            raise ValueError("scale_factor cannot be 0")

        data = np.asarray(data, dtype=np.float64)
        perturbed = data * scale_factor

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.SCALE,
            perturbation_params={"scale_factor": scale_factor},
        )

    def apply_shift(
        self,
        data: NDArray[np.float64],
        offset: float,
    ) -> PerturbationResult:
        """
        Add constant offset to all values (test baseline invariance).

        Args:
            data: Input data array
            offset: Constant to add to all values

        Returns:
            PerturbationResult with shifted data
        """
        data = np.asarray(data, dtype=np.float64)
        perturbed = data + offset

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.SHIFT,
            perturbation_params={"offset": offset},
        )

    def apply_missing(
        self,
        data: NDArray[np.float64],
        missing_rate: float,
        missing_value: float = np.nan,
    ) -> PerturbationResult:
        """
        Introduce missing values at random positions.

        Args:
            data: Input data array
            missing_rate: Fraction of values to make missing (0-1)
            missing_value: Value to use for missing (default NaN)

        Returns:
            PerturbationResult with missing values
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")

        data = np.asarray(data, dtype=np.float64)
        n_missing = int(len(data) * missing_rate)

        if n_missing == 0:
            return PerturbationResult(
                original_data=data,
                perturbed_data=data.copy(),
                perturbation_type=PerturbationType.MISSING,
                perturbation_params={"missing_rate": missing_rate},
                affected_indices=[],
            )

        # Select random indices for missing values
        missing_indices_arr = self.rng.choice(len(data), size=n_missing, replace=False)
        missing_indices: list[int] = sorted(missing_indices_arr.tolist())

        perturbed = data.copy()
        perturbed[missing_indices] = missing_value

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.MISSING,
            perturbation_params={"missing_rate": missing_rate, "missing_value": missing_value},
            affected_indices=missing_indices,
        )

    def apply_outliers(
        self,
        data: NDArray[np.float64],
        outlier_rate: float,
        outlier_magnitude: float = 10.0,
    ) -> PerturbationResult:
        """
        Inject extreme outlier values.

        Args:
            data: Input data array
            outlier_rate: Fraction of values to replace with outliers (0-1)
            outlier_magnitude: Outliers will be this many std from mean

        Returns:
            PerturbationResult with injected outliers
        """
        if not 0 <= outlier_rate <= 1:
            raise ValueError(f"outlier_rate must be in [0, 1], got {outlier_rate}")
        if outlier_magnitude <= 0:
            raise ValueError(f"outlier_magnitude must be > 0, got {outlier_magnitude}")

        data = np.asarray(data, dtype=np.float64)
        n_outliers = int(len(data) * outlier_rate)

        if n_outliers == 0:
            return PerturbationResult(
                original_data=data,
                perturbed_data=data.copy(),
                perturbation_type=PerturbationType.OUTLIERS,
                perturbation_params={"outlier_rate": outlier_rate},
                affected_indices=[],
            )

        # Select random indices for outliers
        outlier_indices_arr = self.rng.choice(len(data), size=n_outliers, replace=False)
        outlier_indices: list[int] = sorted(outlier_indices_arr.tolist())

        perturbed = data.copy()
        data_mean = float(np.mean(data))
        data_std = float(np.std(data))
        if data_std == 0:
            data_std = abs(data_mean) * 0.1 if data_mean != 0 else 1.0

        for idx in outlier_indices:
            # Random direction (positive or negative)
            direction = self.rng.choice([-1, 1])
            perturbed[idx] = data_mean + direction * outlier_magnitude * data_std

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.OUTLIERS,
            perturbation_params={
                "outlier_rate": outlier_rate,
                "outlier_magnitude": outlier_magnitude,
            },
            affected_indices=outlier_indices,
        )

    def apply_truncation(
        self,
        data: NDArray[np.float64],
        keep_fraction: float,
        from_end: bool = False,
    ) -> PerturbationResult:
        """
        Truncate data to a fraction of original length.

        Args:
            data: Input data array
            keep_fraction: Fraction of data to keep (0-1)
            from_end: If True, keep last portion; else keep first portion

        Returns:
            PerturbationResult with truncated data
        """
        if not 0 < keep_fraction <= 1:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

        data = np.asarray(data, dtype=np.float64)
        n_keep = max(1, int(len(data) * keep_fraction))

        if from_end:
            perturbed = data[-n_keep:]
            affected = list(range(len(data) - n_keep))
        else:
            perturbed = data[:n_keep]
            affected = list(range(n_keep, len(data)))

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.TRUNCATION,
            perturbation_params={"keep_fraction": keep_fraction, "from_end": from_end},
            affected_indices=affected,
        )

    def apply_local_shuffle(
        self,
        data: NDArray[np.float64],
        window_size: int = 5,
        shuffle_rate: float = 0.2,
    ) -> PerturbationResult:
        """
        Shuffle values within local windows (test temporal robustness).

        Args:
            data: Input data array
            window_size: Size of windows for local shuffling
            shuffle_rate: Fraction of windows to shuffle

        Returns:
            PerturbationResult with locally shuffled data
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if not 0 <= shuffle_rate <= 1:
            raise ValueError(f"shuffle_rate must be in [0, 1], got {shuffle_rate}")

        data = np.asarray(data, dtype=np.float64)
        perturbed = data.copy()
        n_windows = len(data) // window_size

        if n_windows == 0:
            return PerturbationResult(
                original_data=data,
                perturbed_data=perturbed,
                perturbation_type=PerturbationType.REORDER,
                perturbation_params={"window_size": window_size, "shuffle_rate": shuffle_rate},
                affected_indices=[],
            )

        n_shuffle = max(1, int(n_windows * shuffle_rate))
        shuffle_windows = self.rng.choice(n_windows, size=n_shuffle, replace=False)

        affected_indices: list[int] = []
        for window_idx in shuffle_windows:
            start = window_idx * window_size
            end = min(start + window_size, len(data))
            window_indices = list(range(start, end))
            affected_indices.extend(window_indices)

            # Shuffle the window
            window_data = perturbed[start:end].copy()
            self.rng.shuffle(window_data)
            perturbed[start:end] = window_data

        return PerturbationResult(
            original_data=data,
            perturbed_data=perturbed,
            perturbation_type=PerturbationType.REORDER,
            perturbation_params={"window_size": window_size, "shuffle_rate": shuffle_rate},
            affected_indices=sorted(affected_indices),
        )

    def apply_all_perturbations(
        self,
        data: NDArray[np.float64],
        config: RobustnessConfig,
    ) -> list[PerturbationResult]:
        """
        Apply all perturbation types with configured levels.

        Args:
            data: Input data array
            config: Robustness testing configuration

        Returns:
            List of PerturbationResults for all perturbations
        """
        results: list[PerturbationResult] = []

        # Noise perturbations
        for level in config.noise_levels:
            results.append(self.apply_noise(data, level))

        # Precision perturbations
        for figures in config.precision_levels:
            results.append(self.apply_precision(data, figures))

        # Scale perturbations
        for factor in config.scale_factors:
            results.append(self.apply_scale(data, factor))

        # Shift perturbations
        for offset in config.shift_offsets:
            results.append(self.apply_shift(data, offset))

        # Missing value perturbations
        for rate in config.missing_rates:
            results.append(self.apply_missing(data, rate))

        # Outlier perturbations
        for rate in config.outlier_rates:
            results.append(self.apply_outliers(data, rate))

        return results


class AdversarialInputGenerator:
    """
    Generate adversarial and edge-case inputs for stress testing.

    Adversarial inputs are designed to expose:
    - Numerical edge cases (overflow, underflow, precision loss)
    - Tokenization issues (very long numbers, unusual formats)
    - Distribution assumptions (extreme skew, multimodal)
    - Size edge cases (very short, very long sequences)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_edge_cases(self) -> list[SyntheticDataset]:
        """
        Generate numerical edge case datasets.

        Returns:
            List of SyntheticDataset objects with edge cases
        """
        datasets: list[SyntheticDataset] = []

        # All zeros
        datasets.append(
            SyntheticDataset(
                name="edge_all_zeros",
                data=np.zeros(100),
                ground_truth={"pattern": "all_zeros", "edge_type": "constant"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Single value (constant)
        datasets.append(
            SyntheticDataset(
                name="edge_constant",
                data=np.full(100, 42.0),
                ground_truth={"pattern": "constant", "edge_type": "constant"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Single element
        datasets.append(
            SyntheticDataset(
                name="edge_single_element",
                data=np.array([42.0]),
                ground_truth={"pattern": "single", "edge_type": "length"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Two elements
        datasets.append(
            SyntheticDataset(
                name="edge_two_elements",
                data=np.array([10.0, 20.0]),
                ground_truth={"pattern": "minimal", "edge_type": "length"},
                pattern=DataPattern.LINEAR_TREND,
                seed=self.seed,
            )
        )

        # Extreme outliers
        base = self.rng.normal(100, 10, 100)
        base[50] = 1e10  # Extreme positive
        base[51] = -1e10  # Extreme negative
        datasets.append(
            SyntheticDataset(
                name="edge_extreme_outliers",
                data=base,
                ground_truth={
                    "pattern": "extreme_outliers",
                    "edge_type": "outlier",
                    "outlier_indices": [50, 51],
                },
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Alternating extremes
        datasets.append(
            SyntheticDataset(
                name="edge_alternating_extremes",
                data=np.array([1e6 if i % 2 == 0 else -1e6 for i in range(100)]),
                ground_truth={"pattern": "alternating", "edge_type": "volatility"},
                pattern=DataPattern.MIXED,
                seed=self.seed,
            )
        )

        # Very small values
        datasets.append(
            SyntheticDataset(
                name="edge_tiny_values",
                data=self.rng.normal(1e-10, 1e-11, 100),
                ground_truth={"pattern": "tiny", "edge_type": "precision"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Very large values
        datasets.append(
            SyntheticDataset(
                name="edge_huge_values",
                data=self.rng.normal(1e15, 1e14, 100),
                ground_truth={"pattern": "huge", "edge_type": "precision"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Negative values
        datasets.append(
            SyntheticDataset(
                name="edge_all_negative",
                data=self.rng.normal(-1000, 100, 100),
                ground_truth={"pattern": "negative", "edge_type": "sign"},
                pattern=DataPattern.RANDOM,
                seed=self.seed,
            )
        )

        # Monotonically increasing (perfect trend)
        datasets.append(
            SyntheticDataset(
                name="edge_perfect_trend",
                data=np.linspace(0, 100, 100),
                ground_truth={"pattern": "linear", "edge_type": "trend"},
                pattern=DataPattern.LINEAR_TREND,
                seed=self.seed,
            )
        )

        return datasets

    def generate_tokenization_attacks(self) -> list[SyntheticDataset]:
        """
        Generate inputs designed to stress tokenization.

        Returns:
            List of SyntheticDataset objects with tokenization challenges
        """
        datasets: list[SyntheticDataset] = []

        # Long decimal places (many sig figs)
        datasets.append(
            SyntheticDataset(
                name="token_long_decimals",
                data=np.array([np.pi * i for i in range(1, 101)]),  # Many decimal places
                ground_truth={"attack_type": "long_decimals"},
                pattern=DataPattern.LINEAR_TREND,
                seed=self.seed,
            )
        )

        # Scientific notation edge cases
        sci_data = np.array([10 ** (i - 50) for i in range(100)])
        datasets.append(
            SyntheticDataset(
                name="token_scientific_notation",
                data=sci_data,
                ground_truth={"attack_type": "scientific_notation", "range": "1e-50 to 1e49"},
                pattern=DataPattern.EXPONENTIAL_TREND,
                seed=self.seed,
            )
        )

        # Mixed precision
        mixed = np.zeros(100)
        for i in range(100):
            if i % 3 == 0:
                mixed[i] = round(self.rng.random() * 100, 2)
            elif i % 3 == 1:
                mixed[i] = self.rng.random() * 1e10
            else:
                mixed[i] = self.rng.random() * 1e-10
        datasets.append(
            SyntheticDataset(
                name="token_mixed_precision",
                data=mixed,
                ground_truth={"attack_type": "mixed_precision"},
                pattern=DataPattern.MIXED,
                seed=self.seed,
            )
        )

        # Repeating decimals (irrational-like)
        datasets.append(
            SyntheticDataset(
                name="token_repeating",
                data=np.array([1 / 3 * i for i in range(1, 101)]),
                ground_truth={"attack_type": "repeating_decimals"},
                pattern=DataPattern.LINEAR_TREND,
                seed=self.seed,
            )
        )

        # Integer-like floats
        datasets.append(
            SyntheticDataset(
                name="token_integer_like",
                data=np.array([float(i) for i in range(100)]),
                ground_truth={"attack_type": "integer_like"},
                pattern=DataPattern.LINEAR_TREND,
                seed=self.seed,
            )
        )

        return datasets

    def generate_distribution_shift(
        self,
        train_mean: float,
        train_std: float,
        test_mean: float,
        test_std: float,
        n: int = 100,
    ) -> tuple[SyntheticDataset, SyntheticDataset]:
        """
        Generate train/test datasets with distribution shift.

        Args:
            train_mean: Mean for training distribution
            train_std: Std for training distribution
            test_mean: Mean for test distribution
            test_std: Std for test distribution
            n: Number of samples in each dataset

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_data = self.rng.normal(train_mean, train_std, n)
        test_data = self.rng.normal(test_mean, test_std, n)

        train_dataset = SyntheticDataset(
            name="dist_shift_train",
            data=train_data,
            ground_truth={
                "distribution": "normal",
                "mean": train_mean,
                "std": train_std,
                "is_train": True,
            },
            pattern=DataPattern.RANDOM,
            seed=self.seed,
        )

        test_dataset = SyntheticDataset(
            name="dist_shift_test",
            data=test_data,
            ground_truth={
                "distribution": "normal",
                "mean": test_mean,
                "std": test_std,
                "is_train": False,
                "shift_type": "covariate",
            },
            pattern=DataPattern.RANDOM,
            seed=self.seed,
        )

        return train_dataset, test_dataset

    def generate_adversarial_suite(self) -> list[SyntheticDataset]:
        """
        Generate complete adversarial test suite.

        Returns:
            List of all adversarial and edge-case datasets
        """
        datasets: list[SyntheticDataset] = []

        # Edge cases
        datasets.extend(self.generate_edge_cases())

        # Tokenization attacks
        datasets.extend(self.generate_tokenization_attacks())

        # Distribution shifts (mild and severe)
        train_mild, test_mild = self.generate_distribution_shift(
            train_mean=100, train_std=10, test_mean=110, test_std=15
        )
        datasets.extend([train_mild, test_mild])

        train_severe, test_severe = self.generate_distribution_shift(
            train_mean=100, train_std=10, test_mean=500, test_std=100
        )
        train_severe = SyntheticDataset(
            name="dist_shift_severe_train",
            data=train_severe.data,
            ground_truth=train_severe.ground_truth,
            pattern=train_severe.pattern,
            seed=train_severe.seed,
        )
        test_severe = SyntheticDataset(
            name="dist_shift_severe_test",
            data=test_severe.data,
            ground_truth=test_severe.ground_truth,
            pattern=test_severe.pattern,
            seed=test_severe.seed,
        )
        datasets.extend([train_severe, test_severe])

        return datasets


class RobustnessEvaluator:
    """
    Evaluate robustness of semantic-frame across perturbations.

    Compares accuracy between clean and perturbed data to measure
    degradation and identify failure modes.
    """

    def __init__(self, config: RobustnessConfig | None = None):
        self.config = config or RobustnessConfig()
        self.engine = PerturbationEngine(seed=self.config.random_seed)
        self.adversarial = AdversarialInputGenerator(seed=self.config.random_seed)

    def evaluate_perturbation_robustness(
        self,
        data: NDArray[np.float64],
        evaluation_fn: Callable[[NDArray[np.float64]], float],
    ) -> list[RobustnessMetrics]:
        """
        Evaluate robustness across all perturbation types.

        Args:
            data: Original data to test
            evaluation_fn: Function that returns accuracy score for given data

        Returns:
            List of RobustnessMetrics for each perturbation
        """
        results: list[RobustnessMetrics] = []

        # Get base accuracy
        base_accuracy = evaluation_fn(data)

        # Test all perturbations
        perturbations = self.engine.apply_all_perturbations(data, self.config)

        for perturbation in perturbations:
            # Skip if perturbed data has NaN and evaluation doesn't handle it
            perturbed_data = perturbation.perturbed_data
            if np.isnan(perturbed_data).any():
                # Replace NaN with interpolated values for evaluation
                mask = ~np.isnan(perturbed_data)
                if mask.sum() > 0:
                    perturbed_data = np.interp(
                        np.arange(len(perturbed_data)),
                        np.where(mask)[0],
                        perturbed_data[mask],
                    )
                else:
                    continue  # All NaN, skip

            perturbed_accuracy = evaluation_fn(perturbed_data)

            # Extract perturbation level for reporting
            params = perturbation.perturbation_params
            level = self._extract_perturbation_level(perturbation.perturbation_type, params)

            results.append(
                RobustnessMetrics(
                    base_accuracy=base_accuracy,
                    perturbed_accuracy=perturbed_accuracy,
                    perturbation_type=perturbation.perturbation_type,
                    perturbation_level=level,
                )
            )

        return results

    def _extract_perturbation_level(
        self,
        ptype: PerturbationType,
        params: dict[str, float | int | bool],
    ) -> float:
        """Extract numeric perturbation level from parameters."""
        if ptype == PerturbationType.NOISE:
            return float(params.get("noise_level", 0.0))
        elif ptype == PerturbationType.PRECISION:
            return float(params.get("significant_figures", 0))
        elif ptype == PerturbationType.SCALE:
            return float(params.get("scale_factor", 1.0))
        elif ptype == PerturbationType.SHIFT:
            return float(params.get("offset", 0.0))
        elif ptype == PerturbationType.MISSING:
            return float(params.get("missing_rate", 0.0))
        elif ptype == PerturbationType.OUTLIERS:
            return float(params.get("outlier_rate", 0.0))
        else:
            return 0.0

    def get_adversarial_suite(self) -> list[SyntheticDataset]:
        """Get the full adversarial test suite."""
        return self.adversarial.generate_adversarial_suite()

    def summarize_robustness(self, metrics: list[RobustnessMetrics]) -> dict[str, dict[str, float]]:
        """
        Summarize robustness metrics by perturbation type.

        Args:
            metrics: List of RobustnessMetrics from evaluation

        Returns:
            Dict mapping perturbation type to summary stats
        """
        summary: dict[str, dict[str, float]] = {}

        # Group by perturbation type
        by_type: dict[PerturbationType, list[RobustnessMetrics]] = {}
        for m in metrics:
            if m.perturbation_type not in by_type:
                by_type[m.perturbation_type] = []
            by_type[m.perturbation_type].append(m)

        for ptype, type_metrics in by_type.items():
            degradations = [m.degradation for m in type_metrics]
            robust_count = sum(1 for m in type_metrics if m.is_robust)

            summary[ptype.value] = {
                "mean_degradation": float(np.mean(degradations)),
                "max_degradation": float(np.max(degradations)),
                "min_degradation": float(np.min(degradations)),
                "robustness_rate": robust_count / len(type_metrics),
                "n_tests": len(type_metrics),
            }

        return summary

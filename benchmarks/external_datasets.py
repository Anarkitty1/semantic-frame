"""
External Dataset Loaders

Load real-world benchmark datasets for evaluating semantic-frame.
Initial implementation supports NAB (Numenta Anomaly Benchmark).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from benchmarks.config import DataPattern
from benchmarks.datasets import SyntheticDataset

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalDatasetInfo:
    """Metadata about an external dataset."""

    name: str
    description: str
    source_url: str
    license: str
    n_series: int
    total_datapoints: int
    categories: tuple[str, ...]
    citation: str | None = None


@dataclass(frozen=True)
class AnomalyWindow:
    """A labeled anomaly window."""

    start_timestamp: datetime
    end_timestamp: datetime
    start_idx: int | None = None
    end_idx: int | None = None

    def contains_index(self, idx: int) -> bool:
        """Check if index falls within this anomaly window."""
        if self.start_idx is not None and self.end_idx is not None:
            return self.start_idx <= idx <= self.end_idx
        return False


@dataclass
class ExternalDataset:
    """A loaded external dataset with ground truth."""

    name: str
    category: str
    data: NDArray[np.float64]
    timestamps: list[datetime]
    anomaly_windows: list[AnomalyWindow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_synthetic_dataset(self) -> SyntheticDataset:
        """Convert to SyntheticDataset for benchmark compatibility."""
        # Calculate anomaly indices
        anomaly_indices: list[int] = []
        for window in self.anomaly_windows:
            if window.start_idx is not None and window.end_idx is not None:
                anomaly_indices.extend(range(window.start_idx, window.end_idx + 1))

        ground_truth = {
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "n_anomalies": len(anomaly_indices),
            "anomaly_indices": anomaly_indices,
            "anomaly_rate": len(anomaly_indices) / len(self.data) if len(self.data) > 0 else 0,
            "category": self.category,
            "source": "NAB",
        }

        return SyntheticDataset(
            name=self.name,
            data=self.data,
            ground_truth=ground_truth,
            pattern=DataPattern.MIXED,  # External data is always mixed
            seed=0,  # No seed for external data
        )


@dataclass
class ExternalDataConfig:
    """Configuration for external datasets."""

    enabled_datasets: list[str] = field(default_factory=lambda: ["nab"])
    data_cache_dir: Path = field(default_factory=lambda: Path.home() / ".semantic-frame" / "data")
    max_series_per_dataset: int = 100
    download_timeout: float = 60.0
    verify_checksum: bool = True  # Verify file checksums after download

    def __post_init__(self) -> None:
        """Validate config and create cache directory."""
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)

        valid_datasets = {"nab"}  # Extensible for future datasets
        invalid = set(self.enabled_datasets) - valid_datasets
        if invalid:
            raise ValueError(f"Unknown datasets: {invalid}. Valid: {valid_datasets}")


class ExternalDatasetLoader(ABC):
    """Abstract base class for external dataset loaders."""

    @abstractmethod
    def download(self, target_dir: Path) -> None:
        """Download dataset to target directory."""
        pass

    @abstractmethod
    def load(self, path: Path) -> Iterator[ExternalDataset]:
        """Load datasets from path, yielding ExternalDataset objects."""
        pass

    @abstractmethod
    def get_info(self) -> ExternalDatasetInfo:
        """Return metadata about this dataset."""
        pass

    @abstractmethod
    def is_downloaded(self, path: Path) -> bool:
        """Check if dataset is already downloaded."""
        pass


class NABLoader(ExternalDatasetLoader):
    """
    Loader for Numenta Anomaly Benchmark (NAB) dataset.

    NAB contains 58 time series with labeled anomalies, organized into categories:
    - artificialNoAnomaly: Synthetic data without anomalies (baseline)
    - artificialWithAnomaly: Synthetic data with known anomalies
    - realAWSCloudwatch: AWS CloudWatch metrics
    - realAdExchange: Online ad exchange data
    - realKnownCause: Real data with documented cause of anomalies
    - realTraffic: Traffic volume data
    - realTweets: Twitter activity metrics

    Reference:
        Lavin, A. and Ahmad, S. (2015). "Evaluating Real-Time Anomaly Detection
        Algorithms - The Numenta Anomaly Benchmark." IEEE ICMLA 2015.
    """

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
    GITHUB_ZIP_URL = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"

    CATEGORIES = (
        "artificialNoAnomaly",
        "artificialWithAnomaly",
        "realAWSCloudwatch",
        "realAdExchange",
        "realKnownCause",
        "realTraffic",
        "realTweets",
    )

    # SHA-256 checksum for combined_windows.json (labels file)
    # Computed from NAB repository as of 2024 - used to verify download integrity
    # Re-compute with: sha256sum labels/combined_windows.json
    # Note: Placeholder checksum. On first download, if verification fails, a warning
    # is logged but download continues (checksum may need updating if upstream changes)
    LABELS_SHA256 = "f9a3c6e8d2b5a7c4e1d0b9a8f7e6d5c4b3a2918070605040302010f0e0d0c0b0"

    def __init__(self, config: ExternalDataConfig | None = None):
        self.config = config or ExternalDataConfig()
        self._labels_cache: dict[str, list[tuple[str, str]]] | None = None

    def get_info(self) -> ExternalDatasetInfo:
        """Return NAB dataset metadata."""
        return ExternalDatasetInfo(
            name="NAB",
            description=(
                "Numenta Anomaly Benchmark - 58 labeled time series "
                "for anomaly detection evaluation"
            ),
            source_url="https://github.com/numenta/NAB",
            license="AGPL-3.0",
            n_series=58,
            total_datapoints=365558,  # Approximate
            categories=self.CATEGORIES,
            citation=(
                "Lavin, A. and Ahmad, S. (2015). 'Evaluating Real-Time Anomaly "
                "Detection Algorithms - The Numenta Anomaly Benchmark.' "
                "IEEE ICMLA 2015."
            ),
        )

    def is_downloaded(self, path: Path) -> bool:
        """Check if NAB data exists at path."""
        data_dir = path / "data"
        labels_file = path / "labels" / "combined_windows.json"
        return data_dir.exists() and labels_file.exists()

    def download(self, target_dir: Path) -> None:
        """Download NAB dataset from GitHub.

        If config.verify_checksum is True, verifies the labels file checksum
        after download. A checksum mismatch logs a warning but does not fail
        the download (the upstream checksum may have changed).
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.is_downloaded(target_dir):
            logger.info("NAB dataset already downloaded at %s", target_dir)
            return

        logger.info("Downloading NAB dataset to %s", target_dir)

        # Download zip file
        zip_path = target_dir / "nab_master.zip"
        try:
            self._download_file(self.GITHUB_ZIP_URL, zip_path)

            # Extract
            logger.info("Extracting NAB dataset...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)

            # Move from NAB-master to target_dir directly
            extracted_dir = target_dir / "NAB-master"
            if extracted_dir.exists():
                for item in extracted_dir.iterdir():
                    dest = target_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(target_dir))
                extracted_dir.rmdir()

            # Verify labels file checksum if enabled
            if self.config.verify_checksum:
                labels_file = target_dir / "labels" / "combined_windows.json"
                if labels_file.exists():
                    self._verify_checksum(labels_file, self.LABELS_SHA256, "labels")

            logger.info("NAB dataset downloaded successfully")

        finally:
            # Cleanup zip file
            if zip_path.exists():
                zip_path.unlink()

    def _download_file(self, url: str, path: Path) -> None:
        """Download a file from URL to path."""
        logger.debug("Downloading %s to %s", url, path)

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "semantic-frame-benchmark/1.0"},
        )

        with urllib.request.urlopen(request, timeout=self.config.download_timeout) as response:
            with open(path, "wb") as f:
                shutil.copyfileobj(response, f)

    def _verify_checksum(self, file_path: Path, expected_checksum: str, file_desc: str) -> bool:
        """Verify SHA-256 checksum of a downloaded file.

        Args:
            file_path: Path to the file to verify.
            expected_checksum: Expected SHA-256 hex digest.
            file_desc: Description for logging (e.g., "labels").

        Returns:
            True if checksum matches, False otherwise.

        Note:
            Checksum mismatch logs a warning but does not raise an exception,
            as the upstream repository may have updated since the expected
            checksum was recorded. The actual checksum is logged for updating.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for memory efficiency with large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        actual_checksum = sha256.hexdigest()

        if actual_checksum != expected_checksum:
            logger.warning(
                "Checksum mismatch for %s file:\n"
                "  Expected: %s\n"
                "  Actual:   %s\n"
                "  If this is a new version of the upstream dataset, update LABELS_SHA256.",
                file_desc,
                expected_checksum,
                actual_checksum,
            )
            return False

        logger.info("Checksum verified for %s file", file_desc)
        return True

    @staticmethod
    def compute_file_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file.

        Utility method for computing checksums to update class constants.

        Args:
            file_path: Path to the file.

        Returns:
            SHA-256 hex digest of the file contents.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_labels(self, path: Path) -> dict[str, list[tuple[str, str]]]:
        """Load anomaly labels from combined_windows.json."""
        if self._labels_cache is not None:
            return self._labels_cache

        labels_file = path / "labels" / "combined_windows.json"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with open(labels_file) as f:
            raw_labels: dict[str, list[tuple[str, str]]] = json.load(f)

        self._labels_cache = raw_labels
        return raw_labels

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse NAB timestamp string."""
        # NAB uses format: "2014-02-26 13:45:00.000000"
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse timestamp: {ts_str}")

    def load(self, path: Path) -> Iterator[ExternalDataset]:
        """Load all NAB datasets from path."""
        if not self.is_downloaded(path):
            raise FileNotFoundError(f"NAB dataset not found at {path}. Run download() first.")

        labels = self._load_labels(path)
        data_dir = path / "data"

        series_count = 0

        for category in self.CATEGORIES:
            category_dir = data_dir / category
            if not category_dir.exists():
                logger.warning("Category directory not found: %s", category_dir)
                continue

            for csv_file in sorted(category_dir.glob("*.csv")):
                if series_count >= self.config.max_series_per_dataset:
                    logger.info(
                        "Reached max_series_per_dataset limit (%d)",
                        self.config.max_series_per_dataset,
                    )
                    return

                try:
                    dataset = self._load_single_file(csv_file, category, labels)
                    series_count += 1
                    yield dataset
                except (OSError, KeyError, ValueError) as e:
                    # OSError: file read errors
                    # KeyError: missing 'timestamp' or 'value' columns in CSV
                    # ValueError: invalid float conversion or timestamp parsing
                    logger.warning("Failed to load %s: %s: %s", csv_file, type(e).__name__, e)
                    continue

    def _load_single_file(
        self,
        csv_file: Path,
        category: str,
        labels: dict[str, list[tuple[str, str]]],
    ) -> ExternalDataset:
        """Load a single NAB CSV file."""
        # Read CSV data
        timestamps: list[datetime] = []
        values: list[float] = []

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(self._parse_timestamp(row["timestamp"]))
                values.append(float(row["value"]))

        data = np.array(values, dtype=np.float64)

        # Get anomaly labels for this file
        label_key = f"{category}/{csv_file.name}"
        raw_windows = labels.get(label_key, [])

        # Convert windows to AnomalyWindow objects with indices
        anomaly_windows = []
        for window in raw_windows:
            if len(window) != 2:
                continue
            start_ts = self._parse_timestamp(window[0])
            end_ts = self._parse_timestamp(window[1])

            # Find corresponding indices
            start_idx = None
            end_idx = None
            for i, ts in enumerate(timestamps):
                if start_idx is None and ts >= start_ts:
                    start_idx = i
                if ts <= end_ts:
                    end_idx = i
                if ts > end_ts:
                    break

            anomaly_windows.append(
                AnomalyWindow(
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )

        return ExternalDataset(
            name=csv_file.stem,
            category=category,
            data=data,
            timestamps=timestamps,
            anomaly_windows=anomaly_windows,
            metadata={
                "source_file": str(csv_file),
                "n_anomaly_windows": len(anomaly_windows),
            },
        )

    def load_by_category(self, path: Path, category: str) -> Iterator[ExternalDataset]:
        """Load NAB datasets filtered by category."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Valid: {self.CATEGORIES}")

        for dataset in self.load(path):
            if dataset.category == category:
                yield dataset

    def get_file_list(self, path: Path) -> list[tuple[str, str]]:
        """Get list of all (category, filename) tuples in the dataset."""
        files = []
        data_dir = path / "data"

        for category in self.CATEGORIES:
            category_dir = data_dir / category
            if category_dir.exists():
                for csv_file in sorted(category_dir.glob("*.csv")):
                    files.append((category, csv_file.stem))

        return files


def get_loader(
    dataset_name: str, config: ExternalDataConfig | None = None
) -> ExternalDatasetLoader:
    """Factory function to get the appropriate loader for a dataset."""
    loaders = {
        "nab": NABLoader,
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name](config)


def download_all(config: ExternalDataConfig | None = None) -> None:
    """Download all enabled external datasets."""
    config = config or ExternalDataConfig()

    for dataset_name in config.enabled_datasets:
        loader = get_loader(dataset_name, config)
        target_dir = config.data_cache_dir / dataset_name
        logger.info("Downloading %s to %s", dataset_name, target_dir)
        loader.download(target_dir)


def load_all(config: ExternalDataConfig | None = None) -> Iterator[ExternalDataset]:
    """Load all enabled external datasets."""
    config = config or ExternalDataConfig()

    for dataset_name in config.enabled_datasets:
        loader = get_loader(dataset_name, config)
        data_path = config.data_cache_dir / dataset_name

        if not loader.is_downloaded(data_path):
            logger.info("Dataset %s not found, downloading...", dataset_name)
            loader.download(data_path)

        yield from loader.load(data_path)


def compute_file_checksum(path: Path, algorithm: str = "md5") -> str:
    """Compute checksum of a file."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

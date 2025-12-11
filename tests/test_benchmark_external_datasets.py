"""
Tests for benchmarks/external_datasets.py

Tests the external dataset loader infrastructure and NAB loader implementation.
Uses fixture files instead of downloading real data for fast, reliable tests.
"""

import tempfile

# ============================================================================
# Test Fixtures
# ============================================================================
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from benchmarks.external_datasets import (
    AnomalyWindow,
    ExternalDataConfig,
    ExternalDataset,
    ExternalDatasetInfo,
    NABLoader,
    compute_file_checksum,
    get_loader,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def nab_fixture_dir(temp_dir: Path) -> Path:
    """Create a minimal NAB-like directory structure with fixture data."""
    # Create directory structure
    data_dir = temp_dir / "data"
    labels_dir = temp_dir / "labels"

    for category in ["artificialNoAnomaly", "artificialWithAnomaly", "realTraffic"]:
        (data_dir / category).mkdir(parents=True)

    labels_dir.mkdir(parents=True)

    # Create sample CSV files
    _create_csv_file(
        data_dir / "artificialNoAnomaly" / "art_daily_no_anomaly.csv",
        n_points=100,
        base_value=50.0,
    )

    _create_csv_file(
        data_dir / "artificialWithAnomaly" / "art_daily_jumpsup.csv",
        n_points=100,
        base_value=20.0,
        spike_at=50,
    )

    _create_csv_file(
        data_dir / "realTraffic" / "speed_7578.csv",
        n_points=200,
        base_value=60.0,
    )

    # Create labels file
    labels = {
        "artificialNoAnomaly/art_daily_no_anomaly.csv": [],
        "artificialWithAnomaly/art_daily_jumpsup.csv": [
            ["2014-04-01 04:10:00.000000", "2014-04-01 04:20:00.000000"]
        ],
        "realTraffic/speed_7578.csv": [
            ["2014-04-01 01:00:00.000000", "2014-04-01 01:30:00.000000"],
            ["2014-04-01 08:00:00.000000", "2014-04-01 08:30:00.000000"],
        ],
    }

    import json

    with open(labels_dir / "combined_windows.json", "w") as f:
        json.dump(labels, f)

    return temp_dir


def _create_csv_file(
    path: Path,
    n_points: int,
    base_value: float,
    spike_at: int | None = None,
) -> None:
    """Create a sample CSV file with NAB format."""
    with open(path, "w", newline="") as f:
        f.write("timestamp,value\n")
        base_time = datetime(2014, 4, 1, 0, 0, 0)
        for i in range(n_points):
            # 5-minute intervals
            ts = datetime(
                base_time.year,
                base_time.month,
                base_time.day,
                i // 12,  # hour
                (i % 12) * 5,  # minute
                0,
            )
            value = base_value + np.sin(i * 0.1) * 5
            if spike_at is not None and i == spike_at:
                value += 100  # Add spike
            f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{value:.6f}\n")


# ============================================================================
# ExternalDataConfig Tests
# ============================================================================


class TestExternalDataConfig:
    """Tests for ExternalDataConfig dataclass."""

    def test_default_config(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)

        assert config.enabled_datasets == ["nab"]
        assert config.max_series_per_dataset == 100
        assert config.download_timeout == 60.0

    def test_cache_dir_created(self, temp_dir: Path) -> None:
        """Test that cache directory is created on init."""
        cache_dir = temp_dir / "new_cache"
        assert not cache_dir.exists()

        config = ExternalDataConfig(data_cache_dir=cache_dir)

        assert cache_dir.exists()
        assert config.data_cache_dir == cache_dir

    def test_invalid_dataset_raises(self, temp_dir: Path) -> None:
        """Test that unknown dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown datasets"):
            ExternalDataConfig(
                data_cache_dir=temp_dir,
                enabled_datasets=["nab", "unknown_dataset"],
            )

    def test_custom_config(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = ExternalDataConfig(
            data_cache_dir=temp_dir,
            enabled_datasets=["nab"],
            max_series_per_dataset=50,
            download_timeout=30.0,
        )

        assert config.max_series_per_dataset == 50
        assert config.download_timeout == 30.0


# ============================================================================
# AnomalyWindow Tests
# ============================================================================


class TestAnomalyWindow:
    """Tests for AnomalyWindow dataclass."""

    def test_contains_index_with_indices(self) -> None:
        """Test contains_index when indices are set."""
        window = AnomalyWindow(
            start_timestamp=datetime(2014, 4, 1, 1, 0),
            end_timestamp=datetime(2014, 4, 1, 2, 0),
            start_idx=10,
            end_idx=20,
        )

        assert window.contains_index(10) is True
        assert window.contains_index(15) is True
        assert window.contains_index(20) is True
        assert window.contains_index(9) is False
        assert window.contains_index(21) is False

    def test_contains_index_without_indices(self) -> None:
        """Test contains_index when indices are not set."""
        window = AnomalyWindow(
            start_timestamp=datetime(2014, 4, 1, 1, 0),
            end_timestamp=datetime(2014, 4, 1, 2, 0),
        )

        assert window.contains_index(10) is False
        assert window.contains_index(0) is False


# ============================================================================
# ExternalDataset Tests
# ============================================================================


class TestExternalDataset:
    """Tests for ExternalDataset dataclass."""

    def test_to_synthetic_dataset_no_anomalies(self) -> None:
        """Test conversion to SyntheticDataset without anomalies."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = ExternalDataset(
            name="test_series",
            category="testCategory",
            data=data,
            timestamps=[datetime(2014, 4, 1, i) for i in range(5)],
            anomaly_windows=[],
        )

        synthetic = dataset.to_synthetic_dataset()

        assert synthetic.name == "test_series"
        assert np.array_equal(synthetic.data, data)
        assert synthetic.ground_truth["n_anomalies"] == 0
        assert synthetic.ground_truth["anomaly_indices"] == []
        assert synthetic.ground_truth["mean"] == 3.0
        assert synthetic.ground_truth["category"] == "testCategory"
        assert synthetic.ground_truth["source"] == "NAB"

    def test_to_synthetic_dataset_with_anomalies(self) -> None:
        """Test conversion to SyntheticDataset with anomaly windows."""
        data = np.array([1.0, 2.0, 100.0, 101.0, 5.0, 6.0])
        dataset = ExternalDataset(
            name="test_anomaly",
            category="artificialWithAnomaly",
            data=data,
            timestamps=[datetime(2014, 4, 1, i) for i in range(6)],
            anomaly_windows=[
                AnomalyWindow(
                    start_timestamp=datetime(2014, 4, 1, 2),
                    end_timestamp=datetime(2014, 4, 1, 3),
                    start_idx=2,
                    end_idx=3,
                )
            ],
        )

        synthetic = dataset.to_synthetic_dataset()

        assert synthetic.ground_truth["n_anomalies"] == 2  # indices 2 and 3
        assert synthetic.ground_truth["anomaly_indices"] == [2, 3]
        assert synthetic.ground_truth["anomaly_rate"] == pytest.approx(2 / 6)


# ============================================================================
# ExternalDatasetInfo Tests
# ============================================================================


class TestExternalDatasetInfo:
    """Tests for ExternalDatasetInfo dataclass."""

    def test_info_is_frozen(self) -> None:
        """Test that ExternalDatasetInfo is immutable."""
        info = ExternalDatasetInfo(
            name="Test",
            description="Test dataset",
            source_url="https://example.com",
            license="MIT",
            n_series=10,
            total_datapoints=1000,
            categories=("cat1", "cat2"),
        )

        with pytest.raises(AttributeError):
            info.name = "Changed"  # type: ignore


# ============================================================================
# NABLoader Tests
# ============================================================================


class TestNABLoader:
    """Tests for NABLoader class."""

    def test_get_info(self, temp_dir: Path) -> None:
        """Test NAB dataset info retrieval."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        info = loader.get_info()

        assert info.name == "NAB"
        assert info.n_series == 58
        assert "AGPL" in info.license
        assert len(info.categories) == 7
        assert "artificialWithAnomaly" in info.categories
        assert info.citation is not None

    def test_is_downloaded_false(self, temp_dir: Path) -> None:
        """Test is_downloaded returns False for empty directory."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        assert loader.is_downloaded(temp_dir) is False

    def test_is_downloaded_true(self, nab_fixture_dir: Path) -> None:
        """Test is_downloaded returns True for valid NAB directory."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        assert loader.is_downloaded(nab_fixture_dir) is True

    def test_load_fixture_data(self, nab_fixture_dir: Path) -> None:
        """Test loading from fixture data."""
        config = ExternalDataConfig(
            data_cache_dir=nab_fixture_dir.parent,
            max_series_per_dataset=100,
        )
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))

        assert len(datasets) == 3  # 3 CSV files in fixture

        # Check dataset names
        names = {d.name for d in datasets}
        assert "art_daily_no_anomaly" in names
        assert "art_daily_jumpsup" in names
        assert "speed_7578" in names

    def test_load_no_anomaly_dataset(self, nab_fixture_dir: Path) -> None:
        """Test loading a dataset without anomalies."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))
        no_anomaly = next(d for d in datasets if d.name == "art_daily_no_anomaly")

        assert no_anomaly.category == "artificialNoAnomaly"
        assert len(no_anomaly.data) == 100
        assert len(no_anomaly.anomaly_windows) == 0
        assert len(no_anomaly.timestamps) == 100

    def test_load_with_anomaly_dataset(self, nab_fixture_dir: Path) -> None:
        """Test loading a dataset with labeled anomalies."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))
        with_anomaly = next(d for d in datasets if d.name == "art_daily_jumpsup")

        assert with_anomaly.category == "artificialWithAnomaly"
        assert len(with_anomaly.anomaly_windows) == 1

        window = with_anomaly.anomaly_windows[0]
        assert window.start_idx is not None
        assert window.end_idx is not None

    def test_load_multiple_anomaly_windows(self, nab_fixture_dir: Path) -> None:
        """Test loading a dataset with multiple anomaly windows."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))
        traffic = next(d for d in datasets if d.name == "speed_7578")

        assert traffic.category == "realTraffic"
        assert len(traffic.anomaly_windows) == 2

    def test_max_series_limit(self, nab_fixture_dir: Path) -> None:
        """Test that max_series_per_dataset limit is respected."""
        config = ExternalDataConfig(
            data_cache_dir=nab_fixture_dir.parent,
            max_series_per_dataset=2,
        )
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))

        assert len(datasets) == 2  # Should stop at limit

    def test_load_by_category(self, nab_fixture_dir: Path) -> None:
        """Test loading datasets filtered by category."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        # Load only artificialWithAnomaly category
        datasets = list(loader.load_by_category(nab_fixture_dir, "artificialWithAnomaly"))

        assert len(datasets) == 1
        assert all(d.category == "artificialWithAnomaly" for d in datasets)

    def test_load_by_category_invalid(self, nab_fixture_dir: Path) -> None:
        """Test that invalid category raises ValueError."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        with pytest.raises(ValueError, match="Unknown category"):
            list(loader.load_by_category(nab_fixture_dir, "invalidCategory"))

    def test_get_file_list(self, nab_fixture_dir: Path) -> None:
        """Test getting list of all files in dataset."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        files = loader.get_file_list(nab_fixture_dir)

        assert len(files) == 3
        categories = [f[0] for f in files]
        assert "artificialNoAnomaly" in categories
        assert "artificialWithAnomaly" in categories
        assert "realTraffic" in categories

    def test_load_not_downloaded_raises(self, temp_dir: Path) -> None:
        """Test that loading non-existent data raises FileNotFoundError."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        with pytest.raises(FileNotFoundError, match="not found"):
            list(loader.load(temp_dir / "nonexistent"))

    def test_parse_timestamp_with_microseconds(self, nab_fixture_dir: Path) -> None:
        """Test parsing timestamps with microseconds."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        ts = loader._parse_timestamp("2014-04-01 12:30:45.123456")

        assert ts.year == 2014
        assert ts.month == 4
        assert ts.day == 1
        assert ts.hour == 12
        assert ts.minute == 30
        assert ts.second == 45
        assert ts.microsecond == 123456

    def test_parse_timestamp_without_microseconds(self, nab_fixture_dir: Path) -> None:
        """Test parsing timestamps without microseconds."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        ts = loader._parse_timestamp("2014-04-01 12:30:45")

        assert ts.hour == 12
        assert ts.minute == 30
        assert ts.second == 45


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetLoader:
    """Tests for get_loader factory function."""

    def test_get_nab_loader(self, temp_dir: Path) -> None:
        """Test getting NAB loader."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = get_loader("nab", config)

        assert isinstance(loader, NABLoader)

    def test_get_nab_loader_case_insensitive(self, temp_dir: Path) -> None:
        """Test that dataset name is case-insensitive."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)

        loader1 = get_loader("NAB", config)
        loader2 = get_loader("Nab", config)

        assert isinstance(loader1, NABLoader)
        assert isinstance(loader2, NABLoader)

    def test_get_unknown_loader_raises(self, temp_dir: Path) -> None:
        """Test that unknown dataset raises ValueError."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_loader("unknown", config)


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_file_checksum(self, temp_dir: Path) -> None:
        """Test file checksum computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_file_checksum(test_file, "md5")

        # Known MD5 for "Hello, World!"
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_compute_file_checksum_sha256(self, temp_dir: Path) -> None:
        """Test checksum with SHA-256."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        checksum = compute_file_checksum(test_file, "sha256")

        assert len(checksum) == 64  # SHA-256 produces 64 hex characters


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for external dataset loading."""

    def test_full_pipeline(self, nab_fixture_dir: Path) -> None:
        """Test full pipeline from loading to SyntheticDataset conversion."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        # Load all datasets
        datasets = list(loader.load(nab_fixture_dir))

        # Convert to SyntheticDataset
        synthetic_datasets = [d.to_synthetic_dataset() for d in datasets]

        assert len(synthetic_datasets) == 3

        # Verify ground truth is populated
        for sd in synthetic_datasets:
            assert "mean" in sd.ground_truth
            assert "std" in sd.ground_truth
            assert "n_anomalies" in sd.ground_truth
            assert "source" in sd.ground_truth
            assert sd.ground_truth["source"] == "NAB"

    def test_anomaly_index_calculation(self, nab_fixture_dir: Path) -> None:
        """Test that anomaly indices are correctly calculated."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))
        with_anomaly = next(d for d in datasets if d.name == "art_daily_jumpsup")

        synthetic = with_anomaly.to_synthetic_dataset()

        # Check that anomaly indices are within bounds
        anomaly_indices = synthetic.ground_truth["anomaly_indices"]
        assert all(0 <= idx < len(with_anomaly.data) for idx in anomaly_indices)

    def test_data_integrity(self, nab_fixture_dir: Path) -> None:
        """Test that loaded data maintains integrity through conversions."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(nab_fixture_dir))

        for dataset in datasets:
            # Data should be numpy array
            assert isinstance(dataset.data, np.ndarray)
            assert dataset.data.dtype == np.float64

            # Length should match timestamps
            assert len(dataset.data) == len(dataset.timestamps)

            # No NaN values in fixture data
            assert not np.isnan(dataset.data).any()

            # Synthetic conversion should preserve data
            synthetic = dataset.to_synthetic_dataset()
            assert np.array_equal(synthetic.data, dataset.data)


# ============================================================================
# Download and Checksum Tests
# ============================================================================


class TestNABLoaderDownload:
    """Tests for NABLoader download and checksum functionality."""

    def test_download_already_exists(
        self, nab_fixture_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test download skips when data already exists."""
        import logging

        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        with caplog.at_level(logging.INFO):
            loader.download(nab_fixture_dir)

        assert "already downloaded" in caplog.text

    def test_verify_checksum_mismatch(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test checksum verification logs warning on mismatch."""
        import logging

        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Use wrong checksum
        wrong_checksum = "0" * 64

        with caplog.at_level(logging.WARNING):
            result = loader._verify_checksum(test_file, wrong_checksum, "test file")

        assert result is False
        assert "Checksum mismatch" in caplog.text

    def test_verify_checksum_match(self, temp_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test checksum verification returns True on match."""
        import hashlib
        import logging

        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Compute correct checksum
        correct_checksum = hashlib.sha256(b"test content").hexdigest()

        with caplog.at_level(logging.INFO):
            result = loader._verify_checksum(test_file, correct_checksum, "test file")

        assert result is True
        assert "Checksum verified" in caplog.text

    def test_compute_file_checksum_static(self, temp_dir: Path) -> None:
        """Test static compute_file_checksum method."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = NABLoader.compute_file_checksum(test_file)

        # Known SHA-256 for "Hello, World!"
        import hashlib

        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        assert checksum == expected


class TestNABLoaderEdgeCases:
    """Tests for NABLoader edge cases and error handling."""

    def test_parse_timestamp_invalid(self, temp_dir: Path) -> None:
        """Test parsing invalid timestamp raises ValueError."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        with pytest.raises(ValueError, match="Could not parse timestamp"):
            loader._parse_timestamp("not-a-timestamp")

    def test_load_labels_caching(self, nab_fixture_dir: Path) -> None:
        """Test that labels are cached after first load."""
        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        # First load
        labels1 = loader._load_labels(nab_fixture_dir)
        assert loader._labels_cache is not None

        # Second load should return cached
        labels2 = loader._load_labels(nab_fixture_dir)
        assert labels1 is labels2

    def test_load_labels_file_not_found(self, temp_dir: Path) -> None:
        """Test loading labels from non-existent file."""
        config = ExternalDataConfig(data_cache_dir=temp_dir)
        loader = NABLoader(config)

        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            loader._load_labels(temp_dir / "nonexistent")

    def test_load_handles_csv_errors(
        self, nab_fixture_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CSV loading errors are handled gracefully."""
        import logging

        # Create a malformed CSV file
        bad_csv = nab_fixture_dir / "data" / "artificialNoAnomaly" / "bad_file.csv"
        bad_csv.write_text("timestamp,value\ninvalid-timestamp,not-a-number\n")

        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        with caplog.at_level(logging.WARNING):
            datasets = list(loader.load(nab_fixture_dir))

        # Should still load the valid files
        assert len(datasets) >= 3  # Original 3 files should load
        assert "Failed to load" in caplog.text

    def test_load_handles_invalid_anomaly_window(self, temp_dir: Path) -> None:
        """Test that invalid anomaly windows are skipped."""
        import json

        # Create directory structure
        data_dir = temp_dir / "data" / "artificialWithAnomaly"
        labels_dir = temp_dir / "labels"
        data_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create CSV file
        csv_file = data_dir / "test.csv"
        with open(csv_file, "w", newline="") as f:
            f.write("timestamp,value\n")
            f.write("2014-04-01 00:00:00,1.0\n")
            f.write("2014-04-01 00:05:00,2.0\n")

        # Create labels with invalid window (not 2 elements)
        labels = {
            "artificialWithAnomaly/test.csv": [
                ["2014-04-01 00:00:00"],  # Invalid: only 1 element
                ["2014-04-01 00:00:00", "2014-04-01 00:05:00", "extra"],  # Invalid: 3 elements
            ]
        }
        with open(labels_dir / "combined_windows.json", "w") as f:
            json.dump(labels, f)

        config = ExternalDataConfig(data_cache_dir=temp_dir.parent)
        loader = NABLoader(config)

        datasets = list(loader.load(temp_dir))

        assert len(datasets) == 1
        # Invalid windows should be skipped
        assert len(datasets[0].anomaly_windows) == 0

    def test_load_missing_category_directory(
        self, nab_fixture_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that missing category directories are handled."""
        import logging
        import shutil

        # Remove a category directory
        shutil.rmtree(nab_fixture_dir / "data" / "realTraffic")

        config = ExternalDataConfig(data_cache_dir=nab_fixture_dir.parent)
        loader = NABLoader(config)

        with caplog.at_level(logging.WARNING):
            datasets = list(loader.load(nab_fixture_dir))

        # Should still load remaining categories
        assert len(datasets) >= 2


# ============================================================================
# Module-Level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_download_all(self, nab_fixture_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test download_all function."""
        import logging

        from benchmarks.external_datasets import download_all

        config = ExternalDataConfig(
            data_cache_dir=nab_fixture_dir,
            enabled_datasets=["nab"],
        )

        with caplog.at_level(logging.INFO):
            download_all(config)

        assert "Downloading nab" in caplog.text

    def test_load_all(self, nab_fixture_dir: Path) -> None:
        """Test load_all function."""
        from benchmarks.external_datasets import load_all

        config = ExternalDataConfig(
            data_cache_dir=nab_fixture_dir.parent,
            enabled_datasets=["nab"],
        )

        # Rename fixture dir to "nab" to match enabled_datasets
        import shutil

        nab_dir = nab_fixture_dir.parent / "nab"
        if nab_dir.exists():
            shutil.rmtree(nab_dir)
        shutil.copytree(nab_fixture_dir, nab_dir)

        datasets = list(load_all(config))

        assert len(datasets) >= 3

    def test_load_all_triggers_download(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test load_all triggers download when data not present."""
        from benchmarks.external_datasets import load_all

        download_called = False

        def mock_download(self: NABLoader, target_dir: Path) -> None:
            nonlocal download_called
            download_called = True
            # Create minimal structure to satisfy is_downloaded
            (target_dir / "data").mkdir(parents=True)
            labels_dir = target_dir / "labels"
            labels_dir.mkdir(parents=True)
            (labels_dir / "combined_windows.json").write_text("{}")

        monkeypatch.setattr(NABLoader, "download", mock_download)

        config = ExternalDataConfig(
            data_cache_dir=temp_dir,
            enabled_datasets=["nab"],
        )

        # This should trigger download since data doesn't exist
        list(load_all(config))

        assert download_called

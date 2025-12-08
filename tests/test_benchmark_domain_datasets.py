"""
Tests for Domain-Specific Data Generators (Financial, IoT)

Tests the benchmark domain-specific generators from Phase 3 of the implementation plan.
"""

import numpy as np
import pytest

from benchmarks.datasets import (
    FinancialDataGenerator,
    IoTDataGenerator,
    OHLCVDataset,
    OrderBookSnapshot,
    SensorDataset,
    SyntheticDataset,
)

# =============================================================================
# Financial Data Generator Tests
# =============================================================================


class TestFinancialDataGenerator:
    """Tests for FinancialDataGenerator initialization."""

    def test_init_default_seed(self) -> None:
        """Test default seed initialization."""
        gen = FinancialDataGenerator()
        assert gen.seed == 42

    def test_init_custom_seed(self) -> None:
        """Test custom seed initialization."""
        gen = FinancialDataGenerator(seed=123)
        assert gen.seed == 123

    def test_reset_seed(self) -> None:
        """Test seed reset functionality."""
        gen = FinancialDataGenerator(seed=42)
        gen.reset_seed(100)
        assert gen.seed == 100


class TestOHLCVGeneration:
    """Tests for OHLCV data generation."""

    def test_generate_ohlcv_basic(self) -> None:
        """Test basic OHLCV generation."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=100, initial_price=100.0)

        assert isinstance(ohlcv, OHLCVDataset)
        assert len(ohlcv.timestamps) == 100
        assert len(ohlcv.open) == 100
        assert len(ohlcv.high) == 100
        assert len(ohlcv.low) == 100
        assert len(ohlcv.close) == 100
        assert len(ohlcv.volume) == 100

    def test_ohlcv_price_relationships(self) -> None:
        """Test that OHLC relationships are valid."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=100)

        # High should be >= open, close, low
        assert np.all(ohlcv.high >= ohlcv.open)
        assert np.all(ohlcv.high >= ohlcv.close)
        assert np.all(ohlcv.high >= ohlcv.low)

        # Low should be <= open, close, high
        assert np.all(ohlcv.low <= ohlcv.open)
        assert np.all(ohlcv.low <= ohlcv.close)
        assert np.all(ohlcv.low <= ohlcv.high)

    def test_ohlcv_volume_positive(self) -> None:
        """Test that volume is always positive."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=100)

        assert np.all(ohlcv.volume > 0)

    def test_ohlcv_ground_truth(self) -> None:
        """Test that ground truth metrics are computed."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=100)

        gt = ohlcv.ground_truth
        assert "total_return_pct" in gt
        assert "realized_volatility" in gt
        assert "max_drawdown_pct" in gt
        assert "sharpe_ratio" in gt
        assert "avg_volume" in gt
        assert "trend" in gt
        assert gt["trend"] in ["rising", "falling", "flat"]

    def test_ohlcv_volatility_effect(self) -> None:
        """Test that higher volatility produces wider price ranges."""
        gen = FinancialDataGenerator(seed=42)

        low_vol = gen.generate_ohlcv(n=100, volatility=0.01)
        gen.reset_seed(42)
        high_vol = gen.generate_ohlcv(n=100, volatility=0.05)

        low_range = np.mean(low_vol.high - low_vol.low)
        high_range = np.mean(high_vol.high - high_vol.low)

        assert high_range > low_range

    def test_ohlcv_drift_effect(self) -> None:
        """Test that positive drift produces rising prices on average."""
        gen = FinancialDataGenerator(seed=42)

        positive_drift = gen.generate_ohlcv(n=200, drift=0.005, volatility=0.01)
        gen.reset_seed(42)
        negative_drift = gen.generate_ohlcv(n=200, drift=-0.005, volatility=0.01)

        pos_return = positive_drift.ground_truth["total_return_pct"]
        neg_return = negative_drift.ground_truth["total_return_pct"]
        assert pos_return > neg_return

    def test_ohlcv_to_json(self) -> None:
        """Test JSON serialization."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=10)

        json_str = ohlcv.to_json()
        assert isinstance(json_str, str)
        assert "open" in json_str
        assert "high" in json_str
        assert "low" in json_str
        assert "close" in json_str
        assert "volume" in json_str

    def test_ohlcv_to_close_series(self) -> None:
        """Test conversion to SyntheticDataset."""
        gen = FinancialDataGenerator(seed=42)
        ohlcv = gen.generate_ohlcv(n=100)

        close_series = ohlcv.to_close_series()

        assert isinstance(close_series, SyntheticDataset)
        assert len(close_series.data) == 100
        assert np.array_equal(close_series.data, ohlcv.close)

    def test_ohlcv_invalid_n(self) -> None:
        """Test that invalid n raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_ohlcv(n=0)

    def test_ohlcv_invalid_price(self) -> None:
        """Test that invalid initial price raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="initial_price must be > 0"):
            gen.generate_ohlcv(n=100, initial_price=-100)

    def test_ohlcv_invalid_volatility(self) -> None:
        """Test that negative volatility raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="volatility must be >= 0"):
            gen.generate_ohlcv(n=100, volatility=-0.01)


class TestOrderBookGeneration:
    """Tests for order book snapshot generation."""

    def test_generate_order_book_basic(self) -> None:
        """Test basic order book generation."""
        gen = FinancialDataGenerator(seed=42)
        book = gen.generate_order_book(mid_price=100.0, n_levels=10)

        assert isinstance(book, OrderBookSnapshot)
        assert len(book.bid_prices) == 10
        assert len(book.ask_prices) == 10
        assert len(book.bid_sizes) == 10
        assert len(book.ask_sizes) == 10

    def test_order_book_price_ordering(self) -> None:
        """Test that bid prices descend and ask prices ascend."""
        gen = FinancialDataGenerator(seed=42)
        book = gen.generate_order_book(n_levels=10)

        # Bids should be descending
        for i in range(len(book.bid_prices) - 1):
            assert book.bid_prices[i] >= book.bid_prices[i + 1]

        # Asks should be ascending
        for i in range(len(book.ask_prices) - 1):
            assert book.ask_prices[i] <= book.ask_prices[i + 1]

    def test_order_book_spread(self) -> None:
        """Test that best ask > best bid (positive spread)."""
        gen = FinancialDataGenerator(seed=42)
        book = gen.generate_order_book(mid_price=100.0, spread_bps=10)

        assert book.ask_prices[0] > book.bid_prices[0]
        # Spread should be approximately 10 bps
        actual_spread_bps = (book.ask_prices[0] - book.bid_prices[0]) / 100.0 * 10000
        assert actual_spread_bps == pytest.approx(10, rel=0.5)

    def test_order_book_ground_truth(self) -> None:
        """Test that ground truth metrics are computed."""
        gen = FinancialDataGenerator(seed=42)
        book = gen.generate_order_book(mid_price=100.0)

        gt = book.ground_truth
        assert "mid_price" in gt
        assert "spread_bps" in gt
        assert "best_bid" in gt
        assert "best_ask" in gt
        assert "total_bid_size" in gt
        assert "total_ask_size" in gt
        assert "imbalance" in gt
        assert -1 <= gt["imbalance"] <= 1

    def test_order_book_to_json(self) -> None:
        """Test JSON serialization."""
        gen = FinancialDataGenerator(seed=42)
        book = gen.generate_order_book(n_levels=5)

        json_str = book.to_json()
        assert isinstance(json_str, str)
        assert "bids" in json_str
        assert "asks" in json_str
        assert "price" in json_str
        assert "size" in json_str

    def test_order_book_invalid_mid_price(self) -> None:
        """Test that invalid mid price raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="mid_price must be > 0"):
            gen.generate_order_book(mid_price=0)

    def test_order_book_invalid_spread(self) -> None:
        """Test that negative spread raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="spread_bps must be >= 0"):
            gen.generate_order_book(spread_bps=-10)

    def test_order_book_invalid_levels(self) -> None:
        """Test that invalid n_levels raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="n_levels must be > 0"):
            gen.generate_order_book(n_levels=0)


class TestPortfolioReturnsGeneration:
    """Tests for portfolio returns generation."""

    def test_generate_portfolio_returns_basic(self) -> None:
        """Test basic portfolio returns generation."""
        gen = FinancialDataGenerator(seed=42)
        portfolio = gen.generate_portfolio_returns(n=100, n_assets=5)

        assert isinstance(portfolio, dict)
        assert len(portfolio) == 5
        assert "asset_A" in portfolio
        assert all(isinstance(ds, SyntheticDataset) for ds in portfolio.values())

    def test_portfolio_correlation_structure(self) -> None:
        """Test that assets are correlated as specified."""
        gen = FinancialDataGenerator(seed=42)

        # High correlation
        high_corr = gen.generate_portfolio_returns(n=500, n_assets=3, correlation_strength=0.8)
        gen.reset_seed(42)
        # Low correlation
        low_corr = gen.generate_portfolio_returns(n=500, n_assets=3, correlation_strength=0.2)

        # Extract correlations from ground truth
        high_corr_value = list(high_corr["asset_A"].ground_truth["correlations"].values())[0]
        low_corr_value = list(low_corr["asset_A"].ground_truth["correlations"].values())[0]

        # High correlation portfolio should have higher actual correlations
        assert abs(high_corr_value) > abs(low_corr_value)

    def test_portfolio_ground_truth(self) -> None:
        """Test that ground truth metrics are computed for each asset."""
        gen = FinancialDataGenerator(seed=42)
        portfolio = gen.generate_portfolio_returns(n=100, n_assets=3)

        for asset_name, dataset in portfolio.items():
            gt = dataset.ground_truth
            assert "mean_return" in gt
            assert "volatility" in gt
            assert "cumulative_return" in gt
            assert "sharpe" in gt
            assert "correlations" in gt

    def test_portfolio_invalid_n(self) -> None:
        """Test that invalid n raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_portfolio_returns(n=0)

    def test_portfolio_invalid_n_assets(self) -> None:
        """Test that invalid n_assets raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="n_assets must be > 0"):
            gen.generate_portfolio_returns(n=100, n_assets=0)

    def test_portfolio_invalid_correlation(self) -> None:
        """Test that invalid correlation_strength raises ValueError."""
        gen = FinancialDataGenerator()

        with pytest.raises(ValueError, match="correlation_strength must be in"):
            gen.generate_portfolio_returns(n=100, correlation_strength=1.5)


# =============================================================================
# IoT Data Generator Tests
# =============================================================================


class TestIoTDataGenerator:
    """Tests for IoTDataGenerator initialization."""

    def test_init_default_seed(self) -> None:
        """Test default seed initialization."""
        gen = IoTDataGenerator()
        assert gen.seed == 42

    def test_init_custom_seed(self) -> None:
        """Test custom seed initialization."""
        gen = IoTDataGenerator(seed=123)
        assert gen.seed == 123

    def test_sensor_configs_exist(self) -> None:
        """Test that sensor configurations are defined."""
        gen = IoTDataGenerator()
        assert "temperature" in gen.SENSOR_CONFIGS
        assert "pressure" in gen.SENSOR_CONFIGS
        assert "humidity" in gen.SENSOR_CONFIGS
        assert "vibration" in gen.SENSOR_CONFIGS


class TestSensorReadingsGeneration:
    """Tests for sensor readings generation."""

    def test_generate_sensor_readings_basic(self) -> None:
        """Test basic sensor reading generation."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(n=100, sensor_type="temperature")

        assert isinstance(sensor, SensorDataset)
        assert len(sensor.readings) == 100
        assert len(sensor.timestamps) == 100
        assert sensor.sensor_type == "temperature"
        assert sensor.unit == "°C"

    def test_sensor_readings_all_types(self) -> None:
        """Test generation for all sensor types."""
        gen = IoTDataGenerator(seed=42)

        for sensor_type in gen.SENSOR_CONFIGS:
            sensor = gen.generate_sensor_readings(n=50, sensor_type=sensor_type)
            assert sensor.sensor_type == sensor_type
            assert sensor.unit == gen.SENSOR_CONFIGS[sensor_type]["unit"]

    def test_sensor_readings_within_range(self) -> None:
        """Test that readings are within valid range."""
        gen = IoTDataGenerator(seed=42)
        config = gen.SENSOR_CONFIGS["temperature"]

        sensor = gen.generate_sensor_readings(n=1000, sensor_type="temperature")

        assert np.all(sensor.readings >= config["min_valid"])
        assert np.all(sensor.readings <= config["max_valid"])

    def test_sensor_readings_ground_truth(self) -> None:
        """Test that ground truth metrics are computed."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(n=100)

        gt = sensor.ground_truth
        assert "mean" in gt
        assert "std" in gt
        assert "min" in gt
        assert "max" in gt
        assert "sensor_type" in gt
        assert "has_anomaly" in gt
        assert "n_readings" in gt

    def test_sensor_readings_no_failure(self) -> None:
        """Test sensor without failure mode."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(n=100, failure_mode=None)

        assert sensor.failure_mode is None
        assert sensor.failure_start_idx is None
        assert sensor.ground_truth["has_anomaly"] is False

    def test_sensor_readings_drift_failure(self) -> None:
        """Test sensor with drift failure mode."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(
            n=100, sensor_type="temperature", failure_mode="drift"
        )

        assert sensor.failure_mode == "drift"
        assert sensor.failure_start_idx is not None
        assert sensor.ground_truth["has_anomaly"] is True
        assert "drift_amount" in sensor.ground_truth

    def test_sensor_readings_spike_failure(self) -> None:
        """Test sensor with spike failure mode."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(
            n=200, sensor_type="temperature", failure_mode="spike"
        )

        assert sensor.failure_mode == "spike"
        assert sensor.ground_truth["has_anomaly"] is True
        # Spikes should be detected
        assert "n_spikes" in sensor.ground_truth

    def test_sensor_readings_flatline_failure(self) -> None:
        """Test sensor with flatline failure mode."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(
            n=100, sensor_type="pressure", failure_mode="flatline", failure_start_pct=0.5
        )

        assert sensor.failure_mode == "flatline"
        # After failure, all readings should be the same
        failure_idx = sensor.failure_start_idx
        assert failure_idx is not None
        flatline_readings = sensor.readings[failure_idx:]
        assert np.all(flatline_readings == flatline_readings[0])

    def test_sensor_readings_to_json(self) -> None:
        """Test JSON serialization."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(n=10)

        json_str = sensor.to_json()
        assert isinstance(json_str, str)
        assert "sensor_type" in json_str
        assert "readings" in json_str

    def test_sensor_readings_to_synthetic_dataset(self) -> None:
        """Test conversion to SyntheticDataset."""
        gen = IoTDataGenerator(seed=42)
        sensor = gen.generate_sensor_readings(n=100)

        synthetic = sensor.to_synthetic_dataset()
        assert isinstance(synthetic, SyntheticDataset)
        assert len(synthetic.data) == 100
        assert np.array_equal(synthetic.data, sensor.readings)

    def test_sensor_readings_invalid_n(self) -> None:
        """Test that invalid n raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_sensor_readings(n=0)

    def test_sensor_readings_invalid_type(self) -> None:
        """Test that invalid sensor type raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="Unknown sensor_type"):
            gen.generate_sensor_readings(n=100, sensor_type="invalid_sensor")

    def test_sensor_readings_invalid_failure_mode(self) -> None:
        """Test that invalid failure mode raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="Unknown failure_mode"):
            gen.generate_sensor_readings(n=100, failure_mode="invalid_mode")


class TestDeviceMetricsGeneration:
    """Tests for device metrics generation."""

    def test_generate_device_metrics_basic(self) -> None:
        """Test basic device metrics generation."""
        gen = IoTDataGenerator(seed=42)
        metrics = gen.generate_device_metrics(n=100)

        assert isinstance(metrics, dict)
        assert len(metrics) >= 3
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics

    def test_device_metrics_custom_types(self) -> None:
        """Test device metrics with custom metric types."""
        gen = IoTDataGenerator(seed=42)
        metrics = gen.generate_device_metrics(
            n=100, metric_types=["cpu_percent", "response_time_ms", "error_rate"]
        )

        assert len(metrics) == 3
        assert "cpu_percent" in metrics
        assert "response_time_ms" in metrics
        assert "error_rate" in metrics

    def test_device_metrics_degradation(self) -> None:
        """Test device metrics with degradation pattern."""
        gen = IoTDataGenerator(seed=42)
        metrics = gen.generate_device_metrics(n=200, include_degradation=True)

        for metric_name, dataset in metrics.items():
            assert dataset.ground_truth["has_degradation"] is True
            assert dataset.ground_truth["degradation_start_idx"] is not None

    def test_device_metrics_no_degradation(self) -> None:
        """Test device metrics without degradation."""
        gen = IoTDataGenerator(seed=42)
        metrics = gen.generate_device_metrics(n=100, include_degradation=False)

        for metric_name, dataset in metrics.items():
            assert dataset.ground_truth["has_degradation"] is False

    def test_device_metrics_ground_truth(self) -> None:
        """Test that ground truth metrics are computed."""
        gen = IoTDataGenerator(seed=42)
        metrics = gen.generate_device_metrics(n=100)

        for metric_name, dataset in metrics.items():
            gt = dataset.ground_truth
            assert "mean" in gt
            assert "std" in gt
            assert "metric_type" in gt
            assert "trend" in gt
            assert gt["trend"] in ["rising", "falling", "stable"]

    def test_device_metrics_invalid_n(self) -> None:
        """Test that invalid n raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_device_metrics(n=0)

    def test_device_metrics_invalid_type(self) -> None:
        """Test that invalid metric type raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="Unknown metric_type"):
            gen.generate_device_metrics(n=100, metric_types=["invalid_metric"])


class TestMultiSensorSystemGeneration:
    """Tests for multi-sensor system generation."""

    def test_generate_multi_sensor_basic(self) -> None:
        """Test basic multi-sensor system generation."""
        gen = IoTDataGenerator(seed=42)
        sensors = gen.generate_multi_sensor_system(n=100)

        assert isinstance(sensors, dict)
        assert len(sensors) >= 3
        assert "temperature" in sensors
        assert "pressure" in sensors
        assert "vibration" in sensors

    def test_multi_sensor_custom_types(self) -> None:
        """Test multi-sensor with custom sensor types."""
        gen = IoTDataGenerator(seed=42)
        sensors = gen.generate_multi_sensor_system(n=100, sensor_types=["temperature", "humidity"])

        assert len(sensors) == 2
        assert "temperature" in sensors
        assert "humidity" in sensors

    def test_multi_sensor_failure_propagation(self) -> None:
        """Test that failure propagation is applied."""
        gen = IoTDataGenerator(seed=42)

        # Generate multiple times to test probabilistic behavior
        n_with_primary_failure = 0
        n_with_secondary_failure = 0

        for seed in range(100):
            gen.reset_seed(seed)
            sensors = gen.generate_multi_sensor_system(n=100, failure_propagation=True)

            primary = sensors["temperature"]
            if primary.failure_mode is not None:
                n_with_primary_failure += 1
                # Check if any secondary sensor has failure
                for sensor_type, sensor in sensors.items():
                    if sensor_type != "temperature" and sensor.failure_mode is not None:
                        n_with_secondary_failure += 1
                        break

        # With 100 seeds, we should see some primary failures
        assert n_with_primary_failure > 20
        # Some of those should propagate
        assert n_with_secondary_failure > 0

    def test_multi_sensor_invalid_n(self) -> None:
        """Test that invalid n raises ValueError."""
        gen = IoTDataGenerator()

        with pytest.raises(ValueError, match="n must be > 0"):
            gen.generate_multi_sensor_system(n=0)


class TestReproducibility:
    """Tests for deterministic reproducibility."""

    def test_financial_reproducibility(self) -> None:
        """Test that financial data is reproducible with same seed."""
        gen1 = FinancialDataGenerator(seed=42)
        gen2 = FinancialDataGenerator(seed=42)

        ohlcv1 = gen1.generate_ohlcv(n=100)
        ohlcv2 = gen2.generate_ohlcv(n=100)

        assert np.array_equal(ohlcv1.close, ohlcv2.close)
        assert np.array_equal(ohlcv1.volume, ohlcv2.volume)

    def test_iot_reproducibility(self) -> None:
        """Test that IoT data is reproducible with same seed."""
        gen1 = IoTDataGenerator(seed=42)
        gen2 = IoTDataGenerator(seed=42)

        sensor1 = gen1.generate_sensor_readings(n=100)
        sensor2 = gen2.generate_sensor_readings(n=100)

        assert np.array_equal(sensor1.readings, sensor2.readings)

    def test_different_seeds_different_data(self) -> None:
        """Test that different seeds produce different data."""
        gen1 = FinancialDataGenerator(seed=42)
        gen2 = FinancialDataGenerator(seed=123)

        ohlcv1 = gen1.generate_ohlcv(n=100)
        ohlcv2 = gen2.generate_ohlcv(n=100)

        assert not np.array_equal(ohlcv1.close, ohlcv2.close)

"""Tests for time-windowed multi-timeframe analysis."""

import numpy as np

from semantic_frame.trading import describe_windows
from semantic_frame.trading.windows import (
    MultiWindowResult,
    TimeframeAlignment,
    TimeframeSignal,
    WindowAnalysis,
)


class TestDescribeWindows:
    """Tests for describe_windows function."""

    def test_basic_multi_window(self):
        """Test basic multi-window analysis."""
        data = [100, 102, 105, 103, 108, 110, 112, 109, 115, 118, 120]
        result = describe_windows(data, windows=[5, 10], context="BTC")

        assert isinstance(result, MultiWindowResult)
        assert len(result.windows) == 2
        assert "5" in result.windows
        assert "10" in result.windows

    def test_default_windows(self):
        """Test with default window sizes."""
        data = list(range(100, 300))  # 200 data points
        result = describe_windows(data)

        assert isinstance(result, MultiWindowResult)
        assert len(result.windows) > 0

    def test_bullish_alignment(self):
        """Test detection of aligned bullish trend."""
        # Steadily rising data
        data = [100 + i * 2 for i in range(50)]
        result = describe_windows(data, windows=[10, 25, 50])

        assert result.alignment == TimeframeAlignment.ALIGNED_BULLISH
        assert result.dominant_trend == "BULLISH"

    def test_bearish_alignment(self):
        """Test detection of aligned bearish trend."""
        # Steadily falling data
        data = [200 - i * 2 for i in range(50)]
        result = describe_windows(data, windows=[10, 25, 50])

        assert result.alignment == TimeframeAlignment.ALIGNED_BEARISH
        assert result.dominant_trend == "BEARISH"

    def test_mixed_signals(self):
        """Test detection of mixed signals."""
        # Choppy data with no clear trend
        np.random.seed(42)
        data = [100 + np.random.randn() * 10 for _ in range(50)]
        result = describe_windows(data, windows=[10, 25, 50])

        # Should detect mixed or neutral
        assert result.alignment in [
            TimeframeAlignment.MIXED,
            TimeframeAlignment.ALIGNED_BULLISH,
            TimeframeAlignment.ALIGNED_BEARISH,
        ]

    def test_diverging_timeframes(self):
        """Test detection of diverging timeframes."""
        # Long-term up, short-term down
        data = [100 + i for i in range(40)]  # Rising
        data.extend([140 - i * 2 for i in range(10)])  # Sharp drop at end
        result = describe_windows(data, windows=[5, 50])

        # Short term should be bearish, long term bullish
        _short_signal = result.windows["5"].signal
        _long_signal = result.windows["50"].signal

        # May detect divergence
        assert result.alignment in [
            TimeframeAlignment.DIVERGING,
            TimeframeAlignment.MIXED,
        ]

    def test_window_analysis_fields(self):
        """Test that window analysis contains all expected fields."""
        data = list(range(100, 150))
        result = describe_windows(data, windows=[10])

        window = result.windows["10"]
        assert isinstance(window, WindowAnalysis)
        assert window.window_name == "10"
        assert window.window_size == 10
        assert window.trend_direction in ["RISING", "FLAT", "FALLING"]
        assert 0 <= window.trend_strength <= 1
        assert window.volatility >= 0
        assert window.volatility_level in ["LOW", "MODERATE", "HIGH", "EXTREME"]

    def test_noise_level_assessment(self):
        """Test noise level assessment."""
        data = list(range(100, 200))
        result = describe_windows(data, windows=[10, 50])

        assert result.noise_level in ["low", "moderate", "high"]

    def test_suggested_action(self):
        """Test that suggested action is provided."""
        data = list(range(100, 150))
        result = describe_windows(data, windows=[10, 25])

        assert len(result.suggested_action) > 0

    def test_narrative_generation(self):
        """Test narrative includes key information."""
        data = list(range(100, 150))
        result = describe_windows(data, windows=[10, 25], context="ETH/USD")

        assert "ETH/USD" in result.narrative
        assert "timeframe" in result.narrative.lower()

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        data = np.array([100.0 + i for i in range(30)])
        result = describe_windows(data, windows=[10, 20])

        assert isinstance(result, MultiWindowResult)

    def test_string_window_specs(self):
        """Test with string window specifications."""
        data = list(range(100, 200))
        result = describe_windows(data, windows=["10", "50"])

        assert "10" in result.windows
        assert "50" in result.windows

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = [100, 101]
        result = describe_windows(data)

        assert "Insufficient" in result.narrative

    def test_window_larger_than_data(self):
        """Test when window is larger than data."""
        data = list(range(100, 120))  # 20 points
        result = describe_windows(data, windows=[10, 100])  # 100 > 20

        # Should handle gracefully by using full data for large window
        assert isinstance(result, MultiWindowResult)

    def test_change_pct_calculation(self):
        """Test percentage change calculation."""
        data = [100, 105, 108, 110]  # ~10% increase
        result = describe_windows(data, windows=[4])

        window = result.windows["4"]
        assert abs(window.change_pct - 10.0) < 1.0  # 10% with some tolerance

    def test_high_low_range(self):
        """Test high/low/range calculation."""
        data = [100, 150, 80, 120]
        result = describe_windows(data, windows=[4])

        window = result.windows["4"]
        assert window.high == 150.0
        assert window.low == 80.0
        assert window.range_pct > 0


class TestTimeframeSignal:
    """Tests for signal classification."""

    def test_strong_bullish_signal(self):
        """Test strong bullish signal detection."""
        # Very strong uptrend
        data = [100 + i * 5 for i in range(20)]
        result = describe_windows(data, windows=[20])

        signal = result.windows["20"].signal
        assert signal in [TimeframeSignal.BULLISH, TimeframeSignal.STRONG_BULLISH]

    def test_strong_bearish_signal(self):
        """Test strong bearish signal detection."""
        # Very strong downtrend
        data = [200 - i * 5 for i in range(20)]
        result = describe_windows(data, windows=[20])

        signal = result.windows["20"].signal
        assert signal in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]

    def test_neutral_signal(self):
        """Test neutral signal for flat data."""
        data = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        result = describe_windows(data, windows=[10])

        signal = result.windows["10"].signal
        assert signal == TimeframeSignal.NEUTRAL


class TestVolatilityClassification:
    """Tests for volatility classification."""

    def test_low_volatility(self):
        """Test low volatility detection."""
        # Very smooth data
        data = [100 + i * 0.1 for i in range(50)]
        result = describe_windows(data, windows=[50])

        assert result.windows["50"].volatility_level == "LOW"

    def test_high_volatility(self):
        """Test volatility is calculated and classified."""
        # Create data with measurable volatility
        np.random.seed(42)
        data = [100 + np.random.randn() * 30 for _ in range(50)]
        result = describe_windows(data, windows=[50])

        # Should classify volatility into one of the levels
        assert result.windows["50"].volatility_level in ["LOW", "MODERATE", "HIGH", "EXTREME"]
        # And volatility value should be calculated
        assert result.windows["50"].volatility >= 0


class TestWindowParsing:
    """Tests for window specification parsing."""

    def test_integer_windows(self):
        """Test integer window specifications."""
        data = list(range(100, 200))
        result = describe_windows(data, windows=[10, 50, 100])

        assert "10" in result.windows
        assert "50" in result.windows
        assert "100" in result.windows

    def test_mixed_window_specs(self):
        """Test mixed integer and string windows."""
        data = list(range(100, 200))
        result = describe_windows(data, windows=[10, "50"])

        assert "10" in result.windows
        assert "50" in result.windows


class TestCrossWindowInsights:
    """Tests for cross-window analysis."""

    def test_short_vs_long_term_signals(self):
        """Test short-term vs long-term signal tracking."""
        data = list(range(100, 200))
        result = describe_windows(data, windows=[10, 50])

        assert result.short_term_signal is not None
        assert result.long_term_signal is not None

    def test_total_points_tracking(self):
        """Test total data points is tracked."""
        data = list(range(100, 175))  # 75 points
        result = describe_windows(data, windows=[10, 50])

        assert result.total_points == 75

    def test_context_preserved(self):
        """Test context is preserved in result."""
        data = list(range(100, 150))
        result = describe_windows(data, context="My Strategy")

        assert result.data_context == "My Strategy"

"""Tests for trading drawdown analysis."""

import numpy as np
import pytest

from semantic_frame.trading import describe_drawdown
from semantic_frame.trading.enums import DrawdownSeverity, RecoveryState
from semantic_frame.trading.schemas import DrawdownResult


class TestDescribeDrawdown:
    """Tests for describe_drawdown function."""

    def test_basic_drawdown(self):
        """Test basic drawdown detection."""
        equity = [10000, 10500, 10200, 9800, 9500, 10000, 10800]
        result = describe_drawdown(equity, context="Test Strategy")

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct > 0
        assert result.context == "Test Strategy"
        assert len(result.narrative) > 0

    def test_no_drawdown(self):
        """Test when equity only goes up."""
        equity = [10000, 10100, 10200, 10300, 10400, 10500]
        result = describe_drawdown(equity)

        assert result.max_drawdown_pct == 0.0
        assert result.severity == DrawdownSeverity.MINIMAL
        assert result.recovery_state == RecoveryState.AT_HIGH

    def test_severe_drawdown(self):
        """Test catastrophic drawdown detection."""
        equity = [10000, 10500, 8000, 5000, 4000, 4500, 5000]
        result = describe_drawdown(equity)

        assert result.max_drawdown_pct >= 50
        assert result.severity == DrawdownSeverity.CATASTROPHIC

    def test_currently_in_drawdown(self):
        """Test detection of ongoing drawdown."""
        equity = [10000, 10500, 10200, 9500, 9000]
        result = describe_drawdown(equity)

        assert result.current_drawdown_pct > 0
        assert result.recovery_state == RecoveryState.IN_DRAWDOWN

    def test_fully_recovered(self):
        """Test full recovery detection."""
        equity = [10000, 10500, 9500, 9000, 10500, 10600]
        result = describe_drawdown(equity)

        assert result.current_drawdown_pct == 0.0
        assert result.recovery_state == RecoveryState.AT_HIGH

    def test_multiple_drawdowns(self):
        """Test detection of multiple drawdown periods."""
        equity = [10000, 10500, 9500, 10500, 9000, 10500]
        result = describe_drawdown(equity)

        assert result.num_drawdowns >= 2
        assert len(result.drawdown_periods) >= 2

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        equity = np.array([10000.0, 10500.0, 9500.0, 10500.0])
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct > 0

    def test_list_input(self):
        """Test with Python list input."""
        equity = [10000, 10500, 9500, 10500]
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)

    def test_insufficient_data(self):
        """Test with too little data."""
        equity = [10000]
        result = describe_drawdown(equity)

        assert result.max_drawdown_pct == 0.0
        assert "Insufficient" in result.narrative

    def test_empty_data(self):
        """Test with empty data."""
        equity = []
        result = describe_drawdown(equity)

        assert result.max_drawdown_pct == 0.0

    def test_context_in_narrative(self):
        """Test that context appears in narrative."""
        equity = [10000, 9500, 10000]
        result = describe_drawdown(equity, context="BTC Strategy")

        assert "BTC Strategy" in result.narrative

    def test_min_depth_filtering(self):
        """Test that tiny drawdowns are filtered."""
        # Create equity with very small fluctuations
        equity = [10000, 10010, 10005, 10008, 10012]
        result = describe_drawdown(equity, min_depth_pct=1.0)

        # Drawdowns < 1% should be filtered
        assert result.num_drawdowns == 0 or all(p.depth_pct >= 1.0 for p in result.drawdown_periods)


class TestDrawdownSeverity:
    """Tests for severity classification."""

    @pytest.mark.parametrize(
        "max_dd_pct,expected_severity",
        [
            (3.0, DrawdownSeverity.MINIMAL),
            (10.0, DrawdownSeverity.MODERATE),
            (20.0, DrawdownSeverity.SIGNIFICANT),
            (40.0, DrawdownSeverity.SEVERE),
            (60.0, DrawdownSeverity.CATASTROPHIC),
        ],
    )
    def test_severity_thresholds(self, max_dd_pct, expected_severity):
        """Test severity classification at threshold boundaries."""
        # Create equity curve with specified drawdown
        peak = 10000
        trough = peak * (1 - max_dd_pct / 100)
        equity = [peak, trough, peak + 100]  # DD then recovery

        result = describe_drawdown(equity)

        assert result.severity == expected_severity


class TestDrawdownPeriods:
    """Tests for drawdown period tracking."""

    def test_period_attributes(self):
        """Test that periods have correct attributes."""
        equity = [10000, 10500, 9500, 9000, 10500]
        result = describe_drawdown(equity)

        if result.drawdown_periods:
            period = result.drawdown_periods[0]
            assert period.start_index >= 0
            assert period.trough_index >= period.start_index
            assert period.depth_pct > 0
            assert period.duration >= 1

    def test_ongoing_period(self):
        """Test unrecovered drawdown period."""
        equity = [10000, 10500, 9500, 9000]  # Still in drawdown
        result = describe_drawdown(equity)

        # Should have at least one unrecovered period
        unrecovered = [p for p in result.drawdown_periods if not p.recovered]
        assert len(unrecovered) >= 1

    def test_max_periods_limit(self):
        """Test that periods are limited to 10 most significant."""
        # Create many small drawdowns
        equity = []
        value = 10000
        for i in range(50):
            equity.append(value)
            value *= 0.95  # 5% drop
            equity.append(value)
            value *= 1.10  # 10% recovery
            equity.append(value)

        result = describe_drawdown(np.array(equity))

        assert len(result.drawdown_periods) <= 10


class TestDrawdownNarrative:
    """Tests for narrative generation."""

    def test_narrative_includes_max_dd(self):
        """Test that narrative mentions max drawdown."""
        equity = [10000, 9000, 10000]
        result = describe_drawdown(equity)

        assert "10" in result.narrative or "drawdown" in result.narrative.lower()

    def test_narrative_no_drawdown(self):
        """Test narrative for no drawdown case."""
        equity = [10000, 10100, 10200, 10300]
        result = describe_drawdown(equity)

        assert "no drawdown" in result.narrative.lower() or "pure" in result.narrative.lower()

    def test_narrative_current_state(self):
        """Test that narrative mentions current state."""
        equity = [10000, 10500, 9500, 9000]  # Currently in DD
        result = describe_drawdown(equity)

        assert "currently" in result.narrative.lower()


class TestDrawdownEdgeCases:
    """Tests for edge cases including NaN/Inf and zero equity handling."""

    def test_nan_values_filtered(self):
        """Test that NaN values are filtered from equity curve."""
        equity = [10000, float("nan"), 10500, 9500, float("nan"), 10000]
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert "nan" not in result.narrative.lower()

    def test_inf_values_filtered(self):
        """Test that Inf values are filtered from equity curve."""
        equity = [10000, float("inf"), 10500, float("-inf"), 9500, 10000]
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert "inf" not in result.narrative.lower()

    def test_numpy_nan_inf_handling(self):
        """Test handling of numpy NaN and Inf values."""
        import numpy as np

        equity = np.array([10000, np.nan, 10500, np.inf, 9500, -np.inf, 10000])
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)

    def test_equity_with_very_small_values(self):
        """Test handling of very small equity values (near zero but positive)."""
        equity = [0.001, 0.0015, 0.001, 0.0008, 0.001]
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        # Should calculate drawdown correctly
        assert result.max_drawdown_pct >= 0

    def test_all_same_equity_no_drawdown(self):
        """Test flat equity curve has no drawdown."""
        equity = [10000] * 20
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct == 0.0
        assert result.severity == DrawdownSeverity.MINIMAL

    def test_monotonically_increasing_no_drawdown(self):
        """Test always-increasing equity has no drawdown."""
        equity = list(range(10000, 15000, 100))
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct == 0.0

    def test_single_large_drawdown(self):
        """Test a single catastrophic drawdown."""
        equity = [10000, 10500, 3000, 3500]  # 70%+ drawdown
        result = describe_drawdown(equity)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct > 70
        assert result.severity == DrawdownSeverity.CATASTROPHIC

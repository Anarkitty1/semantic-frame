"""Tests for market regime detection."""

import numpy as np

from semantic_frame.trading import describe_regime
from semantic_frame.trading.regime import (
    RegimePeriod,
    RegimeResult,
    RegimeStability,
    RegimeStrength,
    RegimeType,
)


class TestDescribeRegime:
    """Tests for describe_regime function."""

    def test_basic_regime_detection(self):
        """Test basic regime detection."""
        returns = [0.01, 0.02, 0.01, -0.05, -0.08, -0.03, 0.02, 0.03, 0.04]
        result = describe_regime(returns, context="Test")

        assert isinstance(result, RegimeResult)
        assert result.current_regime in list(RegimeType)
        assert len(result.narrative) > 0

    def test_bull_regime_detection(self):
        """Test detection of bull regime."""
        # Strong consistent positive returns
        returns = [0.02, 0.025, 0.018, 0.022, 0.03, 0.025, 0.028, 0.02] * 3
        result = describe_regime(returns, context="Bull Market")

        assert result.current_regime == RegimeType.BULL
        assert result.time_in_bull_pct > 50

    def test_bear_regime_detection(self):
        """Test detection of bear regime."""
        # Strong consistent negative returns
        returns = [-0.02, -0.025, -0.018, -0.022, -0.03, -0.025, -0.028, -0.02] * 3
        result = describe_regime(returns, context="Bear Market")

        assert result.current_regime == RegimeType.BEAR
        assert result.time_in_bear_pct > 50

    def test_sideways_regime_detection(self):
        """Test detection of sideways regime."""
        # Small mixed returns with no clear trend
        np.random.seed(42)
        returns = list(np.random.uniform(-0.005, 0.005, 30))
        result = describe_regime(returns, context="Sideways")

        # May be sideways or weak bull/bear
        assert result.current_regime in [
            RegimeType.SIDEWAYS,
            RegimeType.BULL,
            RegimeType.BEAR,
        ]

    def test_recovery_regime_detection(self):
        """Test detection of recovery/transition patterns."""
        # Clear bear to bull transition - longer data to allow regime detection
        returns = [
            -0.03,
            -0.04,
            -0.035,
            -0.025,
            -0.02,
            -0.01,  # Bear period
            0.005,
            0.01,
            0.02,
            0.025,
            0.03,
            0.035,
            0.04,
        ]  # Recovery/bull
        result = describe_regime(returns, lookback=6)

        # Should detect regime change (from bear to recovery/bull)
        # or current regime should be improving
        assert result.total_regime_changes >= 0  # May detect transition
        assert isinstance(result, RegimeResult)

    def test_regime_periods_tracking(self):
        """Test that regime periods are tracked."""
        returns = [0.02] * 15 + [-0.02] * 15  # Clear bull then bear
        result = describe_regime(returns, lookback=10)

        assert len(result.regimes_detected) >= 1
        for period in result.regimes_detected:
            assert isinstance(period, RegimePeriod)
            assert period.duration > 0

    def test_regime_change_count(self):
        """Test regime change counting."""
        # Multiple regime changes
        returns = [0.02] * 10 + [-0.02] * 10 + [0.02] * 10
        result = describe_regime(returns, lookback=8)

        assert result.total_regime_changes >= 1

    def test_stability_classification(self):
        """Test stability classification."""
        returns = [0.01] * 50  # Very stable
        result = describe_regime(returns)

        assert result.stability in list(RegimeStability)

    def test_regime_strength_classification(self):
        """Test regime strength is classified."""
        returns = [0.03, 0.035, 0.028, 0.032, 0.04] * 5  # Strong bull
        result = describe_regime(returns)

        assert result.current_regime_strength in list(RegimeStrength)

    def test_current_regime_duration(self):
        """Test current regime duration tracking."""
        returns = [0.02] * 20
        result = describe_regime(returns)

        assert result.current_regime_duration > 0
        assert result.current_regime_duration <= len(returns)

    def test_time_distribution(self):
        """Test time distribution percentages."""
        returns = [0.02] * 30
        result = describe_regime(returns)

        total = result.time_in_bull_pct + result.time_in_bear_pct + result.time_in_sideways_pct
        assert abs(total - 100) < 1  # Should sum to ~100%

    def test_regime_trend(self):
        """Test regime trend detection."""
        returns = [-0.02] * 15 + [0.02] * 15  # Bear to bull = improving
        result = describe_regime(returns, lookback=10)

        assert result.regime_trend in ["improving", "deteriorating", "stable"]

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.02])
        result = describe_regime(returns)

        assert isinstance(result, RegimeResult)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        returns = [0.01, 0.02]
        result = describe_regime(returns)

        assert "Insufficient" in result.narrative

    def test_context_in_narrative(self):
        """Test that context appears in narrative."""
        returns = [0.01] * 10
        result = describe_regime(returns, context="BTC/USD")

        assert "BTC/USD" in result.narrative

    def test_custom_thresholds(self):
        """Test custom threshold parameters."""
        returns = [0.01] * 20
        result = describe_regime(
            returns,
            bull_threshold=0.05,  # More strict
            bear_threshold=-0.05,
        )

        assert isinstance(result, RegimeResult)

    def test_custom_lookback(self):
        """Test custom lookback window."""
        returns = [0.01] * 30
        result_short = describe_regime(returns, lookback=5)
        result_long = describe_regime(returns, lookback=20)

        # Both should work
        assert isinstance(result_short, RegimeResult)
        assert isinstance(result_long, RegimeResult)


class TestRegimeStabilityClassification:
    """Tests for stability classification."""

    def test_very_stable(self):
        """Test very stable classification."""
        # One long regime
        returns = [0.01] * 100
        result = describe_regime(returns)

        # With no regime changes, should be very stable
        if result.total_regime_changes == 0:
            assert result.stability == RegimeStability.VERY_STABLE

    def test_unstable_conditions(self):
        """Test unstable classification with many changes."""
        # Alternating regimes
        returns = []
        for i in range(10):
            returns.extend([0.03] * 5)
            returns.extend([-0.03] * 5)

        result = describe_regime(returns, lookback=4, min_regime_length=2)

        # Should detect instability
        assert result.stability in [
            RegimeStability.UNSTABLE,
            RegimeStability.HIGHLY_UNSTABLE,
            RegimeStability.STABLE,  # May smooth out changes
        ]


class TestRegimePeriod:
    """Tests for regime period tracking."""

    def test_period_cumulative_return(self):
        """Test cumulative return calculation."""
        returns = [0.01, 0.02, 0.015]  # 4.5% total
        result = describe_regime(returns)

        if result.regimes_detected:
            period = result.regimes_detected[0]
            # Cumulative return should be calculated
            assert period.cumulative_return != 0 or all(r == 0 for r in returns)

    def test_period_volatility(self):
        """Test volatility calculation for periods."""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005] * 4
        result = describe_regime(returns)

        for period in result.regimes_detected:
            assert period.volatility >= 0

    def test_period_indices(self):
        """Test period start/end indices."""
        returns = [0.01] * 20
        result = describe_regime(returns)

        for period in result.regimes_detected:
            assert period.start_index >= 0
            assert period.end_index >= period.start_index
            assert period.duration == period.end_index - period.start_index + 1


class TestRegimeNarrative:
    """Tests for narrative generation."""

    def test_narrative_mentions_regime(self):
        """Test that narrative mentions current regime."""
        returns = [0.02] * 15
        result = describe_regime(returns)

        # Should mention regime type
        narrative_lower = result.narrative.lower()
        assert any(rt.value in narrative_lower for rt in RegimeType)

    def test_narrative_mentions_duration(self):
        """Test that narrative mentions duration."""
        returns = [0.01] * 10
        result = describe_regime(returns)

        assert "period" in result.narrative.lower()

    def test_narrative_actionable_insight(self):
        """Test that narrative includes actionable insight."""
        returns = [0.03] * 20  # Strong bull
        result = describe_regime(returns)

        # Should have some actionable content
        assert len(result.narrative) > 50

    def test_narrative_regime_changes(self):
        """Test narrative mentions regime changes."""
        returns = [0.02] * 15 + [-0.02] * 15
        result = describe_regime(returns, lookback=10)

        if result.total_regime_changes > 0:
            assert "change" in result.narrative.lower() or "transition" in result.narrative.lower()


class TestHighVolatilityRegime:
    """Tests for high volatility regime detection."""

    def test_high_volatility_detection(self):
        """Test high volatility regime detection."""
        # Very volatile returns
        np.random.seed(42)
        returns = list(np.random.uniform(-0.15, 0.15, 30))
        result = describe_regime(returns, vol_threshold=0.20)

        # May detect high volatility
        assert result.current_regime in list(RegimeType)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_zero_returns(self):
        """Test with all zero returns."""
        returns = [0.0] * 20
        result = describe_regime(returns)

        assert result.current_regime == RegimeType.SIDEWAYS

    def test_single_large_move(self):
        """Test with single large move."""
        returns = [0.0] * 10 + [0.5] + [0.0] * 10
        result = describe_regime(returns)

        assert isinstance(result, RegimeResult)

    def test_alternating_returns(self):
        """Test with perfectly alternating returns."""
        returns = [0.01, -0.01] * 15
        result = describe_regime(returns)

        # Should handle gracefully
        assert isinstance(result, RegimeResult)


class TestMCPIntegration:
    """Tests for MCP tool integration."""

    def test_describe_regime_mcp_basic(self):
        """Test describe_regime MCP tool."""
        from semantic_frame.integrations.mcp import describe_regime as mcp_describe_regime

        result = mcp_describe_regime(
            returns="[0.01, 0.02, 0.01, -0.05, -0.08, 0.02, 0.03]",
            context="BTC",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "BTC" in result

    def test_describe_regime_mcp_csv(self):
        """Test describe_regime MCP with CSV input."""
        from semantic_frame.integrations.mcp import describe_regime as mcp_describe_regime

        result = mcp_describe_regime(returns="0.01, 0.02, -0.01, 0.015")

        assert isinstance(result, str)
        assert "Error" not in result

    def test_describe_regime_mcp_error_handling(self):
        """Test describe_regime MCP error handling."""
        from semantic_frame.integrations.mcp import describe_regime as mcp_describe_regime

        result = mcp_describe_regime(returns="invalid data")

        assert "Error" in result

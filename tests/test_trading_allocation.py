"""Tests for position sizing and portfolio allocation."""

import numpy as np
import pytest

from semantic_frame.trading import describe_allocation
from semantic_frame.trading.allocation import (
    AllocationMethod,
    AllocationResult,
    AssetAnalysis,
    CorrelationInsight,
    DiversificationLevel,
    RiskLevel,
)


class TestDescribeAllocation:
    """Tests for describe_allocation function."""

    def test_basic_allocation(self):
        """Test basic allocation analysis."""
        assets = {
            "BTC": [100, 105, 102, 108, 110],
            "ETH": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets, context="Crypto")

        assert isinstance(result, AllocationResult)
        assert len(result.suggested_weights) == 2
        assert sum(result.suggested_weights.values()) <= 1.01  # Allow small rounding

    def test_three_asset_portfolio(self):
        """Test with three assets."""
        assets = {
            "BTC": [100, 105, 102, 108, 110, 115],
            "ETH": [50, 52, 48, 55, 54, 58],
            "SOL": [20, 22, 19, 25, 24, 28],
        }
        result = describe_allocation(assets)

        assert result.num_assets == 3
        assert len(result.suggested_weights) == 3
        assert len(result.asset_analyses) == 3

    def test_single_asset(self):
        """Test with single asset."""
        assets = {"BTC": [100, 105, 102, 108, 110]}
        result = describe_allocation(assets)

        assert result.suggested_weights["BTC"] == 1.0
        assert result.num_assets == 1

    def test_equal_weight_method(self):
        """Test equal weight allocation method."""
        assets = {
            "A": [100, 105, 102],
            "B": [50, 52, 48],
            "C": [20, 22, 19],
        }
        result = describe_allocation(assets, method="equal_weight")

        assert result.allocation_method == AllocationMethod.EQUAL_WEIGHT
        # Each should be ~33%
        for weight in result.suggested_weights.values():
            assert abs(weight - 0.333) < 0.01

    def test_risk_parity_method(self):
        """Test risk parity allocation method."""
        assets = {
            "LowVol": [100, 100.5, 100.2, 100.8, 101],  # Low volatility
            "HighVol": [100, 110, 95, 115, 105],  # High volatility
        }
        result = describe_allocation(assets, method="risk_parity")

        assert result.allocation_method == AllocationMethod.RISK_PARITY
        # Low vol asset should have higher weight
        assert result.suggested_weights["LowVol"] > result.suggested_weights["HighVol"]

    def test_min_variance_method(self):
        """Test minimum variance allocation method."""
        assets = {
            "A": [100, 102, 101, 103, 102],
            "B": [50, 52, 48, 55, 50],
        }
        result = describe_allocation(assets, method="min_variance")

        assert result.allocation_method == AllocationMethod.MIN_VARIANCE

    def test_target_vol_method(self):
        """Test target volatility allocation method."""
        assets = {
            "A": [100, 102, 101, 103, 102],
            "B": [50, 52, 48, 55, 50],
        }
        result = describe_allocation(assets, method="target_vol", target_volatility=15.0)

        assert result.allocation_method == AllocationMethod.TARGET_VOL

    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        assets = {
            "BTC": [100, 105, 102, 108, 110],
            "ETH": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets)

        assert result.portfolio_return is not None
        assert result.portfolio_volatility >= 0
        assert result.risk_level in list(RiskLevel)

    def test_diversification_metrics(self):
        """Test diversification metrics."""
        assets = {
            "BTC": [100, 105, 102, 108, 110],
            "ETH": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets)

        assert 0 <= result.diversification_score <= 1
        assert result.diversification_level in list(DiversificationLevel)
        assert -1 <= result.avg_correlation <= 1

    def test_correlation_insights(self):
        """Test correlation insights generation."""
        assets = {
            "A": [100, 105, 102, 108, 110],
            "B": [50, 52.5, 51, 54, 55],  # Similar pattern
            "C": [30, 28, 32, 27, 33],  # Different pattern
        }
        result = describe_allocation(assets)

        assert len(result.correlation_insights) > 0
        for insight in result.correlation_insights:
            assert isinstance(insight, CorrelationInsight)
            assert -1 <= insight.correlation <= 1

    def test_asset_analyses(self):
        """Test per-asset analysis."""
        assets = {
            "BTC": [100, 105, 102, 108, 110],
            "ETH": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets)

        for analysis in result.asset_analyses:
            assert isinstance(analysis, AssetAnalysis)
            assert analysis.annualized_volatility >= 0
            assert 0 <= analysis.suggested_weight <= 1

    def test_narrative_generation(self):
        """Test narrative includes key information."""
        assets = {
            "BTC": [100, 105, 102, 108, 110],
            "ETH": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets, context="Crypto Portfolio")

        assert "Crypto Portfolio" in result.narrative
        assert "%" in result.narrative  # Should mention percentages

    def test_disclaimer_present(self):
        """Test that disclaimer is always present."""
        assets = {"BTC": [100, 105, 102, 108, 110]}
        result = describe_allocation(assets)

        assert len(result.disclaimer) > 0
        assert "not" in result.disclaimer.lower() or "educational" in result.disclaimer.lower()

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        assets = {
            "BTC": np.array([100.0, 105.0, 102.0, 108.0, 110.0]),
            "ETH": np.array([50.0, 52.0, 48.0, 55.0, 54.0]),
        }
        result = describe_allocation(assets)

        assert isinstance(result, AllocationResult)

    def test_insufficient_data_error(self):
        """Test error with insufficient data."""
        assets = {"BTC": [100, 105]}  # Only 2 points

        with pytest.raises(ValueError):
            describe_allocation(assets)


class TestRiskLevel:
    """Tests for risk level classification."""

    def test_low_volatility_portfolio(self):
        """Test low volatility portfolio classification."""
        # Very stable assets
        assets = {
            "Bond1": [100, 100.1, 100.05, 100.15, 100.1],
            "Bond2": [50, 50.05, 50.02, 50.08, 50.05],
        }
        result = describe_allocation(assets)

        assert result.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]

    def test_high_volatility_portfolio(self):
        """Test high volatility portfolio classification."""
        # Volatile assets
        assets = {
            "Crypto1": [100, 130, 85, 150, 110],
            "Crypto2": [50, 70, 40, 80, 55],
        }
        result = describe_allocation(assets)

        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]


class TestDiversification:
    """Tests for diversification analysis."""

    def test_highly_correlated_assets(self):
        """Test with highly correlated assets."""
        # Assets that move together
        assets = {
            "A": [100, 110, 105, 115, 120],
            "B": [50, 55, 52.5, 57.5, 60],  # Same pattern scaled
        }
        result = describe_allocation(assets)

        assert result.avg_correlation > 0.8
        assert result.diversification_level in [
            DiversificationLevel.POOR,
            DiversificationLevel.LIMITED,
        ]

    def test_uncorrelated_assets(self):
        """Test with uncorrelated assets."""
        np.random.seed(42)
        assets = {
            "A": list(100 + np.cumsum(np.random.randn(20))),
            "B": list(50 + np.cumsum(np.random.randn(20))),
        }
        result = describe_allocation(assets)

        # Should have better diversification
        assert result.diversification_level in list(DiversificationLevel)


class TestCorrelationInsights:
    """Tests for correlation insights."""

    def test_insight_relationship_descriptions(self):
        """Test that relationships are described."""
        assets = {
            "A": [100, 105, 102, 108, 110],
            "B": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets)

        if result.correlation_insights:
            insight = result.correlation_insights[0]
            assert len(insight.relationship) > 0

    def test_multiple_correlation_pairs(self):
        """Test with multiple correlation pairs."""
        assets = {
            "A": [100, 105, 102, 108, 110, 115],
            "B": [50, 52, 48, 55, 54, 58],
            "C": [30, 32, 28, 35, 33, 38],
        }
        result = describe_allocation(assets)

        # Should have insights for multiple pairs
        assert len(result.correlation_insights) <= 3  # Top 3


class TestMCPIntegration:
    """Tests for MCP tool integration."""

    def test_describe_allocation_mcp_basic(self):
        """Test describe_allocation MCP tool."""
        from semantic_frame.integrations.mcp import describe_allocation as mcp_describe

        result = mcp_describe(
            assets='{"BTC": [100, 105, 102, 108], "ETH": [50, 52, 48, 55]}',
            context="Crypto",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Crypto" in result

    def test_describe_allocation_mcp_methods(self):
        """Test different allocation methods via MCP."""
        from semantic_frame.integrations.mcp import describe_allocation as mcp_describe

        for method in ["equal_weight", "risk_parity", "min_variance"]:
            result = mcp_describe(
                assets='{"A": [100, 105, 102], "B": [50, 52, 48]}',
                method=method,
            )
            assert "Error" not in result

    def test_describe_allocation_mcp_disclaimer(self):
        """Test that MCP output includes disclaimer."""
        from semantic_frame.integrations.mcp import describe_allocation as mcp_describe

        result = mcp_describe(
            assets='{"BTC": [100, 105, 102, 108]}',
        )

        assert (
            "⚠️" in result
            or "disclaimer" in result.lower()
            or "not financial advice" in result.lower()
        )

    def test_describe_allocation_mcp_error_handling(self):
        """Test MCP error handling."""
        from semantic_frame.integrations.mcp import describe_allocation as mcp_describe

        result = mcp_describe(assets="not valid json")

        assert "Error" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_identical_prices(self):
        """Test with identical prices (zero volatility)."""
        assets = {
            "Stable": [100, 100, 100, 100, 100],
            "Moving": [50, 52, 48, 55, 54],
        }
        result = describe_allocation(assets)

        # Should handle gracefully
        assert isinstance(result, AllocationResult)

    def test_many_assets(self):
        """Test with many assets."""
        np.random.seed(42)
        assets = {f"Asset{i}": list(100 + np.cumsum(np.random.randn(10))) for i in range(10)}
        result = describe_allocation(assets)

        assert result.num_assets == 10
        assert len(result.suggested_weights) == 10

    def test_negative_returns(self):
        """Test with overall negative returns."""
        assets = {
            "Bear1": [100, 95, 90, 85, 80],
            "Bear2": [50, 48, 45, 43, 40],
        }
        result = describe_allocation(assets)

        assert result.portfolio_return < 0
        assert isinstance(result, AllocationResult)

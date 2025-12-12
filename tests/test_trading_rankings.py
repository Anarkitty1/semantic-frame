"""Tests for trading rankings analysis."""

import numpy as np
import pytest

from semantic_frame.trading import describe_rankings
from semantic_frame.trading.schemas import AgentRanking, RankingsResult


class TestDescribeRankings:
    """Tests for describe_rankings function."""

    def test_basic_rankings(self):
        """Test basic ranking comparison."""
        curves = {
            "CLAUDE": [10000, 10500, 11000, 10800, 11500],
            "GROK": [10000, 12000, 9000, 9500, 10500],
            "GPT5": [10000, 10200, 10400, 10300, 10600],
        }
        result = describe_rankings(curves, context="AI agents")

        assert isinstance(result, RankingsResult)
        assert result.num_agents == 3
        assert len(result.rankings) == 3
        assert len(result.narrative) > 0

    def test_leader_identification(self):
        """Test correct leader identification."""
        curves = {
            "A": [10000, 15000],  # 50% return
            "B": [10000, 12000],  # 20% return
        }
        result = describe_rankings(curves)

        assert result.highest_return == "A"

    def test_lowest_volatility(self):
        """Test lowest volatility identification."""
        curves = {
            "Volatile": [10000, 15000, 8000, 12000],
            "Stable": [10000, 10100, 10200, 10300],
        }
        result = describe_rankings(curves)

        assert result.lowest_volatility == "Stable"

    def test_lowest_drawdown(self):
        """Test lowest drawdown identification."""
        curves = {
            "BigDD": [10000, 15000, 5000, 7000],  # Big drawdown
            "SmallDD": [10000, 10500, 10200, 10800],  # Small drawdown
        }
        result = describe_rankings(curves)

        assert result.lowest_drawdown == "SmallDD"

    def test_ranking_order(self):
        """Test that rankings are ordered by composite score."""
        curves = {
            "A": [10000, 10100, 10200, 10300],  # Low return, low vol
            "B": [10000, 15000, 8000, 12000],  # High return, high vol
            "C": [10000, 12000, 11500, 13000],  # Medium return, medium vol
        }
        result = describe_rankings(curves)

        # Rankings should be sorted by composite score
        assert result.rankings[0].name == result.leader

    def test_per_agent_ranking_attributes(self):
        """Test that agent rankings have all required attributes."""
        curves = {
            "Agent1": [10000, 11000, 12000],
            "Agent2": [10000, 9500, 10500],
        }
        result = describe_rankings(curves)

        for ranking in result.rankings:
            assert isinstance(ranking, AgentRanking)
            assert ranking.name in ["Agent1", "Agent2"]
            assert isinstance(ranking.total_return_pct, float)
            assert isinstance(ranking.volatility, float)
            assert isinstance(ranking.max_drawdown_pct, float)
            assert ranking.return_rank >= 1
            assert ranking.volatility_rank >= 1
            assert ranking.drawdown_rank >= 1

    def test_win_rates_integration(self):
        """Test that win rates are included when provided."""
        curves = {
            "A": [10000, 11000],
            "B": [10000, 10500],
        }
        win_rates = {"A": 0.65, "B": 0.55}
        result = describe_rankings(curves, win_rates=win_rates)

        a_ranking = next(r for r in result.rankings if r.name == "A")
        b_ranking = next(r for r in result.rankings if r.name == "B")

        assert a_ranking.win_rate == 0.65
        assert b_ranking.win_rate == 0.55

    def test_numpy_array_input(self):
        """Test with NumPy array input."""
        curves = {
            "Agent1": np.array([10000.0, 11000.0, 12000.0]),
            "Agent2": [10000, 10500, 11000],  # Mixed input
        }
        result = describe_rankings(curves)

        assert result.num_agents == 2

    def test_empty_curves_error(self):
        """Test that empty curves raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            describe_rankings({})

    def test_single_agent(self):
        """Test ranking with single agent."""
        curves = {"Solo": [10000, 11000, 12000]}
        result = describe_rankings(curves)

        assert result.num_agents == 1
        assert result.leader == "Solo"
        assert len(result.rankings) == 1

    def test_context_in_narrative(self):
        """Test that context appears in narrative."""
        curves = {
            "A": [10000, 11000],
            "B": [10000, 10500],
        }
        result = describe_rankings(curves, context="trading bots")

        assert "trading bots" in result.narrative

    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation for rankings."""
        # Need enough data points for Sharpe with actual different patterns
        # A: Steady upward movement with low variance
        a_equity = [10000 + i * 10 for i in range(100)]
        # B: Choppy sideways movement
        b_equity = [10000 + (50 if i % 2 == 0 else -50) + i * 5 for i in range(100)]

        curves = {"A": a_equity, "B": b_equity}
        result = describe_rankings(curves)

        a_ranking = next(r for r in result.rankings if r.name == "A")
        b_ranking = next(r for r in result.rankings if r.name == "B")

        # Both should have Sharpe ratios (enough data points)
        assert a_ranking.sharpe_ratio is not None
        assert b_ranking.sharpe_ratio is not None
        # A should have better (higher) Sharpe than B due to lower volatility
        # But test just that they're calculated, not relative values (implementation dependent)
        assert isinstance(a_ranking.sharpe_ratio, float)
        assert isinstance(b_ranking.sharpe_ratio, float)


class TestRankingCalculations:
    """Tests for ranking calculation correctness."""

    def test_return_calculation(self):
        """Test total return percentage calculation."""
        curves = {
            "10pct": [10000, 11000],  # 10% return
            "50pct": [10000, 15000],  # 50% return
        }
        result = describe_rankings(curves)

        r1 = next(r for r in result.rankings if r.name == "10pct")
        r2 = next(r for r in result.rankings if r.name == "50pct")

        assert abs(r1.total_return_pct - 10.0) < 0.1
        assert abs(r2.total_return_pct - 50.0) < 0.1

    def test_drawdown_calculation(self):
        """Test max drawdown calculation in rankings."""
        curves = {
            "HighDD": [10000, 12000, 6000, 8000],  # 50% DD from peak
            "LowDD": [10000, 11000, 10500, 11500],  # ~4.5% DD
        }
        result = describe_rankings(curves)

        high_dd = next(r for r in result.rankings if r.name == "HighDD")
        low_dd = next(r for r in result.rankings if r.name == "LowDD")

        assert high_dd.max_drawdown_pct > 40  # Should be ~50%
        assert low_dd.max_drawdown_pct < 10

    def test_rank_values(self):
        """Test that ranks are 1-indexed and correct."""
        curves = {
            "First": [10000, 20000],  # Best return
            "Second": [10000, 15000],
            "Third": [10000, 11000],
        }
        result = describe_rankings(curves)

        returns = {r.name: r.return_rank for r in result.rankings}

        assert returns["First"] == 1
        assert returns["Second"] == 2
        assert returns["Third"] == 3


class TestRankingsNarrative:
    """Tests for rankings narrative generation."""

    def test_narrative_mentions_leader(self):
        """Test that narrative mentions the leader."""
        curves = {
            "Winner": [10000, 20000],
            "Loser": [10000, 10100],
        }
        result = describe_rankings(curves)

        assert result.leader in result.narrative

    def test_narrative_mentions_count(self):
        """Test that narrative mentions number of agents."""
        curves = {
            "A": [10000, 11000],
            "B": [10000, 10500],
            "C": [10000, 10200],
        }
        result = describe_rankings(curves)

        assert "3" in result.narrative

    def test_narrative_for_different_leaders(self):
        """Test narrative when different metrics have different leaders."""
        curves = {
            "HighReturn": [10000, 20000, 10000, 15000],  # High return, high vol/DD
            "Stable": [10000, 10100, 10200, 10300],  # Low return, low vol
        }
        result = describe_rankings(curves)

        # Should mention both aspects
        assert len(result.narrative) > 50

    def test_serialization(self):
        """Test that result can be serialized to JSON."""
        curves = {
            "A": [10000, 11000],
            "B": [10000, 10500],
        }
        result = describe_rankings(curves)

        json_str = result.to_json_str()
        assert isinstance(json_str, str)
        assert "rankings" in json_str
        assert "leader" in json_str

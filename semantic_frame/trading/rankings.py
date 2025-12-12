"""Multi-agent/strategy ranking analysis.

This module provides comparative analysis across multiple trading agents
or strategies, producing rankings and semantic descriptions of relative
performance.

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import numpy as np

from semantic_frame.trading.schemas import AgentRanking, RankingsResult


def _calc_total_return(equity: np.ndarray) -> float:
    """Calculate total return percentage from equity curve."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return float((equity[-1] - equity[0]) / equity[0] * 100)


def _calc_volatility(equity: np.ndarray) -> float:
    """Calculate return volatility from equity curve."""
    if len(equity) < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    return float(np.std(returns))


def _calc_max_drawdown(equity: np.ndarray) -> float:
    """Calculate max drawdown percentage from equity curve."""
    if len(equity) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    return float(np.max(drawdown))


def _calc_sharpe(equity: np.ndarray, risk_free: float = 0.0) -> float | None:
    """Calculate Sharpe ratio from equity curve."""
    if len(equity) < 10:
        return None
    returns = np.diff(equity) / equity[:-1]
    if np.std(returns) == 0:
        return None
    excess = returns - risk_free / 252
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def _generate_rankings_narrative(
    rankings: list[AgentRanking],
    leader: str,
    highest_return: str,
    lowest_vol: str,
    best_sharpe: str,
    lowest_dd: str,
    context: str | None,
) -> str:
    """Generate natural language narrative for rankings."""
    parts: list[str] = []

    n = len(rankings)
    prefix = f"Comparing {n} {context}" if context else f"Comparing {n} agents"

    # Leader
    leader_ranking = next((r for r in rankings if r.name == leader), None)
    if leader_ranking:
        parts.append(
            f"{prefix}: {leader} leads overall with "
            f"{leader_ranking.total_return_pct:.1f}% return."
        )

    # Highlight differences
    if highest_return != leader:
        hr = next((r for r in rankings if r.name == highest_return), None)
        if hr:
            parts.append(f"{highest_return} has highest raw return ({hr.total_return_pct:.1f}%).")

    if best_sharpe != leader and best_sharpe != highest_return:
        bs = next((r for r in rankings if r.name == best_sharpe), None)
        if bs and bs.sharpe_ratio:
            parts.append(f"{best_sharpe} is most risk-efficient (Sharpe: {bs.sharpe_ratio:.2f}).")

    if lowest_vol != leader:
        lv = next((r for r in rankings if r.name == lowest_vol), None)
        if lv:
            parts.append(f"{lowest_vol} is most stable (volatility: {lv.volatility:.1%}).")

    if lowest_dd != leader and lowest_dd != lowest_vol:
        ld = next((r for r in rankings if r.name == lowest_dd), None)
        if ld:
            parts.append(f"{lowest_dd} has smallest drawdown ({ld.max_drawdown_pct:.1f}%).")

    return " ".join(parts)


def describe_rankings(
    equity_curves: dict[str, np.ndarray | list[float]],
    win_rates: dict[str, float] | None = None,
    context: str | None = None,
) -> RankingsResult:
    """Compare multiple agents/strategies and produce rankings.

    Analyzes multiple equity curves to produce comparative rankings across
    multiple dimensions: total return, risk-adjusted return, volatility,
    and maximum drawdown.

    Args:
        equity_curves: Dict mapping agent names to equity curve arrays.
            Each array should be cumulative equity values over time.
        win_rates: Optional dict of win rates per agent (0-1 scale).
        context: Optional label (e.g., "trading agents", "BTC strategies").

    Returns:
        RankingsResult with complete comparative analysis.

    Example:
        >>> curves = {
        ...     "CLAUDE": [10000, 10500, 11000, 10800, 11500],
        ...     "GROK4": [10000, 12000, 9000, 9500, 10500],
        ...     "GPT5": [10000, 10200, 10400, 10300, 10600],
        ... }
        >>> result = describe_rankings(curves, context="AI agents")
        >>> print(result.leader)
        "CLAUDE"
    """
    if not equity_curves:
        raise ValueError("equity_curves cannot be empty")

    # Convert to numpy and calculate metrics for each agent
    agent_data: list[dict] = []

    for name, equity in equity_curves.items():
        if isinstance(equity, list):
            equity = np.array(equity, dtype=float)

        total_return = _calc_total_return(equity)
        volatility = _calc_volatility(equity)
        max_dd = _calc_max_drawdown(equity)
        sharpe = _calc_sharpe(equity)
        win_rate = win_rates.get(name) if win_rates else None

        agent_data.append(
            {
                "name": name,
                "total_return_pct": total_return,
                "volatility": volatility,
                "max_drawdown_pct": max_dd,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
            }
        )

    # Calculate rankings (1 = best)
    n = len(agent_data)

    # Sort by different metrics and assign ranks
    # Return rank: higher is better
    by_return = sorted(agent_data, key=lambda x: x["total_return_pct"], reverse=True)
    for i, d in enumerate(by_return):
        d["return_rank"] = i + 1

    # Sharpe rank: higher is better (None goes to end)
    by_sharpe = sorted(
        agent_data,
        key=lambda x: x["sharpe_ratio"] if x["sharpe_ratio"] is not None else float("-inf"),
        reverse=True,
    )
    for i, d in enumerate(by_sharpe):
        d["risk_adjusted_rank"] = i + 1

    # Volatility rank: lower is better
    by_vol = sorted(agent_data, key=lambda x: x["volatility"])
    for i, d in enumerate(by_vol):
        d["volatility_rank"] = i + 1

    # Drawdown rank: lower is better
    by_dd = sorted(agent_data, key=lambda x: x["max_drawdown_pct"])
    for i, d in enumerate(by_dd):
        d["drawdown_rank"] = i + 1

    # Composite score (lower total rank = better)
    for d in agent_data:
        d["composite_rank"] = (
            d["return_rank"] + d["risk_adjusted_rank"] + d["volatility_rank"] + d["drawdown_rank"]
        )

    # Sort by composite for final ordering
    agent_data.sort(key=lambda x: x["composite_rank"])

    # Build AgentRanking objects
    rankings = [
        AgentRanking(
            name=d["name"],
            total_return_pct=round(d["total_return_pct"], 2),
            volatility=round(d["volatility"], 4),
            sharpe_ratio=round(d["sharpe_ratio"], 3) if d["sharpe_ratio"] else None,
            max_drawdown_pct=round(d["max_drawdown_pct"], 2),
            win_rate=d["win_rate"],
            return_rank=d["return_rank"],
            risk_adjusted_rank=d["risk_adjusted_rank"],
            volatility_rank=d["volatility_rank"],
            drawdown_rank=d["drawdown_rank"],
        )
        for d in agent_data
    ]

    # Identify leaders
    leader = agent_data[0]["name"]
    highest_return = by_return[0]["name"]
    lowest_vol = by_vol[0]["name"]
    best_sharpe = by_sharpe[0]["name"]
    lowest_dd = by_dd[0]["name"]

    # Generate narrative
    narrative = _generate_rankings_narrative(
        rankings, leader, highest_return, lowest_vol, best_sharpe, lowest_dd, context
    )

    return RankingsResult(
        rankings=tuple(rankings),
        leader=leader,
        highest_return=highest_return,
        lowest_volatility=lowest_vol,
        best_risk_adjusted=best_sharpe,
        lowest_drawdown=lowest_dd,
        narrative=narrative,
        context=context,
        num_agents=n,
    )

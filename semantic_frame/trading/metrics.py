"""Trading performance metrics calculations.

This module provides comprehensive trading performance analysis including
win rate, profit factor, Sharpe ratio, and other key trading metrics.

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import numpy as np

from semantic_frame.trading.enums import (
    ConsistencyRating,
    PerformanceRating,
    RiskProfile,
)
from semantic_frame.trading.schemas import TradingMetrics, TradingPerformanceResult


def _calc_streaks(trades: np.ndarray) -> tuple[int, int, int]:
    """Calculate winning/losing streaks.

    Returns:
        Tuple of (max_wins, max_losses, current_streak).
        current_streak is positive for wins, negative for losses.
    """
    if len(trades) == 0:
        return 0, 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    current_streak = 0

    for trade in trades:
        if trade > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif trade < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            # Breakeven trade - doesn't break streaks
            pass

    # Calculate current streak from end (count only consecutive same-sign trades)
    current_streak = 0
    streak_sign: int | None = None
    for trade in reversed(trades):
        if trade > 0:
            if streak_sign is None:
                streak_sign = 1
                current_streak = 1
            elif streak_sign == 1:
                current_streak += 1
            else:
                break  # Different sign, stop
        elif trade < 0:
            if streak_sign is None:
                streak_sign = -1
                current_streak = -1
            elif streak_sign == -1:
                current_streak -= 1
            else:
                break  # Different sign, stop
        else:
            break  # Breakeven ends streak counting

    return max_wins, max_losses, current_streak


def _calc_sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float | None:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns (not cumulative).
        risk_free_rate: Annual risk-free rate (default 0).
        periods_per_year: Trading periods per year (252 for daily, 12 for monthly).

    Returns:
        Sharpe ratio or None if insufficient data/zero volatility.
    """
    if len(returns) < 2:
        return None

    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = float(np.std(excess_returns, ddof=1))

    if std == 0:
        return None

    mean_excess = float(np.mean(excess_returns))
    sharpe = (mean_excess / std) * np.sqrt(periods_per_year)

    return round(float(sharpe), 3)


def _calc_sortino_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float | None:
    """Calculate annualized Sortino ratio (downside deviation).

    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Sortino ratio or None if insufficient data/zero downside.
    """
    if len(returns) < 2:
        return None

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return None  # No downside risk

    downside_std = float(np.std(downside_returns, ddof=1))
    if downside_std == 0:
        return None

    mean_excess = float(np.mean(excess_returns))
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

    return round(float(sortino), 3)


def _classify_performance(
    win_rate: float,
    profit_factor: float | None,
    sharpe: float | None,
) -> PerformanceRating:
    """Classify overall performance based on key metrics."""
    score = 0

    # Win rate contribution (0-35 points)
    if win_rate >= 0.6:
        score += 35
    elif win_rate >= 0.5:
        score += 25
    elif win_rate >= 0.4:
        score += 15
    else:
        score += 5

    # Profit factor contribution (0-35 points)
    if profit_factor is not None:
        if profit_factor >= 2.0:
            score += 35
        elif profit_factor >= 1.5:
            score += 25
        elif profit_factor >= 1.0:
            score += 15
        else:
            score += 5
    else:
        score += 15  # Neutral if no losses

    # Sharpe contribution (0-30 points)
    if sharpe is not None:
        if sharpe >= 2.0:
            score += 30
        elif sharpe >= 1.0:
            score += 22
        elif sharpe >= 0.5:
            score += 15
        elif sharpe >= 0:
            score += 8
        else:
            score += 0
    else:
        score += 15  # Neutral if not calculable

    # Map score to rating
    if score >= 80:
        return PerformanceRating.EXCELLENT
    if score >= 60:
        return PerformanceRating.GOOD
    if score >= 40:
        return PerformanceRating.AVERAGE
    if score >= 20:
        return PerformanceRating.BELOW_AVERAGE
    return PerformanceRating.POOR


def _classify_risk_profile(volatility: float, max_drawdown_pct: float | None) -> RiskProfile:
    """Classify risk profile based on volatility and drawdown."""
    # Volatility thresholds (daily returns std)
    # max_drawdown thresholds

    # Conservative: low vol AND low DD
    if volatility < 0.05:
        if max_drawdown_pct is None or max_drawdown_pct < 10:
            return RiskProfile.CONSERVATIVE
        if max_drawdown_pct < 25:
            return RiskProfile.MODERATE
        return RiskProfile.AGGRESSIVE

    # Moderate vol
    if volatility < 0.15:
        if max_drawdown_pct is None or max_drawdown_pct < 25:
            return RiskProfile.MODERATE
        if max_drawdown_pct < 40:
            return RiskProfile.AGGRESSIVE
        return RiskProfile.VERY_AGGRESSIVE

    # High vol
    if volatility < 0.30:
        if max_drawdown_pct is None or max_drawdown_pct < 40:
            return RiskProfile.AGGRESSIVE
        return RiskProfile.VERY_AGGRESSIVE

    return RiskProfile.VERY_AGGRESSIVE


def _classify_consistency(win_rate: float, max_losses: int, total_trades: int) -> ConsistencyRating:
    """Classify consistency based on win rate stability and streaks."""
    if total_trades < 10:
        return ConsistencyRating.INCONSISTENT  # Too few trades

    # Losing streak relative to total trades
    loss_streak_ratio = max_losses / total_trades if total_trades > 0 else 0

    if win_rate >= 0.55 and loss_streak_ratio < 0.1:
        return ConsistencyRating.HIGHLY_CONSISTENT
    if win_rate >= 0.45 and loss_streak_ratio < 0.2:
        return ConsistencyRating.CONSISTENT
    if win_rate >= 0.35 and loss_streak_ratio < 0.3:
        return ConsistencyRating.INCONSISTENT
    return ConsistencyRating.ERRATIC


def _generate_performance_narrative(
    metrics: TradingMetrics,
    rating: PerformanceRating,
    risk: RiskProfile,
    consistency: ConsistencyRating,
    context: str | None,
) -> str:
    """Generate natural language narrative for trading performance."""
    parts: list[str] = []

    # Context prefix
    name = context if context else "The strategy"

    # Overall rating
    rating_desc = {
        PerformanceRating.EXCELLENT: "excellent",
        PerformanceRating.GOOD: "good",
        PerformanceRating.AVERAGE: "average",
        PerformanceRating.BELOW_AVERAGE: "below average",
        PerformanceRating.POOR: "poor",
    }
    parts.append(f"{name} shows {rating_desc[rating]} performance")

    # Win rate context
    win_pct = metrics.win_rate * 100
    if metrics.profit_factor is not None:
        parts.append(
            f"with {win_pct:.0f}% win rate and {metrics.profit_factor:.2f}x profit factor."
        )
    else:
        parts.append(f"with {win_pct:.0f}% win rate (no losing trades).")

    # Risk profile
    risk_desc = {
        RiskProfile.CONSERVATIVE: "conservative",
        RiskProfile.MODERATE: "moderate",
        RiskProfile.AGGRESSIVE: "aggressive",
        RiskProfile.VERY_AGGRESSIVE: "very aggressive",
    }
    parts.append(f"Risk profile: {risk_desc[risk]}.")

    # Key insight based on metrics
    if metrics.risk_reward_ratio is not None:
        if metrics.risk_reward_ratio < 1.0 and metrics.win_rate > 0.6:
            parts.append(
                f"Warning: High win rate ({win_pct:.0f}%) but poor risk-reward "
                f"({metrics.risk_reward_ratio:.1f}:1) suggests potential overtrading."
            )
        elif metrics.risk_reward_ratio > 2.0 and metrics.win_rate < 0.4:
            parts.append(
                "Strong risk-reward compensates for lower win rate - "
                "typical of trend-following strategies."
            )

    # Streaks
    if metrics.max_consecutive_losses >= 5:
        parts.append(
            f"Max losing streak: {metrics.max_consecutive_losses} trades - "
            "prepare for drawdowns."
        )

    return " ".join(parts)


def describe_trading_performance(
    trades: np.ndarray | list[float],
    context: str | None = None,
    max_drawdown_pct: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> TradingPerformanceResult:
    """Analyze trading performance from a series of trade PnLs.

    Calculates comprehensive trading metrics including win rate, profit factor,
    Sharpe ratio, and provides semantic classification of performance.

    Args:
        trades: Array of trade PnL values (positive = profit, negative = loss).
            Each value represents the profit/loss from a single trade.
        context: Optional label (e.g., "CLAUDE agent", "BTC momentum strategy").
        max_drawdown_pct: Optional max drawdown for risk calculations.
            If not provided, risk profile will be based on volatility only.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 0).
        periods_per_year: Trading periods per year (default 252 for daily).

    Returns:
        TradingPerformanceResult with complete performance analysis.

    Example:
        >>> trades = [100, -50, 75, -25, 150, -30, 80]
        >>> result = describe_trading_performance(trades, context="CLAUDE")
        >>> print(result.narrative)
        "CLAUDE shows good performance with 57% win rate and 2.30x profit factor."
    """
    # Convert to numpy array
    if isinstance(trades, list):
        trades = np.array(trades, dtype=float)

    if len(trades) == 0:
        return TradingPerformanceResult(
            metrics=TradingMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                net_profit=0.0,
                profit_factor=None,
                avg_win=None,
                avg_loss=None,
                avg_trade=0.0,
                risk_reward_ratio=None,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                current_streak=0,
                sharpe_ratio=None,
                sortino_ratio=None,
                calmar_ratio=None,
                recovery_factor=None,
            ),
            performance_rating=PerformanceRating.POOR,
            risk_profile=RiskProfile.MODERATE,
            consistency=ConsistencyRating.ERRATIC,
            narrative="No trades to analyze.",
            context=context,
        )

    # Basic counts
    winning = trades[trades > 0]
    losing = trades[trades < 0]
    total_trades = len(trades)
    num_wins = len(winning)
    num_losses = len(losing)

    # Win/loss stats
    win_rate = num_wins / total_trades if total_trades > 0 else 0.0
    gross_profit = float(np.sum(winning)) if len(winning) > 0 else 0.0
    gross_loss = float(np.sum(losing)) if len(losing) > 0 else 0.0
    net_profit = gross_profit + gross_loss

    # Averages
    avg_win = float(np.mean(winning)) if len(winning) > 0 else None
    avg_loss = float(np.mean(losing)) if len(losing) > 0 else None
    avg_trade = float(np.mean(trades))

    # Ratios
    profit_factor = None
    if gross_loss < 0:
        profit_factor = round(gross_profit / abs(gross_loss), 3)

    risk_reward = None
    if avg_win is not None and avg_loss is not None and avg_loss < 0:
        risk_reward = round(avg_win / abs(avg_loss), 3)

    # Streaks
    max_wins, max_losses, current_streak = _calc_streaks(trades)

    # Risk-adjusted metrics (treating trades as returns)
    # Normalize by initial trade size for ratio calculation
    if avg_win is not None and avg_win > 0:
        normalized_returns = trades / avg_win
    else:
        normalized_returns = trades / max(abs(trades.max()), abs(trades.min()), 1)

    sharpe = _calc_sharpe_ratio(normalized_returns, risk_free_rate, periods_per_year)
    sortino = _calc_sortino_ratio(normalized_returns, risk_free_rate, periods_per_year)

    # Calmar and recovery factor (need max drawdown)
    calmar = None
    recovery = None
    if max_drawdown_pct is not None and max_drawdown_pct > 0:
        # Rough annual return estimate
        annual_return_pct = (net_profit / abs(trades[0]) * 100) if trades[0] != 0 else 0
        calmar = round(annual_return_pct / max_drawdown_pct, 3) if max_drawdown_pct > 0 else None
        recovery = (
            round(net_profit / (max_drawdown_pct / 100 * abs(gross_profit + abs(gross_loss))), 3)
            if max_drawdown_pct > 0
            else None
        )

    # Volatility for risk profile
    volatility = float(np.std(normalized_returns)) if len(trades) > 1 else 0.0

    # Build metrics object
    metrics = TradingMetrics(
        total_trades=total_trades,
        winning_trades=num_wins,
        losing_trades=num_losses,
        win_rate=round(win_rate, 4),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        net_profit=round(net_profit, 2),
        profit_factor=profit_factor,
        avg_win=round(avg_win, 2) if avg_win else None,
        avg_loss=round(avg_loss, 2) if avg_loss else None,
        avg_trade=round(avg_trade, 2),
        risk_reward_ratio=risk_reward,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        current_streak=current_streak,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        recovery_factor=recovery,
    )

    # Classifications
    rating = _classify_performance(win_rate, profit_factor, sharpe)
    risk = _classify_risk_profile(volatility, max_drawdown_pct)
    consistency = _classify_consistency(win_rate, max_losses, total_trades)

    # Generate narrative
    narrative = _generate_performance_narrative(metrics, rating, risk, consistency, context)

    return TradingPerformanceResult(
        metrics=metrics,
        performance_rating=rating,
        risk_profile=risk,
        consistency=consistency,
        narrative=narrative,
        context=context,
    )

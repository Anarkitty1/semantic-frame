"""Drawdown analysis for equity curves.

This module provides comprehensive drawdown analysis for trading equity curves,
including peak-to-trough analysis, recovery tracking, and severity classification.

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import numpy as np

from semantic_frame.trading.enums import DrawdownSeverity, RecoveryState
from semantic_frame.trading.schemas import DrawdownPeriod, DrawdownResult


def _classify_severity(max_dd_pct: float) -> DrawdownSeverity:
    """Classify drawdown severity based on percentage.

    Thresholds:
        - < 5%: MINIMAL
        - 5-15%: MODERATE
        - 15-30%: SIGNIFICANT
        - 30-50%: SEVERE
        - >= 50%: CATASTROPHIC
    """
    if max_dd_pct < 5:
        return DrawdownSeverity.MINIMAL
    if max_dd_pct < 15:
        return DrawdownSeverity.MODERATE
    if max_dd_pct < 30:
        return DrawdownSeverity.SIGNIFICANT
    if max_dd_pct < 50:
        return DrawdownSeverity.SEVERE
    return DrawdownSeverity.CATASTROPHIC


def _determine_recovery_state(equity: np.ndarray, current_dd_pct: float) -> RecoveryState:
    """Determine current recovery state."""
    if current_dd_pct == 0:
        # Check if we're at all-time high
        if equity[-1] >= np.max(equity):
            return RecoveryState.AT_HIGH
        return RecoveryState.FULLY_RECOVERED
    if current_dd_pct < 5:
        return RecoveryState.RECOVERING
    return RecoveryState.IN_DRAWDOWN


def _find_drawdown_periods(equity: np.ndarray, min_depth_pct: float = 1.0) -> list[DrawdownPeriod]:
    """Identify all drawdown periods in the equity curve.

    A drawdown period starts when equity drops below a peak and ends
    when it recovers back to or above that peak.

    Args:
        equity: Array of equity values (cumulative).
        min_depth_pct: Minimum drawdown depth to track (default 1%).

    Returns:
        List of DrawdownPeriod objects.
    """
    if len(equity) < 2:
        return []

    periods: list[DrawdownPeriod] = []
    running_max = equity[0]
    peak_idx = 0
    in_drawdown = False
    trough_idx = 0
    trough_value = equity[0]

    for i, val in enumerate(equity):
        if val >= running_max:
            # New high - close any open drawdown
            if in_drawdown:
                depth_pct = (running_max - trough_value) / running_max * 100
                if depth_pct >= min_depth_pct:
                    periods.append(
                        DrawdownPeriod(
                            start_index=peak_idx,
                            trough_index=trough_idx,
                            end_index=i,
                            depth_pct=round(depth_pct, 2),
                            duration=trough_idx - peak_idx + 1,
                            recovery_duration=i - trough_idx,
                            recovered=True,
                        )
                    )
                in_drawdown = False
            running_max = val
            peak_idx = i
            trough_value = val
            trough_idx = i
        else:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
            if val < trough_value:
                trough_value = val
                trough_idx = i

    # Handle ongoing drawdown at end of series
    if in_drawdown:
        depth_pct = (running_max - trough_value) / running_max * 100
        if depth_pct >= min_depth_pct:
            periods.append(
                DrawdownPeriod(
                    start_index=peak_idx,
                    trough_index=trough_idx,
                    end_index=None,
                    depth_pct=round(depth_pct, 2),
                    duration=trough_idx - peak_idx + 1,
                    recovery_duration=None,
                    recovered=False,
                )
            )

    return periods


def _generate_drawdown_narrative(
    max_dd: float,
    max_dd_duration: int,
    current_dd: float,
    num_drawdowns: int,
    avg_recovery: float | None,
    severity: DrawdownSeverity,
    recovery_state: RecoveryState,
    context: str | None,
) -> str:
    """Generate natural language narrative for drawdown analysis."""
    parts: list[str] = []

    # Context prefix
    prefix = f"The {context}" if context else "The equity curve"

    # Max drawdown
    if max_dd == 0:
        parts.append(f"{prefix} shows no drawdowns - pure upward movement.")
        return " ".join(parts)

    # Severity description
    severity_desc = {
        DrawdownSeverity.MINIMAL: "minimal",
        DrawdownSeverity.MODERATE: "moderate",
        DrawdownSeverity.SIGNIFICANT: "significant",
        DrawdownSeverity.SEVERE: "severe",
        DrawdownSeverity.CATASTROPHIC: "catastrophic",
    }

    parts.append(
        f"{prefix} has {severity_desc[severity]} drawdown risk "
        f"(max {max_dd:.1f}% over {max_dd_duration} periods)."
    )

    # Current state
    if recovery_state == RecoveryState.AT_HIGH:
        parts.append("Currently at equity high.")
    elif recovery_state == RecoveryState.RECOVERING:
        parts.append(f"Currently recovering from {current_dd:.1f}% drawdown.")
    elif recovery_state == RecoveryState.IN_DRAWDOWN:
        parts.append(f"Currently in {current_dd:.1f}% drawdown.")
    else:
        parts.append("Fully recovered from last drawdown.")

    # Recovery stats
    if num_drawdowns > 1:
        if avg_recovery is not None:
            parts.append(
                f"{num_drawdowns} drawdown periods detected, "
                f"avg recovery: {avg_recovery:.0f} periods."
            )
        else:
            parts.append(f"{num_drawdowns} drawdown periods detected.")

    return " ".join(parts)


def describe_drawdown(
    equity: np.ndarray | list[float],
    context: str | None = None,
    min_depth_pct: float = 1.0,
) -> DrawdownResult:
    """Analyze drawdowns in an equity curve.

    Calculates maximum drawdown, identifies all drawdown periods,
    and provides semantic classification of drawdown severity.

    Args:
        equity: Array of equity/balance values (cumulative, not returns).
            Must be positive values representing account balance over time.
        context: Optional label for the data (e.g., "CLAUDE equity").
        min_depth_pct: Minimum drawdown depth to track (default 1%).

    Returns:
        DrawdownResult with complete drawdown analysis.

    Example:
        >>> equity = [10000, 10500, 10200, 9800, 9500, 10000, 10800]
        >>> result = describe_drawdown(equity, context="BTC strategy")
        >>> print(result.narrative)
        "The BTC strategy has moderate drawdown risk (max 9.5% over 3 periods).
         Currently at equity high."
    """
    # Convert to numpy array
    if isinstance(equity, list):
        equity_arr = np.array(equity, dtype=float)
    else:
        equity_arr = equity

    # Filter out NaN and Inf values
    valid_mask = np.isfinite(equity_arr)
    equity_arr = equity_arr[valid_mask]

    if len(equity_arr) < 2:
        return DrawdownResult(
            max_drawdown_pct=0.0,
            max_drawdown_duration=0,
            current_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            num_drawdowns=0,
            avg_recovery_periods=None,
            severity=DrawdownSeverity.MINIMAL,
            recovery_state=RecoveryState.AT_HIGH,
            drawdown_periods=(),
            narrative="Insufficient data for drawdown analysis.",
            context=context,
        )

    # Calculate running maximum and drawdown series
    running_max = np.maximum.accumulate(equity_arr)
    drawdown_series = (running_max - equity_arr) / running_max * 100

    # Max drawdown
    max_dd_pct = float(np.max(drawdown_series))

    # Current drawdown
    current_dd_pct = float(drawdown_series[-1])

    # Find all drawdown periods
    periods = _find_drawdown_periods(equity_arr, min_depth_pct)

    # Calculate aggregate stats
    num_drawdowns = len(periods)
    avg_dd_pct = 0.0
    avg_recovery: float | None = None
    max_dd_duration = 0

    if periods:
        avg_dd_pct = float(np.mean([p.depth_pct for p in periods]))
        recoveries = [p.recovery_duration for p in periods if p.recovery_duration is not None]
        if recoveries:
            avg_recovery = float(np.mean(recoveries))

        # Find duration of max drawdown period
        for p in periods:
            if abs(p.depth_pct - max_dd_pct) < 0.1:  # Match max DD period
                max_dd_duration = p.duration
                break

    # Classifications
    severity = _classify_severity(max_dd_pct)
    recovery_state = _determine_recovery_state(equity_arr, current_dd_pct)

    # Generate narrative
    narrative = _generate_drawdown_narrative(
        max_dd=max_dd_pct,
        max_dd_duration=max_dd_duration,
        current_dd=current_dd_pct,
        num_drawdowns=num_drawdowns,
        avg_recovery=avg_recovery,
        severity=severity,
        recovery_state=recovery_state,
        context=context,
    )

    # Limit periods to 10 most significant (by depth)
    sorted_periods = sorted(periods, key=lambda p: p.depth_pct, reverse=True)[:10]

    return DrawdownResult(
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_duration=max_dd_duration,
        current_drawdown_pct=round(current_dd_pct, 2),
        avg_drawdown_pct=round(avg_dd_pct, 2),
        num_drawdowns=num_drawdowns,
        avg_recovery_periods=round(avg_recovery, 1) if avg_recovery else None,
        severity=severity,
        recovery_state=recovery_state,
        drawdown_periods=tuple(sorted_periods),
        narrative=narrative,
        context=context,
    )

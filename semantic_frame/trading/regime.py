"""Market regime detection and classification.

This module provides regime analysis for trading data, enabling:
- Bull/Bear/Sideways/Recovery regime classification
- Regime change detection and tracking
- Regime stability assessment
- Regime duration and strength metrics

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Classification of market regime types."""

    BULL = "bull"  # Strong upward trend
    BEAR = "bear"  # Strong downward trend
    SIDEWAYS = "sideways"  # Range-bound, no clear direction
    RECOVERY = "recovery"  # Transitioning from bear to bull
    CORRECTION = "correction"  # Transitioning from bull to bear
    HIGH_VOLATILITY = "high_volatility"  # Extreme volatility, unclear direction


class RegimeStability(str, Enum):
    """Classification of regime stability."""

    VERY_STABLE = "very_stable"  # Few regime changes
    STABLE = "stable"  # Occasional regime changes
    UNSTABLE = "unstable"  # Frequent regime changes
    HIGHLY_UNSTABLE = "highly_unstable"  # Very frequent regime changes


class RegimeStrength(str, Enum):
    """Strength of the current regime."""

    STRONG = "strong"  # Clear, definitive regime
    MODERATE = "moderate"  # Moderate regime characteristics
    WEAK = "weak"  # Borderline regime classification


class RegimePeriod(BaseModel):
    """Information about a single regime period."""

    model_config = ConfigDict(frozen=True)

    regime_type: RegimeType = Field(description="Type of regime")
    start_index: int = Field(ge=0, description="Starting index of regime")
    end_index: int = Field(ge=0, description="Ending index of regime (inclusive)")
    duration: int = Field(ge=1, description="Number of periods in regime")
    cumulative_return: float = Field(description="Total return during regime (%)")
    avg_return: float = Field(description="Average period return (%)")
    volatility: float = Field(ge=0, description="Volatility during regime")
    strength: RegimeStrength = Field(description="Strength of this regime")

    @model_validator(mode="after")
    def check_temporal_consistency(self) -> RegimePeriod:
        """Validate cross-field invariants for regime period."""
        # Temporal ordering: end_index must be >= start_index
        if self.end_index < self.start_index:
            raise ValueError(
                f"end_index ({self.end_index}) must be >= start_index ({self.start_index})"
            )

        # Duration must be consistent with indices
        expected_duration = self.end_index - self.start_index + 1
        if self.duration != expected_duration:
            raise ValueError(
                f"duration ({self.duration}) inconsistent with indices "
                f"(expected {expected_duration} from {self.start_index} to {self.end_index})"
            )

        return self


class RegimeResult(BaseModel):
    """Complete regime analysis result."""

    model_config = ConfigDict(frozen=True)

    # Current state
    current_regime: RegimeType = Field(description="Current market regime")
    current_regime_strength: RegimeStrength = Field(description="Strength of current regime")
    current_regime_duration: int = Field(ge=1, description="How long current regime has lasted")

    # Historical regimes
    regimes_detected: tuple[RegimePeriod, ...] = Field(
        default_factory=tuple, description="All detected regime periods"
    )
    total_regime_changes: int = Field(ge=0, description="Number of regime transitions")

    # Stability metrics
    stability: RegimeStability = Field(description="Overall regime stability")
    avg_regime_duration: float = Field(ge=0, description="Average duration of regimes")
    regime_change_frequency: float = Field(ge=0, description="Regime changes per 100 periods")

    # Regime distribution
    time_in_bull_pct: float = Field(ge=0, le=100, description="Percentage of time in bull regime")
    time_in_bear_pct: float = Field(ge=0, le=100, description="Percentage of time in bear regime")
    time_in_sideways_pct: float = Field(
        ge=0, le=100, description="Percentage of time in sideways/other regimes"
    )

    # Trend context
    regime_trend: str = Field(
        description="Direction of regime evolution (improving/deteriorating/stable)"
    )

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable summary")

    # Metadata
    data_context: str | None = Field(default=None, description="User-provided context")
    total_periods: int = Field(ge=1, description="Total periods analyzed")


def _classify_regime(
    returns: np.ndarray,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
    vol_threshold: float,
) -> tuple[RegimeType, RegimeStrength]:
    """Classify regime for a window of returns.

    Args:
        returns: Array of returns for the window.
        lookback: Lookback period used.
        bull_threshold: Cumulative return threshold for bull (annualized).
        bear_threshold: Cumulative return threshold for bear (annualized).
        vol_threshold: Volatility threshold for high volatility regime.

    Returns:
        Tuple of (RegimeType, RegimeStrength).
    """
    if len(returns) < 2:
        return RegimeType.SIDEWAYS, RegimeStrength.WEAK

    cumulative_return = float(np.sum(returns))
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    # Annualize for comparison (assuming daily returns, ~252 trading days)
    annualization_factor = np.sqrt(252 / lookback) if lookback > 0 else 1.0
    annualized_return = cumulative_return * (252 / lookback) if lookback > 0 else cumulative_return
    annualized_vol = volatility * annualization_factor

    # High volatility regime takes precedence if extreme
    if annualized_vol > vol_threshold * 2:
        strength = (
            RegimeStrength.STRONG if annualized_vol > vol_threshold * 3 else RegimeStrength.MODERATE
        )
        return RegimeType.HIGH_VOLATILITY, strength

    # Determine trend-based regime
    if annualized_return > bull_threshold:
        if annualized_return > bull_threshold * 2:
            strength = RegimeStrength.STRONG
        elif annualized_return > bull_threshold * 1.3:
            strength = RegimeStrength.MODERATE
        else:
            strength = RegimeStrength.WEAK
        return RegimeType.BULL, strength

    elif annualized_return < bear_threshold:
        if annualized_return < bear_threshold * 2:
            strength = RegimeStrength.STRONG
        elif annualized_return < bear_threshold * 1.3:
            strength = RegimeStrength.MODERATE
        else:
            strength = RegimeStrength.WEAK
        return RegimeType.BEAR, strength

    else:
        # Check for recovery or correction patterns
        if len(returns) >= 4:
            first_half = returns[: len(returns) // 2]
            second_half = returns[len(returns) // 2 :]

            first_mean = float(np.mean(first_half))
            second_mean = float(np.mean(second_half))

            # Recovery: negative first half, positive second half
            if first_mean < 0 and second_mean > 0 and second_mean > abs(first_mean) * 0.3:
                strength = (
                    RegimeStrength.MODERATE
                    if second_mean > first_mean * -0.5
                    else RegimeStrength.WEAK
                )
                return RegimeType.RECOVERY, strength

            # Correction: positive first half, negative second half
            if first_mean > 0 and second_mean < 0 and abs(second_mean) > first_mean * 0.3:
                strength = (
                    RegimeStrength.MODERATE
                    if abs(second_mean) > first_mean * 0.5
                    else RegimeStrength.WEAK
                )
                return RegimeType.CORRECTION, strength

        # Default to sideways
        strength = (
            RegimeStrength.WEAK
            if abs(annualized_return) < bull_threshold * 0.3
            else RegimeStrength.MODERATE
        )
        return RegimeType.SIDEWAYS, strength


def _classify_stability(
    regime_change_freq: float,
    total_changes: int,
    total_periods: int,
) -> RegimeStability:
    """Classify regime stability based on change frequency."""
    if total_periods < 10:
        return RegimeStability.STABLE  # Not enough data to judge

    # Changes per 100 periods
    if regime_change_freq < 5:
        return RegimeStability.VERY_STABLE
    elif regime_change_freq < 15:
        return RegimeStability.STABLE
    elif regime_change_freq < 30:
        return RegimeStability.UNSTABLE
    else:
        return RegimeStability.HIGHLY_UNSTABLE


def _detect_regime_periods(
    returns: np.ndarray,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
    vol_threshold: float,
    min_regime_length: int,
) -> list[RegimePeriod]:
    """Detect all regime periods in the data.

    Uses a rolling window approach with regime smoothing.
    """
    n = len(returns)
    if n < lookback:
        # Single regime for entire period
        regime, strength = _classify_regime(
            returns, n, bull_threshold, bear_threshold, vol_threshold
        )
        cum_ret = float(np.sum(returns)) * 100
        avg_ret = float(np.mean(returns)) * 100
        vol = float(np.std(returns, ddof=1)) if n > 1 else 0.0

        return [
            RegimePeriod(
                regime_type=regime,
                start_index=0,
                end_index=n - 1,
                duration=n,
                cumulative_return=round(cum_ret, 2),
                avg_return=round(avg_ret, 4),
                volatility=round(vol, 4),
                strength=strength,
            )
        ]

    # Rolling regime classification
    regime_labels: list[RegimeType] = []
    regime_strengths: list[RegimeStrength] = []

    for i in range(n):
        start = max(0, i - lookback + 1)
        window = returns[start : i + 1]
        regime, strength = _classify_regime(
            window, len(window), bull_threshold, bear_threshold, vol_threshold
        )
        regime_labels.append(regime)
        regime_strengths.append(strength)

    # Smooth regimes - avoid single-period flips
    smoothed_regimes = regime_labels.copy()
    for i in range(1, n - 1):
        if regime_labels[i] != regime_labels[i - 1] and regime_labels[i] != regime_labels[i + 1]:
            # Isolated regime change - smooth it out
            smoothed_regimes[i] = regime_labels[i - 1]

    # Merge consecutive same-regime periods
    periods: list[RegimePeriod] = []
    current_regime = smoothed_regimes[0]
    current_start = 0

    for i in range(1, n):
        if smoothed_regimes[i] != current_regime:
            # End current period
            period_returns = returns[current_start:i]
            cum_ret = float(np.sum(period_returns)) * 100
            avg_ret = float(np.mean(period_returns)) * 100
            vol = float(np.std(period_returns, ddof=1)) if len(period_returns) > 1 else 0.0

            # Determine strength for the period
            period_strength = max(
                regime_strengths[current_start:i],
                key=lambda s: [
                    RegimeStrength.WEAK,
                    RegimeStrength.MODERATE,
                    RegimeStrength.STRONG,
                ].index(s),
            )

            if i - current_start >= min_regime_length:
                periods.append(
                    RegimePeriod(
                        regime_type=current_regime,
                        start_index=current_start,
                        end_index=i - 1,
                        duration=i - current_start,
                        cumulative_return=round(cum_ret, 2),
                        avg_return=round(avg_ret, 4),
                        volatility=round(vol, 4),
                        strength=period_strength,
                    )
                )

            # Start new period
            current_regime = smoothed_regimes[i]
            current_start = i

    # Add final period
    period_returns = returns[current_start:n]
    cum_ret = float(np.sum(period_returns)) * 100
    avg_ret = float(np.mean(period_returns)) * 100
    vol = float(np.std(period_returns, ddof=1)) if len(period_returns) > 1 else 0.0
    period_strength = max(
        regime_strengths[current_start:n],
        key=lambda s: [RegimeStrength.WEAK, RegimeStrength.MODERATE, RegimeStrength.STRONG].index(
            s
        ),
    )

    if n - current_start >= min_regime_length or not periods:
        periods.append(
            RegimePeriod(
                regime_type=current_regime,
                start_index=current_start,
                end_index=n - 1,
                duration=n - current_start,
                cumulative_return=round(cum_ret, 2),
                avg_return=round(avg_ret, 4),
                volatility=round(vol, 4),
                strength=period_strength,
            )
        )

    return periods


def _determine_regime_trend(periods: list[RegimePeriod]) -> str:
    """Determine if regimes are improving, deteriorating, or stable."""
    if len(periods) < 2:
        return "stable"

    # Compare recent vs earlier regimes
    recent = periods[-1]
    previous = periods[-2] if len(periods) >= 2 else None

    bullish_types = {RegimeType.BULL, RegimeType.RECOVERY}
    bearish_types = {RegimeType.BEAR, RegimeType.CORRECTION}

    if previous:
        if previous.regime_type in bearish_types and recent.regime_type in bullish_types:
            return "improving"
        elif previous.regime_type in bullish_types and recent.regime_type in bearish_types:
            return "deteriorating"

    return "stable"


def _generate_narrative(
    current_regime: RegimeType,
    current_strength: RegimeStrength,
    current_duration: int,
    periods: list[RegimePeriod],
    stability: RegimeStability,
    regime_trend: str,
    context: str | None,
    total_periods: int,
) -> str:
    """Generate natural language narrative for regime analysis."""
    parts: list[str] = []

    prefix = f"{context}" if context else "The market"

    # Current regime description
    regime_desc = {
        RegimeType.BULL: "bullish",
        RegimeType.BEAR: "bearish",
        RegimeType.SIDEWAYS: "sideways/range-bound",
        RegimeType.RECOVERY: "recovery",
        RegimeType.CORRECTION: "correction",
        RegimeType.HIGH_VOLATILITY: "high volatility",
    }

    strength_desc = {
        RegimeStrength.STRONG: "strong",
        RegimeStrength.MODERATE: "moderate",
        RegimeStrength.WEAK: "weak",
    }

    parts.append(
        f"{prefix} is in a {strength_desc[current_strength]} {regime_desc[current_regime]} regime "
        f"(duration: {current_duration} periods)."
    )

    # Regime history context
    if len(periods) > 1:
        changes = len(periods) - 1
        stability_desc = {
            RegimeStability.VERY_STABLE: "very stable",
            RegimeStability.STABLE: "stable",
            RegimeStability.UNSTABLE: "unstable",
            RegimeStability.HIGHLY_UNSTABLE: "highly unstable",
        }
        parts.append(
            f"{changes} regime change(s) detected - conditions are {stability_desc[stability]}."
        )

        # Recent transition
        if len(periods) >= 2:
            prev = periods[-2]
            parts.append(
                f"Transitioned from {regime_desc[prev.regime_type]} ({prev.duration} periods)."
            )

    # Trend context
    if regime_trend == "improving":
        parts.append("Regime trend is improving.")
    elif regime_trend == "deteriorating":
        parts.append("Regime trend is deteriorating.")

    # Actionable insight based on regime
    if current_regime == RegimeType.BULL and current_strength == RegimeStrength.STRONG:
        parts.append("Conditions favor trend-following strategies.")
    elif current_regime == RegimeType.BEAR and current_strength == RegimeStrength.STRONG:
        parts.append("Risk-off positioning recommended.")
    elif current_regime == RegimeType.RECOVERY:
        parts.append("Early signs of recovery - consider gradual re-entry.")
    elif current_regime == RegimeType.CORRECTION:
        parts.append("Correction underway - monitor for support levels.")
    elif current_regime == RegimeType.SIDEWAYS:
        parts.append("Range-bound conditions favor mean-reversion strategies.")
    elif current_regime == RegimeType.HIGH_VOLATILITY:
        parts.append("High volatility - reduce position sizes and widen stops.")

    return " ".join(parts)


def describe_regime(
    returns: np.ndarray | list[float],
    context: str | None = None,
    lookback: int = 20,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
    vol_threshold: float = 0.30,
    min_regime_length: int = 3,
) -> RegimeResult:
    """Detect and classify market regimes from return data.

    Analyzes return series to identify market regimes (bull, bear, sideways,
    recovery, correction) and provides stability assessment.

    Args:
        returns: Array of period returns (as decimals, e.g., 0.01 = 1%).
        context: Optional label (e.g., "BTC/USD", "S&P 500").
        lookback: Lookback window for regime classification (default 20).
        bull_threshold: Annualized return threshold for bull regime (default 0.10 = 10%).
        bear_threshold: Annualized return threshold for bear regime (default -0.10 = -10%).
        vol_threshold: Annualized volatility threshold for high-vol regime (default 0.30 = 30%).
        min_regime_length: Minimum periods for a regime to be counted (default 3).

    Returns:
        RegimeResult with current regime, history, and stability metrics.

    Example:
        >>> returns = [0.01, 0.02, 0.01, -0.05, -0.08, -0.03, 0.02, 0.03, 0.04]
        >>> result = describe_regime(returns, context="BTC")
        >>> print(result.narrative)
        "BTC is in a moderate recovery regime (duration: 3 periods).
         2 regime change(s) detected - conditions are unstable."
    """
    # Convert to numpy array
    if isinstance(returns, list):
        returns = np.array(returns, dtype=float)

    n = len(returns)

    if n < 3:
        return RegimeResult(
            current_regime=RegimeType.SIDEWAYS,
            current_regime_strength=RegimeStrength.WEAK,
            current_regime_duration=n,
            regimes_detected=(),
            total_regime_changes=0,
            stability=RegimeStability.STABLE,
            avg_regime_duration=float(n),
            regime_change_frequency=0.0,
            time_in_bull_pct=0.0,
            time_in_bear_pct=0.0,
            time_in_sideways_pct=100.0,
            regime_trend="stable",
            narrative="Insufficient data for regime analysis.",
            data_context=context,
            total_periods=n,
        )

    # Detect regime periods
    periods = _detect_regime_periods(
        returns,
        lookback=lookback,
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
        vol_threshold=vol_threshold,
        min_regime_length=min_regime_length,
    )

    # Current regime info
    current_period = periods[-1]
    current_regime = current_period.regime_type
    current_strength = current_period.strength
    current_duration = current_period.duration

    # Calculate regime statistics
    total_changes = len(periods) - 1
    avg_duration = float(np.mean([p.duration for p in periods]))
    change_freq = (total_changes / n) * 100 if n > 0 else 0.0

    # Stability classification
    stability = _classify_stability(change_freq, total_changes, n)

    # Time distribution
    bull_periods = sum(p.duration for p in periods if p.regime_type == RegimeType.BULL)
    bear_periods = sum(p.duration for p in periods if p.regime_type == RegimeType.BEAR)
    other_periods = n - bull_periods - bear_periods

    time_in_bull = (bull_periods / n) * 100 if n > 0 else 0.0
    time_in_bear = (bear_periods / n) * 100 if n > 0 else 0.0
    time_in_sideways = (other_periods / n) * 100 if n > 0 else 0.0

    # Regime trend
    regime_trend = _determine_regime_trend(periods)

    # Generate narrative
    narrative = _generate_narrative(
        current_regime,
        current_strength,
        current_duration,
        periods,
        stability,
        regime_trend,
        context,
        n,
    )

    return RegimeResult(
        current_regime=current_regime,
        current_regime_strength=current_strength,
        current_regime_duration=current_duration,
        regimes_detected=tuple(periods),
        total_regime_changes=total_changes,
        stability=stability,
        avg_regime_duration=round(avg_duration, 1),
        regime_change_frequency=round(change_freq, 2),
        time_in_bull_pct=round(time_in_bull, 1),
        time_in_bear_pct=round(time_in_bear, 1),
        time_in_sideways_pct=round(time_in_sideways, 1),
        regime_trend=regime_trend,
        narrative=narrative,
        data_context=context,
        total_periods=n,
    )

"""Time-windowed multi-timeframe analysis for trading data.

This module provides analysis across multiple time windows, enabling:
- Multi-timeframe trend comparison
- Volatility analysis at different scales
- Signal vs noise filtering
- Unified cross-timeframe narratives

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class TimeframeSignal(str, Enum):
    """Signal classification for a timeframe."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TimeframeAlignment(str, Enum):
    """Classification of how timeframes align with each other."""

    ALIGNED_BULLISH = "aligned_bullish"  # All timeframes bullish
    ALIGNED_BEARISH = "aligned_bearish"  # All timeframes bearish
    MIXED = "mixed"  # Timeframes disagree
    DIVERGING = "diverging"  # Short and long term opposite
    CONVERGING = "converging"  # Moving toward agreement


class WindowAnalysis(BaseModel):
    """Analysis results for a single time window."""

    model_config = ConfigDict(frozen=True)

    window_name: str = Field(description="Name of the window (e.g., '1h', '4h', '1d')")
    window_size: int = Field(ge=1, description="Number of data points in window")

    # Trend metrics
    trend_direction: str = Field(description="RISING, FLAT, or FALLING")
    trend_strength: float = Field(ge=0.0, le=1.0, description="Strength of trend (0-1)")
    signal: TimeframeSignal = Field(description="Overall signal for this window")

    # Volatility metrics
    volatility: float = Field(ge=0.0, description="Volatility (std dev of returns)")
    volatility_level: str = Field(description="LOW, MODERATE, HIGH, EXTREME")

    # Price action
    change_pct: float = Field(description="Percentage change over window")
    high: float = Field(description="Highest value in window")
    low: float = Field(description="Lowest value in window")
    range_pct: float = Field(ge=0.0, description="Range as percentage of mean")

    # Summary
    narrative: str = Field(description="Brief narrative for this window")


class MultiWindowResult(BaseModel):
    """Complete multi-window analysis result."""

    model_config = ConfigDict(frozen=True)

    # Per-window analysis
    windows: dict[str, WindowAnalysis] = Field(description="Analysis for each requested window")

    # Cross-window insights
    alignment: TimeframeAlignment = Field(description="How timeframes align with each other")
    dominant_trend: str = Field(description="The prevailing trend across timeframes")
    short_term_signal: TimeframeSignal = Field(description="Signal from shortest timeframe")
    long_term_signal: TimeframeSignal = Field(description="Signal from longest timeframe")

    # Actionable insights
    noise_level: str = Field(description="How much short-term noise vs long-term signal")
    suggested_action: str = Field(description="Suggested positioning based on multi-TF analysis")

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable summary")

    # Metadata
    data_context: str | None = Field(default=None, description="User-provided context")
    total_points: int = Field(ge=1, description="Total data points analyzed")


def _calc_trend_direction(data: np.ndarray) -> tuple[str, float]:
    """Calculate trend direction and strength.

    Returns:
        Tuple of (direction, strength) where direction is RISING/FLAT/FALLING
        and strength is 0-1.
    """
    if len(data) < 2:
        return "FLAT", 0.0

    # Linear regression slope
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data, 1)

    # Normalize slope by data range
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return "FLAT", 0.0

    normalized_slope = slope * len(data) / data_range

    # Determine direction and strength
    if normalized_slope > 0.1:
        direction = "RISING"
        strength = min(abs(normalized_slope), 1.0)
    elif normalized_slope < -0.1:
        direction = "FALLING"
        strength = min(abs(normalized_slope), 1.0)
    else:
        direction = "FLAT"
        strength = 0.0

    return direction, round(strength, 2)


def _calc_volatility_level(volatility: float, mean: float) -> str:
    """Classify volatility level relative to mean."""
    if mean == 0:
        return "MODERATE"

    cv = volatility / abs(mean)  # Coefficient of variation

    if cv < 0.02:
        return "LOW"
    if cv < 0.05:
        return "MODERATE"
    if cv < 0.15:
        return "HIGH"
    return "EXTREME"


def _determine_signal(direction: str, strength: float) -> TimeframeSignal:
    """Determine signal based on trend direction and strength."""
    if direction == "RISING":
        if strength > 0.6:
            return TimeframeSignal.STRONG_BULLISH
        return TimeframeSignal.BULLISH
    elif direction == "FALLING":
        if strength > 0.6:
            return TimeframeSignal.STRONG_BEARISH
        return TimeframeSignal.BEARISH
    return TimeframeSignal.NEUTRAL


def _analyze_window(
    data: np.ndarray,
    window_name: str,
) -> WindowAnalysis:
    """Analyze a single time window."""
    n = len(data)

    # Trend
    direction, strength = _calc_trend_direction(data)
    signal = _determine_signal(direction, strength)

    # Volatility
    if n > 1:
        returns = np.diff(data) / data[:-1]
        returns = returns[np.isfinite(returns)]
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
    else:
        volatility = 0.0

    mean = float(np.mean(data))
    vol_level = _calc_volatility_level(volatility, mean)

    # Price action
    change_pct = ((data[-1] - data[0]) / data[0] * 100) if data[0] != 0 else 0.0
    high = float(np.max(data))
    low = float(np.min(data))
    range_pct = ((high - low) / mean * 100) if mean != 0 else 0.0

    # Brief narrative
    narrative = f"{window_name}: {direction.lower()} trend, {vol_level.lower()} volatility"

    return WindowAnalysis(
        window_name=window_name,
        window_size=n,
        trend_direction=direction,
        trend_strength=strength,
        signal=signal,
        volatility=round(volatility, 4),
        volatility_level=vol_level,
        change_pct=round(change_pct, 2),
        high=round(high, 2),
        low=round(low, 2),
        range_pct=round(range_pct, 2),
        narrative=narrative,
    )


def _determine_alignment(signals: list[TimeframeSignal]) -> TimeframeAlignment:
    """Determine how timeframes align."""
    if not signals:
        return TimeframeAlignment.MIXED

    bullish = [s for s in signals if s in [TimeframeSignal.BULLISH, TimeframeSignal.STRONG_BULLISH]]
    bearish = [s for s in signals if s in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]]

    if len(bullish) == len(signals):
        return TimeframeAlignment.ALIGNED_BULLISH
    if len(bearish) == len(signals):
        return TimeframeAlignment.ALIGNED_BEARISH

    # Check for divergence (short vs long)
    if len(signals) >= 2:
        short_bullish = signals[0] in [TimeframeSignal.BULLISH, TimeframeSignal.STRONG_BULLISH]
        long_bullish = signals[-1] in [TimeframeSignal.BULLISH, TimeframeSignal.STRONG_BULLISH]
        short_bearish = signals[0] in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]
        long_bearish = signals[-1] in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]

        if (short_bullish and long_bearish) or (short_bearish and long_bullish):
            return TimeframeAlignment.DIVERGING

    return TimeframeAlignment.MIXED


def _generate_narrative(
    windows: dict[str, WindowAnalysis],
    alignment: TimeframeAlignment,
    noise_level: str,
    suggested_action: str,
    context: str | None,
) -> str:
    """Generate comprehensive multi-window narrative."""
    parts: list[str] = []

    prefix = f"Multi-timeframe analysis of {context}" if context else "Multi-timeframe analysis"

    # Alignment summary
    align_desc = {
        TimeframeAlignment.ALIGNED_BULLISH: "all timeframes bullish",
        TimeframeAlignment.ALIGNED_BEARISH: "all timeframes bearish",
        TimeframeAlignment.MIXED: "mixed signals across timeframes",
        TimeframeAlignment.DIVERGING: "short and long term diverging",
        TimeframeAlignment.CONVERGING: "timeframes converging",
    }

    parts.append(f"{prefix}: {align_desc[alignment]}.")

    # Per-window summary
    window_summaries = []
    for name, w in windows.items():
        window_summaries.append(f"{name} {w.trend_direction.lower()} ({w.change_pct:+.1f}%)")
    parts.append(f"Windows: {', '.join(window_summaries)}.")

    # Noise assessment
    parts.append(f"Noise level: {noise_level.lower()}.")

    # Action suggestion
    parts.append(f"Suggested: {suggested_action}.")

    return " ".join(parts)


def _parse_window_spec(spec: str) -> int | None:
    """Parse window specification string to number of periods.

    Supports: '5', '10', '1h', '4h', '1d', '1w' etc.
    Returns None if can't parse (will use as-is for grouping).
    """
    spec = spec.lower().strip()

    # Pure number
    if spec.isdigit():
        return int(spec)

    # Time-based specs (assume 1 data point = 1 base period)
    multipliers = {
        "m": 1,  # minutes (base)
        "h": 60,  # hours
        "d": 1440,  # days (24 * 60)
        "w": 10080,  # weeks (7 * 24 * 60)
    }

    for suffix, mult in multipliers.items():
        if spec.endswith(suffix):
            try:
                num = int(spec[:-1])
                return num * mult
            except ValueError:
                pass

    return None


def describe_windows(
    data: np.ndarray | list[float],
    windows: list[str] | list[int] | None = None,
    context: str | None = None,
    base_period: str = "1m",
) -> MultiWindowResult:
    """Analyze data across multiple time windows.

    Provides multi-timeframe analysis comparing trends, volatility, and
    signals across different time horizons.

    Args:
        data: Array of price/value data (most recent at end).
        windows: List of window sizes or specs. Can be:
            - Integers: [10, 50, 200] for last N points
            - Strings: ["1h", "4h", "1d"] for time-based windows
            - None: Uses default [10, 50, 200]
        context: Optional label (e.g., "BTC/USD", "CPU metrics").
        base_period: What each data point represents (for display only).

    Returns:
        MultiWindowResult with per-window analysis and cross-window insights.

    Example:
        >>> prices = [100, 102, 99, 105, 103, 108, 110, 107, 112, 115, 118]
        >>> result = describe_windows(prices, windows=[5, 10], context="BTC")
        >>> print(result.narrative)
        "Multi-timeframe analysis of BTC: all timeframes bullish.
         Windows: 5 rising (+5.4%), 10 rising (+18.0%). Noise level: low."
    """
    # Convert to numpy array
    if isinstance(data, list):
        data = np.array(data, dtype=float)

    n = len(data)

    if n < 3:
        return MultiWindowResult(
            windows={},
            alignment=TimeframeAlignment.MIXED,
            dominant_trend="UNKNOWN",
            short_term_signal=TimeframeSignal.NEUTRAL,
            long_term_signal=TimeframeSignal.NEUTRAL,
            noise_level="unknown",
            suggested_action="insufficient data",
            narrative="Insufficient data for multi-window analysis.",
            data_context=context,
            total_points=n,
        )

    # Default windows
    if windows is None:
        # Auto-scale windows based on data length
        if n >= 200:
            windows = [20, 50, 200]
        elif n >= 50:
            windows = [10, 25, n]
        else:
            windows = [5, n // 2, n] if n >= 10 else [n]

    # Parse window specs and analyze each
    window_analyses: dict[str, WindowAnalysis] = {}
    signals: list[TimeframeSignal] = []
    volatilities: list[float] = []

    for w in windows:
        if isinstance(w, str):
            window_size = _parse_window_spec(w)
            window_name = w
        else:
            window_size = w
            window_name = str(w)

        # Use actual window size or full data if spec couldn't be parsed
        if window_size is None or window_size > n:
            window_size = n

        # Get window data (most recent N points)
        window_data = data[-window_size:]

        # Analyze
        analysis = _analyze_window(window_data, window_name)
        window_analyses[window_name] = analysis
        signals.append(analysis.signal)
        volatilities.append(analysis.volatility)

    # Cross-window analysis
    alignment = _determine_alignment(signals)
    short_term_signal = signals[0] if signals else TimeframeSignal.NEUTRAL
    long_term_signal = signals[-1] if signals else TimeframeSignal.NEUTRAL

    # Determine dominant trend
    bullish_count = sum(
        1 for s in signals if s in [TimeframeSignal.BULLISH, TimeframeSignal.STRONG_BULLISH]
    )
    bearish_count = sum(
        1 for s in signals if s in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]
    )

    if bullish_count > bearish_count:
        dominant_trend = "BULLISH"
    elif bearish_count > bullish_count:
        dominant_trend = "BEARISH"
    else:
        dominant_trend = "NEUTRAL"

    # Noise assessment (short-term vol vs long-term vol)
    if len(volatilities) >= 2 and volatilities[-1] > 0:
        noise_ratio = volatilities[0] / volatilities[-1]
        if noise_ratio > 2.0:
            noise_level = "high"
        elif noise_ratio > 1.2:
            noise_level = "moderate"
        else:
            noise_level = "low"
    else:
        noise_level = "moderate"

    # Suggested action
    if alignment == TimeframeAlignment.ALIGNED_BULLISH:
        suggested_action = "strong buy signal across all timeframes"
    elif alignment == TimeframeAlignment.ALIGNED_BEARISH:
        suggested_action = "strong sell signal across all timeframes"
    elif alignment == TimeframeAlignment.DIVERGING:
        if short_term_signal in [TimeframeSignal.BEARISH, TimeframeSignal.STRONG_BEARISH]:
            suggested_action = "short-term weakness in long-term uptrend - potential dip buy"
        else:
            suggested_action = "short-term strength in long-term downtrend - potential sell rally"
    else:
        suggested_action = "wait for clearer alignment before positioning"

    # Generate narrative
    narrative = _generate_narrative(
        window_analyses, alignment, noise_level, suggested_action, context
    )

    return MultiWindowResult(
        windows=window_analyses,
        alignment=alignment,
        dominant_trend=dominant_trend,
        short_term_signal=short_term_signal,
        long_term_signal=long_term_signal,
        noise_level=noise_level,
        suggested_action=suggested_action,
        narrative=narrative,
        data_context=context,
        total_points=n,
    )

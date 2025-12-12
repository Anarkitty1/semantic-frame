"""Trading-specific enums for semantic classification.

These enums provide the vocabulary for describing trading performance,
risk characteristics, and drawdown severity in token-efficient natural language.

Each enum includes threshold documentation for reproducibility.
"""

from enum import Enum


class DrawdownSeverity(str, Enum):
    """Classification of drawdown severity.

    Thresholds (max drawdown percentage):
        - MINIMAL: < 5%
        - MODERATE: 5% - 15%
        - SIGNIFICANT: 15% - 30%
        - SEVERE: 30% - 50%
        - CATASTROPHIC: >= 50%
    """

    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class PerformanceRating(str, Enum):
    """Classification of overall trading performance.

    Based on composite score of win rate, profit factor, and risk-adjusted returns.

    Thresholds (composite score 0-100):
        - EXCELLENT: >= 80
        - GOOD: 60-79
        - AVERAGE: 40-59
        - BELOW_AVERAGE: 20-39
        - POOR: < 20
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"


class RiskProfile(str, Enum):
    """Classification of risk-taking behavior.

    Based on volatility of returns and drawdown characteristics.

    Thresholds (return volatility / avg drawdown):
        - CONSERVATIVE: volatility < 5%, max DD < 10%
        - MODERATE: volatility 5-15%, max DD 10-25%
        - AGGRESSIVE: volatility 15-30%, max DD 25-40%
        - VERY_AGGRESSIVE: volatility > 30% or max DD > 40%
    """

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class ConsistencyRating(str, Enum):
    """Classification of return consistency.

    Based on winning streak patterns and return distribution.

    Thresholds (based on win rate stability and streak analysis):
        - HIGHLY_CONSISTENT: Low variance in rolling win rates, few losing streaks
        - CONSISTENT: Moderate variance, occasional losing streaks
        - INCONSISTENT: High variance, frequent losing streaks
        - ERRATIC: Very high variance, unpredictable patterns
    """

    HIGHLY_CONSISTENT = "highly_consistent"
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    ERRATIC = "erratic"


class RecoveryState(str, Enum):
    """Classification of drawdown recovery status.

    Describes the current state of recovery from a drawdown.
    """

    FULLY_RECOVERED = "fully_recovered"
    RECOVERING = "recovering"
    IN_DRAWDOWN = "in_drawdown"
    AT_HIGH = "at_equity_high"

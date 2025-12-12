"""Trading-specific semantic analysis tools.

This module provides trading-optimized analysis functions that extend
semantic-frame's core capabilities for trading agents and financial applications.

Key functions:
- describe_drawdown: Analyze equity curve drawdowns
- describe_trading_performance: Calculate win rate, profit factor, Sharpe, etc.
- describe_rankings: Compare multiple trading agents/strategies
- describe_anomalies: Enhanced anomaly detection with severity and type classification
- describe_windows: Multi-timeframe analysis
- describe_regime: Market regime detection and classification
- describe_allocation: Position sizing and portfolio allocation suggestions
"""

from semantic_frame.trading.allocation import (
    AllocationMethod,
    AllocationResult,
    AssetAnalysis,
    CorrelationInsight,
    DiversificationLevel,
    RiskLevel,
    describe_allocation,
)
from semantic_frame.trading.anomalies import (
    AnomalyFrequency,
    AnomalySeverity,
    AnomalyType,
    EnhancedAnomaly,
    EnhancedAnomalyResult,
    describe_anomalies,
)
from semantic_frame.trading.drawdown import describe_drawdown
from semantic_frame.trading.enums import (
    ConsistencyRating,
    DrawdownSeverity,
    PerformanceRating,
    RecoveryState,
    RiskProfile,
)
from semantic_frame.trading.metrics import describe_trading_performance
from semantic_frame.trading.rankings import describe_rankings
from semantic_frame.trading.regime import (
    RegimePeriod,
    RegimeResult,
    RegimeStability,
    RegimeStrength,
    RegimeType,
    describe_regime,
)
from semantic_frame.trading.schemas import (
    AgentRanking,
    DrawdownPeriod,
    DrawdownResult,
    RankingsResult,
    TradingMetrics,
    TradingPerformanceResult,
)
from semantic_frame.trading.windows import (
    MultiWindowResult,
    TimeframeAlignment,
    TimeframeSignal,
    WindowAnalysis,
    describe_windows,
)

__all__ = [
    # Functions
    "describe_drawdown",
    "describe_trading_performance",
    "describe_rankings",
    "describe_anomalies",
    "describe_windows",
    "describe_regime",
    "describe_allocation",
    # Enums - Drawdown & Performance
    "DrawdownSeverity",
    "PerformanceRating",
    "RiskProfile",
    "ConsistencyRating",
    "RecoveryState",
    # Enums - Anomalies
    "AnomalySeverity",
    "AnomalyType",
    "AnomalyFrequency",
    # Enums - Windows
    "TimeframeSignal",
    "TimeframeAlignment",
    # Enums - Regime
    "RegimeType",
    "RegimeStability",
    "RegimeStrength",
    # Enums - Allocation
    "RiskLevel",
    "DiversificationLevel",
    "AllocationMethod",
    # Schemas - Drawdown & Performance
    "DrawdownPeriod",
    "DrawdownResult",
    "TradingMetrics",
    "TradingPerformanceResult",
    "AgentRanking",
    "RankingsResult",
    # Schemas - Anomalies
    "EnhancedAnomaly",
    "EnhancedAnomalyResult",
    # Schemas - Windows
    "WindowAnalysis",
    "MultiWindowResult",
    # Schemas - Regime
    "RegimePeriod",
    "RegimeResult",
    # Schemas - Allocation
    "AssetAnalysis",
    "CorrelationInsight",
    "AllocationResult",
]

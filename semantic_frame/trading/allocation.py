"""Position sizing and portfolio allocation suggestions.

This module provides allocation analysis for multi-asset portfolios, enabling:
- Risk-based position sizing
- Correlation-aware diversification
- Target volatility optimization
- Portfolio risk metrics

DISCLAIMER: This module provides educational/informational analysis only.
It is NOT financial advice. Always consult qualified professionals.

All calculations are deterministic (NumPy-based) - no LLM involvement.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Portfolio risk classification."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DiversificationLevel(str, Enum):
    """Diversification quality classification."""

    POOR = "poor"  # High correlation, little benefit
    LIMITED = "limited"  # Some diversification
    MODERATE = "moderate"  # Decent diversification
    GOOD = "good"  # Well diversified
    EXCELLENT = "excellent"  # Highly diversified


class AllocationMethod(str, Enum):
    """Method used for allocation calculation."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    TARGET_VOL = "target_vol"


class AssetAnalysis(BaseModel):
    """Analysis for a single asset."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Asset identifier")
    annualized_return: float = Field(description="Annualized return (%)")
    annualized_volatility: float = Field(description="Annualized volatility (%)")
    sharpe_ratio: float | None = Field(default=None, description="Sharpe ratio")
    suggested_weight: float = Field(ge=0, le=1, description="Suggested portfolio weight")
    contribution_to_risk: float = Field(ge=0, description="Contribution to portfolio risk (%)")


class CorrelationInsight(BaseModel):
    """Insight about correlation between assets."""

    model_config = ConfigDict(frozen=True)

    asset_1: str = Field(description="First asset")
    asset_2: str = Field(description="Second asset")
    correlation: float = Field(ge=-1, le=1, description="Correlation coefficient")
    relationship: str = Field(description="Description of relationship")


class AllocationResult(BaseModel):
    """Complete allocation analysis result."""

    model_config = ConfigDict(frozen=True)

    # Suggested allocation
    suggested_weights: dict[str, float] = Field(description="Suggested weight for each asset (0-1)")
    allocation_method: AllocationMethod = Field(description="Method used for allocation")

    # Portfolio metrics
    portfolio_return: float = Field(description="Expected portfolio return (%)")
    portfolio_volatility: float = Field(description="Portfolio volatility (%)")
    portfolio_sharpe: float | None = Field(default=None, description="Portfolio Sharpe ratio")
    risk_level: RiskLevel = Field(description="Overall risk classification")

    # Diversification
    diversification_score: float = Field(ge=0, le=1, description="Diversification score (0-1)")
    diversification_level: DiversificationLevel = Field(description="Diversification quality")

    # Per-asset analysis
    asset_analyses: tuple[AssetAnalysis, ...] = Field(description="Analysis for each asset")

    # Correlation insights
    correlation_insights: tuple[CorrelationInsight, ...] = Field(
        default_factory=tuple, description="Key correlation insights"
    )
    avg_correlation: float = Field(ge=-1, le=1, description="Average pairwise correlation")

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable summary")
    disclaimer: str = Field(
        default="This is educational analysis only, not financial advice.",
        description="Required disclaimer",
    )

    # Metadata
    data_context: str | None = Field(default=None, description="User-provided context")
    num_assets: int = Field(ge=1, description="Number of assets analyzed")

    @model_validator(mode="after")
    def check_allocation_consistency(self) -> AllocationResult:
        """Validate cross-field invariants for allocation."""
        # Weights must sum to approximately 1.0 (small tolerance for floating point)
        weight_sum = sum(self.suggested_weights.values())
        if weight_sum < 0.99 or weight_sum > 1.01:
            raise ValueError(
                f"suggested_weights must sum to approximately 1.0, got {weight_sum:.4f}"
            )

        # num_assets must match suggested_weights count
        if len(self.suggested_weights) != self.num_assets:
            raise ValueError(
                f"num_assets ({self.num_assets}) must match "
                f"suggested_weights count ({len(self.suggested_weights)})"
            )

        # num_assets must match asset_analyses count
        if len(self.asset_analyses) != self.num_assets:
            raise ValueError(
                f"num_assets ({self.num_assets}) must match "
                f"asset_analyses count ({len(self.asset_analyses)})"
            )

        return self


def _calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series."""
    # Suppress warnings for division by zero/inf - we filter them afterward
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(prices) / prices[:-1]
    non_finite_count = int(np.sum(~np.isfinite(returns)))
    if non_finite_count > 0:
        logger.warning(
            "Filtered %d non-finite return values (likely due to zero/negative prices)",
            non_finite_count,
        )
    result: np.ndarray = returns[np.isfinite(returns)]
    return result


def _annualize_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualize returns."""
    if len(returns) == 0:
        return 0.0
    mean_return = float(np.mean(returns))
    return mean_return * periods_per_year * 100  # As percentage


def _annualize_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualize volatility."""
    if len(returns) < 2:
        logger.warning("Insufficient data for volatility calculation (<2 returns), returning 0.0")
        return 0.0
    std = float(np.std(returns, ddof=1))
    return float(std * np.sqrt(periods_per_year) * 100)  # As percentage


def _calculate_sharpe(ann_return: float, ann_vol: float, risk_free: float = 2.0) -> float | None:
    """Calculate Sharpe ratio."""
    if ann_vol == 0:
        return None
    return (ann_return - risk_free) / ann_vol


def _calculate_correlation_matrix(returns_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Calculate correlation matrix from returns."""
    assets = list(returns_dict.keys())
    n = len(assets)

    if n < 2:
        return np.array([[1.0]])

    # Align returns to same length
    min_len = min(len(r) for r in returns_dict.values())
    aligned = np.column_stack([returns_dict[a][-min_len:] for a in assets])

    # Suppress warnings for zero-variance assets (identical prices)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(aligned.T)


def _calculate_covariance_matrix(
    returns_dict: dict[str, np.ndarray],
    periods_per_year: int = 252,
) -> np.ndarray:
    """Calculate annualized covariance matrix."""
    assets = list(returns_dict.keys())
    n = len(assets)

    if n < 2:
        vol = np.std(list(returns_dict.values())[0], ddof=1) if returns_dict else 0
        return np.array([[vol**2 * periods_per_year]])

    min_len = min(len(r) for r in returns_dict.values())
    aligned = np.column_stack([returns_dict[a][-min_len:] for a in assets])

    return np.cov(aligned.T) * periods_per_year


def _equal_weight_allocation(n_assets: int) -> np.ndarray:
    """Simple equal weight allocation."""
    return np.ones(n_assets) / n_assets


def _risk_parity_allocation(volatilities: np.ndarray) -> np.ndarray:
    """Risk parity: weight inversely proportional to volatility."""
    if np.all(volatilities == 0):
        n = len(volatilities)
        logger.warning("All assets have zero volatility - falling back to equal-weight allocation")
        result: np.ndarray = np.ones(n) / n
        return result

    inv_vol = 1.0 / np.maximum(volatilities, 1e-8)
    result = inv_vol / np.sum(inv_vol)
    return result


def _min_variance_allocation(cov_matrix: np.ndarray) -> np.ndarray:
    """Minimum variance portfolio (analytical solution)."""
    n = len(cov_matrix)
    if n == 1:
        return np.array([1.0])

    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        # Ensure non-negative weights (long-only)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        return weights
    except np.linalg.LinAlgError:
        logger.warning(
            "Covariance matrix inversion failed (matrix may be singular). "
            "Falling back to equal-weight allocation."
        )
        return _equal_weight_allocation(n)


def _target_vol_allocation(
    volatilities: np.ndarray,
    target_vol: float,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Scale risk parity allocation to target volatility.

    Adjusts weights to achieve approximate target volatility while
    maintaining full allocation (weights sum to 1.0).
    """
    # Start with risk parity
    weights = _risk_parity_allocation(volatilities)

    # Calculate current portfolio vol
    port_var = weights @ cov_matrix @ weights
    port_vol = np.sqrt(port_var) * 100  # As percentage

    if port_vol > 0:
        # Scale factor to achieve target
        scale = target_vol / port_vol
        # Cap scaling to avoid extreme leverage
        scale = min(scale, 2.0)
        weights = weights * scale

    # Always normalize to ensure weights sum to 1.0
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum

    return weights


def _classify_risk_level(portfolio_vol: float) -> RiskLevel:
    """Classify portfolio risk based on volatility."""
    if portfolio_vol < 5:
        return RiskLevel.VERY_LOW
    elif portfolio_vol < 10:
        return RiskLevel.LOW
    elif portfolio_vol < 20:
        return RiskLevel.MODERATE
    elif portfolio_vol < 35:
        return RiskLevel.HIGH
    else:
        return RiskLevel.VERY_HIGH


def _calculate_diversification_score(
    weights: np.ndarray,
    volatilities: np.ndarray,
    portfolio_vol: float,
) -> float:
    """Calculate diversification ratio.

    Ratio of weighted average vol to portfolio vol.
    Higher = more diversification benefit.
    """
    if portfolio_vol == 0:
        return 0.0

    weighted_avg_vol = np.sum(weights * volatilities)
    ratio = weighted_avg_vol / portfolio_vol

    # Normalize to 0-1 scale (ratio of 1 = no diversification, >1 = good)
    # Score: (ratio - 1) / ratio, capped at 0-1
    score = float(max(0, min(1, (ratio - 1) / max(ratio, 1))))
    return score


def _classify_diversification(score: float, avg_corr: float) -> DiversificationLevel:
    """Classify diversification quality."""
    if avg_corr > 0.8 or score < 0.1:
        return DiversificationLevel.POOR
    elif avg_corr > 0.6 or score < 0.2:
        return DiversificationLevel.LIMITED
    elif avg_corr > 0.4 or score < 0.35:
        return DiversificationLevel.MODERATE
    elif avg_corr > 0.2 or score < 0.5:
        return DiversificationLevel.GOOD
    else:
        return DiversificationLevel.EXCELLENT


def _get_correlation_insights(
    assets: list[str],
    corr_matrix: np.ndarray,
    top_n: int = 3,
) -> list[CorrelationInsight]:
    """Extract key correlation insights."""
    insights: list[CorrelationInsight] = []
    n = len(assets)

    if n < 2:
        return insights

    # Collect all pairs with correlations
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            # Skip NaN correlations (can happen with zero variance assets)
            if np.isnan(corr):
                continue
            pairs.append((assets[i], assets[j], corr))

    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for asset_1, asset_2, corr in pairs[:top_n]:
        if corr > 0.7:
            relationship = "highly correlated (move together)"
        elif corr > 0.3:
            relationship = "moderately correlated"
        elif corr > -0.3:
            relationship = "weakly correlated (independent)"
        elif corr > -0.7:
            relationship = "negatively correlated (hedge potential)"
        else:
            relationship = "strongly negatively correlated (strong hedge)"

        insights.append(
            CorrelationInsight(
                asset_1=asset_1,
                asset_2=asset_2,
                correlation=round(float(corr), 3),
                relationship=relationship,
            )
        )

    return insights


def _calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate marginal risk contribution for each asset."""
    port_var = weights @ cov_matrix @ weights
    if port_var == 0:
        return np.zeros(len(weights))

    # Marginal contribution to risk
    mcr = cov_matrix @ weights
    # Risk contribution
    rc = weights * mcr / np.sqrt(port_var)
    # As percentage of total (use absolute values to avoid validation errors)
    rc_abs = np.abs(rc)
    rc_pct = rc_abs / np.sum(rc_abs) * 100 if np.sum(rc_abs) > 0 else np.zeros(len(weights))

    return rc_pct


def _generate_narrative(
    weights: dict[str, float],
    portfolio_return: float,
    portfolio_vol: float,
    risk_level: RiskLevel,
    div_level: DiversificationLevel,
    avg_corr: float,
    correlation_insights: list[CorrelationInsight],
    method: AllocationMethod,
    target_vol: float | None,
    context: str | None,
) -> str:
    """Generate natural language narrative."""
    parts = []

    prefix = f"Portfolio analysis for {context}" if context else "Portfolio analysis"
    parts.append(f"{prefix}:")

    # Allocation summary
    total_allocated = sum(weights.values())
    top_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    holding_str = ", ".join(f"{name} ({w*100:.0f}%)" for name, w in top_holdings)

    if total_allocated < 0.99:
        cash_pct = (1 - total_allocated) * 100
        parts.append(f"Suggested allocation: {holding_str}, Cash ({cash_pct:.0f}%).")
    else:
        parts.append(f"Suggested allocation: {holding_str}.")

    # Risk/return
    parts.append(
        f"Expected return: {portfolio_return:.1f}%, "
        f"volatility: {portfolio_vol:.1f}% ({risk_level.value} risk)."
    )

    # Diversification
    div_desc = {
        DiversificationLevel.POOR: "poor - assets highly correlated",
        DiversificationLevel.LIMITED: "limited diversification benefit",
        DiversificationLevel.MODERATE: "moderate diversification",
        DiversificationLevel.GOOD: "well diversified",
        DiversificationLevel.EXCELLENT: "excellent diversification",
    }
    parts.append(f"Diversification: {div_desc[div_level]} (avg correlation: {avg_corr:.2f}).")

    # Key correlation insight
    if correlation_insights:
        top_insight = correlation_insights[0]
        parts.append(
            f"{top_insight.asset_1}/{top_insight.asset_2}: {top_insight.relationship} "
            f"(r={top_insight.correlation:.2f})."
        )

    # Method note
    if method == AllocationMethod.RISK_PARITY:
        parts.append("Risk parity approach balances risk contribution across assets.")
    elif method == AllocationMethod.TARGET_VOL and target_vol:
        parts.append(f"Allocation scaled to target {target_vol:.0f}% volatility.")

    return " ".join(parts)


def describe_allocation(
    assets: dict[str, np.ndarray | list[float]],
    context: str | None = None,
    method: str = "risk_parity",
    target_volatility: float | None = None,
    risk_free_rate: float = 2.0,
    periods_per_year: int = 252,
) -> AllocationResult:
    """Analyze multi-asset portfolio and suggest allocation.

    Provides risk-based position sizing with correlation-aware diversification
    analysis. Supports multiple allocation methods.

    DISCLAIMER: This is educational/informational analysis only, NOT financial advice.

    Args:
        assets: Dict mapping asset names to price arrays.
                Example: {"BTC": [100, 105, 102, ...], "ETH": [50, 52, 48, ...]}
        context: Optional label (e.g., "Crypto Portfolio").
        method: Allocation method - "equal_weight", "risk_parity", "min_variance", "target_vol".
        target_volatility: Target portfolio volatility (%) for target_vol method.
        risk_free_rate: Annual risk-free rate (%) for Sharpe calculation.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        AllocationResult with suggested weights, risk metrics, and narrative.

    Example:
        >>> assets = {
        ...     "BTC": [100, 105, 102, 108, 110],
        ...     "ETH": [50, 52, 48, 55, 54],
        ...     "SOL": [20, 22, 19, 25, 24]
        ... }
        >>> result = describe_allocation(assets, context="Crypto")
        >>> print(result.narrative)
        "Portfolio analysis for Crypto: Suggested allocation: BTC (40%), ETH (35%), SOL (25%).
         Expected return: 85.2%, volatility: 42.1% (high risk).
         Diversification: limited - assets highly correlated..."
    """
    # Convert to numpy arrays and calculate returns
    asset_names = list(assets.keys())
    n_assets = len(asset_names)

    if n_assets == 0:
        raise ValueError("At least one asset required")

    returns_dict: dict[str, np.ndarray] = {}
    volatilities = []
    ann_returns = []

    for name, prices in assets.items():
        if isinstance(prices, list):
            prices = np.array(prices, dtype=float)

        if len(prices) < 3:
            raise ValueError(f"Asset {name} needs at least 3 price points")

        returns = _calculate_returns(prices)
        returns_dict[name] = returns

        ann_ret = _annualize_return(returns, periods_per_year)
        ann_vol = _annualize_volatility(returns, periods_per_year)

        ann_returns.append(ann_ret)
        volatilities.append(ann_vol)

    volatilities_arr = np.array(volatilities)
    returns_arr = np.array(ann_returns)

    # Calculate matrices
    corr_matrix = _calculate_correlation_matrix(returns_dict)
    cov_matrix = _calculate_covariance_matrix(returns_dict, periods_per_year)

    # Determine allocation method
    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    if method_lower == "equal_weight":
        weights = _equal_weight_allocation(n_assets)
        alloc_method = AllocationMethod.EQUAL_WEIGHT
    elif method_lower == "min_variance":
        weights = _min_variance_allocation(cov_matrix)
        alloc_method = AllocationMethod.MIN_VARIANCE
    elif method_lower == "target_vol" and target_volatility:
        weights = _target_vol_allocation(volatilities_arr, target_volatility, cov_matrix)
        alloc_method = AllocationMethod.TARGET_VOL
    else:  # Default to risk parity
        weights = _risk_parity_allocation(volatilities_arr)
        alloc_method = AllocationMethod.RISK_PARITY

    # Calculate portfolio metrics
    port_return = float(np.sum(weights * returns_arr))
    port_var = weights @ cov_matrix @ weights
    port_vol = float(np.sqrt(port_var)) * 100  # Convert to percentage
    port_sharpe = _calculate_sharpe(port_return, port_vol, risk_free_rate)

    # Risk classification
    risk_level = _classify_risk_level(port_vol)

    # Diversification
    div_score = _calculate_diversification_score(weights, volatilities_arr, port_vol / 100)

    # Average correlation (off-diagonal)
    if n_assets > 1:
        mask = ~np.eye(n_assets, dtype=bool)
        off_diag = corr_matrix[mask]
        # Filter out NaN values
        valid_corrs = off_diag[~np.isnan(off_diag)]
        avg_corr = float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else 0.0
    else:
        avg_corr = 1.0

    div_level = _classify_diversification(div_score, avg_corr)

    # Correlation insights
    corr_insights = _get_correlation_insights(asset_names, corr_matrix)

    # Risk contribution
    risk_contrib = _calculate_risk_contribution(weights, cov_matrix)

    # Build per-asset analysis
    asset_analyses = []
    for i, name in enumerate(asset_names):
        sharpe = _calculate_sharpe(ann_returns[i], volatilities[i], risk_free_rate)
        asset_analyses.append(
            AssetAnalysis(
                name=name,
                annualized_return=round(ann_returns[i], 2),
                annualized_volatility=round(volatilities[i], 2),
                sharpe_ratio=round(sharpe, 2) if sharpe else None,
                suggested_weight=round(weights[i], 4),
                contribution_to_risk=round(risk_contrib[i], 2),
            )
        )

    # Build weights dict
    weights_dict = {name: round(weights[i], 4) for i, name in enumerate(asset_names)}

    # Generate narrative
    narrative = _generate_narrative(
        weights_dict,
        port_return,
        port_vol,
        risk_level,
        div_level,
        avg_corr,
        corr_insights,
        alloc_method,
        target_volatility,
        context,
    )

    return AllocationResult(
        suggested_weights=weights_dict,
        allocation_method=alloc_method,
        portfolio_return=round(port_return, 2),
        portfolio_volatility=round(port_vol, 2),
        portfolio_sharpe=round(port_sharpe, 2) if port_sharpe else None,
        risk_level=risk_level,
        diversification_score=round(div_score, 3),
        diversification_level=div_level,
        asset_analyses=tuple(asset_analyses),
        correlation_insights=tuple(corr_insights),
        avg_correlation=round(avg_corr, 3),
        narrative=narrative,
        disclaimer=(
            "This is educational analysis only, not financial advice. "
            "Consult a qualified professional."
        ),
        data_context=context,
        num_assets=n_assets,
    )

"""Pydantic models for trading analysis results.

These models define the API contract for trading-specific analysis,
ensuring type safety and consistent output format for trading agents.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_frame.trading.enums import (
    ConsistencyRating,
    DrawdownSeverity,
    PerformanceRating,
    RecoveryState,
    RiskProfile,
)


class DrawdownPeriod(BaseModel):
    """Information about a single drawdown period."""

    model_config = ConfigDict(frozen=True)

    start_index: int = Field(ge=0, description="Index where drawdown started (peak)")
    trough_index: int = Field(ge=0, description="Index of maximum drawdown (trough)")
    end_index: int | None = Field(
        default=None, description="Index where fully recovered (None if ongoing)"
    )
    depth_pct: float = Field(ge=0.0, le=100.0, description="Maximum depth as percentage (0-100)")
    duration: int = Field(ge=1, description="Duration in periods from peak to trough")
    recovery_duration: int | None = Field(
        default=None, description="Periods from trough to recovery (None if ongoing)"
    )
    recovered: bool = Field(description="Whether this drawdown has fully recovered")

    @model_validator(mode="after")
    def check_consistency(self) -> DrawdownPeriod:
        """Validate cross-field invariants for drawdown period."""
        # Temporal ordering: start <= trough
        if self.start_index > self.trough_index:
            raise ValueError(
                f"start_index ({self.start_index}) must be <= trough_index ({self.trough_index})"
            )

        # If recovered, end_index and recovery_duration must be set
        if self.recovered:
            if self.end_index is None:
                raise ValueError("recovered period must have end_index")
            if self.recovery_duration is None:
                raise ValueError("recovered period must have recovery_duration")
            # Temporal ordering: trough <= end
            if self.trough_index > self.end_index:
                raise ValueError(
                    f"trough_index ({self.trough_index}) must be <= end_index ({self.end_index})"
                )
        return self


class DrawdownResult(BaseModel):
    """Complete drawdown analysis result."""

    model_config = ConfigDict(frozen=True)

    # Summary metrics
    max_drawdown_pct: float = Field(
        ge=0.0, le=100.0, description="Maximum drawdown percentage (0-100)"
    )
    max_drawdown_duration: int = Field(ge=0, description="Duration of worst drawdown in periods")
    current_drawdown_pct: float = Field(
        ge=0.0, le=100.0, description="Current drawdown from peak (0 if at high)"
    )
    avg_drawdown_pct: float = Field(
        ge=0.0, le=100.0, description="Average drawdown depth percentage"
    )
    num_drawdowns: int = Field(ge=0, description="Total number of drawdown periods")
    avg_recovery_periods: float | None = Field(
        default=None, description="Average periods to recover (None if no recoveries)"
    )

    # Classifications
    severity: DrawdownSeverity = Field(description="Classification of drawdown severity")
    recovery_state: RecoveryState = Field(description="Current recovery status")

    # Detailed periods
    drawdown_periods: tuple[DrawdownPeriod, ...] = Field(
        default_factory=tuple, description="Individual drawdown periods (max 10)"
    )

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable summary")

    # Metadata
    context: str | None = Field(default=None, description="User-provided context label")


class TradingMetrics(BaseModel):
    """Core trading performance metrics."""

    model_config = ConfigDict(frozen=True)

    # Win/Loss metrics
    total_trades: int = Field(ge=0, description="Total number of trades")
    winning_trades: int = Field(ge=0, description="Number of profitable trades")
    losing_trades: int = Field(ge=0, description="Number of losing trades")
    win_rate: float = Field(ge=0.0, le=1.0, description="Percentage of winning trades (0-1)")

    # Profit metrics
    gross_profit: float = Field(description="Total profit from winning trades")
    gross_loss: float = Field(le=0.0, description="Total loss from losing trades (negative)")
    net_profit: float = Field(description="Net profit/loss")
    profit_factor: float | None = Field(
        default=None, description="Gross profit / |gross loss| (None if no losses)"
    )

    # Average trade metrics
    avg_win: float | None = Field(default=None, description="Average profit per winning trade")
    avg_loss: float | None = Field(default=None, description="Average loss per losing trade")
    avg_trade: float = Field(description="Average profit/loss per trade (expectancy)")
    risk_reward_ratio: float | None = Field(
        default=None, description="Average win / |average loss|"
    )

    # Streak metrics
    max_consecutive_wins: int = Field(ge=0, description="Longest winning streak")
    max_consecutive_losses: int = Field(ge=0, description="Longest losing streak")
    current_streak: int = Field(description="Current streak (positive=wins, negative=losses)")

    # Risk-adjusted metrics
    sharpe_ratio: float | None = Field(
        default=None, description="Sharpe ratio (None if insufficient data)"
    )
    sortino_ratio: float | None = Field(
        default=None, description="Sortino ratio (None if insufficient data)"
    )
    calmar_ratio: float | None = Field(
        default=None, description="Annual return / max drawdown (None if no drawdown)"
    )
    recovery_factor: float | None = Field(
        default=None, description="Net profit / max drawdown (None if no drawdown)"
    )

    @model_validator(mode="after")
    def check_trade_counts(self) -> TradingMetrics:
        """Validate that trade counts are consistent."""
        # winning + losing cannot exceed total (breakeven trades exist)
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValueError(
                f"winning_trades ({self.winning_trades}) + losing_trades ({self.losing_trades}) "
                f"cannot exceed total_trades ({self.total_trades})"
            )
        return self


class TradingPerformanceResult(BaseModel):
    """Complete trading performance analysis result."""

    model_config = ConfigDict(frozen=True)

    # Core metrics
    metrics: TradingMetrics = Field(description="Detailed trading metrics")

    # Classifications
    performance_rating: PerformanceRating = Field(description="Overall performance classification")
    risk_profile: RiskProfile = Field(description="Risk-taking behavior classification")
    consistency: ConsistencyRating = Field(description="Return consistency classification")

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable performance summary")

    # Metadata
    context: str | None = Field(default=None, description="User-provided context label")

    def to_json_str(self) -> str:
        """Serialize to JSON string for API responses."""
        return self.model_dump_json(indent=2)


class AgentRanking(BaseModel):
    """Ranking information for a single agent/strategy."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Agent/strategy name")
    total_return_pct: float = Field(description="Total return percentage")
    volatility: float = Field(ge=0.0, description="Return volatility (std dev)")
    sharpe_ratio: float | None = Field(default=None, description="Risk-adjusted return")
    max_drawdown_pct: float = Field(ge=0.0, description="Maximum drawdown percentage")
    win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Win rate if available"
    )

    # Rankings (1 = best)
    return_rank: int = Field(ge=1, description="Rank by total return")
    risk_adjusted_rank: int = Field(ge=1, description="Rank by Sharpe ratio")
    volatility_rank: int = Field(ge=1, description="Rank by volatility (1 = lowest)")
    drawdown_rank: int = Field(ge=1, description="Rank by max drawdown (1 = lowest)")


class RankingsResult(BaseModel):
    """Complete multi-agent ranking analysis result."""

    model_config = ConfigDict(frozen=True)

    # Individual rankings
    rankings: tuple[AgentRanking, ...] = Field(
        description="Per-agent ranking information, sorted by composite score"
    )

    # Leaders
    leader: str = Field(description="Best overall performer (composite)")
    highest_return: str = Field(description="Highest absolute return")
    lowest_volatility: str = Field(description="Most stable returns")
    best_risk_adjusted: str = Field(description="Best risk-adjusted return (Sharpe)")
    lowest_drawdown: str = Field(description="Smallest maximum drawdown")

    # Natural language
    narrative: str = Field(min_length=1, description="Human/LLM-readable ranking summary")

    # Metadata
    context: str | None = Field(default=None, description="User-provided context label")
    num_agents: int = Field(ge=1, description="Number of agents compared")

    @model_validator(mode="after")
    def check_leaders_exist(self) -> RankingsResult:
        """Validate that all leader names exist in rankings."""
        names = {r.name for r in self.rankings}

        if self.leader not in names:
            raise ValueError(f"leader '{self.leader}' not found in rankings")
        if self.highest_return not in names:
            raise ValueError(f"highest_return '{self.highest_return}' not found in rankings")
        if self.lowest_volatility not in names:
            raise ValueError(f"lowest_volatility '{self.lowest_volatility}' not found in rankings")
        if self.best_risk_adjusted not in names:
            raise ValueError(
                f"best_risk_adjusted '{self.best_risk_adjusted}' not found in rankings"
            )
        if self.lowest_drawdown not in names:
            raise ValueError(f"lowest_drawdown '{self.lowest_drawdown}' not found in rankings")

        # Validate num_agents matches rankings length
        if len(self.rankings) != self.num_agents:
            raise ValueError(
                f"num_agents ({self.num_agents}) must match rankings length ({len(self.rankings)})"
            )
        return self

    def to_json_str(self) -> str:
        """Serialize to JSON string for API responses."""
        return self.model_dump_json(indent=2)

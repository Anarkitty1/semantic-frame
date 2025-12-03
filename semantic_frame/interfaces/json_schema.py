"""Pydantic models for structured output.

These models define the API contract for semantic analysis results,
ensuring type safety and consistent output format for LLM integration.

All models are frozen (immutable) after construction to prevent accidental
modification of analysis results.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from semantic_frame.core.enums import (
    AnomalyState,
    DataQuality,
    DistributionShape,
    SeasonalityState,
    TrendState,
    VolatilityState,
)

# Maximum number of anomalies to store in results
MAX_ANOMALIES = 5


class AnomalyInfo(BaseModel):
    """Information about a detected anomaly/outlier."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Position of the anomaly in the data series")
    value: float = Field(description="The anomalous value")
    z_score: float = Field(
        ge=0.0, description="Absolute z-score or equivalent deviation measure"
    )


class SeriesProfile(BaseModel):
    """Statistical profile of a data series.

    Note: min_val and max_val are aliased as "min" and "max" in JSON
    serialization to avoid shadowing Python builtins in attribute access.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    count: int = Field(ge=0, description="Total number of data points")
    mean: float = Field(description="Arithmetic mean of the data")
    median: float = Field(description="Median value of the data")
    std: float = Field(ge=0.0, description="Standard deviation (always non-negative)")
    min_val: float = Field(alias="min", description="Minimum value")
    max_val: float = Field(alias="max", description="Maximum value")
    missing_pct: float = Field(
        ge=0.0, le=100.0, description="Percentage of missing/null values (0-100)"
    )

    @model_validator(mode="after")
    def validate_min_max(self) -> "SeriesProfile":
        """Ensure min_val does not exceed max_val."""
        if self.min_val > self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) cannot exceed max_val ({self.max_val})"
            )
        return self


class SemanticResult(BaseModel):
    """Complete semantic analysis result.

    This is the primary output model containing both the natural language
    narrative and structured analysis data for programmatic access.

    The model is immutable after construction to prevent accidental modification.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Natural language output
    narrative: str = Field(min_length=1, description="Human/LLM-readable semantic description")

    # Categorical classifications
    trend: TrendState = Field(description="Direction and intensity of data trend")
    volatility: VolatilityState = Field(description="Variability classification")
    data_quality: DataQuality = Field(description="Data completeness assessment")
    anomaly_state: AnomalyState = Field(description="Outlier presence and severity")

    # Optional classifications (may require more data)
    seasonality: Optional[SeasonalityState] = Field(
        default=None, description="Cyclic pattern detection result"
    )
    distribution: Optional[DistributionShape] = Field(
        default=None, description="Distribution shape classification"
    )

    # Detailed data
    anomalies: tuple[AnomalyInfo, ...] = Field(
        default_factory=tuple,
        description=f"Tuple of detected anomalies (max {MAX_ANOMALIES})",
    )
    profile: SeriesProfile = Field(description="Statistical profile of the data")

    # Metadata
    context: Optional[str] = Field(
        default=None, description="User-provided context label for the data"
    )
    compression_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Token reduction ratio (0.0 = no reduction, 1.0 = 100% reduction). "
        "Estimated using ~2 tokens/number and ~1 token/word approximation.",
    )

    @field_validator("anomalies", mode="before")
    @classmethod
    def convert_anomalies_to_tuple(cls, v: list | tuple) -> tuple:
        """Convert list to tuple and enforce max length."""
        if isinstance(v, list):
            v = tuple(v)
        if len(v) > MAX_ANOMALIES:
            raise ValueError(
                f"anomalies cannot exceed {MAX_ANOMALIES} items, got {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_anomaly_consistency(self) -> "SemanticResult":
        """Ensure anomaly_state is consistent with anomalies tuple."""
        if self.anomaly_state == AnomalyState.NONE and len(self.anomalies) > 0:
            raise ValueError(
                "anomaly_state is NONE but anomalies tuple is not empty"
            )
        return self

    def to_prompt(self) -> str:
        """Format result as an LLM-ready prompt injection."""
        return f"DATA CONTEXT: {self.narrative}"

    def to_json_str(self) -> str:
        """Serialize to JSON string for API responses."""
        return self.model_dump_json(indent=2, by_alias=True)

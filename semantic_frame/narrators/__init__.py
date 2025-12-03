"""Narrative generation for semantic output."""

from semantic_frame.narrators.distribution import generate_distribution_narrative
from semantic_frame.narrators.time_series import generate_time_series_narrative

__all__ = ["generate_time_series_narrative", "generate_distribution_narrative"]

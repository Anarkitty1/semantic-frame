"""Narrative generation for correlation insights.

This module generates natural language descriptions for correlations
between columns in DataFrame analysis.
"""

from __future__ import annotations

from semantic_frame.core.enums import CorrelationState

# Template strings for correlation narrative construction
CORRELATION_TEMPLATES = {
    CorrelationState.STRONG_POSITIVE: "{a} and {b} move together strongly (r={r:.2f})",
    CorrelationState.MODERATE_POSITIVE: "{a} and {b} show positive relationship (r={r:.2f})",
    CorrelationState.WEAK: "{a} and {b} have weak/no clear relationship (r={r:.2f})",
    CorrelationState.MODERATE_NEGATIVE: "{a} and {b} show inverse relationship (r={r:.2f})",
    CorrelationState.STRONG_NEGATIVE: "{a} and {b} are strongly inverse (r={r:.2f})",
}


def generate_correlation_narrative(
    column_a: str,
    column_b: str,
    r_value: float,
    state: CorrelationState,
) -> str:
    """Generate narrative for a single correlation.

    Args:
        column_a: First column name.
        column_b: Second column name.
        r_value: Correlation coefficient.
        state: Classified correlation state.

    Returns:
        Human-readable correlation description.
    """
    template = CORRELATION_TEMPLATES.get(state, CORRELATION_TEMPLATES[CorrelationState.WEAK])
    return template.format(a=column_a, b=column_b, r=r_value)


def generate_dataframe_summary(
    column_count: int,
    significant_correlations: int,
    key_insights: list[str],
    context: str | None = None,
) -> str:
    """Generate summary narrative for DataFrame analysis.

    Args:
        column_count: Number of numeric columns analyzed.
        significant_correlations: Number of significant correlations found.
        key_insights: List of notable findings (correlation narratives).
        context: Optional context label.

    Returns:
        Summary narrative string.
    """
    ctx = context or "DataFrame"
    parts: list[str] = []

    parts.append(f"Analyzed {column_count} numeric column(s) in {ctx}.")

    if significant_correlations > 0:
        parts.append(f"Found {significant_correlations} significant correlation(s).")
    else:
        parts.append("No strong correlations detected between columns.")

    if key_insights:
        # Limit to top 3 insights
        top_insights = key_insights[:3]
        parts.append("Key findings: " + "; ".join(top_insights) + ".")

    return " ".join(parts)

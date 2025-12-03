"""LLM integration templates and prompt helpers.

This module provides ready-made templates for integrating semantic
analysis results into LLM prompts, supporting popular frameworks
like LangChain and direct API usage.
"""

from __future__ import annotations

from typing import Any

from semantic_frame.interfaces.json_schema import SemanticResult

# System prompt templates
SYSTEM_PROMPT_TEMPLATE = """\
You are analyzing data that has been pre-processed by a semantic analyzer.

DATA CONTEXT:
{narrative}

STRUCTURED ANALYSIS:
- Trend: {trend}
- Volatility: {volatility}
- Data Quality: {data_quality}
- Anomalies: {anomaly_state}

Use this context to inform your response. The analysis is mathematically verified."""


CONCISE_CONTEXT_TEMPLATE = """[DATA: {context}] {narrative}"""


LANGCHAIN_TOOL_DESCRIPTION = """\
Semantic data analysis tool that converts numerical data into natural language insights.

Input: Raw numerical data (CSV, JSON array, or description of data source)
Output: Semantic description including trend, volatility, anomalies, and data quality.

Use this tool to understand patterns in numerical data without processing raw numbers."""


def format_for_system_prompt(result: SemanticResult) -> str:
    """Format SemanticResult for injection into LLM system prompt.

    Creates a structured context block suitable for system prompts,
    providing both narrative and key metrics.

    Args:
        result: SemanticResult from describe_series.

    Returns:
        Formatted string for system prompt.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        narrative=result.narrative,
        trend=result.trend.value,
        volatility=result.volatility.value,
        data_quality=result.data_quality.value,
        anomaly_state=result.anomaly_state.value,
    )


def format_for_context(result: SemanticResult) -> str:
    """Format SemanticResult as concise context injection.

    Creates a minimal one-line context suitable for user messages
    or tool results.

    Args:
        result: SemanticResult from describe_series.

    Returns:
        Concise context string.
    """
    context = result.context or "Analysis"
    return CONCISE_CONTEXT_TEMPLATE.format(
        context=context,
        narrative=result.narrative,
    )


def format_for_langchain(result: SemanticResult) -> dict[str, Any]:
    """Format SemanticResult for LangChain tool output.

    Creates a dict structure suitable for LangChain tool returns,
    with both human-readable and structured components.

    Args:
        result: SemanticResult from describe_series.

    Returns:
        Dict with 'output' and 'metadata' keys.
    """
    return {
        "output": result.narrative,
        "metadata": {
            "trend": result.trend.value,
            "volatility": result.volatility.value,
            "data_quality": result.data_quality.value,
            "anomaly_count": len(result.anomalies),
            "compression_ratio": result.compression_ratio,
            "context": result.context,
        },
    }


def get_analysis_prompt(data_description: str) -> str:
    """Generate a prompt for requesting data analysis.

    Creates a structured prompt that can be sent to an LLM
    when requesting analysis of specific data.

    Args:
        data_description: Description of the data to analyze.

    Returns:
        Formatted analysis request prompt.
    """
    return f"""Analyze the following data and provide insights:

Data: {data_description}

Please describe:
1. Overall trend and direction
2. Volatility and stability
3. Any anomalies or outliers
4. Data quality assessment

Provide actionable insights based on the patterns observed."""


def create_agent_context(results: dict[str, SemanticResult]) -> str:
    """Create comprehensive context from multiple analysis results.

    Useful when analyzing a DataFrame with multiple columns,
    creating a unified context block for agent consumption.

    Args:
        results: Dict mapping column names to SemanticResult objects.

    Returns:
        Combined context string for all columns.
    """
    if not results:
        return "No data analysis available."

    parts = ["MULTI-COLUMN DATA ANALYSIS:", ""]

    for col_name, result in results.items():
        parts.append(f"## {col_name}")
        parts.append(result.narrative)
        parts.append("")

    # Summary of concerning items
    concerns = []
    for col_name, result in results.items():
        if result.anomaly_state.value != "no anomalies":
            concerns.append(f"- {col_name}: {result.anomaly_state.value}")
        if result.volatility.value in ("expanding", "extreme"):
            concerns.append(f"- {col_name}: {result.volatility.value} volatility")

    if concerns:
        parts.append("ATTENTION REQUIRED:")
        parts.extend(concerns)

    return "\n".join(parts)

"""Correlation analysis functions for multi-column data.

This module contains functions for detecting and classifying
relationships between numeric columns in DataFrames.

Key functions:
- classify_correlation: Map correlation coefficient to CorrelationState
- calc_correlation_matrix: Calculate pairwise correlations
- identify_significant_correlations: Filter meaningful relationships
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

from semantic_frame.core.enums import CorrelationState

logger = logging.getLogger(__name__)


def classify_correlation(r_value: float) -> CorrelationState:
    """Classify correlation coefficient to CorrelationState.

    Thresholds:
        - r > 0.7: STRONG_POSITIVE
        - 0.4 < r <= 0.7: MODERATE_POSITIVE
        - -0.4 <= r <= 0.4: WEAK
        - -0.7 <= r < -0.4: MODERATE_NEGATIVE
        - r < -0.7: STRONG_NEGATIVE

    Args:
        r_value: Pearson or Spearman correlation coefficient (-1 to 1).

    Returns:
        CorrelationState enum value.
    """
    if r_value > 0.7:
        return CorrelationState.STRONG_POSITIVE
    if r_value > 0.4:
        return CorrelationState.MODERATE_POSITIVE
    if r_value < -0.7:
        return CorrelationState.STRONG_NEGATIVE
    if r_value < -0.4:
        return CorrelationState.MODERATE_NEGATIVE
    return CorrelationState.WEAK


def calc_correlation_matrix(
    values_dict: dict[str, np.ndarray],
    method: str = "pearson",
) -> dict[tuple[str, str], float]:
    """Calculate pairwise correlation matrix for numeric columns.

    Args:
        values_dict: Dict mapping column names to numpy arrays.
            All arrays should have the same length.
        method: Correlation method - "pearson" (default) or "spearman".

    Returns:
        Dict mapping (col_a, col_b) tuples to correlation coefficients.
        Only includes unique pairs where col_a < col_b lexicographically.
        Returns empty dict if fewer than 2 columns provided.
    """
    if len(values_dict) < 2:
        return {}

    corr_func = pearsonr if method == "pearson" else spearmanr
    columns = sorted(values_dict.keys())
    correlations: dict[tuple[str, str], float] = {}

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            values_a = values_dict[col_a]
            values_b = values_dict[col_b]

            # Align arrays - use common valid indices (no NaN in either)
            valid_mask = ~(np.isnan(values_a) | np.isnan(values_b))
            clean_a = values_a[valid_mask]
            clean_b = values_b[valid_mask]

            if len(clean_a) < 3:
                logger.debug(
                    "Insufficient data for correlation between %s and %s (need >= 3 pairs)",
                    col_a,
                    col_b,
                )
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = corr_func(clean_a, clean_b)
                    # Handle both old and new scipy return types
                    r = result[0] if hasattr(result, "__getitem__") else result.statistic
                if not np.isnan(r):
                    correlations[(col_a, col_b)] = float(r)
            except (ValueError, FloatingPointError) as e:
                logger.debug(
                    "Correlation calculation failed for %s vs %s: %s",
                    col_a,
                    col_b,
                    str(e),
                )

    return correlations


def identify_significant_correlations(
    correlations: dict[tuple[str, str], float],
    threshold: float = 0.5,
) -> list[tuple[str, str, float, CorrelationState]]:
    """Filter correlations to significant relationships.

    Args:
        correlations: Output from calc_correlation_matrix.
        threshold: Minimum absolute correlation to include (default 0.5).
            Only relationships with |r| >= threshold are returned.

    Returns:
        List of (col_a, col_b, r_value, state) tuples, sorted by |r| descending.
        Returns empty list if no significant correlations found.
    """
    significant: list[tuple[str, str, float, CorrelationState]] = []

    for (col_a, col_b), r in correlations.items():
        if abs(r) >= threshold:
            state = classify_correlation(r)
            significant.append((col_a, col_b, r, state))

    # Sort by absolute correlation, strongest first
    return sorted(significant, key=lambda x: abs(x[2]), reverse=True)

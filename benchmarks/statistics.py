"""
Statistical Significance Testing Module

Provides statistical analysis tools for benchmark validation including:
- Paired t-tests for comparing baseline vs treatment conditions
- Bonferroni correction for multiple comparisons
- Cohen's d effect size calculation
- Wilson score intervals for proportions
- Confidence intervals for continuous metrics

References:
    - Student's t-test: Student (1908). "The Probable Error of a Mean."
      Biometrika, 6(1), 1-25. DOI: 10.2307/2331554
    - Bonferroni correction: Dunn, O.J. (1961). "Multiple Comparisons Among Means."
      Journal of the American Statistical Association, 56(293), 52-64.
    - Cohen's d: Cohen, J. (1988). "Statistical Power Analysis for the
      Behavioral Sciences" (2nd ed.). Lawrence Erlbaum Associates.
    - Wilson score interval: Wilson, E.B. (1927). "Probable Inference, the Law of
      Succession, and Statistical Inference." Journal of the American Statistical
      Association, 22(158), 209-212.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalTestResult:
    """Result of a statistical significance test.

    This dataclass is frozen (immutable) for thread safety and hashability.
    """

    test_name: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: float | None
    is_significant: bool
    confidence_level: float
    effect_size: float | None = None
    effect_size_interpretation: str | None = None
    corrected_p_value: float | None = None  # After Bonferroni correction
    sample_size_baseline: int | None = None
    sample_size_treatment: int | None = None


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval for a metric.

    This dataclass is frozen (immutable) for thread safety and hashability.
    """

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str  # e.g., "t-distribution", "Wilson score", "bootstrap"


@dataclass(frozen=True)
class EffectSizeResult:
    """Effect size calculation result.

    This dataclass is frozen (immutable) for thread safety and hashability.
    """

    value: float
    interpretation: str  # "negligible", "small", "medium", "large"
    method: str  # "cohens_d", "cohens_d_paired", "glass_delta"


class StatisticalAnalyzer:
    """Statistical analysis tools for benchmark evaluation.

    Provides methods for significance testing, effect size calculation,
    and confidence interval estimation.

    Example:
        >>> analyzer = StatisticalAnalyzer(alpha=0.05)
        >>> result = analyzer.paired_t_test(baseline_scores, treatment_scores)
        >>> print(f"p-value: {result.p_value:.4f}, significant: {result.is_significant}")
    """

    # Cohen's d interpretation thresholds (Cohen, 1988)
    EFFECT_SIZE_THRESHOLDS = {
        "negligible": 0.2,
        "small": 0.5,
        "medium": 0.8,
        # >= 0.8 is "large"
    }

    def __init__(self, alpha: float = 0.05):
        """Initialize the statistical analyzer.

        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
        self.confidence_level = 1 - alpha

    def paired_t_test(
        self,
        baseline: list[float] | np.ndarray,
        treatment: list[float] | np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> StatisticalTestResult:
        """Perform paired t-test comparing baseline and treatment conditions.

        Use this when the same subjects/samples are measured under both conditions.
        This is the appropriate test for within-subjects experimental designs.

        Args:
            baseline: Scores/metrics from baseline condition
            treatment: Scores/metrics from treatment condition (same order as baseline)
            alternative: Type of alternative hypothesis
                - "two-sided": treatment ≠ baseline (default)
                - "less": treatment < baseline
                - "greater": treatment > baseline

        Returns:
            StatisticalTestResult with test statistics and interpretation

        Raises:
            ValueError: If baseline and treatment have different lengths or < 2 samples

        Example:
            >>> baseline = [0.70, 0.65, 0.72, 0.68]
            >>> treatment = [0.85, 0.82, 0.88, 0.84]
            >>> result = analyzer.paired_t_test(baseline, treatment)
        """
        baseline_arr = np.asarray(baseline)
        treatment_arr = np.asarray(treatment)

        if len(baseline_arr) != len(treatment_arr):
            raise ValueError(
                f"Baseline and treatment must have same length: "
                f"{len(baseline_arr)} vs {len(treatment_arr)}"
            )

        if len(baseline_arr) < 2:
            raise ValueError(f"Need at least 2 samples for t-test, got {len(baseline_arr)}")

        # Perform paired t-test
        # Suppress scipy warnings about precision loss from nearly identical data
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Precision loss occurred in moment calculation",
                category=RuntimeWarning,
            )
            t_stat, p_value = stats.ttest_rel(treatment_arr, baseline_arr, alternative=alternative)

        # Calculate degrees of freedom
        df = len(baseline_arr) - 1

        # Calculate Cohen's d for paired samples
        effect_size_result = self.cohens_d_paired(baseline_arr, treatment_arr)

        return StatisticalTestResult(
            test_name="Paired t-test",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=float(df),
            is_significant=bool(p_value < self.alpha),
            confidence_level=self.confidence_level,
            effect_size=effect_size_result.value,
            effect_size_interpretation=effect_size_result.interpretation,
            sample_size_baseline=len(baseline_arr),
            sample_size_treatment=len(treatment_arr),
        )

    def independent_t_test(
        self,
        group1: list[float] | np.ndarray,
        group2: list[float] | np.ndarray,
        equal_var: bool = False,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> StatisticalTestResult:
        """Perform independent samples t-test (Welch's t-test by default).

        Use this when comparing two independent groups of samples.
        Welch's t-test (equal_var=False) is more robust to unequal variances.

        Args:
            group1: Scores/metrics from first group
            group2: Scores/metrics from second group
            equal_var: If True, use Student's t-test (assumes equal variance)
                       If False, use Welch's t-test (default, more robust)
            alternative: Type of alternative hypothesis

        Returns:
            StatisticalTestResult with test statistics and interpretation

        Raises:
            ValueError: If either group has < 2 samples
        """
        group1_arr = np.asarray(group1)
        group2_arr = np.asarray(group2)

        if len(group1_arr) < 2 or len(group2_arr) < 2:
            raise ValueError("Need at least 2 samples in each group for t-test")

        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(
            group1_arr, group2_arr, equal_var=equal_var, alternative=alternative
        )

        # Calculate degrees of freedom (Welch-Satterthwaite if unequal variance)
        if equal_var:
            df = len(group1_arr) + len(group2_arr) - 2
        else:
            # Welch-Satterthwaite approximation
            v1, v2 = np.var(group1_arr, ddof=1), np.var(group2_arr, ddof=1)
            n1, n2 = len(group1_arr), len(group2_arr)
            num = (v1 / n1 + v2 / n2) ** 2
            denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            df = num / denom if denom > 0 else n1 + n2 - 2

        # Calculate Cohen's d
        effect_size_result = self.cohens_d(group1_arr, group2_arr)

        test_name = "Student's t-test" if equal_var else "Welch's t-test"

        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=float(df),
            is_significant=bool(p_value < self.alpha),
            confidence_level=self.confidence_level,
            effect_size=effect_size_result.value,
            effect_size_interpretation=effect_size_result.interpretation,
            sample_size_baseline=len(group1_arr),
            sample_size_treatment=len(group2_arr),
        )

    def bonferroni_correction(
        self,
        p_values: list[float],
        n_comparisons: int | None = None,
    ) -> list[float]:
        """Apply Bonferroni correction for multiple comparisons.

        The Bonferroni correction adjusts p-values to control the family-wise
        error rate when conducting multiple statistical tests.

        Formula:
            adjusted_p = min(p * n_comparisons, 1.0)

        Args:
            p_values: List of raw p-values from multiple tests
            n_comparisons: Number of comparisons (defaults to len(p_values))

        Returns:
            List of corrected p-values

        Example:
            >>> raw_p = [0.01, 0.03, 0.05]
            >>> corrected = analyzer.bonferroni_correction(raw_p, n_comparisons=3)
            >>> # corrected = [0.03, 0.09, 0.15]
        """
        if n_comparisons is None:
            n_comparisons = len(p_values)

        if n_comparisons <= 0:
            raise ValueError(f"n_comparisons must be positive, got {n_comparisons}")

        return [min(p * n_comparisons, 1.0) for p in p_values]

    def holm_bonferroni_correction(
        self,
        p_values: list[float],
    ) -> list[tuple[int, float, float]]:
        """Apply Holm-Bonferroni step-down correction for multiple comparisons.

        The Holm-Bonferroni method is uniformly more powerful than the standard
        Bonferroni correction while still controlling the family-wise error rate.

        Args:
            p_values: List of raw p-values from multiple tests

        Returns:
            List of tuples (original_index, raw_p, corrected_p) sorted by raw p-value

        Example:
            >>> raw_p = [0.04, 0.01, 0.03]
            >>> results = analyzer.holm_bonferroni_correction(raw_p)
            >>> # Returns [(1, 0.01, 0.03), (2, 0.03, 0.06), (0, 0.04, 0.04)]
        """
        n = len(p_values)
        if n == 0:
            return []

        # Sort p-values and track original indices
        indexed_p = [(i, p) for i, p in enumerate(p_values)]
        sorted_p = sorted(indexed_p, key=lambda x: x[1])

        # Apply step-down correction
        results = []
        max_corrected = 0.0

        for rank, (orig_idx, raw_p) in enumerate(sorted_p):
            # Correction factor decreases as rank increases
            correction_factor = n - rank
            corrected_p = min(raw_p * correction_factor, 1.0)
            # Enforce monotonicity
            corrected_p = max(corrected_p, max_corrected)
            max_corrected = corrected_p
            results.append((orig_idx, raw_p, corrected_p))

        return results

    def cohens_d(
        self,
        group1: list[float] | np.ndarray,
        group2: list[float] | np.ndarray,
    ) -> EffectSizeResult:
        """Calculate Cohen's d effect size for independent samples.

        Cohen's d measures the standardized difference between two group means.
        Uses pooled standard deviation.

        Formula:
            d = (mean1 - mean2) / pooled_std
            pooled_std = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

        Args:
            group1: Scores/metrics from first group
            group2: Scores/metrics from second group

        Returns:
            EffectSizeResult with value and interpretation
        """
        group1_arr = np.asarray(group1)
        group2_arr = np.asarray(group2)

        n1, n2 = len(group1_arr), len(group2_arr)
        mean1, mean2 = np.mean(group1_arr), np.mean(group2_arr)
        var1, var2 = np.var(group1_arr, ddof=1), np.var(group2_arr, ddof=1)

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)

        if pooled_std == 0:
            d = 0.0 if mean1 == mean2 else float("inf")
        else:
            d = (mean1 - mean2) / pooled_std

        interpretation = self._interpret_effect_size(abs(d))

        return EffectSizeResult(
            value=float(d),
            interpretation=interpretation,
            method="cohens_d",
        )

    def cohens_d_paired(
        self,
        pre: list[float] | np.ndarray,
        post: list[float] | np.ndarray,
    ) -> EffectSizeResult:
        """Calculate Cohen's d for paired/repeated measures.

        Uses the standard deviation of the difference scores in the denominator.

        Formula:
            d = mean(post - pre) / std(post - pre)

        Args:
            pre: Pre-treatment/baseline scores
            post: Post-treatment scores (same subjects, same order)

        Returns:
            EffectSizeResult with value and interpretation

        Raises:
            ValueError: If pre and post have different lengths
        """
        pre_arr = np.asarray(pre)
        post_arr = np.asarray(post)

        if len(pre_arr) != len(post_arr):
            raise ValueError("Pre and post must have same length for paired analysis")

        diff = post_arr - pre_arr
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        if std_diff == 0:
            d = 0.0 if mean_diff == 0 else float("inf")
        else:
            d = mean_diff / std_diff

        interpretation = self._interpret_effect_size(abs(d))

        return EffectSizeResult(
            value=float(d),
            interpretation=interpretation,
            method="cohens_d_paired",
        )

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size magnitude.

        Thresholds (Cohen, 1988):
            |d| < 0.2: negligible
            0.2 <= |d| < 0.5: small
            0.5 <= |d| < 0.8: medium
            |d| >= 0.8: large
        """
        d_abs = abs(d)
        if d_abs < self.EFFECT_SIZE_THRESHOLDS["negligible"]:
            return "negligible"
        elif d_abs < self.EFFECT_SIZE_THRESHOLDS["small"]:
            return "small"
        elif d_abs < self.EFFECT_SIZE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "large"

    def wilson_score_interval(
        self,
        successes: int,
        n: int,
        confidence: float | None = None,
    ) -> ConfidenceInterval:
        """Calculate Wilson score confidence interval for a proportion.

        The Wilson score interval has better coverage properties than the
        normal approximation interval, especially for small samples or
        proportions near 0 or 1.

        Formula:
            p̂ = successes / n
            center = (p̂ + z²/2n) / (1 + z²/n)
            margin = z * sqrt(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)

        Args:
            successes: Number of successes (e.g., correct answers)
            n: Total number of trials
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            ConfidenceInterval for the proportion

        Raises:
            ValueError: If successes > n or either is negative
        """
        if confidence is None:
            confidence = self.confidence_level

        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if successes < 0 or successes > n:
            raise ValueError(f"successes must be in [0, n], got {successes} with n={n}")

        p_hat = successes / n
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return ConfidenceInterval(
            point_estimate=p_hat,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            method="Wilson score",
        )

    def t_confidence_interval(
        self,
        values: list[float] | np.ndarray,
        confidence: float | None = None,
    ) -> ConfidenceInterval:
        """Calculate t-distribution confidence interval for a mean.

        Uses the t-distribution which is appropriate for small samples
        or when population variance is unknown.

        Formula:
            CI = mean ± t_{α/2, n-1} * (std / sqrt(n))

        Args:
            values: Sample values
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            ConfidenceInterval for the mean

        Raises:
            ValueError: If fewer than 2 values provided
        """
        if confidence is None:
            confidence = self.confidence_level

        values_arr = np.asarray(values)
        n = len(values_arr)

        if n < 2:
            raise ValueError(f"Need at least 2 values for CI, got {n}")

        mean = np.mean(values_arr)
        std = np.std(values_arr, ddof=1)
        se = std / math.sqrt(n)

        # t critical value
        df = n - 1
        t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)

        margin = t_crit * se

        return ConfidenceInterval(
            point_estimate=float(mean),
            lower_bound=float(mean - margin),
            upper_bound=float(mean + margin),
            confidence_level=confidence,
            method="t-distribution",
        )

    def bootstrap_confidence_interval(
        self,
        values: list[float] | np.ndarray,
        confidence: float | None = None,
        n_bootstrap: int = 10000,
        statistic: str = "mean",
        seed: int | None = None,
    ) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval using percentile method.

        Bootstrap resampling provides robust confidence intervals that don't
        assume normality. Useful for medians and other non-standard statistics.

        Args:
            values: Sample values
            confidence: Confidence level (default: self.confidence_level)
            n_bootstrap: Number of bootstrap resamples (default: 10000)
            statistic: Statistic to estimate ("mean", "median", "std")
            seed: Random seed for reproducibility

        Returns:
            ConfidenceInterval using bootstrap percentile method

        Raises:
            ValueError: If fewer than 2 values or invalid statistic
        """
        if confidence is None:
            confidence = self.confidence_level

        values_arr = np.asarray(values)
        n = len(values_arr)

        if n < 2:
            raise ValueError(f"Need at least 2 values for bootstrap CI, got {n}")

        stat_funcs: dict[str, Any] = {
            "mean": np.mean,
            "median": np.median,
            "std": lambda x: np.std(x, ddof=1),
        }

        if statistic not in stat_funcs:
            raise ValueError(f"Unknown statistic '{statistic}'. Valid: {list(stat_funcs.keys())}")

        stat_func = stat_funcs[statistic]
        point_estimate = float(stat_func(values_arr))

        # Generate bootstrap samples
        rng = np.random.default_rng(seed)
        bootstrap_stats_list: list[float] = []

        for _ in range(n_bootstrap):
            resample = rng.choice(values_arr, size=n, replace=True)
            bootstrap_stats_list.append(float(stat_func(resample)))

        bootstrap_stats = np.array(bootstrap_stats_list)

        # Percentile method
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return ConfidenceInterval(
            point_estimate=point_estimate,
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=confidence,
            method=f"bootstrap ({statistic})",
        )

    def mcnemar_test(
        self,
        baseline_correct: list[bool],
        treatment_correct: list[bool],
    ) -> StatisticalTestResult:
        """McNemar's test for paired nominal data.

        Tests whether the proportion of correct responses differs between
        baseline and treatment conditions for paired observations.

        Use this when you have binary outcomes (correct/incorrect) for the
        same items under two conditions.

        Args:
            baseline_correct: Whether each item was correct under baseline
            treatment_correct: Whether each item was correct under treatment

        Returns:
            StatisticalTestResult for McNemar's test

        Raises:
            ValueError: If lists have different lengths or are empty
        """
        baseline_arr = np.asarray(baseline_correct, dtype=bool)
        treatment_arr = np.asarray(treatment_correct, dtype=bool)

        if len(baseline_arr) != len(treatment_arr):
            raise ValueError("Baseline and treatment must have same length")

        if len(baseline_arr) == 0:
            raise ValueError("Cannot perform test on empty arrays")

        # Count discordant pairs
        # b = baseline correct, treatment incorrect
        # c = baseline incorrect, treatment correct
        b = np.sum(baseline_arr & ~treatment_arr)
        c = np.sum(~baseline_arr & treatment_arr)

        # McNemar's test statistic with continuity correction
        if b + c == 0:
            # No discordant pairs, no difference
            return StatisticalTestResult(
                test_name="McNemar's test",
                test_statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                sample_size_baseline=len(baseline_arr),
                sample_size_treatment=len(treatment_arr),
            )

        # Chi-square statistic with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return StatisticalTestResult(
            test_name="McNemar's test",
            test_statistic=float(chi2),
            p_value=float(p_value),
            degrees_of_freedom=1.0,
            is_significant=bool(p_value < self.alpha),
            confidence_level=self.confidence_level,
            sample_size_baseline=len(baseline_arr),
            sample_size_treatment=len(treatment_arr),
        )

    def summarize_multiple_tests(
        self,
        test_results: list[StatisticalTestResult],
        apply_correction: bool = True,
    ) -> dict:
        """Summarize results from multiple statistical tests.

        Args:
            test_results: List of StatisticalTestResult objects
            apply_correction: Whether to apply Bonferroni correction

        Returns:
            Dictionary with summary statistics including:
            - n_tests: Number of tests performed
            - n_significant_raw: Number significant before correction
            - n_significant_corrected: Number significant after correction
            - corrected_results: Results with corrected p-values
            - mean_effect_size: Average effect size across tests
        """
        if not test_results:
            return {
                "n_tests": 0,
                "n_significant_raw": 0,
                "n_significant_corrected": 0,
                "corrected_results": [],
                "mean_effect_size": None,
            }

        n_tests = len(test_results)
        p_values = [r.p_value for r in test_results]

        # Count significant before correction
        n_sig_raw = sum(1 for r in test_results if r.is_significant)

        # Apply Bonferroni correction if requested
        if apply_correction:
            corrected_p = self.bonferroni_correction(p_values, n_tests)
            n_sig_corrected = sum(1 for p in corrected_p if p < self.alpha)
        else:
            corrected_p = p_values
            n_sig_corrected = n_sig_raw

        # Calculate mean effect size
        effect_sizes = [r.effect_size for r in test_results if r.effect_size is not None]
        mean_effect = np.mean(effect_sizes) if effect_sizes else None

        # Create corrected results
        corrected_results = []
        for result, corr_p in zip(test_results, corrected_p):
            corrected_results.append(
                StatisticalTestResult(
                    test_name=result.test_name,
                    test_statistic=result.test_statistic,
                    p_value=result.p_value,
                    degrees_of_freedom=result.degrees_of_freedom,
                    is_significant=bool(corr_p < self.alpha),
                    confidence_level=result.confidence_level,
                    effect_size=result.effect_size,
                    effect_size_interpretation=result.effect_size_interpretation,
                    corrected_p_value=corr_p,
                    sample_size_baseline=result.sample_size_baseline,
                    sample_size_treatment=result.sample_size_treatment,
                )
            )

        return {
            "n_tests": n_tests,
            "n_significant_raw": n_sig_raw,
            "n_significant_corrected": n_sig_corrected,
            "corrected_results": corrected_results,
            "mean_effect_size": float(mean_effect) if mean_effect is not None else None,
        }


def compare_conditions(
    baseline_metrics: dict[str, list[float]],
    treatment_metrics: dict[str, list[float]],
    alpha: float = 0.05,
    paired: bool = True,
) -> dict[str, StatisticalTestResult]:
    """Compare baseline and treatment conditions across multiple metrics.

    Convenience function for running multiple comparisons with appropriate
    corrections.

    Args:
        baseline_metrics: Dict mapping metric names to baseline values
        treatment_metrics: Dict mapping metric names to treatment values
        alpha: Significance level
        paired: Whether to use paired t-test (True) or independent (False)

    Returns:
        Dict mapping metric names to StatisticalTestResult objects

    Example:
        >>> baseline = {"accuracy": [0.7, 0.72, 0.68], "latency": [100, 110, 105]}
        >>> treatment = {"accuracy": [0.9, 0.88, 0.91], "latency": [50, 55, 52]}
        >>> results = compare_conditions(baseline, treatment, paired=True)
    """
    analyzer = StatisticalAnalyzer(alpha=alpha)
    results = {}

    # Get common metrics
    common_metrics = set(baseline_metrics.keys()) & set(treatment_metrics.keys())

    for metric_name in common_metrics:
        baseline_vals = baseline_metrics[metric_name]
        treatment_vals = treatment_metrics[metric_name]

        if paired:
            result = analyzer.paired_t_test(baseline_vals, treatment_vals)
        else:
            result = analyzer.independent_t_test(baseline_vals, treatment_vals)

        results[metric_name] = result

    return results


def format_statistical_report(
    results: dict[str, StatisticalTestResult],
    include_effect_sizes: bool = True,
) -> str:
    """Format statistical test results as a readable report.

    Args:
        results: Dict mapping test names to StatisticalTestResult objects
        include_effect_sizes: Whether to include effect size information

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("Statistical Analysis Report")
    lines.append("=" * 60)
    lines.append("")

    for name, result in results.items():
        lines.append(f"Metric: {name}")
        lines.append("-" * 40)
        lines.append(f"  Test: {result.test_name}")
        lines.append(f"  t-statistic: {result.test_statistic:.4f}")
        lines.append(f"  p-value: {result.p_value:.4f}")

        if result.corrected_p_value is not None:
            lines.append(f"  Corrected p-value: {result.corrected_p_value:.4f}")

        sig_marker = "✓" if result.is_significant else "✗"
        lines.append(f"  Significant at α={1 - result.confidence_level:.2f}: {sig_marker}")

        if include_effect_sizes and result.effect_size is not None:
            lines.append(f"  Effect size (d): {result.effect_size:.3f}")
            lines.append(f"  Interpretation: {result.effect_size_interpretation}")

        lines.append("")

    return "\n".join(lines)

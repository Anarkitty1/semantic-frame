"""Performance benchmark tests for semantic_frame.

These tests validate performance characteristics at various data scales.
Use pytest markers to run specific benchmark groups:

    # Run all benchmarks (excludes slow 1M tests)
    uv run pytest tests/test_benchmarks.py -v

    # Run including 1M point tests (slow)
    uv run pytest tests/test_benchmarks.py -v -m "benchmark or slow"

    # Run only series benchmarks
    uv run pytest tests/test_benchmarks.py -v -k "Series"
"""

import sys
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from semantic_frame import describe_dataframe, describe_series

# Check optional dependency availability
try:
    from semantic_frame.integrations.mcp import describe_data as mcp_describe_data

    mcp_available = True
except ImportError:
    mcp_available = False
    mcp_describe_data = None  # type: ignore[assignment]


# Performance thresholds (seconds)
THRESHOLDS = {
    "small": 0.1,  # 100 points
    "medium": 1.0,  # 10,000 points
    "large": 10.0,  # 100,000 points
    "very_large": 60.0,  # 1,000,000 points
}


def generate_random_data(n: int, seed: int = 42) -> np.ndarray:
    """Generate random data with some noise."""
    rng = np.random.default_rng(seed)
    return rng.random(n) * 100


def generate_trend_data(n: int, seed: int = 42) -> np.ndarray:
    """Generate data with clear trend."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    noise = rng.normal(0, 0.5, n)
    return x * 5 + noise + 50


def generate_seasonal_data(n: int, seed: int = 42) -> np.ndarray:
    """Generate data with seasonal pattern."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10 * np.pi, n)
    noise = rng.normal(0, 0.3, n)
    return np.sin(x) * 20 + noise + 50


def generate_anomaly_data(n: int, seed: int = 42) -> np.ndarray:
    """Generate data with anomalies."""
    rng = np.random.default_rng(seed)
    data = rng.normal(50, 5, n)
    # Add anomalies at 1% of points
    anomaly_indices = rng.choice(n, size=max(1, n // 100), replace=False)
    data[anomaly_indices] = rng.choice([0, 100, 150], size=len(anomaly_indices))
    return data


def time_function(func: Callable, *args, **kwargs) -> tuple[float, object]:
    """Time a function execution and return (elapsed_time, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


@pytest.mark.benchmark
class TestDescribeSeriesBenchmarks:
    """Benchmark tests for describe_series at various data sizes."""

    @pytest.mark.parametrize(
        "size,threshold",
        [
            (100, THRESHOLDS["small"]),
            (10_000, THRESHOLDS["medium"]),
            (100_000, THRESHOLDS["large"]),
        ],
    )
    @pytest.mark.parametrize(
        "generator,pattern",
        [
            (generate_random_data, "random"),
            (generate_trend_data, "trend"),
            (generate_seasonal_data, "seasonal"),
            (generate_anomaly_data, "anomaly"),
        ],
    )
    def test_series_performance(
        self,
        size: int,
        threshold: float,
        generator: Callable,
        pattern: str,
    ) -> None:
        """Test describe_series performance within threshold.

        Tests various data patterns at different sizes.
        """
        data = generator(size)
        elapsed, result = time_function(describe_series, data)

        assert elapsed < threshold, (
            f"{pattern} data with {size:,} points took {elapsed:.2f}s " f"(threshold: {threshold}s)"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_series_small_comprehensive(self) -> None:
        """Test small dataset comprehensively."""
        data = generate_trend_data(100)
        elapsed, result = time_function(describe_series, data, context="Small Test")

        assert elapsed < THRESHOLDS["small"]
        assert "Small Test" in result

    def test_series_medium_with_json(self) -> None:
        """Test medium dataset with JSON output."""
        data = generate_random_data(10_000)
        elapsed, result = time_function(describe_series, data, context="Medium Test", output="json")

        assert elapsed < THRESHOLDS["medium"]
        assert isinstance(result, dict)
        assert "trend" in result or "volatility" in result


@pytest.mark.benchmark
@pytest.mark.slow
class TestVeryLargeSeriesBenchmarks:
    """Benchmark tests for 1M+ data points.

    These tests are slow and opt-in only.
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize(
        "generator,pattern",
        [
            (generate_random_data, "random"),
            (generate_trend_data, "trend"),
        ],
    )
    def test_million_points(self, generator: Callable, pattern: str) -> None:
        """Test with 1 million data points."""
        data = generator(1_000_000)
        elapsed, result = time_function(describe_series, data)

        assert elapsed < THRESHOLDS["very_large"], (
            f"1M {pattern} points took {elapsed:.2f}s " f"(threshold: {THRESHOLDS['very_large']}s)"
        )
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.benchmark
class TestDescribeDataframeBenchmarks:
    """Benchmark tests for describe_dataframe at various sizes."""

    def _create_dataframe(self, n_rows: int, n_cols: int, seed: int = 42) -> pd.DataFrame:
        """Create a test DataFrame with specified dimensions."""
        rng = np.random.default_rng(seed)
        data = {f"col_{i}": rng.random(n_rows) * 100 for i in range(n_cols)}
        return pd.DataFrame(data)

    @pytest.mark.parametrize(
        "n_rows,n_cols,threshold",
        [
            (100, 3, THRESHOLDS["small"]),
            (100, 10, THRESHOLDS["small"]),
            (1000, 3, THRESHOLDS["small"]),
            (10_000, 3, THRESHOLDS["medium"]),
            (10_000, 10, THRESHOLDS["medium"]),
            (10_000, 25, THRESHOLDS["medium"] * 2),  # More columns need more time
            (100_000, 3, THRESHOLDS["large"]),
        ],
    )
    def test_dataframe_performance(
        self,
        n_rows: int,
        n_cols: int,
        threshold: float,
    ) -> None:
        """Test describe_dataframe performance within threshold."""
        df = self._create_dataframe(n_rows, n_cols)
        elapsed, result = time_function(describe_dataframe, df)

        assert elapsed < threshold, (
            f"DataFrame ({n_rows:,} rows x {n_cols} cols) took {elapsed:.2f}s "
            f"(threshold: {threshold}s)"
        )
        # describe_dataframe returns DataFrameResult object
        assert hasattr(result, "summary_narrative")
        assert len(result.summary_narrative) > 0

    def test_dataframe_with_correlations(self) -> None:
        """Test DataFrame analysis with correlation detection."""
        rng = np.random.default_rng(42)
        n = 1000

        # Create correlated columns
        x = rng.random(n) * 100
        df = pd.DataFrame(
            {
                "base": x,
                "positive_corr": x + rng.normal(0, 5, n),
                "negative_corr": 100 - x + rng.normal(0, 5, n),
                "uncorrelated": rng.random(n) * 100,
            }
        )

        elapsed, result = time_function(describe_dataframe, df)

        assert elapsed < THRESHOLDS["small"]
        # describe_dataframe returns DataFrameResult object
        assert hasattr(result, "summary_narrative")
        assert hasattr(result, "correlations")


@pytest.mark.benchmark
@pytest.mark.slow
class TestVeryLargeDataframeBenchmarks:
    """Benchmark tests for large DataFrames.

    These tests are slow and opt-in only.
    """

    def test_large_dataframe(self) -> None:
        """Test with 1M row DataFrame."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "col_1": rng.random(1_000_000) * 100,
                "col_2": rng.random(1_000_000) * 100,
                "col_3": rng.random(1_000_000) * 100,
            }
        )

        elapsed, result = time_function(describe_dataframe, df)

        assert elapsed < THRESHOLDS["very_large"], (
            f"1M row DataFrame took {elapsed:.2f}s " f"(threshold: {THRESHOLDS['very_large']}s)"
        )
        # describe_dataframe returns DataFrameResult object
        assert hasattr(result, "summary_narrative")


@pytest.mark.benchmark
class TestFrameworkIntegrationBenchmarks:
    """Benchmark tests for framework integration overhead."""

    @pytest.mark.skipif(not mcp_available, reason="mcp not installed")
    def test_mcp_tool_overhead(self) -> None:
        """Test MCP describe_data tool overhead vs direct call."""
        import json

        data = generate_random_data(1000)
        data_str = json.dumps(data.tolist())

        # Direct call baseline
        direct_start = time.perf_counter()
        _ = describe_series(data)
        direct_time = time.perf_counter() - direct_start

        # MCP tool call (includes parsing)
        mcp_start = time.perf_counter()
        _ = mcp_describe_data(data_str, "Benchmark")
        mcp_time = time.perf_counter() - mcp_start

        # MCP overhead should be less than 2x direct call
        overhead_ratio = mcp_time / direct_time if direct_time > 0 else float("inf")
        assert overhead_ratio < 2.0, (
            f"MCP overhead too high: {overhead_ratio:.1f}x "
            f"(direct: {direct_time:.3f}s, MCP: {mcp_time:.3f}s)"
        )

    def test_langchain_tool_overhead(self) -> None:
        """Test LangChain tool overhead vs direct call."""
        import json

        from semantic_frame.integrations.langchain import SemanticAnalysisTool

        data = generate_random_data(1000)
        data_str = json.dumps(data.tolist())

        # Direct call baseline
        direct_start = time.perf_counter()
        _ = describe_series(data)
        direct_time = time.perf_counter() - direct_start

        # LangChain tool call
        tool = SemanticAnalysisTool(context="Benchmark")
        lc_start = time.perf_counter()
        _ = tool._run(data_str)
        lc_time = time.perf_counter() - lc_start

        # LangChain overhead should be less than 2x direct call
        overhead_ratio = lc_time / direct_time if direct_time > 0 else float("inf")
        assert overhead_ratio < 2.0, (
            f"LangChain overhead too high: {overhead_ratio:.1f}x "
            f"(direct: {direct_time:.3f}s, LangChain: {lc_time:.3f}s)"
        )

    def test_crewai_function_overhead(self) -> None:
        """Test CrewAI function overhead vs direct call."""
        import json

        from semantic_frame.integrations.crewai import semantic_analysis

        data = generate_random_data(1000)
        data_str = json.dumps(data.tolist())

        # Direct call baseline
        direct_start = time.perf_counter()
        _ = describe_series(data)
        direct_time = time.perf_counter() - direct_start

        # CrewAI function call
        crewai_start = time.perf_counter()
        _ = semantic_analysis(data_str, "Benchmark")
        crewai_time = time.perf_counter() - crewai_start

        # CrewAI overhead should be less than 2x direct call
        overhead_ratio = crewai_time / direct_time if direct_time > 0 else float("inf")
        assert overhead_ratio < 2.0, (
            f"CrewAI overhead too high: {overhead_ratio:.1f}x "
            f"(direct: {direct_time:.3f}s, CrewAI: {crewai_time:.3f}s)"
        )


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark tests for memory efficiency."""

    def test_output_smaller_than_input(self) -> None:
        """Verify output is significantly smaller than input."""
        data = generate_random_data(10_000)

        # Input size (approximate)
        input_size = sys.getsizeof(data) + data.nbytes

        # Output size
        result = describe_series(data)
        output_size = sys.getsizeof(result)

        # Output should be much smaller than input
        compression_ratio = 1 - (output_size / input_size)
        assert compression_ratio > 0.9, (
            f"Compression ratio too low: {compression_ratio:.2%} "
            f"(input: {input_size:,} bytes, output: {output_size:,} bytes)"
        )

    def test_result_object_size(self) -> None:
        """Test SemanticResult object memory efficiency."""
        from semantic_frame import describe_series

        data = generate_random_data(10_000)
        result = describe_series(data, output="full")

        # SemanticResult should be compact
        result_size = sys.getsizeof(result)

        # Result object should be under 10KB regardless of input size
        assert (
            result_size < 10_000
        ), f"SemanticResult too large: {result_size:,} bytes (expected < 10KB)"

    @pytest.mark.parametrize(
        "size",
        [100, 1_000, 10_000, 100_000],
    )
    def test_compression_scales(self, size: int) -> None:
        """Verify compression ratio improves with data size."""
        data = generate_random_data(size)

        # Input tokens (rough estimate: 1 number ≈ 3-4 tokens)
        input_tokens_estimate = size * 4

        # Output tokens (rough estimate from result length)
        result = describe_series(data)
        output_tokens_estimate = len(result.split())

        compression_ratio = 1 - (output_tokens_estimate / input_tokens_estimate)

        # Compression should be better for larger datasets
        if size >= 1000:
            assert compression_ratio > 0.95, (
                f"Compression ratio for {size:,} points: {compression_ratio:.2%} "
                f"(expected > 95%)"
            )
        else:
            assert compression_ratio > 0.5, (
                f"Compression ratio for {size:,} points: {compression_ratio:.2%} "
                f"(expected > 50%)"
            )


@pytest.mark.benchmark
class TestEdgeCasePerformance:
    """Test performance with edge case data."""

    def test_constant_data_fast(self) -> None:
        """Constant data should be analyzed quickly."""
        data = np.full(10_000, 42.0)
        elapsed, result = time_function(describe_series, data)

        # Constant data should be very fast (skip many analyses)
        assert (
            elapsed < THRESHOLDS["small"]
        ), f"Constant data took {elapsed:.2f}s (expected < {THRESHOLDS['small']}s)"

    def test_two_point_data_fast(self) -> None:
        """Minimal data should be analyzed quickly."""
        data = np.array([10.0, 20.0])
        elapsed, result = time_function(describe_series, data)

        assert elapsed < THRESHOLDS["small"]

    def test_high_variance_data(self) -> None:
        """High variance data should still perform well."""
        rng = np.random.default_rng(42)
        # Extreme variance: values from 0 to 1 million
        data = rng.uniform(0, 1_000_000, 10_000)
        elapsed, result = time_function(describe_series, data)

        assert elapsed < THRESHOLDS["medium"]

    def test_many_anomalies(self) -> None:
        """Data with many anomalies should still perform well."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 10_000)
        # 10% anomalies (extreme case)
        anomaly_indices = rng.choice(10_000, size=1000, replace=False)
        data[anomaly_indices] = rng.choice([-100, 200], size=1000)

        elapsed, result = time_function(describe_series, data)

        assert elapsed < THRESHOLDS["medium"]

"""Integration tests for the complete Semantic Frame library."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from semantic_frame import describe_dataframe, describe_series
from semantic_frame.interfaces.json_schema import DataFrameResult, SemanticResult
from semantic_frame.interfaces.llm_templates import (
    create_agent_context,
    format_for_context,
    format_for_langchain,
    format_for_system_prompt,
)
from semantic_frame.main import compression_stats


class TestDescribeSeries:
    """Integration tests for describe_series function."""

    def test_pandas_series(self):
        """Should work with pandas Series."""
        data = pd.Series([100, 102, 99, 101, 500, 100, 98])
        result = describe_series(data, context="Server Latency")

        assert isinstance(result, str)
        assert "Server Latency" in result

    def test_numpy_array(self):
        """Should work with numpy array."""
        data = np.array([100.0, 102.0, 99.0, 101.0, 500.0, 100.0, 98.0])
        result = describe_series(data, context="Response Time")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_polars_series(self):
        """Should work with polars Series."""
        data = pl.Series("values", [100, 102, 99, 101, 500, 100, 98])
        result = describe_series(data, context="Memory Usage")

        assert isinstance(result, str)
        assert "Memory Usage" in result

    def test_python_list(self):
        """Should work with Python list."""
        data = [100, 102, 99, 101, 500, 100, 98]
        result = describe_series(data, context="CPU Load")

        assert isinstance(result, str)

    def test_output_text(self):
        """Text output should return string."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="text")

        assert isinstance(result, str)

    def test_output_json(self):
        """JSON output should return dict."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="json")

        assert isinstance(result, dict)
        assert "narrative" in result
        assert "trend" in result
        assert "profile" in result

    def test_output_full(self):
        """Full output should return SemanticResult."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="full")

        assert isinstance(result, SemanticResult)
        assert hasattr(result, "narrative")
        assert hasattr(result, "trend")
        assert hasattr(result, "profile")

    def test_json_serializable(self):
        """JSON output should be fully serializable."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="json")

        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["narrative"] == result["narrative"]

    def test_detects_outlier(self):
        """Should detect clear outliers."""
        # Data with obvious outlier at position 4
        data = pd.Series([100, 102, 99, 101, 500, 100, 98])
        result = describe_series(data, output="full")

        assert len(result.anomalies) >= 1
        # The value 500 should be detected
        outlier_values = [a.value for a in result.anomalies]
        assert 500.0 in outlier_values

    def test_detects_trend(self):
        """Should detect rising trend."""
        data = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        result = describe_series(data, output="full")

        assert "rising" in result.trend.value

    def test_compression_ratio(self):
        """Should achieve significant compression."""
        # Large dataset
        data = pd.Series(np.random.normal(100, 10, 1000))
        result = describe_series(data, output="full")

        # Should compress well (>80% reduction)
        assert result.compression_ratio > 0.8


class TestDescribeDataframe:
    """Integration tests for describe_dataframe function."""

    def test_pandas_dataframe(self):
        """Should analyze all numeric columns in pandas DataFrame."""
        df = pd.DataFrame(
            {
                "cpu": [40, 42, 41, 95, 40, 41],
                "memory": [60, 61, 60, 60, 61, 60],
                "name": ["a", "b", "c", "d", "e", "f"],  # Non-numeric
            }
        )

        result = describe_dataframe(df, context="Server Metrics")

        assert isinstance(result, DataFrameResult)
        assert "cpu" in result.columns
        assert "memory" in result.columns
        assert "name" not in result.columns  # Should skip non-numeric

        assert isinstance(result.columns["cpu"], SemanticResult)
        assert isinstance(result.columns["memory"], SemanticResult)

    def test_polars_dataframe(self):
        """Should work with polars DataFrame."""
        df = pl.DataFrame(
            {
                "temperature": [20.5, 21.0, 20.8, 21.2, 20.9],
                "humidity": [45, 46, 44, 45, 46],
            }
        )

        result = describe_dataframe(df, context="Sensor Data")

        assert isinstance(result, DataFrameResult)
        assert "temperature" in result.columns
        assert "humidity" in result.columns

    def test_context_propagation(self):
        """Context should be propagated to each column."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
            }
        )

        result = describe_dataframe(df, context="Test")

        assert "Test - col1" in result.columns["col1"].context
        assert "Test - col2" in result.columns["col2"].context

    def test_returns_dataframe_result(self):
        """Should return DataFrameResult type."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
            }
        )
        result = describe_dataframe(df)

        assert isinstance(result, DataFrameResult)
        assert hasattr(result, "columns")
        assert hasattr(result, "correlations")
        assert hasattr(result, "summary_narrative")


class TestDataframeCorrelations:
    """Integration tests for describe_dataframe correlation analysis."""

    def test_detects_strong_positive_correlation(self):
        """Should detect strong positive correlation."""
        df = pd.DataFrame(
            {
                "sales": [100, 200, 300, 400, 500],
                "revenue": [1000, 2000, 3000, 4000, 5000],
            }
        )
        result = describe_dataframe(df)

        assert len(result.correlations) >= 1
        assert result.correlations[0].state.value == "strongly correlated"
        assert result.correlations[0].correlation > 0.99

    def test_detects_inverse_correlation(self):
        """Should detect inverse correlation (Sales UP, Inventory DOWN)."""
        df = pd.DataFrame(
            {
                "sales": [100, 200, 300, 400, 500],
                "inventory": [500, 400, 300, 200, 100],
            }
        )
        result = describe_dataframe(df)

        assert len(result.correlations) >= 1
        assert "inverse" in result.correlations[0].state.value
        assert result.correlations[0].correlation < -0.99

    def test_correlation_threshold(self):
        """Should respect correlation threshold parameter."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
            }
        )

        # Default threshold (0.5) - random data should have weak correlation
        result = describe_dataframe(df)
        assert len(result.correlations) == 0

        # Very low threshold - should find the weak correlation
        result_low = describe_dataframe(df, correlation_threshold=0.0)
        assert len(result_low.correlations) >= 1

    def test_summary_narrative_present(self):
        """Should generate summary narrative."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = describe_dataframe(df)

        assert len(result.summary_narrative) > 0
        assert "column" in result.summary_narrative.lower()

    def test_summary_mentions_correlations(self):
        """Summary should mention significant correlations."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
            }
        )
        result = describe_dataframe(df)

        assert "correlation" in result.summary_narrative.lower()
        assert "1" in result.summary_narrative  # Should mention count

    def test_no_correlations_for_single_column(self):
        """Single column should have no correlations."""
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
        result = describe_dataframe(df)

        assert len(result.correlations) == 0
        assert "no strong correlation" in result.summary_narrative.lower()

    def test_correlation_narrative_included(self):
        """Each correlation should have a narrative."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
            }
        )
        result = describe_dataframe(df)

        if result.correlations:
            assert len(result.correlations[0].narrative) > 0
            assert "a" in result.correlations[0].narrative
            assert "b" in result.correlations[0].narrative

    def test_polars_correlation(self):
        """Correlation analysis should work with polars DataFrame."""
        df = pl.DataFrame(
            {
                "metric_a": [10, 20, 30, 40, 50],
                "metric_b": [50, 40, 30, 20, 10],
            }
        )
        result = describe_dataframe(df)

        assert isinstance(result, DataFrameResult)
        assert len(result.correlations) >= 1
        assert result.correlations[0].correlation < -0.99


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_compression_stats(self):
        """Should calculate detailed compression stats."""
        data = pd.Series(np.random.normal(100, 10, 500))
        result = describe_series(data, output="full")

        stats = compression_stats(data, result)

        assert stats["original_data_points"] == 500
        assert stats["original_tokens_estimate"] == 1000  # 500 * 2
        assert stats["narrative_tokens"] > 0
        assert 0 <= stats["narrative_compression_ratio"] <= 1

    def test_compression_stats_with_real_tokenizer(self):
        """Should use tiktoken when use_real_tokenizer=True."""
        pytest.importorskip("tiktoken")

        # Use larger dataset to ensure positive compression ratio
        data = pd.Series(np.random.normal(100, 10, 500))
        result = describe_series(data, output="full")

        stats = compression_stats(data, result, use_real_tokenizer=True)

        assert stats["tokenizer"] == "tiktoken"
        assert stats["original_data_points"] == 500
        assert stats["original_tokens_estimate"] > 0
        assert stats["narrative_tokens"] > 0
        assert stats["json_tokens"] > 0
        # With 500 data points, we expect significant compression
        assert stats["narrative_compression_ratio"] > 0.5
        assert stats["json_compression_ratio"] > 0

    def test_compression_stats_real_tokenizer_different_from_estimate(self):
        """Real tokenizer should produce different counts than estimate."""
        pytest.importorskip("tiktoken")

        data = pd.Series(np.random.normal(100, 10, 100))
        result = describe_series(data, output="full")

        stats_estimate = compression_stats(data, result, use_real_tokenizer=False)
        stats_real = compression_stats(data, result, use_real_tokenizer=True)

        assert stats_estimate["tokenizer"] == "estimate"
        assert stats_real["tokenizer"] == "tiktoken"
        # Token counts will differ between estimation and actual tokenization
        assert stats_estimate["original_tokens_estimate"] != stats_real["original_tokens_estimate"]

    def test_compression_stats_real_tokenizer_fallback(self, monkeypatch):
        """Should fallback to estimate when tiktoken is not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("tiktoken not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = describe_series(data, output="full")

        stats = compression_stats(data, result, use_real_tokenizer=True)

        # Should fallback to estimate
        assert stats["tokenizer"] == "estimate"
        assert stats["original_tokens_estimate"] == 10  # 5 * 2


class TestLLMTemplates:
    """Tests for LLM integration templates."""

    def test_format_for_system_prompt(self):
        """Should format result for system prompt injection."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="full")

        prompt = format_for_system_prompt(result)

        assert "DATA CONTEXT:" in prompt
        assert "Trend:" in prompt
        assert "Volatility:" in prompt

    def test_format_for_context(self):
        """Should create concise context string."""
        data = pd.Series([10, 20, 30])
        result = describe_series(data, context="Test", output="full")

        context = format_for_context(result)

        assert "[DATA: Test]" in context

    def test_format_for_langchain(self):
        """Should format for LangChain tool output."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = describe_series(data, output="full")

        output = format_for_langchain(result)

        assert "output" in output
        assert "metadata" in output
        assert output["output"] == result.narrative

    def test_create_agent_context(self):
        """Should create multi-column context for agents."""
        df = pd.DataFrame(
            {
                "metric1": [10, 20, 30],
                "metric2": [100, 200, 300],
            }
        )

        result = describe_dataframe(df)
        context = create_agent_context(result.columns)

        assert "MULTI-COLUMN" in context
        assert "metric1" in context
        assert "metric2" in context


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_unsupported_input_type_raises_error(self):
        """Unsupported input types should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            describe_series({"not": "supported"})
        assert "Unsupported data type" in str(excinfo.value)

    def test_invalid_output_format_raises_error(self):
        """Invalid output format should raise ValueError."""
        data = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError) as excinfo:
            describe_series(data, output="invalid")
        assert "Invalid output format" in str(excinfo.value)
        assert "invalid" in str(excinfo.value)
        assert "text" in str(excinfo.value)  # Should mention valid options

    def test_inf_values_handled(self):
        """Inf values should be filtered out like NaN."""
        data = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])
        result = describe_series(data, output="full")

        # Should treat Inf as missing (20% like NaN test)
        assert result.profile.missing_pct == 20.0
        # Mean should be calculated from clean values only
        assert result.profile.mean == 3.0

    def test_negative_inf_handled(self):
        """Negative Inf values should also be filtered."""
        data = pd.Series([1.0, 2.0, -np.inf, 4.0, 5.0])
        result = describe_series(data, output="full")

        assert result.profile.missing_pct == 20.0

    def test_all_inf_values(self):
        """All-Inf series should be handled like all-NaN."""
        data = pd.Series([np.inf, np.inf, -np.inf])
        result = describe_series(data, output="full")

        assert result.data_quality.value == "fragmented"
        assert "no valid data" in result.narrative.lower()

    def test_non_numeric_pandas_series_raises_error(self):
        """Non-numeric pandas Series should raise TypeError."""
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError) as excinfo:
            describe_series(data)
        assert "numeric" in str(excinfo.value).lower()

    def test_non_numeric_polars_series_raises_error(self):
        """Non-numeric polars Series should raise TypeError."""
        data = pl.Series("strings", ["a", "b", "c"])
        with pytest.raises(TypeError) as excinfo:
            describe_series(data)
        assert "numeric" in str(excinfo.value).lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_series(self):
        """Should handle empty series gracefully."""
        data = pd.Series([], dtype=float)
        result = describe_series(data, output="full")

        assert "no valid data" in result.narrative.lower()

    def test_single_value(self):
        """Should handle single value."""
        data = pd.Series([42.0])
        result = describe_series(data, output="full")

        assert isinstance(result, SemanticResult)
        assert result.profile.count == 1

    def test_all_same_values(self):
        """Should handle constant data."""
        data = pd.Series([5.0] * 100)
        result = describe_series(data, output="full")

        assert result.trend.value == "flat/stationary"
        assert result.volatility.value == "compressed"

    def test_with_nan_values(self):
        """Should handle NaN values correctly."""
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = describe_series(data, output="full")

        assert result.profile.missing_pct == 20.0
        assert result.profile.mean == 3.0  # Mean of clean values

    def test_all_nan(self):
        """Should handle all-NaN series."""
        data = pd.Series([np.nan, np.nan, np.nan])
        result = describe_series(data, output="full")

        assert result.data_quality.value == "fragmented"

    def test_very_large_dataset(self):
        """Should handle large datasets efficiently."""
        data = pd.Series(np.random.normal(100, 10, 100000))
        result = describe_series(data, output="full")

        # Should complete and achieve high compression
        assert result.compression_ratio > 0.99

    def test_extreme_values(self):
        """Should handle extreme numerical values."""
        data = pd.Series([1e-10, 1e10, 1e-10, 1e10])
        result = describe_series(data, output="full")

        assert isinstance(result, SemanticResult)


class TestRealWorldScenarios:
    """Tests simulating real-world use cases."""

    def test_server_latency_with_spike(self):
        """Simulate server latency data with spike detection."""
        # Normal latency with one spike
        np.random.seed(42)
        normal = np.random.normal(50, 5, 100)
        data = pd.Series(np.concatenate([normal, [250], normal]))

        result = describe_series(data, context="API Latency (ms)", output="full")

        assert len(result.anomalies) >= 1
        assert result.anomaly_state.value != "no anomalies"

    def test_sales_trend(self):
        """Simulate sales data with growth trend."""
        # Steadily increasing sales
        base = np.linspace(1000, 2000, 30)
        noise = np.random.normal(0, 50, 30)
        data = pd.Series(base + noise)

        result = describe_series(data, context="Daily Sales ($)", output="full")

        assert "rising" in result.trend.value

    def test_temperature_seasonality(self):
        """Simulate temperature data with seasonal pattern."""
        # Sinusoidal pattern simulating daily temperature variation
        t = np.linspace(0, 4 * np.pi, 100)
        data = pd.Series(20 + 10 * np.sin(t))

        result = describe_series(data, context="Temperature (C)", output="full")

        # Should detect some seasonality
        assert result.seasonality is not None

    def test_multi_metric_dashboard(self):
        """Simulate multi-metric monitoring dashboard."""
        df = pd.DataFrame(
            {
                "cpu_pct": np.random.normal(40, 10, 50),
                "memory_pct": np.random.normal(60, 5, 50),
                "disk_io": np.random.exponential(100, 50),
                "network_bytes": np.random.normal(1e6, 1e5, 50),
            }
        )

        result = describe_dataframe(df, context="Server Health")

        # All metrics should be analyzed
        assert len(result.columns) == 4

        # Should be able to create combined context from columns dict
        context = create_agent_context(result.columns)
        assert "MULTI-COLUMN" in context

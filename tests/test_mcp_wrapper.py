"""Tests for MCP wrapper utilities.

These tests validate the semantic output wrapper functionality
for transforming numerical data from any source.
"""

import numpy as np
import pytest

from semantic_frame.integrations.mcp_wrapper import (
    SemanticMCPWrapper,
    _extract_numeric_array,
    transform_to_semantic,
    wrap_numeric_output,
)


class TestExtractNumericArray:
    """Tests for _extract_numeric_array utility."""

    def test_extract_from_list(self) -> None:
        """Should extract from plain list."""
        result = _extract_numeric_array([1, 2, 3, 4, 5])
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_extract_from_tuple(self) -> None:
        """Should extract from tuple."""
        result = _extract_numeric_array((1.5, 2.5, 3.5))
        assert result == [1.5, 2.5, 3.5]

    def test_extract_from_numpy_array(self) -> None:
        """Should extract from numpy array."""
        arr = np.array([10, 20, 30])
        result = _extract_numeric_array(arr)
        assert result == [10.0, 20.0, 30.0]

    def test_extract_from_dict_with_data_key(self) -> None:
        """Should extract from dict with 'data' key."""
        data = {"data": [1, 2, 3], "other": "value"}
        result = _extract_numeric_array(data)
        assert result == [1.0, 2.0, 3.0]

    def test_extract_from_dict_with_values_key(self) -> None:
        """Should extract from dict with 'values' key."""
        data = {"values": [4, 5, 6]}
        result = _extract_numeric_array(data)
        assert result == [4.0, 5.0, 6.0]

    def test_extract_from_dict_with_readings_key(self) -> None:
        """Should extract from dict with 'readings' key."""
        data = {"readings": [7, 8, 9]}
        result = _extract_numeric_array(data)
        assert result == [7.0, 8.0, 9.0]

    def test_extract_from_json_string(self) -> None:
        """Should extract from JSON string."""
        result = _extract_numeric_array("[1, 2, 3]")
        assert result == [1.0, 2.0, 3.0]

    def test_returns_none_for_non_numeric(self) -> None:
        """Should return None for non-numeric data."""
        assert _extract_numeric_array("not numeric") is None
        assert _extract_numeric_array({"no": "numbers"}) is None
        assert _extract_numeric_array(42) is None


class TestTransformToSemantic:
    """Tests for transform_to_semantic function."""

    def test_transform_list(self) -> None:
        """Should transform list to narrative."""
        result = transform_to_semantic([1, 2, 3, 4, 5])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transform_with_context(self) -> None:
        """Should include context in narrative."""
        result = transform_to_semantic([100, 102, 99, 101, 98], context="Temperature")
        assert isinstance(result, str)
        assert "Temperature" in result

    def test_transform_to_json(self) -> None:
        """Should return dict for json output."""
        result = transform_to_semantic([1, 2, 3, 4, 5], output_format="json")
        assert isinstance(result, dict)
        assert "narrative" in result

    def test_transform_dict_with_data(self) -> None:
        """Should transform dict with data key."""
        result = transform_to_semantic({"data": [10, 20, 30]}, context="Metrics")
        assert isinstance(result, str)
        assert "Metrics" in result

    def test_invalid_data_raises_error(self) -> None:
        """Should raise ValueError for invalid data."""
        with pytest.raises(ValueError, match="Could not extract"):
            transform_to_semantic("not numeric data")


class TestWrapNumericOutput:
    """Tests for wrap_numeric_output decorator."""

    def test_wraps_list_output(self) -> None:
        """Should wrap function returning list."""

        @wrap_numeric_output(context="Test Data")
        def get_numbers():
            return [1, 2, 3, 4, 5]

        result = get_numbers()
        assert isinstance(result, str)
        assert "Test Data" in result

    def test_wraps_numpy_output(self) -> None:
        """Should wrap function returning numpy array."""

        @wrap_numeric_output(context="Array Data")
        def get_array():
            return np.array([10, 20, 30, 40])

        result = get_array()
        assert isinstance(result, str)
        assert "Array Data" in result

    def test_wraps_dict_output_preserves_structure(self) -> None:
        """Should preserve dict structure and add narrative."""

        @wrap_numeric_output(context="Dict Data")
        def get_dict():
            return {"data": [1, 2, 3], "metadata": "info"}

        result = get_dict()
        assert isinstance(result, dict)
        assert "semantic_narrative" in result
        assert "metadata" in result
        assert result["metadata"] == "info"

    def test_dynamic_context_from_key(self) -> None:
        """Should use context from dict key."""

        @wrap_numeric_output(context_key="metric_name")
        def get_metrics():
            return {"metric_name": "CPU Load", "values": [45, 47, 95, 44]}

        result = get_metrics()
        assert isinstance(result, dict)
        assert "CPU Load" in result["semantic_narrative"]

    def test_static_context_as_fallback(self) -> None:
        """Should use static context when key missing."""

        @wrap_numeric_output(context="Fallback", context_key="missing_key")
        def get_data():
            return {"values": [1, 2, 3]}

        result = get_data()
        assert isinstance(result, dict)
        assert "Fallback" in result["semantic_narrative"]

    def test_json_output_format(self) -> None:
        """Should return JSON for json format."""

        @wrap_numeric_output(context="JSON Test", output_format="json")
        def get_numbers():
            return [1, 2, 3, 4, 5]

        result = get_numbers()
        assert isinstance(result, dict)
        assert "narrative" in result
        assert "trend" in result

    def test_passthrough_on_failure_enabled(self) -> None:
        """Should return original on failure when passthrough enabled."""

        @wrap_numeric_output(passthrough_on_failure=True)
        def get_non_numeric():
            return "not numbers"

        result = get_non_numeric()
        assert result == "not numbers"

    def test_passthrough_on_failure_disabled(self) -> None:
        """Should raise error on failure when passthrough disabled."""

        @wrap_numeric_output(passthrough_on_failure=False)
        def get_non_numeric():
            return "not numbers"

        with pytest.raises(ValueError, match="Could not extract"):
            get_non_numeric()

    def test_preserves_function_metadata(self) -> None:
        """Should preserve wrapped function's metadata."""

        @wrap_numeric_output(context="Test")
        def my_function_with_docstring():
            """This is a docstring."""
            return [1, 2, 3]

        assert my_function_with_docstring.__name__ == "my_function_with_docstring"
        assert "docstring" in (my_function_with_docstring.__doc__ or "")

    def test_passes_arguments_through(self) -> None:
        """Should pass arguments to wrapped function."""

        @wrap_numeric_output(context="Args Test")
        def get_range(start: int, end: int):
            return list(range(start, end))

        result = get_range(1, 6)
        assert isinstance(result, str)
        assert "Args Test" in result


class TestSemanticMCPWrapper:
    """Tests for SemanticMCPWrapper class."""

    def test_initialization_defaults(self) -> None:
        """Should initialize with default values."""
        wrapper = SemanticMCPWrapper()
        assert wrapper.default_context is None
        assert wrapper.default_format == "text"
        assert wrapper.passthrough_on_failure is True

    def test_initialization_with_options(self) -> None:
        """Should store initialization options."""
        wrapper = SemanticMCPWrapper(
            default_context="Sensors",
            default_format="json",
            passthrough_on_failure=False,
        )
        assert wrapper.default_context == "Sensors"
        assert wrapper.default_format == "json"
        assert wrapper.passthrough_on_failure is False

    def test_transform_method(self) -> None:
        """Should transform data using default context."""
        wrapper = SemanticMCPWrapper(default_context="Default Context")
        result = wrapper.transform([1, 2, 3, 4, 5])
        assert isinstance(result, str)
        assert "Default Context" in result

    def test_transform_with_override(self) -> None:
        """Should allow context override in transform."""
        wrapper = SemanticMCPWrapper(default_context="Default")
        result = wrapper.transform([1, 2, 3], context="Override")
        assert "Override" in result

    def test_transform_json_format(self) -> None:
        """Should support json format override."""
        wrapper = SemanticMCPWrapper(default_format="text")
        result = wrapper.transform([1, 2, 3], output_format="json")
        assert isinstance(result, dict)

    def test_wrap_decorator(self) -> None:
        """Should provide decorator via wrap method."""
        wrapper = SemanticMCPWrapper(default_context="Wrapper Context")

        @wrapper.wrap()
        def get_numbers():
            return [1, 2, 3, 4, 5]

        result = get_numbers()
        assert isinstance(result, str)
        assert "Wrapper Context" in result

    def test_wrap_with_override(self) -> None:
        """Should allow context override in wrap."""
        wrapper = SemanticMCPWrapper(default_context="Default")

        @wrapper.wrap(context="Specific Context")
        def get_numbers():
            return [10, 20, 30]

        result = get_numbers()
        assert "Specific Context" in result


class TestIntegrationWithMCP:
    """Integration tests simulating MCP tool usage."""

    def test_wrap_cpu_metrics_tool(self) -> None:
        """Should wrap a CPU metrics tool."""

        @wrap_numeric_output(context="CPU Usage %")
        def get_cpu_usage():
            # Simulates MCP tool returning metrics
            return [45, 47, 46, 95, 44, 45, 46, 43]

        result = get_cpu_usage()
        assert isinstance(result, str)
        assert "CPU Usage %" in result
        # Should detect the spike
        assert "anomal" in result.lower() or "95" in result

    def test_wrap_price_feed_tool(self) -> None:
        """Should wrap a price feed tool."""

        @wrap_numeric_output(context_key="symbol")
        def get_prices():
            return {
                "symbol": "BTC/USD",
                "prices": [42000, 42500, 41000, 43000, 44000],
                "timestamp": "2025-01-01T00:00:00Z",
            }

        result = get_prices()
        assert isinstance(result, dict)
        assert "BTC/USD" in result["semantic_narrative"]
        # Original data preserved
        assert result["timestamp"] == "2025-01-01T00:00:00Z"

    def test_wrap_sensor_readings(self) -> None:
        """Should wrap IoT sensor readings."""
        wrapper = SemanticMCPWrapper(default_context="Temperature (C)")

        @wrapper.wrap()
        def read_temperature():
            return np.array([22.1, 22.3, 22.0, 22.2, 35.5, 22.1])

        result = read_temperature()
        assert isinstance(result, str)
        assert "Temperature (C)" in result
        # Should detect the temperature spike
        assert "anomal" in result.lower() or "35" in result

    def test_batch_processing_simulation(self) -> None:
        """Should handle batch processing of multiple series."""
        wrapper = SemanticMCPWrapper()

        datasets = {
            "cpu": [45, 47, 46, 44, 45],
            "memory": [60, 61, 60, 62, 61],
            "disk": [80, 81, 80, 80, 81],
        }

        results = {}
        for name, data in datasets.items():
            results[name] = wrapper.transform(data, context=name.upper())

        assert len(results) == 3
        assert all(isinstance(v, str) for v in results.values())
        assert "CPU" in results["cpu"]
        assert "MEMORY" in results["memory"]
        assert "DISK" in results["disk"]

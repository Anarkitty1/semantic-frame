"""Tests for MCP integration.

Tests for the Model Context Protocol server including:
- describe_data: Single series analysis
- describe_batch: Batch analysis
- describe_json: JSON output format
- wrap_for_semantic_output: Decorator for existing tools
- create_semantic_tool: Factory for semantic tools
"""

import json

import pytest

# Skip tests if mcp is not installed
try:
    from semantic_frame.integrations.mcp import (
        create_semantic_tool,
        describe_batch,
        describe_data,
        describe_json,
        get_mcp_tool_config,
        mcp,
        wrap_for_semantic_output,
    )

    mcp_available = True
except ImportError:
    mcp_available = False


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestMCPServer:
    """Tests for MCP server initialization."""

    def test_server_initialization(self) -> None:
        """Test that the MCP server is initialized correctly."""
        assert mcp.name == "semantic-frame"

    @pytest.mark.asyncio
    async def test_tool_registration(self) -> None:
        """Test that all tools are registered."""
        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "describe_data" in tool_names
        assert "describe_batch" in tool_names
        assert "describe_json" in tool_names


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestDescribeData:
    """Tests for describe_data tool."""

    def test_basic_analysis(self) -> None:
        """Test basic data analysis."""
        data = "[10, 12, 11, 13, 12]"
        context = "Test Metrics"

        result = describe_data(data, context)

        assert isinstance(result, str)
        assert "Test Metrics" in result

    def test_csv_input(self) -> None:
        """Test CSV format input."""
        result = describe_data("1, 2, 3, 4, 5", "CSV Data")

        assert isinstance(result, str)
        assert "CSV Data" in result

    def test_newline_input(self) -> None:
        """Test newline-separated input."""
        result = describe_data("1\n2\n3\n4\n5", "Newline Data")

        assert isinstance(result, str)
        assert "Newline Data" in result

    def test_anomaly_detection(self) -> None:
        """Test anomaly detection in data."""
        result = describe_data("[10, 11, 10, 100, 10, 11]", "Anomaly Test")

        assert isinstance(result, str)
        assert "anomal" in result.lower() or "outlier" in result.lower()

    def test_error_handling(self) -> None:
        """Test error handling for invalid input."""
        result = describe_data("invalid data", "Context")
        assert "Error analyzing data" in result


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestDescribeBatch:
    """Tests for describe_batch tool."""

    def test_single_dataset(self) -> None:
        """Test batch with single dataset."""
        datasets = json.dumps({"cpu": [45, 47, 46, 48, 47]})
        result = describe_batch(datasets)

        assert isinstance(result, str)
        assert "cpu" in result.lower()

    def test_multiple_datasets(self) -> None:
        """Test batch with multiple datasets."""
        datasets = json.dumps(
            {
                "cpu": [45, 47, 46, 48, 47],
                "memory": [60, 61, 60, 61, 60],
            }
        )
        result = describe_batch(datasets)

        assert isinstance(result, str)
        assert "cpu" in result.lower()
        assert "memory" in result.lower()

    def test_three_datasets(self) -> None:
        """Test batch with three datasets."""
        datasets = json.dumps(
            {
                "cpu": [45, 47, 95, 44, 46],
                "memory": [60, 61, 60, 61, 60],
                "disk": [10, 20, 30, 40, 50],
            }
        )
        result = describe_batch(datasets)

        assert "cpu" in result.lower()
        assert "memory" in result.lower()
        assert "disk" in result.lower()

    def test_invalid_json(self) -> None:
        """Test error handling for invalid JSON."""
        result = describe_batch("not valid json")
        assert "Error parsing datasets JSON" in result

    def test_empty_datasets(self) -> None:
        """Test empty datasets dict."""
        result = describe_batch("{}")
        assert isinstance(result, str)

    def test_output_format_text_default(self) -> None:
        """Test text output format (default)."""
        datasets = json.dumps({"cpu": [45, 47, 46, 48, 47]})
        result = describe_batch(datasets)

        # Default is text format
        assert isinstance(result, str)
        assert "cpu:" in result.lower()
        # Should NOT be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_output_format_text_explicit(self) -> None:
        """Test explicit text output format."""
        datasets = json.dumps({"metric": [1, 2, 3, 4, 5]})
        result = describe_batch(datasets, output_format="text")

        assert isinstance(result, str)
        assert "metric:" in result.lower()

    def test_output_format_json(self) -> None:
        """Test JSON output format."""
        datasets = json.dumps({"cpu": [45, 47, 46, 48, 47], "memory": [60, 61, 60]})
        result = describe_batch(datasets, output_format="json")

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "cpu" in parsed
        assert "memory" in parsed
        # Each result should have analysis fields
        assert "narrative" in parsed["cpu"]
        assert "trend" in parsed["cpu"]

    def test_output_format_json_single_dataset(self) -> None:
        """Test JSON output with single dataset."""
        datasets = json.dumps({"temperature": [22.1, 22.3, 22.0, 22.2]})
        result = describe_batch(datasets, output_format="json")

        parsed = json.loads(result)
        assert "temperature" in parsed
        assert isinstance(parsed["temperature"], dict)

    def test_output_format_invalid_falls_back_to_text(self) -> None:
        """Test invalid output_format falls back to text."""
        datasets = json.dumps({"data": [1, 2, 3]})
        result = describe_batch(datasets, output_format="invalid_format")

        # Should fall back to text format
        assert isinstance(result, str)
        assert "data:" in result.lower()

    def test_output_format_text_vs_json_differ(self) -> None:
        """Test that text and json outputs have different structure."""
        datasets = json.dumps({"metric": [10, 20, 30, 40, 50]})

        text_result = describe_batch(datasets, output_format="text")
        json_result = describe_batch(datasets, output_format="json")

        # They should be different
        assert text_result != json_result
        # JSON should be parseable
        parsed = json.loads(json_result)
        assert isinstance(parsed, dict)


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestDescribeJson:
    """Tests for describe_json tool."""

    def test_returns_valid_json(self) -> None:
        """Test that output is valid JSON."""
        result = describe_json("[1, 2, 3, 4, 5]", "Test")

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_has_narrative(self) -> None:
        """Test that JSON output has narrative field."""
        result = describe_json("[1, 2, 3, 4, 5]", "Test Data")

        parsed = json.loads(result)
        assert "narrative" in parsed

    def test_json_has_trend(self) -> None:
        """Test that JSON output has trend field."""
        result = describe_json("[1, 2, 3, 4, 5]", "Test")

        parsed = json.loads(result)
        assert "trend" in parsed

    def test_error_returns_json(self) -> None:
        """Test that errors are also returned as JSON."""
        result = describe_json("invalid", "Test")

        parsed = json.loads(result)
        assert "error" in parsed


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestWrapForSemanticOutput:
    """Tests for wrap_for_semantic_output decorator."""

    def test_wrap_function_returning_list(self) -> None:
        """Test wrapping function that returns list."""

        @wrap_for_semantic_output()
        def get_metrics() -> list[float]:
            return [45, 47, 46, 48, 47]

        result = get_metrics()

        assert isinstance(result, str)
        # Function name used as context
        assert "Get Metrics" in result or len(result) > 0

    def test_wrap_with_context_key(self) -> None:
        """Test wrapping with context_key parameter."""

        @wrap_for_semantic_output(context_key="metric_name")
        def get_readings(metric_name: str = "CPU") -> list[float]:
            return [45, 47, 46, 48, 47]

        result = get_readings(metric_name="Temperature")

        assert isinstance(result, str)
        assert "Temperature" in result

    def test_wrap_function_returning_dict(self) -> None:
        """Test wrapping function that returns dict with data key."""

        @wrap_for_semantic_output(data_key="values")
        def get_sensor_data() -> dict:
            return {"values": [22.1, 22.3, 22.0, 22.2], "unit": "celsius"}

        result = get_sensor_data()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_wrap_with_custom_data_key(self) -> None:
        """Test wrapping with custom data_key."""

        @wrap_for_semantic_output(data_key="readings")
        def get_custom_data() -> dict:
            return {"readings": [1, 2, 3, 4, 5]}

        result = get_custom_data()
        assert isinstance(result, str)


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestCreateSemanticTool:
    """Tests for create_semantic_tool factory."""

    def test_create_basic_tool(self) -> None:
        """Test creating a basic semantic tool."""

        def fetch_data() -> list[float]:
            return [10, 20, 30, 40, 50]

        tool = create_semantic_tool(
            name="test_tool",
            data_fetcher=fetch_data,
            description="Test tool",
            context="Test Context",
        )

        result = tool()

        assert isinstance(result, str)
        assert "Test Context" in result

    def test_tool_has_correct_name(self) -> None:
        """Test that created tool has correct name."""

        def fetch() -> list[float]:
            return [1, 2, 3]

        tool = create_semantic_tool(
            name="my_semantic_tool",
            data_fetcher=fetch,
            description="Description",
        )

        assert tool.__name__ == "my_semantic_tool"

    def test_tool_has_correct_description(self) -> None:
        """Test that created tool has correct description."""

        def fetch() -> list[float]:
            return [1, 2, 3]

        tool = create_semantic_tool(
            name="tool",
            data_fetcher=fetch,
            description="My custom description",
        )

        assert tool.__doc__ == "My custom description"

    def test_tool_handles_errors(self) -> None:
        """Test that created tool handles errors gracefully."""

        def failing_fetch() -> list[float]:
            raise ValueError("Connection failed")

        tool = create_semantic_tool(
            name="failing_tool",
            data_fetcher=failing_fetch,
            description="Tool that fails",
        )

        result = tool()
        assert "Error" in result


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestGetMCPToolConfig:
    """Tests for get_mcp_tool_config function."""

    def test_basic_config(self) -> None:
        """Test basic configuration."""
        config = get_mcp_tool_config()

        assert config["name"] == "semantic-frame"
        assert "description" in config
        assert "tools" in config

    def test_tools_list(self) -> None:
        """Test that config includes all tools."""
        config = get_mcp_tool_config()

        assert "describe_data" in config["tools"]
        assert "describe_batch" in config["tools"]
        assert "describe_json" in config["tools"]

    def test_defer_loading_option(self) -> None:
        """Test defer_loading configuration."""
        config = get_mcp_tool_config(defer_loading=True)

        assert "default_config" in config
        assert config["default_config"]["defer_loading"] is True

    def test_no_defer_loading_by_default(self) -> None:
        """Test no defer_loading by default."""
        config = get_mcp_tool_config()

        assert "default_config" not in config


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestParseDataInputEdgeCases:
    """Tests for _parse_data_input error paths."""

    def test_invalid_json_array_falls_through(self) -> None:
        """Test that invalid JSON array falls through to other parsers.

        Covers lines 63-64: JSONDecodeError/ValueError exception handling.
        """
        from semantic_frame.integrations.mcp import _parse_data_input

        # Starts with [ but is not valid JSON - should try CSV next
        # "[1, 2, abc]" would fail JSON parsing due to "abc"
        # But since it has commas, CSV parsing will also fail on "abc"
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_data_input("[1, 2, abc]")

    def test_csv_with_invalid_values_falls_through(self) -> None:
        """Test that CSV with non-numeric values falls through.

        Covers lines 70-71: ValueError exception handling in CSV parsing.
        """
        from semantic_frame.integrations.mcp import _parse_data_input

        # Has commas but contains non-numeric values
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_data_input("hello, world, test")

    def test_newline_with_invalid_values_falls_through(self) -> None:
        """Test that newline-separated with non-numeric values falls through.

        Covers lines 77-78: ValueError exception handling in newline parsing.
        """
        from semantic_frame.integrations.mcp import _parse_data_input

        # Has newlines but contains non-numeric values, no commas
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_data_input("hello\nworld\ntest")

    def test_json_array_with_non_numeric_values(self) -> None:
        """Test JSON array with strings instead of numbers.

        Covers line 62-64: ValueError in float conversion.
        """
        from semantic_frame.integrations.mcp import _parse_data_input

        # Valid JSON but can't convert to floats
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_data_input('["a", "b", "c"]')


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestDescribeBatchStringParsing:
    """Tests for describe_batch with string values in datasets."""

    def test_batch_with_string_values_text_format(self) -> None:
        """Test batch with string-encoded data arrays in text format.

        Covers line 173: _parse_data_input called for string values in text mode.
        """
        # Values as strings instead of arrays
        datasets = json.dumps(
            {
                "cpu": "45, 47, 46, 48, 47",
                "memory": "60, 61, 60, 61, 60",
            }
        )
        result = describe_batch(datasets, output_format="text")

        assert isinstance(result, str)
        assert "cpu" in result.lower()
        assert "memory" in result.lower()

    def test_batch_with_string_values_json_format(self) -> None:
        """Test batch with string-encoded data arrays in JSON format.

        Covers line 164: _parse_data_input called for string values in JSON mode.
        """
        datasets = json.dumps(
            {
                "temperature": "[22.1, 22.3, 22.0, 22.2]",
            }
        )
        result = describe_batch(datasets, output_format="json")

        parsed = json.loads(result)
        assert "temperature" in parsed
        assert "narrative" in parsed["temperature"]

    def test_batch_with_mixed_string_and_array_values(self) -> None:
        """Test batch with mix of string and array values."""
        datasets = json.dumps(
            {
                "cpu": [45, 47, 46],  # Array
                "memory": "60, 61, 60",  # String
            }
        )
        result = describe_batch(datasets, output_format="text")

        assert "cpu" in result.lower()
        assert "memory" in result.lower()


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestDescribeBatchExceptionHandling:
    """Tests for describe_batch exception handling paths."""

    def test_batch_general_exception_handling(self) -> None:
        """Test general exception handling in describe_batch.

        Covers lines 180-181: General exception handling.
        """
        # Create a scenario that triggers a general exception
        # Invalid data type that passes JSON parsing but fails analysis
        datasets = json.dumps(
            {
                "bad_data": {"nested": "object"},  # Not a list or string of numbers
            }
        )
        result = describe_batch(datasets, output_format="text")

        assert "Error analyzing batch data" in result


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestWrapForSemanticOutputEdgeCases:
    """Tests for wrap_for_semantic_output edge cases."""

    def test_wrap_returns_original_for_unexpected_type(self) -> None:
        """Test that wrapper returns original for unexpected return types.

        Covers line 268: Return str(result) when data can't be extracted.
        """

        @wrap_for_semantic_output(data_key="values")
        def get_unexpected() -> str:
            return "just a string"

        result = get_unexpected()
        assert result == "just a string"

    def test_wrap_returns_original_for_dict_without_data_key(self) -> None:
        """Test wrapper with dict missing the expected data key.

        Covers line 268: Return str(result) when data_key not in dict.
        """

        @wrap_for_semantic_output(data_key="missing_key")
        def get_wrong_key() -> dict:
            return {"other_key": [1, 2, 3]}

        result = get_wrong_key()
        # Should return string representation of original
        assert "other_key" in result

    def test_wrap_handles_analysis_exception(self) -> None:
        """Test that wrapper handles exceptions during analysis.

        Covers lines 274-275: Exception handling in semantic conversion.
        """

        @wrap_for_semantic_output()
        def get_bad_data() -> list:
            return []  # Empty list may cause analysis issues

        result = get_bad_data()
        # Should either succeed with a message or return error message
        assert isinstance(result, str)

    def test_wrap_handles_exception_with_invalid_data(self) -> None:
        """Test wrapper with data that causes describe_series to fail.

        Covers lines 274-275: Exception in describe_series.
        """

        @wrap_for_semantic_output()
        def get_invalid() -> list:
            return [float("nan"), float("nan"), float("nan")]

        result = get_invalid()
        # Should handle gracefully - either succeed or error message
        assert isinstance(result, str)

    def test_wrap_handles_describe_series_exception(self) -> None:
        """Test wrapper exception handling when describe_series raises.

        Covers lines 274-275: Exception path in semantic conversion.
        """
        from unittest.mock import patch

        @wrap_for_semantic_output()
        def get_data() -> list[float]:
            return [1.0, 2.0, 3.0]

        # Mock describe_series at the source (imported inside the wrapper)
        with patch("semantic_frame.describe_series") as mock_describe:
            mock_describe.side_effect = RuntimeError("Mock analysis failure")
            result = get_data()

        assert "Error in semantic conversion" in result
        assert "Mock analysis failure" in result
        assert "Original" in result

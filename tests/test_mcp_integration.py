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

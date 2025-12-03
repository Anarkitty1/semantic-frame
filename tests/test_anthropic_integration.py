"""Tests for Anthropic native tool integration.

These tests validate the Anthropic tool wrapper functionality.
Tests that require the anthropic SDK are skipped if not available.
"""

import json

import pytest

from semantic_frame.integrations.anthropic import (
    ANTHROPIC_TOOL_SCHEMA,
    AnthropicSemanticTool,
    _parse_data_input,
    create_tool_result,
    get_anthropic_tool,
    handle_tool_call,
)


class TestParseDataInput:
    """Tests for data parsing utility."""

    def test_parse_list_of_floats(self) -> None:
        """Should handle list of floats directly."""
        result = _parse_data_input([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_list_of_ints(self) -> None:
        """Should convert ints to floats."""
        result = _parse_data_input([1, 2, 3, 4, 5])
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_json_array(self) -> None:
        """Should parse JSON array format."""
        result = _parse_data_input("[1, 2, 3, 4, 5]")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_csv(self) -> None:
        """Should parse CSV format."""
        result = _parse_data_input("1, 2, 3, 4, 5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_newline_separated(self) -> None:
        """Should parse newline-separated format."""
        result = _parse_data_input("1\n2\n3\n4\n5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_with_whitespace(self) -> None:
        """Should handle whitespace."""
        result = _parse_data_input("  [1, 2, 3]  ")
        assert result == [1.0, 2.0, 3.0]

    def test_parse_floats(self) -> None:
        """Should parse floating point numbers."""
        result = _parse_data_input([1.5, 2.7, 3.14])
        assert result == [1.5, 2.7, 3.14]

    def test_invalid_input_raises_error(self) -> None:
        """Should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            _parse_data_input("not valid data")


class TestAnthropicToolSchema:
    """Tests for the Anthropic tool schema."""

    def test_schema_has_required_fields(self) -> None:
        """Should have name, description, and input_schema."""
        assert "name" in ANTHROPIC_TOOL_SCHEMA
        assert "description" in ANTHROPIC_TOOL_SCHEMA
        assert "input_schema" in ANTHROPIC_TOOL_SCHEMA

    def test_schema_name(self) -> None:
        """Should have semantic_analysis as name."""
        assert ANTHROPIC_TOOL_SCHEMA["name"] == "semantic_analysis"

    def test_schema_description(self) -> None:
        """Should have meaningful description."""
        desc = ANTHROPIC_TOOL_SCHEMA["description"]
        assert len(desc) > 0
        assert "analyze" in desc.lower() or "analysis" in desc.lower()

    def test_input_schema_structure(self) -> None:
        """Should have proper JSON schema structure."""
        input_schema = ANTHROPIC_TOOL_SCHEMA["input_schema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema

    def test_data_property_is_required(self) -> None:
        """Data should be a required property."""
        assert "data" in ANTHROPIC_TOOL_SCHEMA["input_schema"]["required"]

    def test_data_property_schema(self) -> None:
        """Data property should be array of numbers."""
        data_prop = ANTHROPIC_TOOL_SCHEMA["input_schema"]["properties"]["data"]
        assert data_prop["type"] == "array"
        assert data_prop["items"]["type"] == "number"

    def test_optional_properties(self) -> None:
        """Should have optional context and output_format."""
        props = ANTHROPIC_TOOL_SCHEMA["input_schema"]["properties"]
        assert "context" in props
        assert "output_format" in props


class TestGetAnthropicTool:
    """Tests for get_anthropic_tool function."""

    def test_returns_dict(self) -> None:
        """Should return a dictionary."""
        tool = get_anthropic_tool()
        assert isinstance(tool, dict)

    def test_returns_copy(self) -> None:
        """Should return a copy, not the original."""
        tool1 = get_anthropic_tool()
        tool2 = get_anthropic_tool()
        tool1["name"] = "modified"
        assert tool2["name"] == "semantic_analysis"

    def test_has_all_required_fields(self) -> None:
        """Should have all fields for Anthropic API."""
        tool = get_anthropic_tool()
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool


class TestHandleToolCall:
    """Tests for handle_tool_call function."""

    def test_basic_analysis(self) -> None:
        """Should analyze basic data array."""
        tool_input = {"data": [1, 2, 3, 4, 5]}
        result = handle_tool_call(tool_input)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_context(self) -> None:
        """Should include context in analysis."""
        tool_input = {"data": [100, 102, 99, 101, 98], "context": "Temperature"}
        result = handle_tool_call(tool_input)

        assert isinstance(result, str)
        assert "Temperature" in result

    def test_default_context(self) -> None:
        """Should use default context when not in input."""
        tool_input = {"data": [1, 2, 3, 4, 5]}
        result = handle_tool_call(tool_input, default_context="Default Label")

        assert isinstance(result, str)
        assert "Default Label" in result

    def test_input_context_overrides_default(self) -> None:
        """Input context should override default."""
        tool_input = {"data": [1, 2, 3, 4, 5], "context": "Input Context"}
        result = handle_tool_call(tool_input, default_context="Default Label")

        assert "Input Context" in result
        assert "Default Label" not in result

    def test_json_output_format(self) -> None:
        """Should return JSON when requested."""
        tool_input = {"data": [1, 2, 3, 4, 5], "output_format": "json"}
        result = handle_tool_call(tool_input)

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_text_output_format(self) -> None:
        """Should return text narrative by default."""
        tool_input = {"data": [1, 2, 3, 100, 4, 5]}
        result = handle_tool_call(tool_input)

        # Should be a narrative string, not JSON
        assert isinstance(result, str)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_with_anomaly_data(self) -> None:
        """Should detect anomalies in data."""
        tool_input = {"data": [10, 11, 10, 9, 100, 10, 11]}
        result = handle_tool_call(tool_input)

        assert isinstance(result, str)
        # Should mention anomaly or outlier
        assert (
            "anomal" in result.lower() or "outlier" in result.lower() or "spike" in result.lower()
        )

    def test_empty_data(self) -> None:
        """Should handle empty data array."""
        tool_input = {"data": []}
        result = handle_tool_call(tool_input)

        assert isinstance(result, str)


class TestCreateToolResult:
    """Tests for create_tool_result function."""

    def test_creates_tool_result_dict(self) -> None:
        """Should create proper tool_result structure."""
        result = create_tool_result("tool_123", "Analysis result text")

        assert isinstance(result, dict)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_123"
        assert result["content"] == "Analysis result text"

    def test_preserves_tool_use_id(self) -> None:
        """Should preserve the exact tool_use_id."""
        result = create_tool_result("toolu_abc123xyz", "Result")
        assert result["tool_use_id"] == "toolu_abc123xyz"

    def test_handles_json_content(self) -> None:
        """Should handle JSON string content."""
        json_content = '{"trend": "rising", "volatility": "low"}'
        result = create_tool_result("tool_1", json_content)
        assert result["content"] == json_content


class TestAnthropicSemanticTool:
    """Tests for AnthropicSemanticTool helper class."""

    def test_initialization_without_context(self) -> None:
        """Should initialize without context."""
        tool = AnthropicSemanticTool()
        assert tool.context is None

    def test_initialization_with_context(self) -> None:
        """Should store context."""
        tool = AnthropicSemanticTool(context="Sensor Data")
        assert tool.context == "Sensor Data"

    def test_get_tool(self) -> None:
        """Should return tool definition."""
        tool = AnthropicSemanticTool()
        definition = tool.get_tool()

        assert isinstance(definition, dict)
        assert definition["name"] == "semantic_analysis"

    def test_handle(self) -> None:
        """Should handle tool input."""
        tool = AnthropicSemanticTool(context="Test Context")
        result = tool.handle({"data": [1, 2, 3, 4, 5]})

        assert isinstance(result, str)
        assert "Test Context" in result

    def test_handle_with_input_context(self) -> None:
        """Input context should override default."""
        tool = AnthropicSemanticTool(context="Default")
        result = tool.handle({"data": [1, 2, 3], "context": "Override"})

        assert "Override" in result

    def test_create_result(self) -> None:
        """Should create tool result."""
        tool = AnthropicSemanticTool()
        result = tool.create_result("tool_abc", "Analysis text")

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_abc"
        assert result["content"] == "Analysis text"


class TestAnthropicSDKIntegration:
    """Tests requiring anthropic SDK to be installed."""

    @pytest.fixture
    def anthropic_available(self) -> bool:
        """Check if anthropic is available."""
        try:
            import anthropic  # noqa: F401

            return True
        except ImportError:
            return False

    def test_anthropic_import_check(self) -> None:
        """Verify import checking works."""
        from semantic_frame.integrations.anthropic import _check_anthropic

        # Should return bool
        result = _check_anthropic()
        assert isinstance(result, bool)

    def test_tool_schema_matches_anthropic_format(self) -> None:
        """Tool schema should be compatible with Anthropic API format."""
        tool = get_anthropic_tool()

        # Required top-level fields
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

        # input_schema must be valid JSON Schema
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

        # Name must be valid identifier
        assert tool["name"].replace("_", "").isalnum()

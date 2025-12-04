"""Tests for Anthropic native tool integration.

These tests validate the Anthropic tool wrapper functionality,
including Advanced Tool Use features (Tool Search, Programmatic Calling, Examples).

Tests that require the anthropic SDK are skipped if not available.
"""

import json

import pytest

from semantic_frame.integrations.anthropic import (
    ANTHROPIC_TOOL_SCHEMA,
    TOOL_USE_EXAMPLES,
    AnthropicSemanticTool,
    _parse_data_input,
    create_tool_result,
    get_advanced_tool,
    get_anthropic_tool,
    get_tool_for_batch_processing,
    get_tool_for_discovery,
    handle_batch_tool_calls,
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


class TestToolUseExamples:
    """Tests for the input_examples feature."""

    def test_examples_exist(self) -> None:
        """Should have examples defined."""
        assert len(TOOL_USE_EXAMPLES) > 0

    def test_examples_have_required_structure(self) -> None:
        """Each example should have input and expected_output."""
        for example in TOOL_USE_EXAMPLES:
            assert "input" in example
            assert "expected_output" in example

    def test_examples_have_valid_inputs(self) -> None:
        """Example inputs should have data array."""
        for example in TOOL_USE_EXAMPLES:
            input_data = example["input"]
            assert "data" in input_data
            assert isinstance(input_data["data"], list)
            assert all(isinstance(x, int | float) for x in input_data["data"])

    def test_examples_cover_different_scenarios(self) -> None:
        """Examples should cover various use cases."""
        contexts = [ex.get("input", {}).get("context") for ex in TOOL_USE_EXAMPLES]
        # Should have some with context and some without
        assert any(c is not None for c in contexts)
        assert any(c is None for c in contexts)

    def test_example_with_json_output(self) -> None:
        """Should have at least one JSON output example."""
        has_json = any(
            ex.get("input", {}).get("output_format") == "json" for ex in TOOL_USE_EXAMPLES
        )
        assert has_json


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

    def test_excludes_examples_by_default(self) -> None:
        """Should exclude input_examples by default for standard API compatibility."""
        tool = get_anthropic_tool()
        assert "input_examples" not in tool

    def test_include_examples_when_requested(self) -> None:
        """Should include examples when include_examples=True (for beta API)."""
        tool = get_anthropic_tool(include_examples=True)
        assert "input_examples" in tool
        assert len(tool["input_examples"]) > 0

    def test_defer_loading_option(self) -> None:
        """Should add defer_loading when requested."""
        tool = get_anthropic_tool(defer_loading=True)
        assert tool.get("defer_loading") is True

    def test_no_defer_loading_by_default(self) -> None:
        """Should not have defer_loading by default."""
        tool = get_anthropic_tool()
        assert "defer_loading" not in tool or tool.get("defer_loading") is False

    def test_allowed_callers_option(self) -> None:
        """Should add allowed_callers when requested."""
        tool = get_anthropic_tool(allowed_callers=["code_execution"])
        assert tool.get("allowed_callers") == ["code_execution"]

    def test_no_allowed_callers_by_default(self) -> None:
        """Should not have allowed_callers by default."""
        tool = get_anthropic_tool()
        assert "allowed_callers" not in tool


class TestAdvancedToolGetters:
    """Tests for convenience functions for advanced tool use."""

    def test_get_tool_for_discovery(self) -> None:
        """Should return tool with defer_loading=True."""
        tool = get_tool_for_discovery()
        assert tool.get("defer_loading") is True
        assert "input_examples" in tool

    def test_get_tool_for_batch_processing(self) -> None:
        """Should return tool with code_execution caller."""
        tool = get_tool_for_batch_processing()
        assert tool.get("allowed_callers") == ["code_execution"]
        assert "input_examples" in tool

    def test_get_advanced_tool(self) -> None:
        """Should return tool with all advanced features."""
        tool = get_advanced_tool()
        assert tool.get("defer_loading") is True
        assert tool.get("allowed_callers") == ["code_execution"]
        assert "input_examples" in tool


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


class TestHandleBatchToolCalls:
    """Tests for handle_batch_tool_calls function."""

    def test_batch_single_input(self) -> None:
        """Should handle single input in batch."""
        inputs = [{"data": [1, 2, 3, 4, 5], "context": "Series A"}]
        results = handle_batch_tool_calls(inputs)

        assert len(results) == 1
        assert "Series A" in results[0]

    def test_batch_multiple_inputs(self) -> None:
        """Should handle multiple inputs in batch."""
        inputs = [
            {"data": [1, 2, 3, 4, 5], "context": "Series A"},
            {"data": [10, 20, 30, 40, 50], "context": "Series B"},
            {"data": [100, 99, 100, 101, 100], "context": "Series C"},
        ]
        results = handle_batch_tool_calls(inputs)

        assert len(results) == 3
        assert "Series A" in results[0]
        assert "Series B" in results[1]
        assert "Series C" in results[2]

    def test_batch_with_default_context(self) -> None:
        """Should apply default context to all inputs."""
        inputs = [
            {"data": [1, 2, 3]},
            {"data": [4, 5, 6]},
        ]
        results = handle_batch_tool_calls(inputs, default_context="Batch Context")

        assert "Batch Context" in results[0]
        assert "Batch Context" in results[1]

    def test_batch_empty_list(self) -> None:
        """Should handle empty input list."""
        results = handle_batch_tool_calls([])
        assert results == []


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

    def test_initialization_with_advanced_options(self) -> None:
        """Should store advanced tool use options."""
        tool = AnthropicSemanticTool(
            context="Test",
            defer_loading=True,
            allowed_callers=["code_execution"],
            include_examples=False,
        )
        assert tool.defer_loading is True
        assert tool.allowed_callers == ["code_execution"]
        assert tool.include_examples is False

    def test_get_tool(self) -> None:
        """Should return tool definition."""
        tool = AnthropicSemanticTool()
        definition = tool.get_tool()

        assert isinstance(definition, dict)
        assert definition["name"] == "semantic_analysis"

    def test_get_tool_with_advanced_options(self) -> None:
        """Should return tool with advanced options."""
        tool = AnthropicSemanticTool(
            defer_loading=True,
            allowed_callers=["code_execution"],
        )
        definition = tool.get_tool()

        assert definition.get("defer_loading") is True
        assert definition.get("allowed_callers") == ["code_execution"]

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

    def test_handle_batch(self) -> None:
        """Should handle batch tool calls."""
        tool = AnthropicSemanticTool(context="Batch Default")
        inputs = [
            {"data": [1, 2, 3]},
            {"data": [4, 5, 6]},
        ]
        results = tool.handle_batch(inputs)

        assert len(results) == 2
        assert all("Batch Default" in r for r in results)

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

        # input_examples should NOT be present by default (requires beta API)
        assert "input_examples" not in tool

        # input_schema must be valid JSON Schema
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

        # Name must be valid identifier
        assert tool["name"].replace("_", "").isalnum()

    def test_advanced_tool_schema_format(self) -> None:
        """Advanced tool schema should have all beta fields."""
        tool = get_advanced_tool()

        # Standard fields
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

        # Advanced Tool Use fields
        assert "defer_loading" in tool
        assert "allowed_callers" in tool
        assert "input_examples" in tool

        # Values
        assert tool["defer_loading"] is True
        assert "code_execution" in tool["allowed_callers"]
        assert len(tool["input_examples"]) >= 3

"""Tests for LangChain integration.

These tests validate the LangChain tool wrapper functionality.
Tests that require langchain to be installed are skipped if not available.
"""

import pytest

from semantic_frame.integrations.langchain import (
    SemanticAnalysisTool,
    _parse_data_input,
)


class TestParseDataInput:
    """Tests for data parsing utility."""

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
        result = _parse_data_input("[1.5, 2.7, 3.14]")
        assert result == [1.5, 2.7, 3.14]

    def test_invalid_input_raises_error(self) -> None:
        """Should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            _parse_data_input("not valid data")


class TestParseDataInputEdgeCases:
    """Tests for exception paths in _parse_data_input.

    Covers lines 64-65, 71-72, 78-79: ValueError handling in each parser.
    """

    def test_json_with_non_numeric_values(self) -> None:
        """JSON array with non-numeric strings should fail.

        Tests lines 64-65: JSONDecodeError when parsing non-numeric JSON.
        """
        with pytest.raises(ValueError):
            _parse_data_input('["a", "b", "c"]')

    def test_csv_with_non_numeric_values(self) -> None:
        """CSV with letters should fail.

        Tests lines 71-72: CSV ValueError handling.
        """
        with pytest.raises(ValueError):
            _parse_data_input("x, y, z")

    def test_newline_with_non_numeric_values(self) -> None:
        """Newline-separated with non-numeric should fail.

        Tests lines 78-79: Newline ValueError handling.
        """
        with pytest.raises(ValueError):
            _parse_data_input("one\ntwo\nthree")

    def test_empty_string_raises_error(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            _parse_data_input("")

    def test_all_parsers_fail(self) -> None:
        """Should raise ValueError when all parsers fail."""
        with pytest.raises(ValueError) as excinfo:
            _parse_data_input("abc def ghi")
        assert "parse" in str(excinfo.value).lower()


class TestSemanticAnalysisTool:
    """Tests for SemanticAnalysisTool class."""

    def test_tool_attributes(self) -> None:
        """Should have required tool attributes."""
        tool = SemanticAnalysisTool()
        assert tool.name == "semantic_analysis"
        assert len(tool.description) > 0
        assert "analyze" in tool.description.lower()

    def test_context_initialization(self) -> None:
        """Should accept context parameter."""
        tool = SemanticAnalysisTool(context="Test Data")
        assert tool.context == "Test Data"

    def test_run_with_json_input(self) -> None:
        """Should analyze JSON array input."""
        tool = SemanticAnalysisTool(context="Test")
        result = tool._run("[1, 2, 3, 4, 100, 5, 6]")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test" in result

    def test_run_with_csv_input(self) -> None:
        """Should analyze CSV input."""
        tool = SemanticAnalysisTool()
        result = tool._run("10, 20, 30, 40, 50")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_without_context(self) -> None:
        """Should work without context."""
        tool = SemanticAnalysisTool()
        result = tool._run("[1, 2, 3, 4, 5]")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_async_run(self) -> None:
        """Async run should delegate to sync run."""
        import asyncio

        tool = SemanticAnalysisTool()
        result = asyncio.get_event_loop().run_until_complete(tool._arun("[1, 2, 3]"))

        assert isinstance(result, str)


class TestLangChainIntegration:
    """Tests requiring langchain to be installed."""

    @pytest.fixture
    def langchain_available(self) -> bool:
        """Check if langchain is available."""
        try:
            from langchain.tools import BaseTool  # noqa: F401

            return True
        except ImportError:
            return False

    def test_as_langchain_tool_without_langchain(self) -> None:
        """Should raise ImportError if langchain not installed."""
        # This test runs regardless of langchain availability
        # to verify error handling
        tool = SemanticAnalysisTool()

        try:
            from langchain.tools import BaseTool  # noqa: F401

            # langchain is available, tool should work
            lc_tool = tool.as_langchain_tool()
            assert lc_tool is not None
        except ImportError:
            # langchain not available, should raise ImportError
            with pytest.raises(ImportError) as excinfo:
                tool.as_langchain_tool()
            assert "langchain" in str(excinfo.value).lower()

    def test_get_semantic_tool_factory(self) -> None:
        """Test factory function."""
        try:
            from langchain.tools import BaseTool

            from semantic_frame.integrations.langchain import get_semantic_tool

            tool = get_semantic_tool(context="Factory Test")
            assert isinstance(tool, BaseTool)
        except ImportError:
            pytest.skip("langchain not installed")

    def test_langchain_tool_execution(self) -> None:
        """Test tool execution when langchain is available."""
        try:
            from semantic_frame.integrations.langchain import get_semantic_tool

            tool = get_semantic_tool()
            result = tool.run("[10, 20, 30, 40, 50]")
            assert isinstance(result, str)
            assert len(result) > 0
        except ImportError:
            pytest.skip("langchain not installed")

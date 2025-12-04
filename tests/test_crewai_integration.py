"""Tests for CrewAI integration.

These tests validate the CrewAI tool wrapper functionality.
Tests that require crewai to be installed are skipped if not available.
"""

import pytest

from semantic_frame.integrations.crewai import (
    _parse_data_input,
    semantic_analysis,
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

    def test_invalid_input_raises_error(self) -> None:
        """Should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            _parse_data_input("invalid data")


class TestSemanticAnalysisFunction:
    """Tests for the standalone semantic_analysis function."""

    def test_basic_analysis(self) -> None:
        """Should analyze data and return narrative."""
        result = semantic_analysis("[1, 2, 3, 4, 5]", "Test Data")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Data" in result

    def test_with_default_context(self) -> None:
        """Should use default context when not provided."""
        result = semantic_analysis("[10, 20, 30, 40, 50]")

        assert isinstance(result, str)
        assert "Data" in result

    def test_detects_anomaly(self) -> None:
        """Should mention anomaly in narrative."""
        result = semantic_analysis("[1, 2, 3, 100, 4, 5]", "Sensor")

        assert isinstance(result, str)
        # Should detect the outlier
        assert "anomal" in result.lower() or "outlier" in result.lower()

    def test_csv_input(self) -> None:
        """Should handle CSV input."""
        result = semantic_analysis("10, 20, 30, 40, 50", "Metrics")

        assert isinstance(result, str)
        assert "Metrics" in result


class TestParseDataInputEdgeCases:
    """Tests for exception paths in _parse_data_input.

    Covers lines 68-69, 75-76, 82-83: ValueError handling in each parser.
    """

    def test_json_decode_error_then_csv_parse(self) -> None:
        """Malformed JSON should fall back to CSV parsing.

        Tests lines 68-69: JSONDecodeError handling.
        """
        # This looks like JSON but isn't valid - should try CSV
        result = _parse_data_input("1, 2, 3")
        assert result == [1.0, 2.0, 3.0]

    def test_all_parsers_fail_raises_error(self) -> None:
        """Should raise ValueError when all parsers fail.

        Tests the final raise at line 128.
        """
        with pytest.raises(ValueError) as excinfo:
            _parse_data_input("abc, def, ghi")
        assert "parse" in str(excinfo.value).lower()

    def test_json_with_non_numeric_values(self) -> None:
        """JSON array with non-numeric strings should fail all parsers.

        Tests lines 75-76, 82-83: ValueError from float conversion.
        """
        with pytest.raises(ValueError):
            _parse_data_input('["a", "b", "c"]')

    def test_csv_with_non_numeric_values(self) -> None:
        """CSV with letters should fail parsing.

        Tests lines 75-76: CSV ValueError handling.
        """
        with pytest.raises(ValueError):
            _parse_data_input("x, y, z")

    def test_newline_with_non_numeric_values(self) -> None:
        """Newline-separated with non-numeric should fail.

        Tests lines 82-83: Newline ValueError handling.
        """
        with pytest.raises(ValueError):
            _parse_data_input("one\ntwo\nthree")

    def test_empty_string_raises_error(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            _parse_data_input("")

    def test_whitespace_only_raises_error(self) -> None:
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError):
            _parse_data_input("   \n\t  ")


class TestCrewAIIntegration:
    """Tests requiring crewai to be installed."""

    def test_get_crewai_tool_without_crewai(self) -> None:
        """Should raise ImportError if crewai not installed."""
        from semantic_frame.integrations.crewai import get_crewai_tool

        try:
            # crewai >= 1.0 moved tool to crewai.tools
            try:
                from crewai.tools import tool  # noqa: F401
            except ImportError:
                from crewai import tool  # noqa: F401

            # crewai is available, tool should work
            crewai_tool = get_crewai_tool()
            assert crewai_tool is not None
        except ImportError:
            # crewai not available, should raise ImportError
            with pytest.raises(ImportError) as excinfo:
                get_crewai_tool()
            assert "crewai" in str(excinfo.value).lower()

    def test_crewai_tool_execution(self) -> None:
        """Test tool execution when crewai is available."""
        try:
            from semantic_frame.integrations.crewai import get_crewai_tool

            tool = get_crewai_tool()
            # crewai Tool objects use .run() method, not direct call
            result = tool.run(data="[10, 20, 30, 40, 50]", context="Test")
            assert isinstance(result, str)
            assert len(result) > 0
        except ImportError:
            pytest.skip("crewai not installed")

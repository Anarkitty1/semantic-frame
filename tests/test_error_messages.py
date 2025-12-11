"""Tests for error message quality and helpfulness.

These tests verify that error messages:
1. Explain what went wrong
2. Include the actual problematic value/type
3. Suggest how to fix the issue
4. Reference valid alternatives
"""

import numpy as np
import pytest


class TestToNumpyErrorMessages:
    """Tests for _to_numpy conversion error messages."""

    def test_unsupported_type_shows_type_name(self):
        """Error should include the actual type name provided."""
        from semantic_frame.main import _to_numpy

        with pytest.raises(TypeError) as exc_info:
            _to_numpy({"not": "supported"})

        error_msg = str(exc_info.value)
        assert "dict" in error_msg.lower(), "Error should mention 'dict'"

    def test_unsupported_type_lists_valid_alternatives(self):
        """Error should mention valid input types."""
        from semantic_frame.main import _to_numpy

        with pytest.raises(TypeError) as exc_info:
            _to_numpy({"not": "supported"})

        error_msg = str(exc_info.value)
        # Should mention at least some valid types
        assert any(
            valid_type in error_msg
            for valid_type in ["pandas.Series", "numpy.ndarray", "polars.Series", "list"]
        ), "Error should mention valid input types"

    def test_non_numeric_list_error_is_clear(self):
        """Error for non-numeric list should be clear about the problem."""
        from semantic_frame.main import _to_numpy

        with pytest.raises(TypeError) as exc_info:
            _to_numpy(["a", "b", "c"])

        error_msg = str(exc_info.value)
        assert "non-numeric" in error_msg.lower() or "list" in error_msg.lower()
        assert "numbers" in error_msg.lower() or "numeric" in error_msg.lower()

    def test_non_numeric_numpy_array_error(self):
        """Error for non-numeric numpy array should explain the issue."""
        from semantic_frame.main import _to_numpy

        with pytest.raises(TypeError) as exc_info:
            _to_numpy(np.array(["a", "b", "c"]))

        error_msg = str(exc_info.value)
        assert "numeric" in error_msg.lower()


class TestDescribeSeriesErrorMessages:
    """Tests for describe_series error messages."""

    def test_invalid_output_format_error(self):
        """Error for invalid output format should list valid options."""
        from semantic_frame import describe_series

        with pytest.raises(ValueError) as exc_info:
            describe_series([1, 2, 3], output="invalid_format")  # type: ignore

        error_msg = str(exc_info.value)
        assert "invalid_format" in error_msg or "output" in error_msg.lower()
        # Should mention at least some valid formats
        assert any(
            fmt in error_msg for fmt in ["text", "json", "full"]
        ), "Error should mention valid output formats"

    def test_empty_data_returns_helpful_message(self):
        """Empty data should return a helpful message (not raise error)."""
        from semantic_frame import describe_series

        # describe_series handles empty data gracefully with a message
        result = describe_series([])
        assert "no valid data" in result.lower() or "empty" in result.lower()


class TestBenchmarkConfigErrorMessages:
    """Tests for benchmark configuration error messages."""

    def test_api_key_missing_error_mentions_env_var(self):
        """Missing API key error should mention ANTHROPIC_API_KEY."""
        from benchmarks.claude_client import ClaudeClient
        from benchmarks.config import BenchmarkConfig

        try:
            from anthropic import Anthropic  # noqa: F401
        except ImportError:
            pytest.skip("anthropic not installed")

        config = BenchmarkConfig(api_key=None)

        with pytest.raises(ValueError) as exc_info:
            ClaudeClient(config)

        error_msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in error_msg, "Error should mention environment variable name"


class TestMCPErrorMessages:
    """Tests for MCP integration error messages."""

    def test_parse_data_input_error_shows_preview(self):
        """Parse error should show a preview of the problematic input."""
        try:
            from semantic_frame.integrations.mcp import _parse_data_input
        except ImportError:
            pytest.skip("mcp not installed")

        with pytest.raises(ValueError) as exc_info:
            _parse_data_input("not valid data format at all")

        error_msg = str(exc_info.value)
        assert "not valid" in error_msg, "Error should show preview of input"
        assert any(
            fmt in error_msg.lower() for fmt in ["json", "csv", "newline"]
        ), "Error should mention valid formats"


class TestAnalyzerErrorMessages:
    """Tests for analyzer function error messages."""

    def test_negative_z_threshold_error_is_helpful(self):
        """Error for negative z_threshold should explain the constraint."""
        from semantic_frame.core.analyzers import detect_anomalies

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError) as exc_info:
            detect_anomalies(values, z_threshold=-1.0)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()
        assert "z_threshold" in error_msg.lower()


class TestIntegrationImportErrorMessages:
    """Tests for optional dependency import error messages."""

    def test_langchain_import_error_mentions_pip_install(self):
        """LangChain import error should mention installation command."""
        # Skip if langchain is actually installed
        try:
            import langchain  # noqa: F401

            pytest.skip("langchain is installed")
        except ImportError:
            pass

        # The module can be imported, but using it should raise
        from semantic_frame.integrations.langchain import get_semantic_tool

        with pytest.raises(ImportError) as exc_info:
            get_semantic_tool()

        error_msg = str(exc_info.value)
        # Should mention how to install
        assert "langchain" in error_msg.lower()
        assert "pip" in error_msg.lower() or "install" in error_msg.lower()

    def test_crewai_import_error_mentions_pip_install(self):
        """CrewAI import error should mention installation command."""
        try:
            import crewai  # noqa: F401

            pytest.skip("crewai is installed")
        except ImportError:
            pass

        from semantic_frame.integrations.crewai import get_crewai_tool

        with pytest.raises(ImportError) as exc_info:
            get_crewai_tool()

        error_msg = str(exc_info.value)
        assert "crewai" in error_msg.lower()
        assert "pip" in error_msg.lower() or "install" in error_msg.lower()


class TestErrorMessageConsistency:
    """Tests for error message consistency across the codebase."""

    def test_type_errors_use_clear_language(self):
        """Type errors should use consistent, clear language."""
        from semantic_frame import describe_series

        invalid_inputs = [
            {"dict": "input"},
            42,  # Just a number, not iterable
            "not_a_list_of_numbers",
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                describe_series(invalid_input)  # type: ignore

            error_msg = str(exc_info.value)
            # All errors should be complete sentences or clear phrases
            assert len(error_msg) > 10, f"Error message too short: {error_msg}"
            # Should not be raw exception types or stack traces
            assert "Traceback" not in error_msg
            assert "raise " not in error_msg


class TestErrorMessageFormatting:
    """Tests for error message formatting quality."""

    def test_errors_dont_have_nested_quotes_issues(self):
        """Error messages should handle nested values correctly."""
        from semantic_frame.main import _to_numpy

        # Input with special characters that might cause formatting issues
        problematic_inputs = [
            "string with 'quotes'",
            {"key": 'value with "quotes"'},
        ]

        for problematic_input in problematic_inputs:
            try:
                _to_numpy(problematic_input)  # type: ignore
            except TypeError as e:
                error_msg = str(e)
                # Error message should be well-formed
                assert (
                    error_msg.count("'") % 2 == 0 or error_msg.count('"') % 2 == 0
                ), f"Unbalanced quotes in error message: {error_msg}"

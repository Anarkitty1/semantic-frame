"""Tests for benchmarks/claude_client.py.

Tests Claude API client wrapper and mock client for testing.
"""

from unittest import mock

import pytest

# Conditional import for optional anthropic dependency
try:
    from anthropic import APIConnectionError, APITimeoutError

    anthropic_available = True
except ImportError:
    anthropic_available = False
    # Create placeholder classes for type hints when anthropic not installed
    APIConnectionError = Exception  # type: ignore[misc, assignment]
    APITimeoutError = Exception  # type: ignore[misc, assignment]

from benchmarks.claude_client import (
    BackendType,
    ClaudeCodeClient,
    ClaudeResponse,
    MockClaudeClient,
    get_client,
)
from benchmarks.config import BenchmarkConfig

# Conditionally import ClaudeClient which requires anthropic
if anthropic_available:
    from benchmarks.claude_client import ClaudeClient


class TestClaudeResponse:
    """Tests for ClaudeResponse dataclass."""

    def test_create_response(self) -> None:
        """Test creating a ClaudeResponse."""
        response = ClaudeResponse(
            content="Test content",
            input_tokens=100,
            output_tokens=50,
            latency_ms=150.0,
            model="claude-sonnet-4-20250514",
            parsed={"answer": 42},
        )

        assert response.content == "Test content"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.latency_ms == 150.0
        assert response.error is None

    def test_response_with_error(self) -> None:
        """Test response with error."""
        response = ClaudeResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            model="claude-sonnet-4-20250514",
            parsed={},
            error="API rate limit exceeded",
        )

        assert response.error == "API rate limit exceeded"


@pytest.mark.skipif(not anthropic_available, reason="anthropic not installed")
class TestClaudeClient:
    """Tests for ClaudeClient class."""

    def test_init_without_api_key_raises(self) -> None:
        """Test initialization without API key raises error."""
        config = BenchmarkConfig(api_key=None)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            ClaudeClient(config)

    @mock.patch("anthropic.Anthropic")
    def test_init_with_api_key(self, mock_anthropic: mock.Mock) -> None:
        """Test initialization with API key."""
        config = BenchmarkConfig(api_key="test-api-key")
        _ = ClaudeClient(config)

        mock_anthropic.assert_called_once_with(api_key="test-api-key")

    @mock.patch("anthropic.Anthropic")
    def test_query_success(self, mock_anthropic: mock.Mock) -> None:
        """Test successful query."""
        # Setup mock response
        mock_response = mock.Mock()
        mock_response.content = [mock.Mock(text="- Answer: 42\n- Confidence: high")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key")
        client = ClaudeClient(config)
        response = client.query("Test prompt")

        assert response.content == "- Answer: 42\n- Confidence: high"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.error is None
        assert response.parsed["answer"] == 42.0

    @mock.patch("anthropic.Anthropic")
    def test_query_with_system(self, mock_anthropic: mock.Mock) -> None:
        """Test query with system message."""
        mock_response = mock.Mock()
        mock_response.content = [mock.Mock(text="Response")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25

        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key")
        client = ClaudeClient(config)
        client.query("Test prompt", system="Be helpful")

        # Verify system was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"

    @mock.patch("anthropic.Anthropic")
    def test_query_retry_on_failure(self, mock_anthropic: mock.Mock) -> None:
        """Test query retries on transient failures."""
        mock_client = mock.Mock()
        # Fail twice with retryable Anthropic error, succeed on third try
        mock_client.messages.create.side_effect = [
            APITimeoutError(request=mock.Mock()),
            APITimeoutError(request=mock.Mock()),
            mock.Mock(
                content=[mock.Mock(text="Success")],
                usage=mock.Mock(input_tokens=50, output_tokens=25),
            ),
        ]
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key", retry_attempts=3, retry_delay=0.01)
        client = ClaudeClient(config)
        response = client.query("Test")

        assert response.content == "Success"
        assert response.error is None
        assert mock_client.messages.create.call_count == 3

    @mock.patch("anthropic.Anthropic")
    def test_query_all_retries_fail(self, mock_anthropic: mock.Mock) -> None:
        """Test query returns error after all retries fail."""
        mock_client = mock.Mock()
        # Use APIConnectionError which is in _RETRYABLE_ERRORS
        mock_client.messages.create.side_effect = APIConnectionError(request=mock.Mock())
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key", retry_attempts=3, retry_delay=0.01)
        client = ClaudeClient(config)
        response = client.query("Test")

        assert response.error is not None
        assert "Connection error" in response.error or "failed" in response.error.lower()
        assert mock_client.messages.create.call_count == 3

    @mock.patch("anthropic.Anthropic")
    def test_query_baseline(self, mock_anthropic: mock.Mock) -> None:
        """Test query_baseline formats prompt correctly."""
        mock_response = mock.Mock()
        mock_response.content = [mock.Mock(text="- Answer: 50")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 20

        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key")
        client = ClaudeClient(config)
        client.query_baseline("[1, 2, 3]", "What is the mean?")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "[1, 2, 3]" in prompt
        assert "What is the mean?" in prompt

    @mock.patch("anthropic.Anthropic")
    def test_query_treatment(self, mock_anthropic: mock.Mock) -> None:
        """Test query_treatment formats prompt correctly."""
        mock_response = mock.Mock()
        mock_response.content = [mock.Mock(text="- Answer: 50")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20

        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key")
        client = ClaudeClient(config)
        client.query_treatment("Mean: 2.0, Trend: rising", "What is the mean?")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "Mean: 2.0" in prompt
        assert "What is the mean?" in prompt


class TestMockClaudeClient:
    """Tests for MockClaudeClient class."""

    def test_init(self) -> None:
        """Test mock client initialization."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        assert client.call_count == 0

    def test_query_increments_call_count(self) -> None:
        """Test query increments call count."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        client.query("Test prompt")
        assert client.call_count == 1

        client.query("Another prompt")
        assert client.call_count == 2

    def test_query_returns_claude_response(self) -> None:
        """Test query returns ClaudeResponse."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        response = client.query("QUERY: What is the mean?")

        assert isinstance(response, ClaudeResponse)
        assert response.content != ""
        assert response.latency_ms > 0
        assert response.error is None

    def test_query_parses_response(self) -> None:
        """Test response is parsed."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        response = client.query("QUERY: What is the mean?")

        assert "answer" in response.parsed
        assert "confidence" in response.parsed or response.parsed.get("answer") is not None

    def test_baseline_lower_accuracy(self) -> None:
        """Test baseline condition has lower accuracy rate."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        # Run many trials to check accuracy distribution
        correct_count = 0
        for _ in range(100):
            client._is_treatment = False
            if client._should_be_correct():
                correct_count += 1

        # Should be around 70% (with some variance)
        assert 50 < correct_count < 90

    def test_treatment_higher_accuracy(self) -> None:
        """Test treatment condition has higher accuracy rate."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        # Run many trials to check accuracy distribution
        correct_count = 0
        for _ in range(100):
            client._is_treatment = True
            if client._should_be_correct():
                correct_count += 1

        # Should be around 95% (with some variance)
        assert correct_count > 85

    def test_query_baseline_sets_treatment_false(self) -> None:
        """Test query_baseline sets treatment flag to False."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        client.query_baseline("[1, 2, 3]", "What is the mean?")

        assert client._is_treatment is False

    def test_query_treatment_sets_treatment_true(self) -> None:
        """Test query_treatment sets treatment flag to True."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        client.query_treatment("Mean: 2.0", "What is the mean?")

        assert client._is_treatment is True

    def test_extract_ground_truth_from_prompt_mean(self) -> None:
        """Test extraction of mean from prompt."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        prompt = "The data has a mean: 42.5 and std: 10.0"
        hints = client._extract_ground_truth_from_prompt(prompt)

        assert hints.get("mean") == 42.5
        assert hints.get("std") == 10.0

    def test_extract_ground_truth_from_prompt_trend(self) -> None:
        """Test extraction of trend from prompt."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        prompt = "The data shows a rising trend"
        hints = client._extract_ground_truth_from_prompt(prompt)

        assert hints.get("trend") == "rising"

    def test_generate_answer_mean_correct(self) -> None:
        """Test answer generation for mean query when correct."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        answer = client._generate_answer(
            query="What is the mean?",
            hints={"mean": 42.5},
            correct=True,
        )

        assert "42.5" in answer

    def test_generate_answer_mean_incorrect(self) -> None:
        """Test answer generation for mean query when incorrect."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        answer = client._generate_answer(
            query="What is the mean?",
            hints={"mean": 42.5},
            correct=False,
        )

        # Should not be 42.5 exactly
        assert "42.50" not in answer or float(answer) != 42.5

    def test_generate_answer_trend(self) -> None:
        """Test answer generation for trend query."""
        config = BenchmarkConfig(random_seed=42)
        client = MockClaudeClient(config)

        answer = client._generate_answer(
            query="What is the trend direction?",
            hints={"trend": "rising"},
            correct=True,
        )

        assert answer == "rising"

    def test_deterministic_with_same_seed(self) -> None:
        """Test mock client is deterministic with same seed."""
        config1 = BenchmarkConfig(random_seed=42)
        client1 = MockClaudeClient(config1)

        config2 = BenchmarkConfig(random_seed=42)
        client2 = MockClaudeClient(config2)

        # Same sequence of calls should produce same results
        response1 = client1.query("QUERY: What is the mean?")
        response2 = client2.query("QUERY: What is the mean?")

        assert response1.content == response2.content


class TestClaudeCodeClient:
    """Tests for ClaudeCodeClient class."""

    @mock.patch("subprocess.run")
    def test_init_verifies_cli(self, mock_run: mock.Mock) -> None:
        """Test initialization verifies CLI is available."""
        config = BenchmarkConfig(random_seed=42)

        # Mock successful version check
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")

        # Should not raise if CLI is available
        client = ClaudeCodeClient(config)
        assert client.config == config
        # Verify version check was called
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["claude", "--version"]

    def test_init_without_cli_raises(self) -> None:
        """Test initialization raises if CLI not available."""
        config = BenchmarkConfig(random_seed=42)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("claude not found")
            with pytest.raises(RuntimeError, match="claude CLI not found"):
                ClaudeCodeClient(config)

    @mock.patch("subprocess.run")
    def test_get_model_alias_haiku(self, mock_run: mock.Mock) -> None:
        """Test model alias extraction for haiku."""
        from benchmarks.config import ModelConfig

        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(
            random_seed=42, model=ModelConfig(model="claude-haiku-4-5-20251001")
        )
        client = ClaudeCodeClient(config)

        assert client._get_model_alias() == "haiku"

    @mock.patch("subprocess.run")
    def test_get_model_alias_opus(self, mock_run: mock.Mock) -> None:
        """Test model alias extraction for opus."""
        from benchmarks.config import ModelConfig

        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(
            random_seed=42, model=ModelConfig(model="claude-opus-4-5-20251101")
        )
        client = ClaudeCodeClient(config)

        assert client._get_model_alias() == "opus"

    @mock.patch("subprocess.run")
    def test_get_model_alias_sonnet(self, mock_run: mock.Mock) -> None:
        """Test model alias extraction for sonnet (default)."""
        from benchmarks.config import ModelConfig

        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(
            random_seed=42, model=ModelConfig(model="claude-sonnet-4-20250514")
        )
        client = ClaudeCodeClient(config)

        assert client._get_model_alias() == "sonnet"

    @mock.patch("subprocess.run")
    def test_parse_cli_response_success(self, mock_run: mock.Mock) -> None:
        """Test parsing successful CLI JSON response."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = ClaudeCodeClient(config)

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": 1500,
            "result": "4",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "modelUsage": {"claude-sonnet-4-20250514": {}}
        }"""
        mock_result.stderr = ""

        response = client._parse_cli_response(mock_result, 0.0)

        assert response.content == "4"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.latency_ms == 1500
        assert response.error is None

    @mock.patch("subprocess.run")
    def test_parse_cli_response_error(self, mock_run: mock.Mock) -> None:
        """Test parsing CLI error response."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = ClaudeCodeClient(config)

        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "CLI error occurred"

        response = client._parse_cli_response(mock_result, 0.0)

        assert response.content == ""
        assert response.error == "CLI error occurred"

    @mock.patch("subprocess.run")
    def test_parse_cli_response_is_error_true(self, mock_run: mock.Mock) -> None:
        """Test parsing CLI response with is_error=true."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = ClaudeCodeClient(config)

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "type": "result",
            "is_error": true,
            "duration_ms": 500,
            "result": "Rate limit exceeded"
        }"""
        mock_result.stderr = ""

        response = client._parse_cli_response(mock_result, 0.0)

        assert response.error == "Rate limit exceeded"

    @mock.patch("subprocess.run")
    def test_parse_cli_response_invalid_json(self, mock_run: mock.Mock) -> None:
        """Test parsing invalid JSON response."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = ClaudeCodeClient(config)

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json"
        mock_result.stderr = ""

        response = client._parse_cli_response(mock_result, 0.0)

        assert response.error is not None
        assert "Failed to parse CLI JSON" in response.error

    @mock.patch("subprocess.run")
    def test_is_retryable_error(self, mock_run: mock.Mock) -> None:
        """Test retryable error detection."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = ClaudeCodeClient(config)

        assert client._is_retryable_error("Rate limit exceeded")
        assert client._is_retryable_error("Request timeout")
        assert client._is_retryable_error("Connection refused")
        assert client._is_retryable_error("Server temporarily unavailable")
        assert not client._is_retryable_error("Invalid API key")
        assert not client._is_retryable_error("Malformed request")

    @mock.patch("subprocess.run")
    def test_query_success(self, mock_run: mock.Mock) -> None:
        """Test successful query via CLI."""
        config = BenchmarkConfig(random_seed=42)

        # Mock version check
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")

        client = ClaudeCodeClient(config)

        # Mock actual query
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="""{
                "type": "result",
                "is_error": false,
                "duration_ms": 1000,
                "result": "- Answer: 4",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "modelUsage": {"claude-sonnet-4-20250514": {}}
            }""",
            stderr="",
        )

        response = client.query("What is 2+2?")

        assert response.content == "- Answer: 4"
        assert response.error is None

    @mock.patch("subprocess.run")
    def test_query_timeout_retry(self, mock_run: mock.Mock) -> None:
        """Test query retries on timeout."""
        import subprocess

        config = BenchmarkConfig(random_seed=42, retry_attempts=2, retry_delay=0.01)

        # First call is version check (success)
        # Second call times out, third succeeds
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="1.0.0", stderr=""),  # version check
            subprocess.TimeoutExpired(cmd="claude", timeout=60),  # first query
            mock.Mock(  # retry succeeds
                returncode=0,
                stdout='{"is_error": false, "result": "4", "usage": {}, "modelUsage": {}}',
                stderr="",
            ),
        ]

        client = ClaudeCodeClient(config)
        response = client.query("What is 2+2?")

        assert response.content == "4"
        assert response.error is None

    @mock.patch("subprocess.run")
    def test_query_baseline(self, mock_run: mock.Mock) -> None:
        """Test query_baseline formats prompt correctly."""
        config = BenchmarkConfig(random_seed=42)

        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout='{"is_error": false, "result": "50", "usage": {}, "modelUsage": {}}',
            stderr="",
        )

        client = ClaudeCodeClient(config)
        client.query_baseline("[1, 2, 3]", "What is the mean?")

        # Check prompt was passed via stdin
        call_args = mock_run.call_args
        assert "[1, 2, 3]" in call_args.kwargs.get("input", "")
        assert "What is the mean?" in call_args.kwargs.get("input", "")

    @mock.patch("subprocess.run")
    def test_query_treatment(self, mock_run: mock.Mock) -> None:
        """Test query_treatment formats prompt correctly."""
        config = BenchmarkConfig(random_seed=42)

        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout='{"is_error": false, "result": "50", "usage": {}, "modelUsage": {}}',
            stderr="",
        )

        client = ClaudeCodeClient(config)
        client.query_treatment("Mean: 2.0, Trend: rising", "What is the mean?")

        # Check prompt was passed via stdin
        call_args = mock_run.call_args
        assert "Mean: 2.0" in call_args.kwargs.get("input", "")
        assert "What is the mean?" in call_args.kwargs.get("input", "")

    @mock.patch("subprocess.run")
    def test_query_all_retries_exhausted(self, mock_run: mock.Mock) -> None:
        """Test error message when all retries are exhausted."""
        import subprocess

        config = BenchmarkConfig(random_seed=42, retry_attempts=3, retry_delay=0.01)

        # Version check succeeds, all queries time out
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="1.0.0", stderr=""),  # version check
            subprocess.TimeoutExpired(cmd="claude", timeout=60),  # first query
            subprocess.TimeoutExpired(cmd="claude", timeout=60),  # second query
            subprocess.TimeoutExpired(cmd="claude", timeout=60),  # third query
        ]

        client = ClaudeCodeClient(config)
        response = client.query("What is 2+2?")

        assert response.error is not None
        assert "failed after 3 attempts" in response.error
        assert mock_run.call_count == 4  # 1 version + 3 query attempts

    def test_init_cli_returns_error_raises(self) -> None:
        """Test initialization raises if CLI version check fails."""
        config = BenchmarkConfig(random_seed=42)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=1,
                stdout="",
                stderr="Permission denied",
            )
            with pytest.raises(RuntimeError, match="claude CLI returned error"):
                ClaudeCodeClient(config)

    @mock.patch("subprocess.run")
    def test_query_os_error(self, mock_run: mock.Mock) -> None:
        """Test query handles OSError (non-retryable)."""
        config = BenchmarkConfig(random_seed=42)

        # Version check succeeds, query fails with OSError
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="1.0.0", stderr=""),
            OSError("No such file or directory"),
        ]

        client = ClaudeCodeClient(config)
        response = client.query("test")

        assert response.error is not None
        assert "OS error" in response.error

    def test_init_cli_timeout_raises(self) -> None:
        """Test initialization raises if CLI times out during version check."""
        import subprocess

        config = BenchmarkConfig(random_seed=42)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=10)
            with pytest.raises(RuntimeError, match="timed out during version check"):
                ClaudeCodeClient(config)

    @mock.patch("subprocess.run")
    def test_query_retry_on_rate_limit_in_response(self, mock_run: mock.Mock) -> None:
        """Test query retries when rate limit error is in JSON response."""
        config = BenchmarkConfig(random_seed=42, retry_attempts=2, retry_delay=0.01)

        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="1.0.0", stderr=""),  # version check
            mock.Mock(  # first query - rate limit in response
                returncode=0,
                stdout='{"is_error": true, "result": "Rate limit exceeded"}',
                stderr="",
            ),
            mock.Mock(  # retry succeeds
                returncode=0,
                stdout='{"is_error": false, "result": "4", "usage": {}, "modelUsage": {}}',
                stderr="",
            ),
        ]

        client = ClaudeCodeClient(config)
        response = client.query("test")

        assert response.content == "4"
        assert response.error is None


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_values(self) -> None:
        """Test backend type values."""
        assert BackendType.API.value == "api"
        assert BackendType.CLAUDE_CODE.value == "claude-code"
        assert BackendType.MOCK.value == "mock"

    def test_backend_from_string(self) -> None:
        """Test creating BackendType from string."""
        assert BackendType("api") == BackendType.API
        assert BackendType("claude-code") == BackendType.CLAUDE_CODE
        assert BackendType("mock") == BackendType.MOCK


class TestGetClient:
    """Tests for get_client factory function."""

    def test_get_client_mock_backend(self) -> None:
        """Test get_client returns MockClaudeClient with mock backend."""
        config = BenchmarkConfig(random_seed=42)
        client = get_client(config, backend="mock")

        assert isinstance(client, MockClaudeClient)

    def test_get_client_mock_backend_enum(self) -> None:
        """Test get_client with BackendType enum."""
        config = BenchmarkConfig(random_seed=42)
        client = get_client(config, backend=BackendType.MOCK)

        assert isinstance(client, MockClaudeClient)

    @mock.patch("subprocess.run")
    def test_get_client_claude_code_backend(self, mock_run: mock.Mock) -> None:
        """Test get_client returns ClaudeCodeClient with claude-code backend."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="1.0.0", stderr="")
        config = BenchmarkConfig(random_seed=42)
        client = get_client(config, backend="claude-code")

        assert isinstance(client, ClaudeCodeClient)

    @pytest.mark.skipif(not anthropic_available, reason="anthropic not installed")
    @mock.patch("anthropic.Anthropic")
    def test_get_client_api_backend(self, mock_anthropic: mock.Mock) -> None:
        """Test get_client returns ClaudeClient with api backend."""
        config = BenchmarkConfig(api_key="test-key")
        client = get_client(config, backend="api")

        assert isinstance(client, ClaudeClient)

    def test_get_client_invalid_backend(self) -> None:
        """Test get_client raises for invalid backend."""
        config = BenchmarkConfig(random_seed=42)

        with pytest.raises(ValueError, match="Invalid backend"):
            get_client(config, backend="invalid")

    @pytest.mark.skipif(not anthropic_available, reason="anthropic not installed")
    def test_get_client_default_api(self) -> None:
        """Test get_client defaults to API backend."""
        config = BenchmarkConfig(api_key=None)

        # Should raise because no API key (default is API backend)
        with pytest.raises(ValueError):
            get_client(config)

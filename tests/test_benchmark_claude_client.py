"""Tests for benchmarks/claude_client.py.

Tests Claude API client wrapper and mock client for testing.
"""

from unittest import mock

import pytest

from benchmarks.claude_client import (
    ClaudeClient,
    ClaudeResponse,
    MockClaudeClient,
    get_client,
)
from benchmarks.config import BenchmarkConfig


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
        # Fail twice, succeed on third try
        mock_client.messages.create.side_effect = [
            Exception("Transient error"),
            Exception("Transient error"),
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
        mock_client.messages.create.side_effect = Exception("Persistent error")
        mock_anthropic.return_value = mock_client

        config = BenchmarkConfig(api_key="test-key", retry_attempts=3, retry_delay=0.01)
        client = ClaudeClient(config)
        response = client.query("Test")

        assert response.error is not None
        assert "Persistent error" in response.error
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


class TestGetClient:
    """Tests for get_client factory function."""

    def test_get_client_mock(self) -> None:
        """Test get_client returns MockClaudeClient when mock=True."""
        config = BenchmarkConfig(random_seed=42)
        client = get_client(config, mock=True)

        assert isinstance(client, MockClaudeClient)

    @mock.patch("anthropic.Anthropic")
    def test_get_client_real(self, mock_anthropic: mock.Mock) -> None:
        """Test get_client returns ClaudeClient when mock=False."""
        config = BenchmarkConfig(api_key="test-key")
        client = get_client(config, mock=False)

        assert isinstance(client, ClaudeClient)

    def test_get_client_default_mock_false(self) -> None:
        """Test get_client defaults to real client."""
        config = BenchmarkConfig(api_key=None)

        # Should raise because no API key
        with pytest.raises(ValueError):
            get_client(config, mock=False)

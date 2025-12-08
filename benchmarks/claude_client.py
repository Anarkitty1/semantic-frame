"""
Claude API Client

Wrapper for Anthropic API calls with retry logic and response parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmarks.config import BASELINE_PROMPT_TEMPLATE, TREATMENT_PROMPT_TEMPLATE, BenchmarkConfig
from benchmarks.metrics import count_tokens, parse_llm_response

if TYPE_CHECKING:
    from anthropic import Anthropic

# Try to import specific Anthropic exceptions for better error handling
try:
    from anthropic import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )

    _ANTHROPIC_ERRORS: tuple[type[Exception], ...] = (
        APIError,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )
    _RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:
    # If anthropic not installed, use generic exceptions
    _ANTHROPIC_ERRORS = (Exception,)
    _RETRYABLE_ERRORS = (Exception,)


@dataclass
class ClaudeResponse:
    """Structured response from Claude API."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    parsed: dict[str, Any]
    error: str | None = None


class ClaudeClient:
    """Client for interacting with Claude API."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client: Anthropic | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic

            if not self.config.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it as an environment variable or pass it to BenchmarkConfig."
                )

            self._client = Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. " "Install with: pip install anthropic"
            )

    def query(
        self,
        prompt: str,
        system: str | None = None,
    ) -> ClaudeResponse:
        """
        Send a query to Claude and return structured response.

        Includes retry logic for transient failures.
        """
        messages = [{"role": "user", "content": prompt}]

        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                start_time = time.perf_counter()

                kwargs: dict[str, Any] = {
                    "model": self.config.model.model,
                    "max_tokens": self.config.model.max_tokens,
                    "temperature": self.config.model.temperature,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                assert self._client is not None, "Client not initialized"
                response = self._client.messages.create(**kwargs)

                latency_ms = (time.perf_counter() - start_time) * 1000

                content = response.content[0].text if response.content else ""

                return ClaudeResponse(
                    content=content,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency_ms,
                    model=self.config.model.model,
                    parsed=parse_llm_response(content),
                )

            except _RETRYABLE_ERRORS as e:
                # Transient errors - retry with exponential backoff
                last_error = str(e)
                error_type = type(e).__name__
                if attempt < self.config.retry_attempts - 1:
                    retries = self.config.retry_attempts
                    delay = self.config.retry_delay * (2**attempt)  # Exponential backoff
                    print(f"  {error_type} (attempt {attempt + 1}/{retries}): {e}")
                    print(f"    Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                continue
            except _ANTHROPIC_ERRORS as e:
                # Non-retryable API errors - fail immediately
                error_type = type(e).__name__
                error_msg = f"{error_type}: {e}"
                print(f"ERROR: Non-retryable API error: {error_msg}", flush=True)
                return ClaudeResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    model=self.config.model.model,
                    parsed={},
                    error=error_msg,
                )
            except Exception as e:
                # Unexpected errors - log and fail
                last_error = str(e)
                error_type = type(e).__name__
                if attempt < self.config.retry_attempts - 1:
                    retries = self.config.retry_attempts
                    print(f"  Unexpected {error_type} (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                continue

        # All retries failed - log error prominently
        error_msg = f"API call failed after {self.config.retry_attempts} attempts: {last_error}"
        print(f"ERROR: {error_msg}", flush=True)

        return ClaudeResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            model=self.config.model.model,
            parsed={},
            error=error_msg,
        )

    def query_baseline(
        self,
        raw_data: str,
        query: str,
    ) -> ClaudeResponse:
        """
        Query Claude with raw data (baseline condition).
        """
        prompt = BASELINE_PROMPT_TEMPLATE.format(
            data=raw_data,
            query=query,
        )
        return self.query(prompt)

    def query_treatment(
        self,
        semantic_frame_output: str,
        query: str,
    ) -> ClaudeResponse:
        """
        Query Claude with Semantic Frame output (treatment condition).
        """
        prompt = TREATMENT_PROMPT_TEMPLATE.format(
            semantic_frame_output=semantic_frame_output,
            query=query,
        )
        return self.query(prompt)


class MockClaudeClient:
    """
    Smart mock client for testing without API calls.

    Simulates realistic baseline vs treatment accuracy differences.
    Uses ground truth hints from semantic frame output to generate
    responses that demonstrate expected benchmark outcomes.
    """

    # Expected accuracy rates based on research
    BASELINE_ACCURACY = 0.70  # 70% accuracy for baseline
    TREATMENT_ACCURACY = 0.95  # 95% accuracy for treatment

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.call_count = 0
        self._rng = __import__("random").Random(config.random_seed)
        self._current_ground_truth = None
        self._is_treatment = False

    def _extract_ground_truth_from_prompt(self, prompt: str) -> dict[str, float | str]:
        """Try to extract ground truth hints from prompt content."""
        import re

        hints: dict[str, float | str] = {}

        # Look for statistical values in semantic frame output
        patterns = {
            "mean": r"mean[:\s]+([+-]?\d+\.?\d*)",
            "median": r"median[:\s]+([+-]?\d+\.?\d*)",
            "std": r"(?:std|deviation)[:\s]+([+-]?\d+\.?\d*)",
            "min": r"min(?:imum)?[:\s]+([+-]?\d+\.?\d*)",
            "max": r"max(?:imum)?[:\s]+([+-]?\d+\.?\d*)",
            "count": r"(?:count|points?|samples?)[:\s]+(\d+)",
            "trend": r"(rising|falling|flat|cyclical|upward|downward|stable)",
            "volatility": r"(high|moderate|low|stable)\s*(?:volatility|variability)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, prompt.lower())
            if match:
                val = match.group(1)
                try:
                    hints[key] = float(val)
                except ValueError:
                    hints[key] = val

        return hints

    def _should_be_correct(self) -> bool:
        """Determine if this response should be correct based on condition."""
        accuracy = self.TREATMENT_ACCURACY if self._is_treatment else self.BASELINE_ACCURACY
        return bool(self._rng.random() < accuracy)

    def _generate_answer(self, query: str, hints: dict, correct: bool) -> str:
        """Generate an answer based on query type and correctness."""
        query_lower = query.lower()

        # Determine query type and generate appropriate answer
        if "mean" in query_lower or "average" in query_lower:
            if correct and "mean" in hints:
                answer = f"{hints['mean']:.2f}"
            else:
                answer = f"{self._rng.uniform(30, 70):.2f}"

        elif "median" in query_lower:
            if correct and "median" in hints:
                answer = f"{hints['median']:.2f}"
            else:
                answer = f"{self._rng.uniform(30, 70):.2f}"

        elif "standard deviation" in query_lower or "std" in query_lower:
            if correct and "std" in hints:
                answer = f"{hints['std']:.2f}"
            else:
                answer = f"{self._rng.uniform(5, 20):.2f}"

        elif "minimum" in query_lower or "min" in query_lower:
            if correct and "min" in hints:
                answer = f"{hints['min']:.2f}"
            else:
                answer = f"{self._rng.uniform(0, 30):.2f}"

        elif "maximum" in query_lower or "max" in query_lower:
            if correct and "max" in hints:
                answer = f"{hints['max']:.2f}"
            else:
                answer = f"{self._rng.uniform(70, 100):.2f}"

        elif "count" in query_lower or "how many" in query_lower:
            if correct and "count" in hints:
                answer = str(int(hints["count"]))
            else:
                answer = str(self._rng.randint(50, 200))

        elif "trend" in query_lower:
            if correct and "trend" in hints:
                answer = hints["trend"]
            else:
                answer = self._rng.choice(["rising", "falling", "flat", "cyclical"])

        elif "anomal" in query_lower:
            # Anomaly presence - usually there are some
            if correct:
                answer = "yes"
            else:
                answer = self._rng.choice(["yes", "no"])

        elif (
            "percentile" in query_lower
            or "p25" in query_lower
            or "p75" in query_lower
            or "p95" in query_lower
        ):
            if correct:
                # Approximate based on mean and std if available
                mean = hints.get("mean", 50)
                answer = f"{mean + self._rng.uniform(-10, 10):.2f}"
            else:
                answer = f"{self._rng.uniform(30, 70):.2f}"

        elif "range" in query_lower or "iqr" in query_lower:
            if correct:
                answer = f"{self._rng.uniform(15, 40):.2f}"
            else:
                answer = f"{self._rng.uniform(5, 60):.2f}"

        elif "skew" in query_lower:
            if correct:
                answer = self._rng.choice(["positive", "negative", "none"])
            else:
                answer = self._rng.choice(["positive", "negative", "none", "unknown"])

        elif "series a" in query_lower or "series b" in query_lower:
            # Comparative queries
            if correct:
                answer = "Series B"  # Often B has higher values in test data
            else:
                answer = self._rng.choice(["Series A", "Series B"])

        elif "correlat" in query_lower:
            if correct:
                answer = "positively correlated"
            else:
                answer = self._rng.choice(
                    ["positively correlated", "negatively correlated", "uncorrelated"]
                )

        elif "volatilit" in query_lower:
            if correct and "volatility" in hints:
                answer = hints["volatility"]
            else:
                answer = self._rng.choice(["high", "moderate", "low"])

        else:
            answer = "unknown"

        return answer

    def query(
        self,
        prompt: str,
        system: str | None = None,
    ) -> ClaudeResponse:
        """Return smart mock response."""
        self.call_count += 1

        # Simulate some latency
        time.sleep(0.001)

        # Extract hints from prompt
        hints = self._extract_ground_truth_from_prompt(prompt)

        # Determine if response should be correct
        correct = self._should_be_correct()

        # Extract query from prompt
        query_match = __import__("re").search(r"QUERY:\s*(.+?)(?:\n|$)", prompt)
        query = query_match.group(1) if query_match else prompt

        # Generate answer
        answer = self._generate_answer(query, hints, correct)
        confidence = "high" if correct else "medium"
        reasoning = "Based on analysis of the data."

        content = f"- Answer: {answer}\n- Confidence: {confidence}\n- Reasoning: {reasoning}"

        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(content)

        return ClaudeResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=10.0,
            model="mock-model",
            parsed=parse_llm_response(content),
        )

    def query_baseline(
        self,
        raw_data: str,
        query: str,
    ) -> ClaudeResponse:
        """Query with raw data (baseline condition)."""
        self._is_treatment = False
        prompt = BASELINE_PROMPT_TEMPLATE.format(data=raw_data, query=query)
        return self.query(prompt)

    def query_treatment(
        self,
        semantic_frame_output: str,
        query: str,
    ) -> ClaudeResponse:
        """Query with Semantic Frame output (treatment condition)."""
        self._is_treatment = True
        prompt = TREATMENT_PROMPT_TEMPLATE.format(
            semantic_frame_output=semantic_frame_output,
            query=query,
        )
        return self.query(prompt)


def get_client(config: BenchmarkConfig, mock: bool = False) -> ClaudeClient | MockClaudeClient:
    """Get appropriate client based on configuration."""
    if mock:
        return MockClaudeClient(config)
    return ClaudeClient(config)

"""
Claude API Client

Wrapper for Anthropic API calls with retry logic and response parsing.
Supports multiple backends: API (paid), Claude Code CLI (free on Max), and Mock.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from benchmarks.config import BASELINE_PROMPT_TEMPLATE, TREATMENT_PROMPT_TEMPLATE, BenchmarkConfig
from benchmarks.metrics import count_tokens, parse_llm_response


class BackendType(Enum):
    """Available backend types for Claude queries."""

    API = "api"  # Anthropic API (paid)
    CLAUDE_CODE = "claude-code"  # Claude Code CLI (free on Max plan)
    MOCK = "mock"  # Mock client for testing


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
    # If anthropic not installed, only catch network errors (not programming bugs)
    _ANTHROPIC_ERRORS = (ConnectionError, TimeoutError, OSError)
    _RETRYABLE_ERRORS = (ConnectionError, TimeoutError)


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
                "anthropic package not installed. Install with: pip install anthropic"
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
                response = self._client.messages.create(**kwargs)  # type: ignore[attr-defined]

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
            # Let programming errors (TypeError, AttributeError, etc.) propagate immediately

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


class ClaudeCodeClient:
    """
    Client using Claude Code CLI instead of paid API.

    Enables free iteration on Max plans ($100-200/month) with final
    validation through the API. Uses subprocess calls to the `claude` CLI.

    Requirements:
        - Claude Code CLI installed and authenticated
        - Max plan subscription for sufficient quota

    Rate Limits (Max plans):
        - Max 5x: 50-200 prompts per 5 hours
        - Max 20x: 200-800 prompts per 5 hours
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._verify_cli_available()

    def _verify_cli_available(self) -> None:
        """Check that claude CLI is available and working."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI returned error: {result.stderr or result.stdout}")
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install from: https://claude.ai/code\n"
                "After installation, run 'claude' to authenticate."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "claude CLI timed out during version check (10s). "
                "This may indicate the CLI is not properly installed or "
                "is waiting for authentication. "
                "Try running 'claude --version' manually to diagnose."
            )

    def _get_model_alias(self) -> str:
        """Convert model config to CLI alias."""
        model = self.config.model.model.lower()
        if "haiku" in model:
            return "haiku"
        elif "opus" in model:
            return "opus"
        else:
            return "sonnet"  # Default to sonnet

    def _parse_cli_response(
        self,
        result: subprocess.CompletedProcess[str],
        start_time: float,
    ) -> ClaudeResponse:
        """Parse CLI JSON output to ClaudeResponse."""
        latency_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            if result.stderr and result.stderr.strip():
                error_msg = result.stderr.strip()
            else:
                error_msg = f"CLI execution failed with exit code {result.returncode}"
                if result.stdout:
                    error_msg += f"\nOutput: {result.stdout[:500]}"
            return ClaudeResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                model=self.config.model.model,
                parsed={},
                error=error_msg,
            )

        try:
            data = json.loads(result.stdout)

            # Check for error in response
            if data.get("is_error", False):
                return ClaudeResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=data.get("duration_ms", latency_ms),
                    model=self.config.model.model,
                    parsed={},
                    error=data.get("result", "Unknown CLI error"),
                )

            # Extract usage info
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Get actual model used from modelUsage
            model_usage = data.get("modelUsage", {})
            actual_model = self.config.model.model
            if model_usage:
                # Get first model from usage (primary model used)
                actual_model = next(iter(model_usage.keys()), actual_model)

            content = data.get("result", "")

            return ClaudeResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=data.get("duration_ms", latency_ms),
                model=actual_model,
                parsed=parse_llm_response(content),
            )

        except json.JSONDecodeError as e:
            return ClaudeResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                model=self.config.model.model,
                parsed={},
                error=f"Failed to parse CLI JSON response: {e}\nOutput: {result.stdout[:500]}",
            )
        except (KeyError, TypeError) as e:
            return ClaudeResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                model=self.config.model.model,
                parsed={},
                error=f"Unexpected CLI response structure: {e}",
            )

    def _calculate_timeout(self, prompt: str) -> float:
        """
        Calculate timeout based on prompt size.

        Larger prompts (e.g., 1000+ data points) need more time.
        Returns timeout in seconds.
        """
        base_timeout = self.config.model.timeout
        extra_timeout = (len(prompt) / 1000) * self.config.model.timeout_per_1k_chars
        return min(base_timeout + extra_timeout, self.config.model.max_timeout)

    def warmup(self) -> bool:
        """
        Send minimal query to trigger CLI cache creation.

        The first CLI call incurs ~40K token cache creation overhead.
        A warmup query before benchmarks starts amortizes this cost.

        Returns:
            True if warmup succeeded, False otherwise.
        """
        try:
            response = self.query("Respond with just the word 'ready'.")
            return response.error is None
        except Exception:
            return False

    def query(
        self,
        prompt: str,
        system: str | None = None,
    ) -> ClaudeResponse:
        """
        Execute query via Claude Code CLI subprocess.

        Uses --tools "" to disable all tools for pure LLM response,
        matching API behavior for benchmarking.
        """
        cmd = [
            "claude",
            "-p",  # Non-interactive print mode
            "--output-format",
            "json",
            "--tools",
            "",  # Disable all tools for pure LLM response
            "--model",
            self._get_model_alias(),
        ]

        if system:
            cmd.extend(["--system-prompt", system])

        # Calculate timeout based on prompt size (larger prompts need more time)
        timeout_seconds = self._calculate_timeout(prompt)

        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                start_time = time.perf_counter()

                # Pass prompt via stdin to handle special characters safely
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )

                response = self._parse_cli_response(result, start_time)

                # Check if we got an error
                if response.error:
                    if self._is_retryable_error(response.error):
                        last_error = response.error
                        if attempt < self.config.retry_attempts - 1:
                            delay = self.config.retry_delay * (2**attempt)
                            print(
                                f"  CLI error (attempt {attempt + 1}/"
                                f"{self.config.retry_attempts}): "
                                f"{response.error[:100]}"
                            )
                            print(f"    Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        else:
                            # Final attempt failed with retryable error
                            error_msg = (
                                f"CLI call failed after {self.config.retry_attempts} attempts: "
                                f"{response.error}"
                            )
                            print(f"ERROR: {error_msg}", flush=True)
                            return ClaudeResponse(
                                content="",
                                input_tokens=0,
                                output_tokens=0,
                                latency_ms=response.latency_ms,
                                model=self.config.model.model,
                                parsed={},
                                error=error_msg,
                            )
                    else:
                        # Non-retryable error - log and return immediately
                        print(
                            f"ERROR: CLI returned non-retryable error: {response.error}",
                            flush=True,
                        )
                        return response

                return response

            except subprocess.TimeoutExpired:
                last_error = f"CLI timed out after {timeout_seconds}s"
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    print(
                        f"  Timeout (attempt {attempt + 1}/{self.config.retry_attempts}): "
                        f"{last_error}"
                    )
                    print(f"    Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                continue

            except OSError as e:
                # Non-retryable OS error
                error_msg = f"OS error running CLI: {e}"
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

        # All retries failed
        error_msg = f"CLI call failed after {self.config.retry_attempts} attempts: {last_error}"
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

    def _is_retryable_error(self, error: str) -> bool:
        """Check if an error is retryable."""
        retryable_patterns = [
            "rate limit",
            "timeout",
            "overloaded",
            "temporarily unavailable",
            "connection",
            "network",
        ]
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)

    def query_baseline(
        self,
        raw_data: str,
        query: str,
    ) -> ClaudeResponse:
        """Query Claude Code CLI with raw data (baseline condition)."""
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
        """Query Claude Code CLI with Semantic Frame output (treatment condition)."""
        prompt = TREATMENT_PROMPT_TEMPLATE.format(
            semantic_frame_output=semantic_frame_output,
            query=query,
        )
        return self.query(prompt)


# Type alias for any client type
ClientType = ClaudeClient | MockClaudeClient | ClaudeCodeClient


def get_client(
    config: BenchmarkConfig,
    backend: BackendType | str = BackendType.API,
) -> ClientType:
    """
    Get appropriate client based on backend configuration.

    Args:
        config: Benchmark configuration
        backend: Backend type - "api", "claude-code", or "mock"
                 (or BackendType enum)

    Returns:
        Client instance for the specified backend

    Examples:
        >>> client = get_client(config, "mock")  # For testing
        >>> client = get_client(config, "claude-code")  # Free on Max plan
        >>> client = get_client(config, "api")  # Paid API
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        try:
            backend = BackendType(backend.lower())
        except ValueError:
            valid = [b.value for b in BackendType]
            raise ValueError(f"Invalid backend '{backend}'. Must be one of: {valid}")

    if backend == BackendType.MOCK:
        return MockClaudeClient(config)
    elif backend == BackendType.CLAUDE_CODE:
        return ClaudeCodeClient(config)
    elif backend == BackendType.API:
        return ClaudeClient(config)
    else:
        raise ValueError(f"Unknown backend type: {backend}")

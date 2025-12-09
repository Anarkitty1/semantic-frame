# Claude Code CLI Backend Research

**Date:** 2025-12-09
**Status:** Research Complete - Ready for Implementation

## Executive Summary

Claude Code CLI can serve as an alternative backend for benchmarks, enabling free iteration on Max plans ($100-200/month) with final validation through the paid API. The CLI provides all necessary features: non-interactive mode, JSON output, model selection, and tool control.

## CLI Capabilities

### Core Command Structure

```bash
# Basic non-interactive query
claude -p "prompt text"

# With JSON output (recommended for programmatic parsing)
claude -p "prompt" --output-format json

# Piped input
echo "prompt" | claude -p --output-format json
```

### JSON Output Format

The `--output-format json` flag returns structured data:

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "duration_ms": 3217,
  "duration_api_ms": 3021,
  "num_turns": 1,
  "result": "The response text...",
  "session_id": "cf23c5db-e6b1-4999-a418-b8cd77245759",
  "total_cost_usd": 0.25558,
  "usage": {
    "input_tokens": 3,
    "cache_creation_input_tokens": 40871,
    "cache_read_input_tokens": 0,
    "output_tokens": 5
  },
  "modelUsage": {
    "claude-opus-4-5-20251101": {
      "inputTokens": 3,
      "outputTokens": 5,
      "cacheReadInputTokens": 0,
      "cacheCreationInputTokens": 40871,
      "costUSD": 0.25558
    }
  }
}
```

### Model Selection

```bash
# Use specific model aliases
claude -p "prompt" --model haiku    # Fast, cheap
claude -p "prompt" --model sonnet   # Balanced
claude -p "prompt" --model opus     # Most capable

# Full model names also work
claude -p "prompt" --model claude-sonnet-4-5-20250929
```

### Tool Control (Critical for Benchmarks)

```bash
# Disable all tools (pure LLM response, no code execution)
claude -p "prompt" --tools ""

# Limit to specific tools
claude -p "prompt" --allowedTools "Read,Grep"

# Disallow specific tools
claude -p "prompt" --disallowedTools "Bash,Edit"
```

For benchmarks, we should use `--tools ""` to get pure LLM responses without agentic tool use, matching the API behavior.

### Agentic Turn Control

```bash
# Limit conversation turns (cost control)
claude -p "prompt" --max-turns 1
```

### Custom System Prompts

```bash
claude -p "prompt" --system-prompt "You are a data analyst..."
claude -p "prompt" --append-system-prompt "Always respond in JSON format"
```

## Parallel Execution

**Native Support:** None. CLI processes one request at a time.

**Shell-Based Parallelization:** Works well with background processes.

```bash
# Sequential: ~5.4s for 2 queries
claude -p "query1" --model haiku && claude -p "query2" --model haiku

# Parallel: ~3.1s for 2 queries (42% faster)
claude -p "query1" --model haiku &
claude -p "query2" --model haiku &
wait
```

**Implementation Options:**
1. Python `subprocess` with `concurrent.futures.ThreadPoolExecutor`
2. Python `asyncio.create_subprocess_exec` for async parallelism
3. Sequential execution with progress reporting

## Rate Limits (Max Plans)

### Max 5x ($100/month)
- **Prompts per 5 hours:** ~50-200 with Claude Code
- **Weekly limits:** 140-280 hours Sonnet 4, 15-35 hours Opus 4
- **Auto-switch:** Opus → Sonnet at 20% usage

### Max 20x ($200/month)
- **Prompts per 5 hours:** ~200-800 with Claude Code
- **Weekly limits:** 240-480 hours Sonnet 4, 24-40 hours Opus 4
- **Auto-switch:** Opus → Sonnet at 50% usage

### Reset Cycles
- 5-hour rolling window for prompt limits
- Weekly reset for hour-based limits
- Parallel instances consume quota faster

### Benchmark Implications
- Full benchmark suite (6 tasks × 10 trials × 2 conditions = 120 queries) fits within Max 5x limits
- Quick validation mode (fewer trials) recommended for iterative development
- Use `--model haiku` for rapid iteration (cheaper, faster)
- Save `sonnet` for final validation runs

## Token Counting Differences

The CLI reports tokens differently from the raw API:

| Metric | CLI Field | Notes |
|--------|-----------|-------|
| Input tokens | `usage.input_tokens` | Excludes cache tokens |
| Output tokens | `usage.output_tokens` | Same as API |
| Cache creation | `usage.cache_creation_input_tokens` | CLI-specific (system prompt caching) |
| Cache read | `usage.cache_read_input_tokens` | CLI-specific |
| Total cost | `total_cost_usd` | Includes all token types |

For benchmark comparisons, use `input_tokens + output_tokens` for consistency with API baseline.

## ClaudeCodeClient Design

### Interface Compatibility

Must match existing `ClaudeClient` and `MockClaudeClient` interfaces:

```python
@dataclass
class ClaudeResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    parsed: dict[str, Any]
    error: str | None = None

class ClaudeCodeClient:
    def __init__(self, config: BenchmarkConfig) -> None: ...
    def query(self, prompt: str, system: str | None = None) -> ClaudeResponse: ...
    def query_baseline(self, raw_data: str, query: str) -> ClaudeResponse: ...
    def query_treatment(self, semantic_frame_output: str, query: str) -> ClaudeResponse: ...
```

### Implementation Strategy

```python
import subprocess
import json
from typing import Any

class ClaudeCodeClient:
    """Client using Claude Code CLI instead of API."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._verify_cli_available()

    def _verify_cli_available(self) -> None:
        """Check that claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("claude CLI not available")
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install from: https://claude.ai/code"
            )

    def query(self, prompt: str, system: str | None = None) -> ClaudeResponse:
        """Execute query via CLI subprocess."""
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--tools", "",  # Disable tools for pure LLM response
            "--max-turns", "1",  # Single turn only
            "--model", self._get_model_alias(),
        ]

        if system:
            cmd.extend(["--system-prompt", system])

        # Pass prompt via stdin to handle special characters
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.config.model.timeout,
        )

        return self._parse_response(result)

    def _get_model_alias(self) -> str:
        """Convert model config to CLI alias."""
        model = self.config.model.model
        if "haiku" in model:
            return "haiku"
        elif "opus" in model:
            return "opus"
        else:
            return "sonnet"  # Default

    def _parse_response(self, result: subprocess.CompletedProcess) -> ClaudeResponse:
        """Parse CLI JSON output to ClaudeResponse."""
        if result.returncode != 0:
            return ClaudeResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                model=self.config.model.model,
                parsed={},
                error=result.stderr or "CLI execution failed",
            )

        try:
            data = json.loads(result.stdout)
            usage = data.get("usage", {})

            return ClaudeResponse(
                content=data.get("result", ""),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=data.get("duration_ms", 0),
                model=data.get("modelUsage", {}).keys().__iter__().__next__(),
                parsed=parse_llm_response(data.get("result", "")),
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ClaudeResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                model=self.config.model.model,
                parsed={},
                error=f"Failed to parse CLI response: {e}",
            )
```

### Parallel Execution Option

For faster benchmark runs with parallel queries:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ClaudeCodeClient:
    def __init__(self, config: BenchmarkConfig, parallel: int = 1) -> None:
        self.config = config
        self.parallel = parallel
        self._executor = ThreadPoolExecutor(max_workers=parallel)

    def query_batch(self, prompts: list[str]) -> list[ClaudeResponse]:
        """Execute multiple queries in parallel."""
        futures = [
            self._executor.submit(self.query, prompt)
            for prompt in prompts
        ]
        return [f.result() for f in as_completed(futures)]
```

## CLI Flags Summary

| Flag | Purpose | Benchmark Use |
|------|---------|---------------|
| `-p` / `--print` | Non-interactive mode | Required |
| `--output-format json` | Structured output | Required for parsing |
| `--tools ""` | Disable all tools | Required for pure LLM |
| `--max-turns 1` | Single turn | Prevents runaway |
| `--model <alias>` | Model selection | Match API model |
| `--system-prompt` | Custom system prompt | Optional |
| `--timeout` | Request timeout | Implicit via subprocess |

## Run Benchmark Integration

```bash
# Proposed CLI additions
python -m benchmarks.run_benchmark --backend api      # Default: paid API
python -m benchmarks.run_benchmark --backend claude-code  # Free on Max plan
python -m benchmarks.run_benchmark --backend mock     # Existing mock mode

# Combined with other flags
python -m benchmarks.run_benchmark --backend claude-code --quick --model haiku
```

## Recommendations

1. **Start with sequential execution** - simpler implementation, sufficient for quick validation
2. **Use `--tools ""` always** - ensures pure LLM responses matching API behavior
3. **Use `--model haiku` for iteration** - fastest, cheapest, saves quota
4. **Add progress bar** - CLI is slower than API, users need feedback
5. **Handle errors gracefully** - CLI can timeout, return non-JSON, etc.
6. **Document quota impact** - warn users about limit consumption

## Sources

- [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)
- [Claude Code Headless mode](https://code.claude.com/docs/en/headless.md)
- [Max plan limits](https://support.claude.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan)
- [Rate limits documentation](https://docs.claude.com/en/api/rate-limits)

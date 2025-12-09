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

### Implementation

The complete `ClaudeCodeClient` implementation is in `benchmarks/claude_client.py`.
Key design decisions documented in the class and method docstrings:

- Verifies CLI availability on initialization
- Uses `--tools ""` for pure LLM responses (no agentic behavior)
- Passes prompts via stdin for safe handling of special characters
- Implements retry logic with exponential backoff for transient errors
- Parses JSON output to match `ClaudeResponse` interface

### Parallel Execution

**Note:** Parallel execution was considered but not implemented. Sequential execution
is sufficient for benchmark validation and avoids rate limit concerns. The
Recommendations section below explains this design decision.

## CLI Flags Summary

| Flag | Purpose | Benchmark Use |
|------|---------|---------------|
| `-p` / `--print` | Non-interactive mode | Required |
| `--output-format json` | Structured output | Required for parsing |
| `--tools ""` | Disable all tools | Required for pure LLM |
| `--model <alias>` | Model selection | Match API model |
| `--system-prompt` | Custom system prompt | Optional |

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

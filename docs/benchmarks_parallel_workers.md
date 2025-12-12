# Running Benchmarks with Parallel Workers

This guide covers how to run semantic-frame benchmarks using parallel execution with the Claude Code CLI backend.

## Quick Start

```bash
# Basic parallel execution (2x speedup)
python -m benchmarks.run_benchmark --backend claude-code --parallel

# Multiple trials in parallel (up to 4x speedup)
python -m benchmarks.run_benchmark --backend claude-code --trial-parallelism 4

# Combined: parallel conditions + parallel trials
python -m benchmarks.run_benchmark --backend claude-code --parallel --trial-parallelism 4
```

## Backend Options

| Backend | Cost | Speed | Use Case |
|---------|------|-------|----------|
| `--backend api` | Paid (per token) | Fast | Final validation, production |
| `--backend claude-code` | Free (Max plan) | Medium | Development iteration |
| `--backend mock` | Free | Instant | Testing pipeline |

## Parallelism Modes

### 1. Condition Parallelism (`--parallel`)

Runs baseline and treatment conditions concurrently within each trial.

```bash
python -m benchmarks.run_benchmark --backend claude-code --parallel
```

**How it works:**
- Each trial compares baseline (raw data) vs treatment (semantic-frame output)
- With `--parallel`, both API calls happen simultaneously
- **Speedup:** ~2x per trial

**Implementation:** Uses Python `ThreadPoolExecutor` with 2 workers to run `query_baseline()` and `query_treatment()` concurrently.

### 2. Trial Parallelism (`--trial-parallelism N`)

Runs multiple trials concurrently across the benchmark suite.

```bash
# Run 4 trials at once
python -m benchmarks.run_benchmark --backend claude-code --trial-parallelism 4
```

**How it works:**
- Multiple independent trials execute simultaneously
- Each trial is a complete dataset + query combination
- **Speedup:** Up to Nx (where N is worker count)

**Rate Limit Warning:** Using more than 4 workers may hit Claude Code CLI rate limits:
- Max 5x plan: 50-200 prompts per 5 hours
- Max 20x plan: 200-800 prompts per 5 hours

### 3. Combined Parallelism

Maximum throughput by combining both modes:

```bash
python -m benchmarks.run_benchmark --backend claude-code --parallel --trial-parallelism 4
```

**Effective parallelism:** Up to 8 concurrent API calls (4 trials x 2 conditions each)

## Configuration Options

### Command Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--backend` | API backend: `api`, `claude-code`, `mock` | `api` |
| `--parallel` | Enable baseline/treatment parallelism | Disabled |
| `--trial-parallelism N` | Run N trials concurrently | 1 |
| `--max-data-size N` | Limit dataset size (points) | Auto for CLI |
| `--skip-baseline-above N` | Skip baseline for large datasets | 5000 |
| `--quick` | Quick mode (fewer trials) | Full mode |
| `--trials N` | Override trial count | 30 (5 in quick) |

### Programmatic Configuration

```python
from benchmarks.config import BenchmarkConfig
from benchmarks.runner import BenchmarkRunner
from benchmarks.claude_client import BackendType

config = BenchmarkConfig.quick_mode()
config.parallel_workers = 2        # Enable condition parallelism
config.trial_parallelism = 4       # 4 concurrent trials

runner = BenchmarkRunner(config, backend=BackendType.CLAUDE_CODE)
results = runner.run_all()
```

## Claude Code CLI Backend Details

### Prerequisites

1. **Claude Code CLI installed:**
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Authenticated:**
   ```bash
   claude  # Follow authentication prompts
   ```

3. **Max plan subscription** (for sufficient quota)

### How It Works

The `ClaudeCodeClient` executes queries via subprocess:

```bash
claude -p "prompt" --output-format json --tools "" --model sonnet
```

Key flags:
- `-p`: Non-interactive print mode
- `--output-format json`: Structured output for parsing
- `--tools ""`: Disable all tools (pure LLM response)
- `--model`: Model alias (haiku, sonnet, opus)

### Rate Limits

| Plan | Prompts per 5 hours | Recommended Parallelism |
|------|---------------------|------------------------|
| Max 5x ($100/mo) | 50-200 | 2 workers |
| Max 20x ($200/mo) | 200-800 | 4 workers |

**Full benchmark suite:** 6 tasks x 10 trials x 2 conditions = 120 queries

### CLI-Specific Optimizations

When using `--backend claude-code`:

1. **Dataset size limiting:** Automatically limits datasets to prevent timeouts
   ```bash
   # Default CLI limit: 1000 points
   # Override with:
   python -m benchmarks.run_benchmark --backend claude-code --max-data-size 500
   ```

2. **Warmup query:** First CLI call creates ~40K token cache overhead. The runner sends a warmup query before benchmarks to amortize this cost.

3. **Dynamic timeouts:** Larger prompts get proportionally longer timeouts:
   ```python
   timeout = base_timeout + (prompt_length / 1000) * timeout_per_1k_chars
   ```

## Performance Comparison

### Sequential vs Parallel (Typical Results)

| Mode | Time (30 trials) | Speedup |
|------|-----------------|---------|
| Sequential | ~15 min | 1x |
| `--parallel` | ~8 min | ~2x |
| `--trial-parallelism 4` | ~4 min | ~4x |
| Combined | ~3 min | ~5x |

### Backend Comparison

| Backend | Latency per Query | Cost per 1K Queries |
|---------|------------------|---------------------|
| API (Sonnet) | 2-5s | ~$15 |
| Claude Code CLI | 3-8s | $0 (included in Max) |
| Mock | <1ms | $0 |

## Example Workflows

### Rapid Iteration (Minimal Cost)

Fastest feedback for debugging/development:

```bash
# Single trial, single task (fastest)
python -m benchmarks.run_benchmark \
    --trials 1 --backend claude-code --parallel --task trend --no-viz

# Single trial, all tasks
python -m benchmarks.run_benchmark \
    --trials 1 --backend claude-code --parallel --no-viz

# Pipeline test only (no API calls)
python -m benchmarks.run_benchmark --mock --no-viz
```

### Development Iteration

Fast feedback loop during development:

```bash
# Quick validation with CLI backend
python -m benchmarks.run_benchmark \
    --backend claude-code \
    --quick \
    --parallel \
    --task statistical

# Output: ~2 minutes, 10 trials
```

### Full Benchmark Suite

Comprehensive evaluation:

```bash
# Full suite with maximum parallelism
python -m benchmarks.run_benchmark \
    --backend claude-code \
    --parallel \
    --trial-parallelism 4 \
    --format all

# Output: ~10 minutes, 180 trials across all tasks
```

### Production Validation

Final validation with paid API (more reliable, faster):

```bash
# Use API backend for final numbers
export ANTHROPIC_API_KEY="sk-..."
python -m benchmarks.run_benchmark \
    --backend api \
    --parallel \
    --format all
```

### Large Dataset Testing

Test with large datasets (treatment only):

```bash
# Skip baseline for datasets > 2000 points
python -m benchmarks.run_benchmark \
    --backend claude-code \
    --skip-baseline-above 2000 \
    --task scaling
```

## Error Handling

### Rate Limit Errors

If you hit rate limits:

1. Reduce parallelism:
   ```bash
   --trial-parallelism 2  # Instead of 4
   ```

2. Add delays between runs:
   ```bash
   # Wait 5 minutes between task types
   python -m benchmarks.run_benchmark --task statistical
   sleep 300
   python -m benchmarks.run_benchmark --task trend
   ```

3. Use quick mode:
   ```bash
   --quick  # 5 trials instead of 30
   ```

### Timeout Errors

For large datasets:

```bash
# Limit dataset size
--max-data-size 500

# Or skip baseline for large datasets
--skip-baseline-above 1000
```

### Consecutive Error Handling

The runner aborts after 3 consecutive API errors to prevent wasting quota:

```
ERROR: Aborting: 3 consecutive API errors. Total errors: 5.
```

**Fix:** Check CLI authentication, rate limits, or network connectivity.

## Architecture

### Parallelism Implementation

```
BenchmarkRunner
    |
    +-- run_all()
    |       |
    |       +-- warmup() [CLI backend only]
    |       |
    |       +-- for each task:
    |               |
    |               +-- run_task()
    |                       |
    |                       +-- BaseTask.run()

BaseTask.run()
    |
    +-- [if trial_parallelism > 1]
    |       |
    |       +-- ThreadPoolExecutor(max_workers=N)
    |               |
    |               +-- submit(_run_trial_wrapper) x N trials
    |
    +-- [for each trial]
            |
            +-- run_single_trial()
                    |
                    +-- [if parallel_workers > 1]
                    |       |
                    |       +-- ThreadPoolExecutor(max_workers=2)
                    |               |
                    |               +-- query_baseline() | query_treatment()
                    |
                    +-- [else: sequential]
                            |
                            +-- query_baseline()
                            +-- query_treatment()
```

### Client Interface

All backends implement the same interface:

```python
class ClientType(Protocol):
    def query(self, prompt: str, system: str | None = None) -> ClaudeResponse: ...
    def query_baseline(self, raw_data: str, query: str) -> ClaudeResponse: ...
    def query_treatment(self, semantic_output: str, query: str) -> ClaudeResponse: ...
```

## Troubleshooting

### "claude CLI not found"

Install Claude Code CLI:
```bash
npm install -g @anthropic-ai/claude-code
```

### "CLI timed out"

Reduce dataset size or increase timeout:
```bash
--max-data-size 500
```

### "Rate limit exceeded"

Reduce parallelism or wait for rate limit reset:
```bash
--trial-parallelism 2
# Or wait 5 minutes and retry
```

### "warmup failed"

CLI may need re-authentication:
```bash
claude  # Re-authenticate interactively
```

## See Also

- [Claude Code CLI Research](./claude_code_cli_research.md) - Detailed CLI capabilities
- [Benchmark Framework](../benchmarks/README.md) - Full benchmark documentation
- [CLAUDE.md](../CLAUDE.md) - Project configuration

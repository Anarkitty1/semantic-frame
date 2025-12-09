# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**semantic-frame** is a Python library that converts raw numerical data (NumPy, Pandas, Polars) into token-efficient natural language descriptions optimized for LLM consumption. Instead of sending thousands of data points to an AI agent, send a 50-word semantic summary.

Core value proposition: 95%+ token reduction, zero hallucination risk (deterministic math via NumPy/scipy, not LLM guesses).

## Build and Development Commands

```bash
# Install dependencies
uv sync

# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_analyzers.py

# Run single test
uv run pytest tests/test_analyzers.py::TestClassifyTrend::test_rising_sharp -v

# Type checking
uv run mypy semantic_frame

# Linting
uv run ruff check semantic_frame

# Format
uv run ruff format semantic_frame

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Install docs dependencies
uv sync --group docs

# Build documentation
uv run mkdocs build

# Serve docs locally (with live reload, runs on port 8001)
uv run mkdocs serve
```

## Pre-commit Hooks

Pre-commit hooks are configured for code quality. Install with:

```bash
uv run pre-commit install
```

Hooks run automatically on `git commit`:
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML syntax
- **check-added-large-files**: Prevent large file commits
- **check-merge-conflict**: Prevent committing conflict markers
- **ruff**: Linting with auto-fix
- **ruff-format**: Code formatting
- **mypy**: Type checking (excludes tests/)

## Architecture

The library follows a 4-stage pipeline:

```
Input (NumPy/Pandas/Polars) → Profiler → Classifier → Narrator → Output (text/json/SemanticResult)
```

### Module Structure

```
semantic_frame/
├── main.py              # Public API: describe_series(), describe_dataframe()
├── core/
│   ├── enums.py         # Semantic vocabulary (TrendState, VolatilityState, StructuralChange, etc.)
│   ├── analyzers.py     # Math engine (NumPy/scipy stats, no LLMs)
│   ├── correlations.py  # Cross-column correlation analysis (Pearson/Spearman)
│   └── translator.py    # Orchestrates pipeline: profile → analyze → narrate
├── narrators/
│   ├── time_series.py   # Generates narratives for ordered data
│   ├── distribution.py  # Generates narratives for unordered data
│   └── correlation.py   # Generates narratives for column relationships
├── interfaces/
│   ├── json_schema.py   # Pydantic models (SemanticResult, AnomalyInfo, etc.)
│   └── llm_templates.py # LangChain/agent integration helpers
└── integrations/
    ├── anthropic.py     # Native Anthropic Claude tool use (optional dep)
    ├── langchain.py     # LangChain BaseTool wrapper (optional dep)
    ├── crewai.py        # CrewAI tool decorator wrapper (optional dep)
    └── mcp.py           # Model Context Protocol server (optional dep)
```

### Data Flow

1. **main.py**: Entry point. Converts any input (Pandas/Polars/NumPy/list) to NumPy array via `_to_numpy()`
2. **translator.py**: `analyze_series()` runs full pipeline:
   - Builds `SeriesProfile` (basic stats: mean, median, std, min, max)
   - Runs analyzers: trend, volatility, anomalies, seasonality, distribution, step changes
   - Generates narrative via `narrators/time_series.py`
   - Returns `SemanticResult` with compression ratio
3. **analyzers.py**: Pure math functions, deterministic, no external calls:
   - `calc_linear_slope()` → normalized slope for trend
   - `classify_trend()` → maps slope to `TrendState` enum
   - `calc_volatility()` → coefficient of variation → `VolatilityState`
   - `detect_anomalies()` → adaptive Z-score/IQR based on sample size
   - `calc_seasonality()` → autocorrelation analysis
   - `calc_distribution_shape()` → skewness/kurtosis classification
   - `detect_step_changes()` → sliding window baseline shift detection
4. **correlations.py**: Cross-column relationship analysis:
   - `classify_correlation()` → maps r-value to `CorrelationState` enum
   - `calc_correlation_matrix()` → pairwise Pearson/Spearman correlations
   - `identify_significant_correlations()` → filters by threshold (default 0.5)

### Key Design Decisions

- **Enum thresholds are documented in enums.py docstrings** (e.g., RISING_SHARP = slope > 0.5)
- **Anomaly detection adapts to sample size**: IQR for <10 samples, Z-score for larger
- **All scipy warnings are suppressed** to avoid noise from edge cases
- **Infinite values are filtered with logging** (`translator.py:58-64`)
- **Compression ratio = 1 - (output_tokens / input_tokens)**, clamped to [0, 1]

## Testing Patterns

Tests are organized to mirror the module structure:
- `test_enums.py` - Enum value tests
- `test_analyzers.py` - Math function unit tests
- `test_correlations.py` - Correlation analysis tests
- `test_translator.py` - Pipeline integration tests
- `test_narrators.py` - Narrative generation tests
- `test_integration.py` - End-to-end with various data types
- `test_advanced_analyzers.py` - Step change detection and advanced analysis tests
- `test_anthropic_integration.py` - Anthropic native tool use tests
- `test_langchain_integration.py` - LangChain tool wrapper tests
- `test_crewai_integration.py` - CrewAI tool wrapper tests
- `test_mcp_integration.py` - MCP server integration tests
- `test_benchmarks.py` - Benchmark pipeline integration tests
- `test_benchmark_config.py` - Configuration validation tests
- `test_benchmark_runner.py` - Runner orchestration tests
- `test_benchmark_datasets.py` - Data generation tests
- `test_benchmark_metrics.py` - Metric calculation tests
- `test_benchmark_reporter.py` - Report generation tests
- `test_benchmark_claude_client.py` - API client tests
- `test_benchmark_tasks.py` - Task implementation tests

### Test Markers

```bash
# Run only benchmark tests
uv run pytest -m benchmark

# Exclude slow tests (default behavior)
uv run pytest

# Include slow tests (1M+ data points)
uv run pytest -m slow
```

When adding new analysis features:
1. Add enum to `core/enums.py` with threshold docstring
2. Add math function to `core/analyzers.py`
3. Integrate in `core/translator.py`
4. Update narrative in `narrators/time_series.py`
5. Add tests for each layer

## Benchmark Framework

The `benchmarks/` directory contains a framework for evaluating semantic-frame's effectiveness with LLMs.

### Running Benchmarks

```bash
# Full benchmark suite (requires ANTHROPIC_API_KEY)
python -m benchmarks.run_benchmark

# Run specific task
python -m benchmarks.run_benchmark --task statistical

# Quick validation (fewer trials)
python -m benchmarks.run_benchmark --quick

# Mock mode (no API calls, for testing pipeline)
python -m benchmarks.run_benchmark --mock

# Output formats
python -m benchmarks.run_benchmark --format json  # or csv, markdown, all
```

### Benchmark Architecture

```
benchmarks/
├── run_benchmark.py    # CLI entry point
├── config.py           # Configuration (TaskType, DataPattern, thresholds)
├── runner.py           # Orchestrates benchmark execution
├── claude_client.py    # Anthropic API wrapper
├── datasets.py         # Synthetic data generation
├── metrics.py          # Accuracy, F1, hallucination detection
├── reporter.py         # JSON/CSV/Markdown report generation
└── tasks/              # Task implementations
    ├── base.py         # BaseBenchmarkTask abstract class
    ├── statistical.py  # T1: Single-value extraction (mean, median, etc.)
    ├── trend.py        # T2: Trend classification
    ├── anomaly.py      # T3: Anomaly detection
    ├── comparative.py  # T4: Multi-series comparison
    ├── multi_step.py   # T5: Multi-step reasoning chains
    └── scaling.py      # T6: Large dataset handling
```

### Task Types

| Task | Code | Description |
|------|------|-------------|
| Statistical | T1 | Extract mean, median, std, percentiles |
| Trend | T2 | Classify trend direction/strength |
| Anomaly | T3 | Detect anomaly count/locations |
| Comparative | T4 | Compare multiple series |
| Multi-step | T5 | Chained reasoning tasks |
| Scaling | T6 | Handle 10K-100K data points |

### Key Thresholds (from `config.py`)

- Token compression: 90% minimum, 95% target
- Hallucination rate: <2% maximum
- Accuracy targets vary by task (see `MetricThresholds`)

### Tracking Results

Benchmark history is tracked in `benchmarks/BENCHMARK_HISTORY.md`. Update this file after significant changes to record milestone results and catch regressions.

Current best results (2025-12-09):
- Accuracy: Treatment outperforms baseline by +2.3%
- Hallucination: 2.3% (treatment) vs 4.5% (baseline)
- Token compression: 96.7%
- Cost savings: 91.3%

### Claude Code CLI Backend ✅ COMPLETE (Dec 2025)

**Goal:** Add `--backend claude-code` option to run benchmarks using Claude Code CLI instead of paid API. This enables free iteration during development (Max plan), with final validation through API.

**Usage:**
```bash
# Use Claude Code CLI backend (free on Max plan)
python -m benchmarks.run_benchmark --backend claude-code

# Use paid API (default)
python -m benchmarks.run_benchmark --backend api

# Use mock backend (no API calls, for testing)
python -m benchmarks.run_benchmark --backend mock
```

**Implementation Details:**
- `ClaudeCodeClient` class in `benchmarks/claude_client.py` with same interface as API client
- Uses `claude -p --output-format json --tools ""` for pure LLM responses
- Supports retry logic for transient errors (rate limits, timeouts)
- Model aliases: haiku, sonnet, opus mapped from config

**Rate Limits (Max plans):**
- Max 5x ($100/mo): 50-200 prompts per 5 hours
- Max 20x ($200/mo): 200-800 prompts per 5 hours

**Research documented in:** `docs/claude_code_cli_research.md`

## Framework Integrations

Optional dependencies for agent frameworks:
- **Anthropic**: `pip install semantic-frame[anthropic]` → `get_anthropic_tool()`, `handle_tool_call()` in `integrations/anthropic.py`
  - Native Claude tool use with Anthropic SDK
  - Provides tool schema and handler for messages API
- **LangChain**: `pip install semantic-frame[langchain]` → `get_semantic_tool()` in `integrations/langchain.py`
- **CrewAI**: `pip install semantic-frame[crewai]` → `get_crewai_tool()` in `integrations/crewai.py`
- **MCP**: `pip install semantic-frame[mcp]` → FastMCP server in `integrations/mcp.py`
  - Run: `mcp run semantic_frame.integrations.mcp:mcp`
  - Exposes `describe_data` tool for MCP clients (Claude Desktop, ElizaOS, Claude Code)
- **Claude Code**: Add as MCP server for native tool access in Claude Code CLI
  - Run: `claude mcp add semantic-frame -- uv run --project /path/to/semantic-frame mcp run /path/to/semantic-frame/semantic_frame/integrations/mcp.py`
  - Restart Claude Code, then use `mcp__semantic-frame__describe_data` tool

All integrations use lazy imports and provide helpful errors if dependencies are missing.

## Dependencies

Core: numpy, pandas, polars, scipy, pydantic
Dev: pytest, pytest-cov, pytest-asyncio, mypy, ruff, pre-commit
Optional: anthropic, langchain, crewai, mcp (for agent framework integrations)

## Releasing to PyPI

The project uses GitHub Actions with trusted publishing for automated PyPI releases.

### Release Workflow

1. **Merge feature branches** to `master` via PR
2. **Bump version** in two places:
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `semantic_frame/__init__.py`: `__version__ = "X.Y.Z"`
3. **Update CHANGELOG.md** with release notes
4. **Commit and push** to `master`
5. **Create GitHub Release**:
   - Go to https://github.com/Anarkitty1/semantic-frame/releases/new
   - Tag: `vX.Y.Z` (e.g., `v0.3.0`)
   - Title: `vX.Y.Z`
   - Description: Copy from CHANGELOG.md
   - Click "Publish release"

The GitHub Actions workflow (`.github/workflows/publish.yml`) automatically:
- Runs tests
- Builds the package
- Publishes to PyPI via trusted publishing (no API tokens needed)

### Manual Release (if needed)

```bash
# Run release script
./scripts/release.sh

# Upload to TestPyPI first
uv run twine upload --repository testpypi dist/*

# Upload to production PyPI
uv run twine upload dist/*
```

### PyPI Links

- Package: https://pypi.org/project/semantic-frame/
- Trusted Publishing: https://pypi.org/manage/project/semantic-frame/settings/publishing/

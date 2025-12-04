# Semantic Frame Commands Reference

Complete reference of all commands used in this project.

---

## Installation

### Core Package

```bash
# Install with pip
pip install semantic-frame

# Install with uv
uv add semantic-frame
```

### Optional Dependencies

```bash
# Anthropic Claude integration
pip install semantic-frame[anthropic]

# LangChain integration
pip install semantic-frame[langchain]

# CrewAI integration
pip install semantic-frame[crewai]

# MCP (Model Context Protocol) integration
pip install semantic-frame[mcp]

# Token validation (tiktoken)
pip install semantic-frame[validation]

# All optional dependencies
pip install semantic-frame[all]
```

---

## Development Setup

### Clone and Install

```bash
git clone https://github.com/Anarkitty1/semantic-frame
cd semantic-frame
uv sync
```

### Install with Extras

```bash
# Install dev dependencies (default group)
uv sync

# Install with specific optional extra
uv sync --extra anthropic
uv sync --extra langchain
uv sync --extra mcp

# Install all optional extras
uv sync --extra all

# Install docs dependencies
uv sync --group docs
```

### Pre-commit Hooks

```bash
# Install hooks (run once after cloning)
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files
```

---

## Testing

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_analyzers.py

# Run single test
uv run pytest tests/test_analyzers.py::TestClassifyTrend::test_rising_sharp -v

# Run with coverage report
uv run pytest --cov=semantic_frame

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "test_trend"
```

### Benchmark Tests

```bash
# Run benchmark tests (excludes slow 1M point tests)
uv run pytest tests/test_benchmarks.py -v --no-cov

# Run ALL benchmarks including slow 1M point tests
uv run pytest tests/test_benchmarks.py -v --no-cov -m "benchmark or slow"

# Run only slow tests (1M+ data points)
uv run pytest tests/test_benchmarks.py -v --no-cov -m "slow"

# Run specific benchmark class
uv run pytest tests/test_benchmarks.py::TestDescribeSeriesBenchmarks -v --no-cov
uv run pytest tests/test_benchmarks.py::TestMemoryBenchmarks -v --no-cov
```

---

## Code Quality

### Linting

```bash
# Check for linting errors
uv run ruff check semantic_frame

# Auto-fix linting errors
uv run ruff check --fix semantic_frame
```

### Formatting

```bash
# Format code
uv run ruff format semantic_frame

# Check formatting without changes
uv run ruff format --check semantic_frame
```

### Type Checking

```bash
# Run mypy type checker
uv run mypy semantic_frame
```

---

## Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Build documentation
uv run mkdocs build

# Serve docs locally (live reload on port 8001)
uv run mkdocs serve
```

---

## MCP Server

### Run MCP Server

```bash
# Run the MCP server
mcp run semantic_frame.integrations.mcp:mcp

# Alternative: run with uv
uv run mcp run semantic_frame/integrations/mcp.py
```

### Claude Code Integration

```bash
# Add MCP server to Claude Code
claude mcp add semantic-frame -- uv run --project /path/to/semantic-frame mcp run /path/to/semantic-frame/semantic_frame/integrations/mcp.py

# List configured MCP servers
claude mcp list

# Remove MCP server
claude mcp remove semantic-frame
```

---

## Git Commands

```bash
# Check status
git status

# Stage changes
git add .

# Commit with conventional format
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in analyzer"
git commit -m "test: add coverage for edge cases"
git commit -m "docs: update API documentation"
git commit -m "refactor: simplify translator logic"
git commit -m "chore: update dependencies"

# Push to remote
git push origin main
```

---

## UV Package Manager Reference

### Dependency Management

```bash
# Sync all dependencies (core + dev)
uv sync

# Add a new runtime dependency
uv add requests

# Add a dev dependency
uv add --dev pytest-mock

# Add to optional dependency group
uv add --optional anthropic "anthropic>=0.20"

# Remove a dependency
uv remove requests

# Update lockfile
uv lock

# Show dependency tree
uv tree
```

### Running Commands

```bash
# Run any command in the virtual environment
uv run <command>

# Examples
uv run python script.py
uv run pytest
uv run mypy semantic_frame
uv run ruff check .
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Run tests | `uv run pytest` |
| Run benchmarks | `uv run pytest tests/test_benchmarks.py -v --no-cov` |
| Run slow benchmarks | `uv run pytest tests/test_benchmarks.py -v --no-cov -m "benchmark or slow"` |
| Type check | `uv run mypy semantic_frame` |
| Lint | `uv run ruff check semantic_frame` |
| Format | `uv run ruff format semantic_frame` |
| Pre-commit | `uv run pre-commit run --all-files` |
| Build docs | `uv run mkdocs build` |
| Serve docs | `uv run mkdocs serve` |
| Run MCP | `mcp run semantic_frame.integrations.mcp:mcp` |

# Semantic Frame Benchmark Suite

A comprehensive benchmark for demonstrating **token reduction** and **accuracy gains** when using Semantic Frame for LLM numerical analysis tasks.

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY='your-key-here'

# Quick validation (5 trials, smaller datasets)
python -m benchmarks.run_benchmark --quick

# Full benchmark suite (30 trials per condition)
python -m benchmarks.run_benchmark

# Run specific task only
python -m benchmarks.run_benchmark --task statistical

# Test pipeline without API calls
python -m benchmarks.run_benchmark --mock
```

## What This Measures

The benchmark compares LLM performance under two conditions:

| Condition | Description |
|-----------|-------------|
| **Baseline** | Raw numerical data (JSON) passed directly to Claude |
| **Treatment** | Semantic Frame preprocessed output passed to Claude |

### Tasks

| Task | Description | Key Metric |
|------|-------------|------------|
| **T1: Statistical** | Single-value extraction (mean, std, percentiles) | Exact match accuracy |
| **T2: Trend** | Trend classification (rising/falling/flat) | Classification accuracy |
| **T3: Anomaly** | Anomaly detection | F1 score |
| **T4: Comparative** | Multi-series comparison | Comparison accuracy |
| **T5: Multi-step** | Multi-step reasoning chains | Chain accuracy |
| **T6: Scaling** | Performance at varying data sizes | Accuracy vs. scale |

### Metrics

- **Token Compression Ratio**: `1 - (treatment_tokens / baseline_tokens)`
- **Accuracy**: Proportion of correct answers vs. ground truth
- **Hallucination Rate**: Numerical claims not derivable from input
- **Cost Savings**: API cost reduction from token compression

## Expected Results

Based on LLM numerical reasoning research and Semantic Frame's architecture:

| Metric | Baseline | Treatment | Expected Improvement |
|--------|----------|-----------|---------------------|
| Statistical Accuracy | ~70% | ~95% | +25pp |
| Trend Classification | ~65% | ~90% | +25pp |
| Anomaly Detection F1 | ~55% | ~80% | +25pp |
| Token Compression | 0% | 90-95% | 90-95% reduction |
| Hallucination Rate | ~20% | <2% | >90% reduction |

## Output Files

Results are saved to `benchmarks/results/`:

- `benchmark_results.json` - Full structured results
- `benchmark_results.csv` - Tabular export for analysis
- `benchmark_report.md` - Human-readable report

## Architecture

```
benchmarks/
├── __init__.py           # Package exports
├── config.py             # Configuration and constants
├── datasets.py           # Synthetic data generation
├── metrics.py            # Evaluation metrics
├── claude_client.py      # Claude API wrapper
├── runner.py             # Main orchestration
├── reporter.py           # Report generation
├── run_benchmark.py      # CLI entry point
├── tasks/
│   ├── __init__.py
│   ├── base.py           # Base task class
│   ├── statistical.py    # T1: Statistical queries
│   ├── trend.py          # T2: Trend classification
│   ├── anomaly.py        # T3: Anomaly detection
│   ├── comparative.py    # T4: Multi-series comparison
│   └── scaling.py        # T6: Scale testing
├── data/                 # Generated/cached datasets
└── results/              # Benchmark outputs
```

## CLI Options

```
--task TASK       Run specific task only (statistical, trend, anomaly, comparative, scaling)
--quick           Quick mode: 5 trials, smaller datasets
--trials N        Override number of trials per condition
--mock            Mock mode: test pipeline without API calls
--output PATH     Custom output directory
--format FORMAT   Output format (json, csv, markdown, all)
--quiet           Suppress progress output
```

## Programmatic Usage

```python
from benchmarks import BenchmarkConfig, BenchmarkRunner, BenchmarkReporter

# Quick mode for development
config = BenchmarkConfig.quick_mode()

# Or full mode for publication
config = BenchmarkConfig.full_mode()

# Run benchmarks
runner = BenchmarkRunner(config)
results = runner.run_all()

# Generate reports
reporter = BenchmarkReporter(results, config)
reporter.generate_markdown_report("results.md")
reporter.print_comparison_table()
```

## Requirements

- Python 3.10+
- semantic-frame (this package)
- anthropic (`pip install anthropic`)
- tiktoken (optional, for accurate token counting)

## Adding New Tasks

1. Create a new task file in `benchmarks/tasks/`
2. Inherit from `BaseTask`
3. Implement `generate_datasets()`, `get_queries()`, `evaluate_answer()`
4. Register in `benchmarks/tasks/__init__.py`
5. Add to `TASK_CLASSES` in `benchmarks/runner.py`

## Notes

- API calls are made sequentially to respect rate limits
- Results are reproducible via fixed random seeds
- Mock mode allows testing the full pipeline without API costs
- Ground truth is computed deterministically (NumPy/SciPy)

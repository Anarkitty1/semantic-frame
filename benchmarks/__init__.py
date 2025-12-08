"""
Semantic Frame Benchmark Suite

A comprehensive benchmark for demonstrating token reduction and accuracy gains
when using Semantic Frame for LLM numerical analysis tasks.

Usage:
    # Run full benchmark suite
    python -m benchmarks.run_benchmark

    # Run specific task
    python -m benchmarks.run_benchmark --task statistical

    # Quick validation run
    python -m benchmarks.run_benchmark --quick

See benchmarks/README.md for full documentation.
"""

from benchmarks.config import BenchmarkConfig
from benchmarks.reporter import BenchmarkReporter
from benchmarks.runner import BenchmarkRunner

__all__ = ["BenchmarkConfig", "BenchmarkRunner", "BenchmarkReporter"]
__version__ = "0.1.0"

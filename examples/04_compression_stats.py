#!/usr/bin/env python3
"""Measure and demonstrate token compression efficiency.

Shows how semantic-frame achieves 95%+ compression by converting
thousands of data points into ~50 word summaries.

Install: pip install semantic-frame
"""

import numpy as np

from semantic_frame import describe_series
from semantic_frame.main import compression_stats


def main() -> None:
    print("=" * 70)
    print("Token Compression Analysis")
    print("=" * 70)
    print()
    print("semantic-frame converts raw numerical data into token-efficient")
    print("natural language descriptions. This example measures the compression.")
    print()

    # Test different dataset sizes
    sizes = [100, 1_000, 10_000, 100_000]

    header = f"{'Data Points':>12} | {'Original Tokens':>16} | "
    header += f"{'Narrative Tokens':>16} | {'Compression':>12}"
    print(header)
    print("-" * 70)

    for size in sizes:
        # Generate random data
        np.random.seed(42)
        data = np.random.normal(100, 15, size)

        # Analyze and get full result
        result = describe_series(data, context="Metric", output="full")

        # Get compression statistics
        stats = compression_stats(data, result)

        print(
            f"{stats['original_data_points']:>12,} | "
            f"{stats['original_tokens_estimate']:>16,} | "
            f"{stats['narrative_tokens']:>16} | "
            f"{stats['narrative_compression_ratio']:>11.1%}"
        )

    print()
    print("Note: Compression uses estimates (2 tokens/number, 1 token/word).")
    print("Actual compression varies by tokenizer (GPT-4, Claude, etc.).")
    print()

    # Show detailed stats for 10K dataset
    print("=" * 70)
    print("Detailed Analysis: 10,000 Data Points")
    print("=" * 70)
    print()

    np.random.seed(42)
    large_data = np.random.normal(100, 15, 10_000)

    result = describe_series(large_data, context="Server Response Time (ms)", output="full")
    stats = compression_stats(large_data, result)

    print("Input:")
    print(f"  Data points: {stats['original_data_points']:,}")
    print(f"  Estimated tokens: {stats['original_tokens_estimate']:,}")
    print()

    print("Output (narrative):")
    print(f"  Word count: {stats['narrative_tokens']}")
    print(f"  Compression: {stats['narrative_compression_ratio']:.1%}")
    print()
    print("  Text:")
    print(f"  {result.narrative}")
    print()

    print("Output (JSON):")
    print(f"  Token count: {stats['json_tokens']}")
    print(f"  Compression: {stats['json_compression_ratio']:.1%}")
    print()

    # Show what the raw data looks like vs summary
    print("=" * 70)
    print("Context Window Impact")
    print("=" * 70)
    print()
    print("Sending 10,000 numbers to an LLM consumes ~20,000 tokens.")
    print("At $0.01/1K tokens (Claude Sonnet), that's $0.20 per analysis.")
    print()
    print("With semantic-frame:")
    print(f"  - Narrative uses ~{stats['narrative_tokens']} tokens")
    print(f"  - Cost reduction: {stats['narrative_compression_ratio']:.1%}")
    print("  - Same analytical insights, fraction of the cost")


if __name__ == "__main__":
    main()

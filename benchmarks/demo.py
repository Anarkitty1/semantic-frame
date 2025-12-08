#!/usr/bin/env python3
"""
Semantic Frame Demo

Demonstrates the core value proposition: token compression with semantic preservation.
Shows a side-by-side comparison of raw data vs Semantic Frame output.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import count_tokens
from semantic_frame import describe_series


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def demo_compression() -> None:
    """Demonstrate token compression on sample data."""
    print_header("SEMANTIC FRAME TOKEN COMPRESSION DEMO")

    # Generate sample time series
    np.random.seed(42)

    datasets = {
        "Small (50 points)": np.random.normal(50, 10, 50),
        "Medium (500 points)": np.random.normal(50, 10, 500),
        "Large (5000 points)": np.random.normal(50, 10, 5000),
    }

    print("\n📊 Comparing raw JSON vs Semantic Frame output:\n")
    print(
        f"{'Dataset':<25} {'Raw Tokens':>12} {'SF Tokens':>12} {'Compression':>12} {'Savings':>10}"
    )
    print("-" * 75)

    for name, data in datasets.items():
        # Raw JSON
        raw_json = json.dumps(data.tolist())
        raw_tokens = count_tokens(raw_json)

        # Semantic Frame output
        sf_output = describe_series(data, context="Time Series")
        sf_tokens = count_tokens(sf_output)

        # Calculate metrics
        compression = 1 - (sf_tokens / raw_tokens)

        line = f"{name:<25} {raw_tokens:>12,} {sf_tokens:>12,}"
        line += f" {compression:>11.1%} {compression:>9.1%} ↓"
        print(line)

    print()

    # Show actual output comparison for medium dataset
    print_header("EXAMPLE: Medium Dataset (500 points)")

    data = datasets["Medium (500 points)"]
    raw_json = json.dumps(data.tolist()[:20])  # Just first 20 for display
    sf_output = describe_series(data, context="Sensor Readings")

    print("\n📝 RAW DATA (first 20 of 500 values):")
    print("-" * 50)
    print(raw_json[:500] + "..." if len(raw_json) > 500 else raw_json)

    print("\n✨ SEMANTIC FRAME OUTPUT:")
    print("-" * 50)
    print(sf_output)

    print("\n📈 KEY METRICS:")
    raw_token_count = count_tokens(json.dumps(data.tolist()))
    sf_token_count = count_tokens(sf_output)
    compression_ratio = 1 - sf_token_count / raw_token_count
    print(f"   • Raw tokens: {raw_token_count:,}")
    print(f"   • SF tokens: {sf_token_count:,}")
    print(f"   • Compression: {compression_ratio:.1%}")
    print("   • Information preserved: statistical properties, trends, anomalies")


def demo_anomaly_detection() -> None:
    """Demonstrate anomaly detection capability."""
    print_header("ANOMALY DETECTION CAPABILITY")

    np.random.seed(123)

    # Create data with obvious anomalies
    data = np.random.normal(50, 5, 100)
    data[25] = 95  # Spike
    data[75] = 10  # Drop

    sf_output = describe_series(data, context="Server CPU %")

    print("\n📊 Data with injected anomalies at indices 25 (spike) and 75 (drop)")
    print("\n✨ SEMANTIC FRAME OUTPUT:")
    print("-" * 50)
    print(sf_output)


def demo_trend_detection() -> None:
    """Demonstrate trend detection capability."""
    print_header("TREND DETECTION CAPABILITY")

    np.random.seed(456)

    trends = {
        "Rising": np.arange(100) * 0.5 + np.random.normal(0, 2, 100),
        "Falling": -np.arange(100) * 0.5 + 100 + np.random.normal(0, 2, 100),
        "Flat": np.ones(100) * 50 + np.random.normal(0, 5, 100),
        "Cyclical": 50 + 20 * np.sin(np.linspace(0, 4 * np.pi, 100)),
    }

    print("\n📊 Trend classification across different patterns:\n")

    for name, data in trends.items():
        sf_output = describe_series(data, context=f"{name} Pattern")
        # Extract just the first line or key info
        first_line = sf_output.split(".")[0] + "."
        print(f"  {name:<10}: {first_line}")


def demo_api_cost_savings() -> None:
    """Calculate potential API cost savings."""
    print_header("ESTIMATED API COST SAVINGS")

    # Anthropic Claude Sonnet pricing (approximate)
    input_cost_per_1k = 0.003  # $3 per 1M input tokens

    # Simulate enterprise usage
    queries_per_day = 10_000
    avg_data_points = 500
    days_per_month = 30

    # Generate sample data
    np.random.seed(789)
    sample_data = np.random.normal(50, 10, avg_data_points)

    raw_tokens = count_tokens(json.dumps(sample_data.tolist()))
    sf_tokens = count_tokens(describe_series(sample_data, context="Data"))

    # Monthly calculations
    monthly_queries = queries_per_day * days_per_month

    # Baseline cost (raw data)
    baseline_input_tokens = raw_tokens * monthly_queries
    baseline_cost = (baseline_input_tokens / 1000) * input_cost_per_1k

    # Treatment cost (Semantic Frame)
    treatment_input_tokens = sf_tokens * monthly_queries
    treatment_cost = (treatment_input_tokens / 1000) * input_cost_per_1k

    savings = baseline_cost - treatment_cost
    savings_pct = savings / baseline_cost * 100

    print(f"""
📊 Usage Scenario:
   • Queries per day: {queries_per_day:,}
   • Average data points per query: {avg_data_points:,}
   • Monthly queries: {monthly_queries:,}

💰 Cost Comparison (Input Tokens Only):

   BASELINE (Raw JSON):
   • Tokens per query: {raw_tokens:,}
   • Monthly tokens: {baseline_input_tokens:,}
   • Monthly cost: ${baseline_cost:,.2f}

   WITH SEMANTIC FRAME:
   • Tokens per query: {sf_tokens:,}
   • Monthly tokens: {treatment_input_tokens:,}
   • Monthly cost: ${treatment_cost:,.2f}

   💵 MONTHLY SAVINGS: ${savings:,.2f} ({savings_pct:.1f}%)
   💵 ANNUAL SAVINGS: ${savings * 12:,.2f}
""")


def main() -> None:
    """Run all demos."""
    print("\n" + "🚀" * 35)
    print("       SEMANTIC FRAME BENCHMARK DEMO")
    print("🚀" * 35)

    demo_compression()
    demo_anomaly_detection()
    demo_trend_detection()
    demo_api_cost_savings()

    print_header("NEXT STEPS")
    print("""
To run the full benchmark with Claude API:

    # Set your API key
    export ANTHROPIC_API_KEY='your-key-here'

    # Quick benchmark (5 trials)
    python -m benchmarks.run_benchmark --quick

    # Full benchmark (30 trials)
    python -m benchmarks.run_benchmark

    # Single task
    python -m benchmarks.run_benchmark --task statistical
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

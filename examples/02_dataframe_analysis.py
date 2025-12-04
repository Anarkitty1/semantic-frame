#!/usr/bin/env python3
"""Multi-column DataFrame analysis with correlations.

Demonstrates analyzing DataFrames to understand:
- Individual column behaviors
- Cross-column correlations
- System-wide patterns

Install: pip install semantic-frame pandas
"""

from pathlib import Path

import pandas as pd

from semantic_frame import describe_dataframe


def main() -> None:
    # Load sample metrics data
    data_path = Path(__file__).parent / "data" / "sample_metrics.csv"

    if data_path.exists():
        print("Loading sample_metrics.csv...")
        df = pd.read_csv(data_path)
        # Select numeric columns only
        numeric_cols = ["cpu_percent", "memory_percent", "latency_ms", "requests_per_sec"]
        df_numeric = df[numeric_cols]
    else:
        print("Creating synthetic server metrics...")
        # Create synthetic data if file doesn't exist
        import numpy as np

        np.random.seed(42)
        n = 24  # 24 hours

        # Correlated metrics: CPU and memory tend to move together
        base_load = np.sin(np.linspace(0, 2 * np.pi, n)) * 20 + 60

        df_numeric = pd.DataFrame(
            {
                "cpu_percent": base_load + np.random.normal(0, 5, n),
                "memory_percent": base_load * 0.9 + np.random.normal(0, 3, n),
                "latency_ms": base_load / 3 + np.random.normal(0, 2, n),
                "requests_per_sec": base_load * 30 + np.random.normal(0, 100, n),
            }
        )

    # Analyze the full DataFrame
    print("=" * 70)
    print("Full DataFrame Analysis (with correlations)")
    print("=" * 70)

    result = describe_dataframe(df_numeric, context="Server Metrics")

    # Print summary
    print(result.summary_narrative)
    print()

    # Print per-column analysis
    print("Per-Column Analysis:")
    for col_name, col_result in result.columns.items():
        print(f"  {col_name}: {col_result.narrative}")
    print()

    # Print correlations
    if result.correlations:
        print("Significant Correlations:")
        for corr in result.correlations:
            print(f"  {corr.narrative}")
    else:
        print("No significant correlations detected.")
    print()

    # Analyze with higher correlation threshold
    print("=" * 70)
    print("DataFrame Analysis (higher correlation threshold: 0.7)")
    print("=" * 70)

    result_high_thresh = describe_dataframe(df_numeric, correlation_threshold=0.7)
    print(result_high_thresh.summary_narrative)
    if result_high_thresh.correlations:
        print("Strong correlations:")
        for corr in result_high_thresh.correlations:
            print(f"  {corr.narrative}")
    print()

    # Analyze specific columns
    print("=" * 70)
    print("Selective Column Analysis (CPU vs Latency)")
    print("=" * 70)

    subset_df = df_numeric[["cpu_percent", "latency_ms"]]
    result_subset = describe_dataframe(subset_df, context="Performance")
    print(result_subset.summary_narrative)
    for corr in result_subset.correlations:
        print(f"  {corr.narrative}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Basic series analysis with semantic-frame.

Demonstrates analyzing single data series from various sources:
- Python lists
- NumPy arrays
- Pandas Series
- Polars Series

Install: pip install semantic-frame
"""

import numpy as np

from semantic_frame import describe_series


def main() -> None:
    # Example 1: Python list
    print("=" * 60)
    print("Example 1: Analyzing a Python list")
    print("=" * 60)

    cpu_readings = [45.2, 42.8, 48.5, 52.3, 78.2, 82.1, 65.8, 48.8]
    result = describe_series(cpu_readings, context="CPU Usage (%)")
    print(result)
    print()

    # Example 2: NumPy array with trend
    print("=" * 60)
    print("Example 2: NumPy array with rising trend")
    print("=" * 60)

    daily_sales = np.array([100, 120, 135, 150, 180, 210, 245, 280, 320, 365])
    result = describe_series(daily_sales, context="Daily Sales ($)")
    print(result)
    print()

    # Example 3: Data with anomalies
    print("=" * 60)
    print("Example 3: Server latency with spike (anomaly detection)")
    print("=" * 60)

    latency_ms = np.array(
        [12.5, 11.2, 10.8, 9.5, 8.9, 125.5, 10.2, 15.8, 22.1, 28.5]  # Spike at index 5
    )
    result = describe_series(latency_ms, context="Server Latency (ms)")
    print(result)
    print()

    # Example 4: Seasonal pattern
    print("=" * 60)
    print("Example 4: Hourly traffic with daily pattern")
    print("=" * 60)

    # Simulated hourly web traffic (2 days)
    hours = np.arange(48)
    traffic = 1000 + 500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 50, 48)
    result = describe_series(traffic, context="Hourly Visitors")
    print(result)
    print()

    # Example 5: Pandas Series (if available)
    try:
        import pandas as pd

        print("=" * 60)
        print("Example 5: Pandas Series")
        print("=" * 60)

        revenue = pd.Series([15000, 18500, 22000, 19500, 25000, 28000, 32000])
        result = describe_series(revenue, context="Weekly Revenue ($)")
        print(result)
        print()
    except ImportError:
        print("Skipping Pandas example (pandas not installed)")
        print()

    # Example 6: Polars Series (if available)
    try:
        import polars as pl

        print("=" * 60)
        print("Example 6: Polars Series")
        print("=" * 60)

        temps = pl.Series("temperature", [22.1, 22.3, 22.0, 35.5, 22.2, 22.4, 22.1])
        result = describe_series(temps, context="Room Temperature (C)")
        print(result)
    except ImportError:
        print("Skipping Polars example (polars not installed)")


if __name__ == "__main__":
    main()

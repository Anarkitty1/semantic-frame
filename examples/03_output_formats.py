#!/usr/bin/env python3
"""Compare output formats: text, JSON, and full structured.

Demonstrates the three output modes:
- "text": Natural language narrative (default)
- "json": Dictionary with all analysis fields
- "full": SemanticResult object with methods

Install: pip install semantic-frame
"""

import json

import numpy as np

from semantic_frame import describe_series


def main() -> None:
    # Sample data with interesting characteristics
    data = np.array(
        [
            100,
            105,
            110,
            108,
            115,
            120,
            118,
            125,
            130,
            128,
            500,  # Anomaly
            135,
            140,
            145,
            150,
        ]
    )
    context = "Daily Revenue ($)"

    # Output format 1: Text (default)
    print("=" * 70)
    print('Output Format: "text" (default)')
    print("=" * 70)

    text_result = describe_series(data, context=context, output="text")
    print(f"Type: {type(text_result).__name__}")
    print()
    print(text_result)
    print()

    # Output format 2: JSON dictionary
    print("=" * 70)
    print('Output Format: "json"')
    print("=" * 70)

    json_result = describe_series(data, context=context, output="json")
    print(f"Type: {type(json_result).__name__}")
    print()
    print("Keys:", list(json_result.keys()))
    print()
    print("Formatted JSON:")
    print(json.dumps(json_result, indent=2, default=str))
    print()

    # Output format 3: Full SemanticResult object
    print("=" * 70)
    print('Output Format: "full" (SemanticResult object)')
    print("=" * 70)

    full_result = describe_series(data, context=context, output="full")
    print(f"Type: {type(full_result).__name__}")
    print()

    # Access individual fields
    print("Individual Fields:")
    print(f"  - context: {full_result.context}")
    print(f"  - trend: {full_result.trend}")
    print(f"  - volatility: {full_result.volatility}")
    print(f"  - anomaly_state: {full_result.anomaly_state}")
    print(f"  - seasonality: {full_result.seasonality}")
    print(f"  - data_quality: {full_result.data_quality}")
    print(f"  - distribution: {full_result.distribution}")
    print(f"  - compression_ratio: {full_result.compression_ratio:.2%}")
    print()

    # Access statistics via profile
    print("Statistics (profile):")
    print(f"  - mean: {full_result.profile.mean:.2f}")
    print(f"  - median: {full_result.profile.median:.2f}")
    print(f"  - std: {full_result.profile.std:.2f}")
    print(f"  - min: {full_result.profile.min_val:.2f}")
    print(f"  - max: {full_result.profile.max_val:.2f}")
    print()

    # Access anomalies
    if full_result.anomalies:
        print("Anomalies:")
        for anomaly in full_result.anomalies:
            print(
                f"  - Index {anomaly.index}: value={anomaly.value}, z_score={anomaly.z_score:.2f}"
            )
    else:
        print("Anomalies: None detected")
    print()

    # Convert to JSON string
    print("Method: to_json_str()")
    json_str = full_result.to_json_str()
    print(f"Length: {len(json_str)} characters")
    print()

    # Get the narrative
    print("Field: narrative")
    print(full_result.narrative)


if __name__ == "__main__":
    main()

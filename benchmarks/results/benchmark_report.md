# Semantic Frame Benchmark Results

Generated: 2025-12-08 17:40:25
Model: claude-sonnet-4-20250514

## Executive Summary

- **Overall Accuracy Improvement**: 0.0%
- **Mean Token Compression**: 97.0%
- **Hallucination Reduction**: 0.0%
- **Estimated Cost Savings**: 95.1%

## Primary Results

| Metric | Baseline (95% CI) | Treatment (95% CI) | Improvement |
|--------|-------------------|--------------------| ------------|
| Accuracy | 4.5% | 4.5% | +0.0% |
| Token Compression | 0% | 97.0% | 97.0% ↓ |
| Hallucination Rate | 85.7% | 85.7% | +0.0% ↓ |
| API Cost | $3.2940 | $0.1605 | 95.1% ↓ |

## Results by Task

### Anomaly

- **Accuracy**: 0.0% → 3.3% (+3.3%)
- **Token Compression**: 97.2%
- **Hallucination Rate**: 100.0% → 100.0%
- **Trials**: 30

### Comparative

- **Accuracy**: 25.0% → 25.0% (+0.0%)
- **Token Compression**: 89.4%
- **Hallucination Rate**: 25.0% → 25.0%
- **Trials**: 8

### Scaling

- **Accuracy**: 0.0% → 4.2% (+4.2%)
- **Token Compression**: 95.9%
- **Hallucination Rate**: 100.0% → 100.0%
- **Trials**: 24

### Statistical

- **Accuracy**: 5.3% → 4.5% (-0.8%)
- **Token Compression**: 97.8%
- **Hallucination Rate**: 90.9% → 90.9%
- **Trials**: 132

### Trend

- **Accuracy**: 0.0% → 0.0% (+0.0%)
- **Token Compression**: 99.2%
- **Hallucination Rate**: 0.0% → 0.0%
- **Trials**: 10

## Methodology

This benchmark compares LLM performance on numerical analysis tasks under two conditions:

1. **Baseline**: Raw numerical data passed directly to Claude
2. **Treatment**: Semantic Frame preprocessed output passed to Claude

Each condition was tested with 2 trials per query type.
Accuracy is measured against deterministically computed ground truth.

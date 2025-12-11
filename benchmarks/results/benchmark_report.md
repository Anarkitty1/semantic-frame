# Semantic Frame Benchmark Results

Generated: 2025-12-11 14:25:10
Model: claude-sonnet-4-20250514

## Executive Summary

- **Overall Accuracy Improvement**: 15.5%
- **Mean Token Compression**: 96.5%
- **Hallucination Reduction**: 0.8%
- **Estimated Cost Savings**: 95.1%

## Primary Results

| Metric | Baseline (95% CI) | Treatment (95% CI) | Improvement |
|--------|-------------------|--------------------| ------------|
| Accuracy | 7.7% | 23.2% | +15.5% |
| Token Compression | 0% | 96.5% | 96.5% ↓ |
| Hallucination Rate | 14.8% | 14.0% | +0.8% ↓ |
| API Cost | $49.1229 | $2.4018 | 95.1% ↓ |

## Results by Task

### Anomaly

- **Accuracy**: 1.3% → 1.1% (-0.2%)
- **Token Compression**: 96.8%
- **Hallucination Rate**: 30.4% → 30.9%
- **Trials**: 450

### Comparative

- **Accuracy**: 22.5% → 18.3% (-4.2%)
- **Token Compression**: 88.0%
- **Hallucination Rate**: 0.0% → 0.0%
- **Trials**: 120

### Scaling

- **Accuracy**: 2.0% → 30.0% (+28.0%)
- **Token Compression**: 96.9%
- **Hallucination Rate**: 19.1% → 20.4%
- **Trials**: 540

### Statistical

- **Accuracy**: 4.2% → 19.7% (+15.5%)
- **Token Compression**: 96.7%
- **Hallucination Rate**: 14.0% → 11.8%
- **Trials**: 1320

### Trend

- **Accuracy**: 26.3% → 55.3% (+28.9%)
- **Token Compression**: 99.0%
- **Hallucination Rate**: 0.0% → 0.0%
- **Trials**: 570

## Methodology

This benchmark compares LLM performance on numerical analysis tasks under two conditions:

1. **Baseline**: Raw numerical data passed directly to Claude
2. **Treatment**: Semantic Frame preprocessed output passed to Claude

Each condition was tested with 30 trials per query type.
Accuracy is measured against deterministically computed ground truth.

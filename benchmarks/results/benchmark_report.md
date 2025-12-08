# Semantic Frame Benchmark Results

Generated: 2025-12-08 22:37:52
Model: claude-sonnet-4-20250514

## Executive Summary

- **Overall Accuracy Improvement**: -0.6%
- **Mean Token Compression**: 97.8%
- **Hallucination Reduction**: 3.9%
- **Estimated Cost Savings**: 96.3%

## Primary Results

| Metric | Baseline (95% CI) | Treatment (95% CI) | Improvement |
|--------|-------------------|--------------------| ------------|
| Accuracy | 5.5% | 4.8% | -0.6% |
| Token Compression | 0% | 97.8% | 97.8% ↓ |
| Hallucination Rate | 21.5% | 17.6% | +3.9% ↓ |
| API Cost | $6.5062 | $0.2381 | 96.3% ↓ |

## Results by Task

### Statistical

- **Accuracy**: 5.5% → 4.8% (-0.6%)
- **Token Compression**: 97.8%
- **Hallucination Rate**: 21.5% → 17.6%
- **Trials**: 330

## Methodology

This benchmark compares LLM performance on numerical analysis tasks under two conditions:

1. **Baseline**: Raw numerical data passed directly to Claude
2. **Treatment**: Semantic Frame preprocessed output passed to Claude

Each condition was tested with 5 trials per query type.
Accuracy is measured against deterministically computed ground truth.

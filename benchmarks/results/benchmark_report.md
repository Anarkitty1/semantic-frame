# Semantic Frame Benchmark Results

Generated: 2025-12-11 14:25:10
Model: claude-sonnet-4-20250514

## Executive Summary

- **Overall Accuracy Improvement**: 17.9%
- **Mean Token Compression**: 97.5%
- **Hallucination Reduction**: 2.4%
- **Estimated Cost Savings**: 96.2%

## Primary Results

| Metric | Baseline (95% CI) | Treatment (95% CI) | Improvement |
|--------|-------------------|--------------------| ------------|
| Accuracy | 4.2% | 22.1% | +17.9% |
| Token Compression | 0% | 97.5% | 97.5% ↓ |
| Hallucination Rate | 12.1% | 9.7% | +2.4% ↓ |
| API Cost | $6.5062 | $0.2441 | 96.2% ↓ |

## Results by Task

### Statistical

- **Accuracy**: 4.2% → 22.1% (+17.9%)
- **Token Compression**: 97.5%
- **Hallucination Rate**: 12.1% → 9.7%
- **Trials**: 330

## Methodology

This benchmark compares LLM performance on numerical analysis tasks under two conditions:

1. **Baseline**: Raw numerical data passed directly to Claude
2. **Treatment**: Semantic Frame preprocessed output passed to Claude

Each condition was tested with 5 trials per query type.
Accuracy is measured against deterministically computed ground truth.

# Benchmark History

This file tracks significant benchmark results over time to monitor improvements and catch regressions.

## How to Read Results

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Baseline % → Treatment % (change) | Treatment > Baseline |
| **Hallucination** | Baseline % → Treatment % | < 5% for both |
| **Compression** | Token reduction percentage | > 90% |
| **Cost Savings** | API cost reduction | > 90% |

---

## 2025-12-09 - Hallucination Detection Fix

**Commit:** `7698bf9` (chore: update gitignore and suppress scipy warning)
**Changes:** Improved hallucination detection to focus on Answer line only

### Statistical Task (88 trials, 1 trial per condition)

| Metric | Baseline | Treatment | Change |
|--------|----------|-----------|--------|
| Accuracy | 45.5% | 47.7% | **+2.3%** |
| Hallucination | 4.5% | 2.3% | **+2.3%** |
| Token Compression | 0% | 96.7% | 96.7% |
| Cost | $0.9116 | $0.0792 | **91.3% savings** |

**Notes:**
- Treatment now outperforms baseline on accuracy
- Hallucination rate dramatically reduced from previous ~70% false positive rate
- Fixed by focusing detection on `Answer:` line instead of all numbers in response

---

## 2025-12-09 - Mean Output Fix (Pre-hallucination fix)

**Commit:** `7c9ec06` (feat: add mean to semantic frame narrative output)
**Changes:** Added mean to semantic frame output alongside median

### Statistical Task (88 trials)

| Metric | Baseline | Treatment | Change |
|--------|----------|-----------|--------|
| Accuracy | 45.5% | 47.7% | **+2.3%** |
| Hallucination | 63.6% | 70.5% | -6.8% |
| Token Compression | 0% | 96.7% | 96.7% |
| Cost | $0.9127 | $0.0791 | **91.3% savings** |

**Notes:**
- First run after adding mean to output
- Accuracy improved but hallucination rate still high (detection issue, not actual hallucinations)

---

## 2025-12-09 - Initial Baseline (Before Fixes)

**Commit:** `e8a52c4` (Merge pull request #7)
**Changes:** None - baseline measurement

### Statistical Task (88 trials)

| Metric | Baseline | Treatment | Change |
|--------|----------|-----------|--------|
| Accuracy | 47.7% | 38.6% | **-9.1%** |
| Hallucination | 63.6% | 81.8% | -18.2% |
| Token Compression | 0% | 97.0% | 97.0% |
| Cost | $0.9146 | $0.0829 | **90.9% savings** |

**Notes:**
- Treatment performed WORSE than baseline on accuracy
- Root cause: Semantic frame output showed "Baseline: {median}" which Claude misinterpreted as mean
- High hallucination rates were false positives from strict detection logic

---

## Running Benchmarks

```bash
# Full statistical benchmark
source .env && uv run python -m benchmarks.run_benchmark --task statistical --trials 3

# Quick validation (1 trial)
source .env && uv run python -m benchmarks.run_benchmark --task statistical --trials 1

# All tasks
source .env && uv run python -m benchmarks.run_benchmark --trials 3
```

## When to Update This File

Update after:
1. Significant code changes to semantic-frame output
2. Changes to benchmark methodology or metrics
3. Before/after major releases
4. When investigating performance regressions

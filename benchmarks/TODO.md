# Benchmarks Framework TODO

## High Priority

### Add Unit Tests
- [ ] `test_metrics.py` - Test token counting, accuracy metrics, aggregation
- [ ] `test_datasets.py` - Test synthetic data generation, anomaly injection
- [ ] `test_claude_client.py` - Test retry logic, mock client behavior
- [ ] `test_runner.py` - Test benchmark orchestration, result aggregation
- [ ] `test_tasks.py` - Test task implementations and evaluation logic

### Fix Type Errors (mypy)
- [ ] Add `-> None` return types to functions missing annotations
  - `config.py:131` - `__post_init__`
  - `datasets.py:45,59` - `__post_init__`, `reset_seed`
  - `claude_client.py:37` - `_initialize_client`
  - `demo.py:22,29,84,103` - various functions
- [ ] Fix numpy floating type assignments in `metrics.py:312-323`
  - Use `float()` wrapper on numpy results
- [ ] Fix `AnomalyDataset` mutable default fields (`datasets.py:42-43`)
  - Use `field(default_factory=list)` instead of `= None`

### Enable Hallucination Detection
Currently disabled in `tasks/base.py:159-164`. To enable:
1. [ ] Add `raw_data: list[float]` field to `TaskResult` dataclass
2. [ ] Pass `dataset.data.tolist()` when creating `TaskResult` in `run_single_trial()`
3. [ ] Uncomment `detect_hallucination` import
4. [ ] Update `convert_to_trial_result()` to call `detect_hallucination()` with actual data

## Medium Priority

### Add Input Validation
- [ ] Add `__post_init__` validation to `BenchmarkConfig`
  - `n_trials > 0`
  - `retry_attempts > 0`
  - `retry_delay >= 0`
- [ ] Add `__post_init__` validation to `DatasetConfig`
  - `small_size < medium_size < large_size < very_large_size`
  - `min_variables <= max_variables`
- [ ] Add `__post_init__` validation to `MetricThresholds`
  - All rates in `[0.0, 1.0]`
- [ ] Add input validation to dataset generators
  - `n > 0`
  - `low < high` for random generation

### Improve Type Safety
- [ ] Replace `task_type: str` with `TaskType` enum in `TrialResult`
- [ ] Replace `condition: str` with `Literal["baseline", "treatment"]` in `TrialResult`
- [ ] Type the `ground_truth` dict with `TypedDict` or dataclass
- [ ] Replace parallel lists in `AnomalyDataset` with `list[Anomaly]` structured type

### Improve Error Handling
- [ ] Catch specific API exceptions instead of broad `Exception` in `claude_client.py`
  - `anthropic.APIError`, `anthropic.RateLimitError`, `anthropic.APITimeoutError`
- [ ] Add file I/O error handling to `save_dataset()` and `load_dataset()`
- [ ] Add timeout parameter to API calls (config exists but unused)

## Low Priority

### Documentation Improvements
- [ ] Fix tiktoken docstring - it's GPT-4 tokenizer, not "Claude-compatible"
- [ ] Add source citations for accuracy threshold values
- [ ] Document Wilson score interval rationale in `metrics.py`
- [ ] Document anomaly injection magnitude choices (3-5 sigma)

### Code Quality
- [ ] Add logging framework instead of `print()` statements
- [ ] Add `tqdm` progress bar for long benchmark runs
- [ ] Consider frozen dataclasses for immutable result types
- [ ] Make tiktoken a required dependency (or warn loudly on fallback)

### Features
- [ ] Add CSV/JSON dataset export
- [ ] Add benchmark comparison tool (compare two runs)
- [ ] Add visualization/plotting support
- [ ] Add confidence interval display in reports

## Completed

- [x] Fix Python 3.9 compatibility (Union syntax) - Added `from __future__ import annotations`
- [x] Fix silent API failures - Added error logging
- [x] Fix silent aggregation errors - Always log failures
- [x] Document disabled hallucination detection - Clear TODO comments added

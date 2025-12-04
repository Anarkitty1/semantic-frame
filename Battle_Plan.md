# Battle Plan: Semantic-Frame

## Status: ENHANCEMENT PHASE COMPLETE

---

## 1. The Architectural Philosophy [COMPLETE]

Before writing code, we must agree on the Data Pipeline. The goal is Compression and Translation.

- [x] Input: Heavy Numerical Data (100KB+ of JSON/CSV).
- [x] Step 1 (The Profiler): Fast, deterministic math (NumPy/Polars).
- [x] Step 2 (The Classifier): Map math results to categorical Enums.
- [x] Step 3 (The Narrator): Inject Enums into logic-based sentence templates.
- [x] Output: High-Density Semantic Context (100 bytes of Text).

---

## 2. The Tech Stack [COMPLETE]

Keep it lightweight. This needs to be a library that people can pip install without bloating their project.

- [x] Core Logic: Python 3.9+
- [x] Math Engine: Pandas + Polars + NumPy (all three supported!)
- [x] Structure: Pydantic v2 (strict output schemas for Agents)
- [x] Package Manager: uv
- [x] Additional: scipy for advanced statistics (skewness, kurtosis, autocorrelation)

---

## 3. The Library Structure (File System) [COMPLETE]

Organize the code so it is modular.

```
semantic_frame/
├── __init__.py              # [x] Package entry point
├── main.py                  # [x] Entry point (describe_series, describe_dataframe)
├── py.typed                 # [x] PEP 561 type marker
├── core/
│   ├── __init__.py          # [x] Core exports
│   ├── analyzers.py         # [x] The Math (Stats, Trends, Z-scores, IQR, Seasonality)
│   ├── enums.py             # [x] The Dictionary (TrendState, VolatilityState, etc.)
│   └── translator.py        # [x] The Logic (Maps Math -> Enums -> Narrative)
├── narrators/
│   ├── __init__.py          # [x] Narrator exports
│   ├── time_series.py       # [x] Narration for ordered data (Sales, Logs, Price)
│   └── distribution.py      # [x] Narration for unordered data (Test scores, Ages)
├── interfaces/
│   ├── __init__.py          # [x] Interface exports
│   ├── llm_templates.py     # [x] Ready-made prompts for LangChain/Agents
│   └── json_schema.py       # [x] Pydantic output models (SemanticResult, etc.)
tests/
├── __init__.py              # [x] Test package
├── test_enums.py            # [x] Enum tests
├── test_analyzers.py        # [x] Analyzer tests
├── test_translator.py       # [x] Pipeline tests
├── test_narrators.py        # [x] Narrator tests
└── test_integration.py      # [x] End-to-end tests
```

---

## 4. Phase 1: The "Dictionary" (Day 1-2) [COMPLETE]

You cannot code the logic until you define the language. Define your Enums in enums.py.

- [x] TrendState: RISING_SHARP, RISING_STEADY, FLAT, FALLING_STEADY, FALLING_SHARP
- [x] VolatilityState: COMPRESSED, STABLE, MODERATE, EXPANDING, EXTREME
- [x] DataQuality: PRISTINE, GOOD, SPARSE, FRAGMENTED
- [x] AnomalyState: NONE, MINOR, SIGNIFICANT, EXTREME
- [x] SeasonalityState: NONE, WEAK, MODERATE, STRONG
- [x] DistributionShape: NORMAL, LEFT_SKEWED, RIGHT_SKEWED, BIMODAL, UNIFORM

---

## 5. Phase 2: The "Math Engine" (Day 3-5) [COMPLETE]

Create analyzers.py. This is where you use NumPy/Pandas. No LLMs here. Just pure math.

Key functions implemented:

- [x] `calc_linear_slope(series)`: Returns normalized slope for trend detection
- [x] `classify_trend(slope)`: Maps slope to TrendState enum
- [x] `calc_volatility(series)`: Coefficient of variation analysis
- [x] `detect_anomalies(series)`: Adaptive Z-Score/IQR method (IQR for <10 samples)
- [x] `classify_anomaly_state(anomalies)`: Maps anomaly count/severity to enum
- [x] `assess_data_quality(series)`: Missing value percentage analysis
- [x] `calc_distribution_shape(series)`: Skewness + Kurtosis analysis
- [x] `calc_seasonality(series)`: Autocorrelation-based cycle detection

---

## 6. Phase 3: The "Narrator" (Day 6-8) [COMPLETE]

This is your product's frontend. It takes the Enums and makes them human/LLM readable.

- [x] Time series narrator with template-based generation
- [x] Distribution narrator for unordered data
- [x] Context injection (user-provided labels)
- [x] Anomaly reporting with positions
- [x] Seasonality mentions
- [x] Data quality warnings
- [x] JSON output via Pydantic models
- [x] LLM integration templates (system prompts, LangChain format)

---

## 7. Phase 4: The MVP Release (Day 9-10) [COMPLETE]

Package it and verify the value proposition.

- [x] 107 tests passing
- [x] 93% code coverage
- [x] 0 warnings
- [x] pyproject.toml configured for PyPI
- [x] README.md with usage examples
- [x] Supports NumPy, Pandas, Polars, and Python lists

**Compression Verification:**
```
Data points: 10,000
Narrative: "The Sensor Readings data shows a flat/stationary pattern with stable
           variability. 5 anomalies detected at indices 5664, 7210, 4729 (+2 more).
           Baseline: 99.85 (range: 60.71-139.88)."
Compression: 99.9% token reduction
```

---

## 8. Strategic Roadmap to "Standardization"

### Milestone 1: The "Describer" (Week 1-2) [COMPLETE]

- [x] Works on a single Pandas Series (One column)
- [x] Can describe Trend, Volatility, and Outliers
- [x] Seasonality detection
- [x] Distribution shape analysis
- [x] Data quality assessment
- [x] Ready for GitHub release

### Milestone 2: The "Frame Worker" (Week 3-4) [COMPLETE]

- [x] Works on a whole DataFrame (`describe_dataframe()`)
- [x] Analyzes all numeric columns
- [x] Context propagation to columns
- [x] Multi-column agent context generation
- [x] Column correlation analysis
- [x] Ready for PyPI release (`pip install semantic-frame`)

### Milestone 3: The "Agent Kit" (Month 2) [COMPLETE]

- [x] LLM template helpers (`format_for_system_prompt`, `format_for_langchain`)
- [x] Agent context creation (`create_agent_context`)
- [x] LangChain tool wrapper (`semantic_frame/integrations/langchain.py`)
- [x] CrewAI integration (`semantic_frame/integrations/crewai.py`)
- [ ] ElizaOS integration (future)

---

## Immediate Next Steps [COMPLETE]

- [x] Initialize your project
- [x] Create a new repo structure
- [x] Define your enums.py first
- [x] Write the README before you write the code
- [x] Implement the full pipeline
- [x] Write comprehensive tests (107 tests, 93% coverage)
- [x] Fix all warnings

---

## 9. Completed Enhancements

### Milestone 2 Completion: Cross-Column Correlation [COMPLETE]

**Goal:** Detect relationships like "Sales UP, Inventory DOWN"

**Completed Tasks:**

1. **[x] New Enum: `CorrelationState`** (`semantic_frame/core/enums.py`)
   - STRONG_POSITIVE, MODERATE_POSITIVE, WEAK, MODERATE_NEGATIVE, STRONG_NEGATIVE

2. **[x] New Module: `semantic_frame/core/correlations.py`**
   - `calc_correlation_matrix(df)`: Pearson/Spearman correlation
   - `identify_significant_correlations(matrix, threshold=0.5)`: Filter meaningful pairs
   - `classify_correlation(r_value)`: Map to CorrelationState enum

3. **[x] New Pydantic Model: `CorrelationInsight`** (`semantic_frame/interfaces/json_schema.py`)

4. **[x] New Pydantic Model: `DataFrameResult`** (`semantic_frame/interfaces/json_schema.py`)

5. **[x] Updated `describe_dataframe()`** to return `DataFrameResult`

6. **[x] New Narrator: `semantic_frame/narrators/correlation.py`**

7. **[x] Tests: `tests/test_correlations.py`** (18 tests)

---

### Milestone 3 Completion: Framework Integrations [COMPLETE]

**Goal:** "Add Semantic Memory to your Agent in 1 line of code"

**Completed Tasks:**

1. **[x] LangChain Tool Wrapper** (`semantic_frame/integrations/langchain.py`)
   - `SemanticAnalysisTool` class
   - `get_semantic_tool()` factory function

2. **[x] CrewAI Tool** (`semantic_frame/integrations/crewai.py`)
   - `semantic_analysis()` function
   - `get_crewai_tool()` factory function

3. **[x] Optional Dependencies** in `pyproject.toml`
   ```bash
   pip install semantic-frame[langchain]
   pip install semantic-frame[crewai]
   pip install semantic-frame[all]
   ```

4. **[x] Integration Tests**
   - `tests/test_langchain_integration.py` (12 tests)
   - `tests/test_crewai_integration.py` (8 tests)

5. **[x] LICENSE file** (MIT)

---

## 10. Anthropic Advanced Tool Use Integration [COMPLETE]

**Goal:** Support Anthropic's Advanced Tool Use features for production-scale agent deployments.

**Reference:** [Anthropic Advanced Tool Use Blog](https://www.anthropic.com/engineering/advanced-tool-use)

**Completed Tasks:**

1. **[x] Tool Use Examples** (`semantic_frame/integrations/anthropic.py`)
   - 5 curated examples covering anomalies, trends, JSON output, minimal input, volatile data
   - +18% parameter accuracy per Anthropic's testing
   - Included by default via `get_anthropic_tool()`

2. **[x] Deferred Loading Support**
   - `get_tool_for_discovery()` returns tool with `defer_loading=True`
   - Enables Tool Search discovery for 1000+ tool agents
   - Keeps context window lean

3. **[x] Programmatic Tool Calling**
   - `get_tool_for_batch_processing()` with `allowed_callers=["code_execution"]`
   - `handle_batch_tool_calls()` for parallel processing
   - Enables Claude to call tool from code for batch analysis

4. **[x] Convenience Functions**
   - `get_advanced_tool()` - All features enabled
   - `AnthropicSemanticTool` class with batch handling

5. **[x] MCP Wrapper Utilities** (`semantic_frame/integrations/mcp_wrapper.py`)
   - `@wrap_numeric_output()` decorator for any function
   - `transform_to_semantic()` for one-off transformations
   - `SemanticMCPWrapper` class for MCP server integration
   - Dynamic context extraction from dict keys

6. **[x] Comprehensive Documentation**
   - `docs/advanced-tool-use.md` - Full integration guide
   - Updated README with Advanced Tool Use section
   - API reference and best practices

7. **[x] Tests**
   - `tests/test_anthropic_integration.py` - 58 tests for advanced features
   - `tests/test_mcp_wrapper.py` - 35 tests for wrapper utilities

---

## 11. Future Enhancements

| Priority | Task | Status |
|----------|------|--------|
| 1 | ElizaOS integration | Not Started |
| 2 | Think Tool integration pattern | Not Started |
| 3 | PyPI release with advanced features | Ready |

---

## Test Results

- **301 tests passing** (3 skipped - optional deps not installed)
- **89% code coverage**
- **All linting passes** (ruff)
- **Package builds successfully**

---

**You have built the Compression Algorithm for Intelligence.**

**Status: ADVANCED TOOL USE COMPLETE**

# Changelog

All notable changes to Semantic Frame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-12-12

### Added

#### Enhanced Anomaly Detection (`describe_anomalies`)
Trading-optimized anomaly analysis with rich context and classification:

- **Severity Classification**: MILD, MODERATE, SEVERE, EXTREME (based on z-score magnitude)
- **Type Classification**: SPIKE, DROP, GAIN, LOSS, OUTLIER_HIGH, OUTLIER_LOW
- **Frequency Assessment**: RARE (<1%), OCCASIONAL (1-3%), FREQUENT (3-5%), PERVASIVE (>5%)
- **Per-anomaly Context**: Human-readable descriptions with deviation multiples
- **PnL Mode**: Use `is_pnl_data=True` for gain/loss terminology
- **Configurable Threshold**: Adjustable z-score threshold (default 2.0)

Example output:
```
"The Trade PnL has occasional anomalies (2 detected in 100 points).
 1 extreme outlier detected - investigate immediately.
 Most significant: index 45 (value: -5000.00, z-score: -4.2, 4.1x typical deviation, largest outlier)."
```

#### Time-Windowed Multi-Timeframe Analysis (`describe_windows`)
Compare trends and volatility across multiple time horizons:

- **Multi-Window Analysis**: Analyze any combination of window sizes (e.g., [10, 50, 200])
- **Timeframe Signals**: STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
- **Alignment Detection**: ALIGNED_BULLISH, ALIGNED_BEARISH, MIXED, DIVERGING, CONVERGING
- **Per-Window Metrics**: Trend direction/strength, volatility level, change %, high/low/range
- **Noise Assessment**: Low/moderate/high based on short-term vs long-term volatility ratio
- **Action Suggestions**: Contextual positioning recommendations based on multi-TF alignment

Example output:
```
"Multi-timeframe analysis of BTC: all timeframes bullish.
 Windows: 10 rising (+5.4%), 50 rising (+18.0%). Noise level: low.
 Suggested: strong buy signal across all timeframes."
```

#### MCP Tools for Phase 2
- `describe_anomalies` - Enhanced anomaly detection with severity/type classification
- `describe_windows` - Multi-timeframe trend and volatility analysis

### Tests
- Added 51 new tests for Phase 2 features
- Enhanced anomaly tests: severity thresholds, type classification, PnL mode, narrative generation
- Time window tests: alignment detection, signal classification, noise assessment, cross-window insights
- MCP integration tests for new tools
- Total: **1,262 tests passing with 94% coverage**

## [0.3.0] - 2025-12-12

### Added

#### Trading Module (`semantic_frame.trading`)
A complete trading intelligence toolkit for agent-based trading systems and financial analysis.

##### Drawdown Analysis (`describe_drawdown`)
- Maximum drawdown percentage and duration calculation
- Drawdown period tracking with start, trough, end indices
- Severity classification: MINIMAL, MODERATE, SIGNIFICANT, SEVERE, CATASTROPHIC
- Recovery state detection: AT_HIGH, RECOVERING, IN_DRAWDOWN, FULLY_RECOVERED
- Natural language narratives for equity curve risk assessment

##### Trading Performance Metrics (`describe_trading_performance`)
- Win rate, profit factor, expectancy calculations
- Average win/loss and risk-reward ratio
- Streak analysis (max consecutive wins/losses, current streak)
- Risk-adjusted metrics: Sharpe ratio, Sortino ratio, Calmar ratio
- Performance rating: EXCELLENT, GOOD, AVERAGE, BELOW_AVERAGE, POOR
- Risk profile classification: CONSERVATIVE, MODERATE, AGGRESSIVE, VERY_AGGRESSIVE
- Consistency rating: HIGHLY_CONSISTENT, CONSISTENT, INCONSISTENT, ERRATIC

##### Multi-Agent Rankings (`describe_rankings`)
- Compare multiple trading agents/strategies simultaneously
- Rankings by: total return, risk-adjusted return (Sharpe), volatility, max drawdown
- Composite scoring for overall leader identification
- Per-agent detailed rankings with win rate integration
- Natural language comparison narratives

#### MCP Trading Tools
All trading functions exposed as MCP tools:
- `describe_drawdown` - Equity curve drawdown analysis
- `describe_trading_performance` - Trade PnL performance metrics
- `describe_rankings` - Multi-agent comparative analysis

### Tests
- Added 80 new tests for trading module
- Test coverage for enums, schemas, drawdown, metrics, rankings, and MCP integration
- Total: **1,202 tests passing with 95% coverage**

## [0.2.1] - 2025-12-10

### Fixed
- Version bump for PyPI release consistency

## [0.2.0] - 2025-12-04

### Added

#### Anthropic Advanced Tool Use Support
- **Tool Use Examples**: 5 curated input examples for +18% parameter accuracy
- **Deferred Loading**: `defer_loading=True` for Tool Search discovery in 1000+ tool agents
- **Programmatic Tool Calling**: `allowed_callers=["code_execution"]` for batch analysis
- New convenience functions:
  - `get_tool_for_discovery()` - Tool Search optimized
  - `get_tool_for_batch_processing()` - Code execution enabled
  - `get_advanced_tool()` - All features enabled
  - `handle_batch_tool_calls()` - Parallel processing support

#### MCP Wrapper Utilities
- New module: `semantic_frame/integrations/mcp_wrapper.py`
- `@wrap_numeric_output()` decorator for automatic semantic transformation
- `transform_to_semantic()` function for one-off transformations
- `SemanticMCPWrapper` class for MCP server integration
- Dynamic context extraction from dict keys via `context_key` parameter

#### Documentation
- New guide: `docs/advanced-tool-use.md` with comprehensive integration examples
- Updated README with Advanced Tool Use section
- API reference for all new functions and classes

### Changed
- Upgraded from Alpha to Beta status
- `AnthropicSemanticTool` class now supports all advanced options
- Improved tool schema description for better Tool Search discovery

### Tests
- Added 35 new tests for MCP wrapper utilities
- Added 25 new tests for Advanced Tool Use features
- Total: 301 tests passing with 89% coverage

## [0.1.0] - 2025-11-15

### Added

#### Core Features
- `describe_series()` - Analyze single data series
- `describe_dataframe()` - Analyze all numeric columns with correlation detection
- Support for NumPy, Pandas, Polars, and Python lists

#### Analysis Capabilities
- Trend detection (RISING_SHARP, RISING_STEADY, FLAT, FALLING_STEADY, FALLING_SHARP)
- Volatility classification (COMPRESSED, STABLE, MODERATE, EXPANDING, EXTREME)
- Anomaly detection with adaptive Z-score/IQR
- Seasonality detection via autocorrelation
- Distribution shape analysis (NORMAL, LEFT_SKEWED, RIGHT_SKEWED, BIMODAL, UNIFORM)
- Step change detection (STEP_UP, STEP_DOWN, NONE)
- Data quality assessment (PRISTINE, GOOD, SPARSE, FRAGMENTED)
- Cross-column correlation analysis

#### Framework Integrations
- Anthropic Claude native tool use (`semantic_frame/integrations/anthropic.py`)
- LangChain BaseTool wrapper (`semantic_frame/integrations/langchain.py`)
- CrewAI tool decorator (`semantic_frame/integrations/crewai.py`)
- MCP server with FastMCP (`semantic_frame/integrations/mcp.py`)

#### Output Formats
- Text narratives (default)
- JSON structured output
- Full SemanticResult objects with compression ratio

### Technical
- 95%+ token compression for large datasets
- Deterministic math via NumPy/scipy (no hallucination risk)
- Pydantic v2 models for type safety
- Full type hints with py.typed marker

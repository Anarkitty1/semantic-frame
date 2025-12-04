# Changelog

All notable changes to Semantic Frame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

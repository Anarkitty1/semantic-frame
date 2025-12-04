# Semantic Frame

**Token-efficient semantic compression for numerical data.**

Semantic Frame converts raw numerical data (NumPy, Pandas, Polars) into natural language descriptions optimized for LLM consumption. Instead of sending thousands of data points to an AI agent, send a 50-word semantic summary.

## The Problem

LLMs are terrible at arithmetic. When you send raw data like `[100, 102, 99, 101, 500, 100, 98]` to GPT-4 or Claude:
- **Token waste**: 1000 data points = ~2000 tokens
- **Hallucination risk**: LLMs guess trends instead of calculating them
- **Context overflow**: Large datasets fill the context window

## The Solution

Semantic Frame provides **deterministic analysis** using NumPy, then translates results into **token-efficient narratives**:

```python
from semantic_frame import describe_series
import pandas as pd

data = pd.Series([100, 102, 99, 101, 500, 100, 98])
print(describe_series(data, context="Server Latency (ms)"))
```

Output:
```
The Server Latency (ms) data shows a flat/stationary pattern with stable
variability. 1 anomaly detected at index 4 (value: 500.00).
Baseline: 100.00 (range: 98.00-500.00).
```

**Result**: 95%+ token reduction, zero hallucination risk.

## Installation

```bash
pip install semantic-frame
```

Or with uv:
```bash
uv add semantic-frame
```

## Quick Start

### Analyze a Series

```python
from semantic_frame import describe_series
import numpy as np

# Works with NumPy arrays
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
result = describe_series(data, context="Daily Sales")
print(result)
# "The Daily Sales data shows a rapidly rising pattern with moderate variability..."
```

### Analyze a DataFrame

```python
from semantic_frame import describe_dataframe
import pandas as pd

df = pd.DataFrame({
    'cpu': [40, 42, 41, 95, 40, 41],
    'memory': [60, 61, 60, 60, 61, 60],
})

results = describe_dataframe(df, context="Server Metrics")
print(results['cpu'].narrative)
# "The Server Metrics - cpu data shows a flat/stationary pattern..."
```

### Get Structured Output

```python
result = describe_series(data, output="full")

print(result.trend)          # TrendState.RISING_SHARP
print(result.volatility)     # VolatilityState.MODERATE
print(result.anomalies)      # [AnomalyInfo(index=4, value=500.0, z_score=4.2)]
print(result.compression_ratio)  # 0.95
```

### JSON Output for APIs

```python
result = describe_series(data, output="json")
# Returns dict ready for JSON serialization
```

## Supported Data Types

- **NumPy**: `np.ndarray`
- **Pandas**: `pd.Series`, `pd.DataFrame`
- **Polars**: `pl.Series`, `pl.DataFrame`
- **Python**: `list`

## Analysis Features

| Feature | Method | Output |
|---------|--------|--------|
| **Trend** | Linear regression slope | RISING_SHARP, RISING_STEADY, FLAT, FALLING_STEADY, FALLING_SHARP |
| **Volatility** | Coefficient of variation | COMPRESSED, STABLE, MODERATE, EXPANDING, EXTREME |
| **Anomalies** | Z-score / IQR adaptive | Index, value, z-score for each outlier |
| **Seasonality** | Autocorrelation | NONE, WEAK, MODERATE, STRONG |
| **Distribution** | Skewness + Kurtosis | NORMAL, LEFT_SKEWED, RIGHT_SKEWED, BIMODAL, UNIFORM |
| **Step Change** | Baseline shift detection | NONE, STEP_UP, STEP_DOWN |
| **Data Quality** | Missing value % | PRISTINE, GOOD, SPARSE, FRAGMENTED |

## LLM Integration

### System Prompt Injection

```python
from semantic_frame.interfaces import format_for_system_prompt

result = describe_series(data, output="full")
prompt = format_for_system_prompt(result)
# Returns formatted context block for system prompts
```

### LangChain Tool Output

```python
from semantic_frame.interfaces import format_for_langchain

output = format_for_langchain(result)
# {"output": "narrative...", "metadata": {...}}
```

### Multi-Column Agent Context

```python
from semantic_frame.interfaces import create_agent_context

results = describe_dataframe(df)
context = create_agent_context(results)
# Combined narrative for all columns with attention flags
```

## Framework Integrations

### Anthropic Claude (Native Tool Use)

```bash
pip install semantic-frame[anthropic]
```

```python
import anthropic
from semantic_frame.integrations.anthropic import get_anthropic_tool, handle_tool_call

client = anthropic.Anthropic()
tool = get_anthropic_tool()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[tool],
    messages=[{"role": "user", "content": "Analyze this sales data: [100, 120, 115, 500, 118]"}]
)

# Handle tool use in response
for block in response.content:
    if block.type == "tool_use" and block.name == "semantic_analysis":
        result = handle_tool_call(block.input)
        print(result)
```

### LangChain

```bash
pip install semantic-frame[langchain]
```

```python
from semantic_frame.integrations.langchain import get_semantic_tool

tool = get_semantic_tool()
# Use as a LangChain BaseTool in your agent
```

### CrewAI

```bash
pip install semantic-frame[crewai]
```

```python
from semantic_frame.integrations.crewai import get_crewai_tool

tool = get_crewai_tool()
# Use with CrewAI agents
```

### MCP (Model Context Protocol)

```bash
pip install semantic-frame[mcp]
```

Run the MCP server:
```bash
mcp run semantic_frame.integrations.mcp:mcp
```

Exposes `describe_data` tool for MCP clients like:
- **ElizaOS**: TypeScript-based agent framework
- **Claude Desktop**: Anthropic's desktop app
- **Claude Code**: Anthropic's CLI for Claude
- Any MCP-compatible client

### Claude Code

Add Semantic Frame as a native tool in [Claude Code](https://claude.ai/code):

```bash
# Install MCP dependencies
pip install semantic-frame[mcp]

# Add MCP server to Claude Code
claude mcp add semantic-frame -- uv run --project /path/to/semantic-frame mcp run /path/to/semantic-frame/semantic_frame/integrations/mcp.py

# Restart Claude Code, then verify connection
claude mcp list
# semantic-frame: ... - ✓ Connected
```

Once configured, ask Claude to analyze data and it will use the `describe_data` tool automatically.

## Advanced Tool Use (Beta)

Semantic Frame supports [Anthropic's Advanced Tool Use features](https://www.anthropic.com/engineering/advanced-tool-use) for efficient tool orchestration in complex agent workflows.

### Features

| Feature | Benefit | API |
|---------|---------|-----|
| **Input Examples** | +18% parameter accuracy | Included by default |
| **Tool Search** | 1000+ tools without context bloat | `defer_loading=True` |
| **Programmatic Calling** | Batch analysis via code execution | `allowed_callers=["code_execution"]` |

### Quick Start (Advanced)

```python
import anthropic
from semantic_frame.integrations.anthropic import get_advanced_tool, handle_tool_call

client = anthropic.Anthropic()
tool = get_advanced_tool()  # All advanced features enabled

response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        {"type": "code_execution_20250825", "name": "code_execution"},
        tool,
    ],
    messages=[{"role": "user", "content": "Analyze all columns in this dataset..."}]
)
```

### Configuration Options

```python
from semantic_frame.integrations.anthropic import (
    get_anthropic_tool,          # Standard (includes examples)
    get_tool_for_discovery,      # For Tool Search
    get_tool_for_batch_processing,  # For code execution
    get_advanced_tool,           # All features enabled
)
```

### MCP Batch Analysis

```python
from semantic_frame.integrations.mcp import describe_batch

# Analyze multiple series in one call
result = describe_batch(
    datasets='{"cpu": [45, 47, 95, 44], "memory": [60, 61, 60, 61]}',
)
```

See [docs/advanced-tool-use.md](docs/advanced-tool-use.md) for complete documentation.

## Use Cases

### Crypto Trading
```python
btc_prices = pd.Series(hourly_btc_prices)
insight = describe_series(btc_prices, context="BTC/USD Hourly")
# "The BTC/USD Hourly data shows a rapidly rising pattern with extreme variability.
#  Step up detected at index 142. 2 anomalies detected at indices 89, 203."
```

### DevOps Monitoring
```python
cpu_data = pd.Series(cpu_readings)
insight = describe_series(cpu_data, context="CPU Usage %")
# "The CPU Usage % data shows a flat/stationary pattern with stable variability
#  until index 850, where a critical anomaly was detected..."
```

### Sales Analytics
```python
sales = pd.Series(daily_sales)
insight = describe_series(sales, context="Daily Revenue")
# "The Daily Revenue data shows a steadily rising pattern with weak cyclic pattern
#  detected. Baseline: $12,450 (range: $8,200-$18,900)."
```

### IoT Sensor Data
```python
temps = pl.Series("temperature", sensor_readings)
insight = describe_series(temps, context="Machine Temperature (C)")
# "The Machine Temperature (C) data is expanding with extreme outliers.
#  3 anomalies detected at indices 142, 156, 161."
```

## API Reference

### `describe_series(data, context=None, output="text")`

Analyze a single data series.

**Parameters:**
- `data`: Input data (NumPy array, Pandas Series, Polars Series, or list)
- `context`: Optional label for the data (appears in narrative)
- `output`: Format - `"text"` (string), `"json"` (dict), or `"full"` (SemanticResult)

**Returns:** Semantic description in requested format.

### `describe_dataframe(df, context=None)`

Analyze all numeric columns in a DataFrame.

**Parameters:**
- `df`: Pandas or Polars DataFrame
- `context`: Optional prefix for column context labels

**Returns:** Dict mapping column names to SemanticResult objects.

### `SemanticResult`

Full analysis result with:
- `narrative`: Human-readable text description
- `trend`: TrendState enum
- `volatility`: VolatilityState enum
- `data_quality`: DataQuality enum
- `anomaly_state`: AnomalyState enum
- `anomalies`: List of AnomalyInfo objects
- `seasonality`: Optional SeasonalityState
- `distribution`: Optional DistributionShape
- `step_change`: Optional StructuralChange (STEP_UP, STEP_DOWN, NONE)
- `step_change_index`: Optional int (index where step change occurred)
- `profile`: SeriesProfile with statistics
- `compression_ratio`: Token reduction ratio

## Development

```bash
# Clone and install
git clone https://github.com/yourusername/semantic-frame
cd semantic-frame
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=semantic_frame
```

## License

MIT License - see LICENSE file.

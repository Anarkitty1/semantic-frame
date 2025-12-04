# Advanced Tool Use Integration

This guide covers how to use Semantic Frame with Anthropic's **Advanced Tool Use** features (beta) for building powerful AI agents.

## Overview

Anthropic's Advanced Tool Use provides three key capabilities:

1. **Tool Search**: Discover tools on-demand instead of loading all upfront
2. **Programmatic Tool Calling**: Execute tools from code for batch processing
3. **Tool Use Examples**: Concrete examples for +18% parameter accuracy

Semantic Frame supports all three, making it ideal for agents that process numerical data at scale.

## Quick Start

### Standard Usage (No Beta Features)

```python
import anthropic
from semantic_frame.integrations.anthropic import (
    get_anthropic_tool,
    handle_tool_call,
)

client = anthropic.Anthropic()
tool = get_anthropic_tool()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[tool],
    messages=[{"role": "user", "content": "Analyze: [100, 102, 99, 500, 101]"}]
)

# Handle tool use
for block in response.content:
    if block.type == "tool_use" and block.name == "semantic_analysis":
        result = handle_tool_call(block.input)
        print(result)
```

### Advanced Usage (All Beta Features)

```python
import anthropic
from semantic_frame.integrations.anthropic import get_advanced_tool

client = anthropic.Anthropic()

response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        # Tool Search: Discover semantic_analysis when needed
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        # Code Execution: Enable batch processing
        {"type": "code_execution_20250825", "name": "code_execution"},
        # Semantic Frame with all advanced features
        get_advanced_tool(),
    ],
    messages=[{"role": "user", "content": "Analyze all columns in this dataset..."}]
)
```

## Feature Details

### 1. Tool Use Examples

Examples boost parameter accuracy from 72% to 90% (per Anthropic's testing). Semantic Frame includes 5 curated examples covering:

- Anomaly detection (spike in stable data)
- Clear upward trends
- JSON output format
- Minimal input (no context)
- High-volatility crypto data

```python
# Examples are included by default
tool = get_anthropic_tool()  # Has input_examples

# Disable if you need smaller context
tool = get_anthropic_tool(include_examples=False)
```

### 2. Deferred Loading (Tool Search)

For agents with 50+ tools, defer loading keeps context lean:

```python
from semantic_frame.integrations.anthropic import get_tool_for_discovery

tool = get_tool_for_discovery()
# Returns: {"name": "semantic_analysis", ..., "defer_loading": True}
```

Claude discovers the tool via search when it needs data analysis, rather than loading the full schema upfront.

### 3. Programmatic Tool Calling

Enable Claude to call tools from code for batch processing:

```python
from semantic_frame.integrations.anthropic import (
    get_tool_for_batch_processing,
    handle_batch_tool_calls,
)

tool = get_tool_for_batch_processing()
# Returns: {"name": "semantic_analysis", ..., "allowed_callers": ["code_execution"]}
```

Claude can now write:

```python
# Claude's generated code
results = await asyncio.gather(*[
    semantic_analysis({"data": col, "context": name})
    for name, col in dataframe.items()
])
```

Handle batch results:

```python
inputs = [
    {"data": [1, 2, 3], "context": "Series A"},
    {"data": [4, 5, 6], "context": "Series B"},
]
results = handle_batch_tool_calls(inputs)
# Returns: ["The Series A data shows...", "The Series B data shows..."]
```

## MCP Wrapper Utilities

Wrap any data source to automatically transform numerical outputs:

### Decorator Pattern

```python
from semantic_frame.integrations.mcp_wrapper import wrap_numeric_output

@wrap_numeric_output(context="CPU Usage %")
def get_cpu_metrics():
    # Your existing logic
    return fetch_cpu_data()  # Returns [45, 47, 46, 95, 44]

# Now returns semantic narrative instead of raw numbers
print(get_cpu_metrics())
# "The CPU Usage % data shows a flat/stationary pattern..."
```

### Dynamic Context

```python
@wrap_numeric_output(context_key="metric_name")
def get_sensor_reading():
    return {
        "metric_name": "Temperature (C)",
        "values": [22.1, 22.3, 35.5, 22.0],
        "sensor_id": "temp-001"
    }

result = get_sensor_reading()
# Result: {
#   "metric_name": "Temperature (C)",
#   "values": [22.1, 22.3, 35.5, 22.0],
#   "sensor_id": "temp-001",
#   "semantic_narrative": "The Temperature (C) data shows... anomaly detected..."
# }
```

### Class-Based Wrapper

```python
from semantic_frame.integrations.mcp_wrapper import SemanticMCPWrapper

wrapper = SemanticMCPWrapper(
    default_context="Sensor Readings",
    default_format="text",
)

# Transform any data
result = wrapper.transform([100, 102, 99, 500, 101])

# Or use as decorator factory
@wrapper.wrap(context="Specific Metric")
def get_specific_data():
    return fetch_data()
```

## Best Practices

### 1. Choose the Right Tool Configuration

| Scenario | Configuration |
|----------|---------------|
| Simple agent, few tools | `get_anthropic_tool()` |
| Large tool library (50+) | `get_tool_for_discovery()` |
| Batch data processing | `get_tool_for_batch_processing()` |
| Complex agent, all features | `get_advanced_tool()` |

### 2. Context Labels Matter

Good context labels improve narrative quality:

```python
# Good
"Server Latency (ms)"
"BTC/USD Hourly"
"Q3 Revenue ($M)"

# Too generic
"Data"
"Values"
"Numbers"
```

### 3. Batch Processing for DataFrames

For multi-column analysis, use batch processing:

```python
# Instead of 20 sequential API calls...
for col in df.columns:
    response = client.messages.create(...)  # Slow!

# Use batch processing
tool = get_tool_for_batch_processing()
# Claude processes all columns in one code execution block
```

### 4. Combine with MCP Servers

Semantic Frame works as a compression layer for any MCP server:

```python
# Your MCP server's raw tool
@mcp.tool()
def get_exchange_rates():
    return {"EUR": [1.08, 1.09, 1.07, 1.10], "GBP": [1.26, 1.27, 1.25, 1.28]}

# Wrapped version with semantic output
@mcp.tool()
@wrap_numeric_output(context_key="currency")
def get_rates_semantic(currency: str):
    rates = fetch_rates(currency)
    return {"currency": currency, "values": rates}
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `get_anthropic_tool()` | Standard tool with examples |
| `get_tool_for_discovery()` | Tool with `defer_loading=True` |
| `get_tool_for_batch_processing()` | Tool with `allowed_callers=["code_execution"]` |
| `get_advanced_tool()` | All advanced features enabled |
| `handle_tool_call(input)` | Process single tool call |
| `handle_batch_tool_calls(inputs)` | Process multiple tool calls |

### Classes

| Class | Description |
|-------|-------------|
| `AnthropicSemanticTool` | Helper class for tool management |
| `SemanticMCPWrapper` | Wrapper for MCP data sources |

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@wrap_numeric_output()` | Transform function output to semantic |

## Troubleshooting

### "Tool not found" in Tool Search

Ensure your tool has good searchable terms in the description:

```python
# Good: Keywords like "analyze", "data", "statistics", "trend"
"Analyze numerical time series or distribution data..."

# Bad: Too vague
"Process numbers"
```

### Batch processing not working

Ensure code execution is enabled:

```python
response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    tools=[
        {"type": "code_execution_20250825", "name": "code_execution"},  # Required!
        get_tool_for_batch_processing(),
    ],
    ...
)
```

### Examples consuming too many tokens

Disable examples for simple use cases:

```python
tool = get_anthropic_tool(include_examples=False)
# Saves ~500 tokens
```

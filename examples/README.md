# Semantic Frame Examples

Runnable examples demonstrating semantic-frame usage patterns.

## Getting Started

Install the library:

```bash
pip install semantic-frame
```

For Anthropic integration examples:

```bash
pip install semantic-frame[anthropic]
```

## Examples

### Core Usage

| Example | Description |
|---------|-------------|
| [01_basic_series.py](01_basic_series.py) | Analyze single data series (NumPy, Pandas, Polars) |
| [02_dataframe_analysis.py](02_dataframe_analysis.py) | Multi-column analysis with correlations |
| [03_output_formats.py](03_output_formats.py) | Compare text, JSON, and full output formats |
| [04_compression_stats.py](04_compression_stats.py) | Measure token compression efficiency |

### Framework Integrations

| Example | Description | Extra Deps |
|---------|-------------|------------|
| [10_anthropic_tool.py](10_anthropic_tool.py) | Claude native tool use | `anthropic` |

#### Anthropic Integration Notes

The standard `get_anthropic_tool()` works with the regular Anthropic API. For the beta Advanced Tool Use features (input examples for +18% parameter accuracy), use:

```python
from semantic_frame.integrations.anthropic import get_anthropic_tool

# Standard API (default)
tool = get_anthropic_tool()

# Beta API with input examples
tool = get_anthropic_tool(include_examples=True)
response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-20250514",
    tools=[tool],
    ...
)
```

## Sample Data

The `data/` directory contains sample datasets:

- `sample_metrics.csv` - Server metrics (CPU, memory, latency)

## Running Examples

```bash
# Run basic example
python examples/01_basic_series.py

# Run Anthropic example (requires API key)
export ANTHROPIC_API_KEY="your-key"
python examples/10_anthropic_tool.py
```

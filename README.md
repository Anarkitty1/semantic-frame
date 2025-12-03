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
| **Distribution** | Skewness + Kurtosis | NORMAL, LEFT_SKEWED, RIGHT_SKEWED, BIMODAL |
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

## Use Cases

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

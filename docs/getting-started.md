# Getting Started

## Installation

Install the package using pip:

```bash
pip install semantic-frame
```

Or with `uv`:

```bash
uv add semantic-frame
```

## Basic Usage

### 1. Analyze a Single Series

The core function is `describe_series`. It accepts a list, NumPy array, or Pandas/Polars Series.

```python
from semantic_frame import describe_series

data = [10, 12, 11, 13, 12, 11, 100, 12]
narrative = describe_series(data, context="User Logins")

print(narrative)
# Output: "The User Logins data shows a flat/stationary pattern... 1 anomaly detected..."
```

### 2. Analyze a DataFrame

Use `describe_dataframe` to analyze an entire dataset at once.

```python
import pandas as pd
from semantic_frame import describe_dataframe

df = pd.DataFrame({
    "price": [100, 101, 102, 103, 104],
    "volume": [500, 520, 480, 510, 490]
})

results = describe_dataframe(df)

print(results["price"].narrative)
print(results["volume"].narrative)
```

### 3. Structured Output

If you need the raw stats instead of text, request the full object:

```python
result = describe_series(data, output="full")

print(result.trend)       # TrendState.FLAT
print(result.volatility)  # VolatilityState.STABLE
print(result.anomalies)   # List of anomalies
```

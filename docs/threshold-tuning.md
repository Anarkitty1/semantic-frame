# Threshold Tuning Guide

This guide explains how to customize semantic-frame's classification thresholds for domain-specific needs. While the default thresholds work well for general-purpose analysis, certain domains may benefit from tuning.

## Why Tune Thresholds?

The default thresholds in semantic-frame are calibrated for general numerical data. However, domain-specific data often has unique characteristics:

- **Financial data**: Stock volatility patterns differ from weather data volatility
- **IoT sensors**: Temperature sensors have different "anomaly" thresholds than network latency
- **Social metrics**: Engagement rates have different scale than manufacturing defect rates

## Current Thresholds Reference

### Trend Classification

Calculated using normalized linear regression slope over the data range.

| State | Threshold | Description |
|-------|-----------|-------------|
| `RISING_SHARP` | slope > 0.5 | Rapid growth pattern |
| `RISING_STEADY` | 0.1 < slope ≤ 0.5 | Consistent upward trend |
| `FLAT` | -0.1 ≤ slope ≤ 0.1 | No significant trend |
| `FALLING_STEADY` | -0.5 ≤ slope < -0.1 | Consistent decline |
| `FALLING_SHARP` | slope < -0.5 | Rapid decline |

**Tuning tips:**
- Narrow the FLAT range (e.g., ±0.05) for data where small trends matter
- Widen the FLAT range (e.g., ±0.2) for naturally noisy data
- Financial data may need tighter bounds to detect meaningful trends

### Acceleration Classification

Calculated using the second derivative of a polynomial fit, normalized by data range.

| State | Threshold | Description |
|-------|-----------|-------------|
| `ACCELERATING_SHARPLY` | second_deriv > 0.3 | Rapidly increasing rate of change |
| `ACCELERATING` | 0.1 < second_deriv ≤ 0.3 | Increasing rate of change |
| `STEADY` | -0.1 ≤ second_deriv ≤ 0.1 | Constant rate of change |
| `DECELERATING` | -0.3 ≤ second_deriv < -0.1 | Decreasing rate of change |
| `DECELERATING_SHARPLY` | second_deriv < -0.3 | Rapidly decreasing rate of change |

**Tuning tips:**
- Use acceleration to detect trend reversals (e.g., "rising but slowing down")
- Widen STEADY range for noisy data where second derivatives fluctuate
- Tighten thresholds for high-frequency trading or real-time systems

### Volatility Classification

Based on Coefficient of Variation (CV = standard deviation / mean).

| State | Threshold | Description |
|-------|-----------|-------------|
| `COMPRESSED` | CV < 0.05 | Extremely tight range (5%) |
| `STABLE` | 0.05 ≤ CV < 0.15 | Normal variation (5-15%) |
| `MODERATE` | 0.15 ≤ CV < 0.30 | Noticeable fluctuation (15-30%) |
| `EXPANDING` | 0.30 ≤ CV < 0.50 | High volatility (30-50%) |
| `EXTREME` | CV ≥ 0.50 | Chaotic/unpredictable (>50%) |

**Tuning tips:**
- Manufacturing tolerances: may need COMPRESSED at CV < 0.01
- Stock markets: STABLE might be CV < 0.20 (higher baseline volatility)
- Temperature data: seasonal variations may require EXPANDING threshold at 0.40

### Anomaly Detection

Uses adaptive Z-score or IQR method based on distribution characteristics.

| State | Condition | Description |
|-------|-----------|-------------|
| `NONE` | 0 outliers | No anomalies detected |
| `MINOR` | 1-2 outliers | Few isolated anomalies |
| `SIGNIFICANT` | 3-5 outliers | Pattern of anomalies |
| `EXTREME` | >5 outliers or z-score > 5 | Many or severe anomalies |

**Z-score threshold:** Default 3.0 (99.7% of normal distribution)

**Tuning tips:**
- Security monitoring: lower threshold (2.5) catches more potential threats
- Quality control: higher threshold (3.5) reduces false positives
- Time series with trends: consider residual-based anomaly detection

### Seasonality Classification

Based on peak autocorrelation at potential seasonal lags.

| State | Threshold | Description |
|-------|-----------|-------------|
| `NONE` | peak < 0.3 | No cyclic pattern |
| `WEAK` | 0.3 ≤ peak < 0.5 | Faint pattern detected |
| `MODERATE` | 0.5 ≤ peak < 0.7 | Clear cyclic behavior |
| `STRONG` | peak ≥ 0.7 | Highly predictable cycles |

**Tuning tips:**
- Retail data: may show STRONG at 0.6 due to weekly patterns
- Weather data: MODERATE threshold might be 0.6 for annual cycles
- Server load: hourly/daily patterns may need WEAK at 0.2

### Data Quality Classification

Based on percentage of missing values.

| State | Threshold | Description |
|-------|-----------|-------------|
| `PRISTINE` | < 1% missing | High quality data |
| `GOOD` | 1-5% missing | Good quality data |
| `SPARSE` | 5-20% missing | Sparse data |
| `FRAGMENTED` | > 20% missing | Fragmented data |

**Tuning tips:**
- Real-time streaming: FRAGMENTED might be at 10% due to packet loss
- Survey data: GOOD might extend to 10% missing (expected non-response)
- Medical data: PRISTINE might be < 0.1% for critical measurements

### Distribution Shape

Based on skewness and kurtosis analysis.

| State | Condition | Description |
|-------|-----------|-------------|
| `UNIFORM` | kurtosis < -1.2 and \|skewness\| < 0.3 | Uniform distribution |
| `BIMODAL` | kurtosis < -1 | Flat-topped (bimodal hint) |
| `NORMAL` | \|skewness\| < 0.5 | Normal/Gaussian distribution |
| `LEFT_SKEWED` | skewness < -0.5 | Left tail (negative skew) |
| `RIGHT_SKEWED` | skewness > 0.5 | Right tail (positive skew) |

**Tuning tips:**
- Financial returns: tighten NORMAL to \|skewness\| < 0.3
- Income data: expect RIGHT_SKEWED; adjust threshold to 1.0 for "extreme" skew
- Ratings data (1-5 scale): BIMODAL common, adjust kurtosis threshold

### Correlation Classification

Based on Pearson correlation coefficient.

| State | Threshold | Description |
|-------|-----------|-------------|
| `STRONG_POSITIVE` | r > 0.7 | Strongly correlated |
| `MODERATE_POSITIVE` | 0.4 < r ≤ 0.7 | Moderately correlated |
| `WEAK` | \|r\| ≤ 0.4 | Weakly related |
| `MODERATE_NEGATIVE` | -0.7 ≤ r < -0.4 | Inversely related |
| `STRONG_NEGATIVE` | r < -0.7 | Strongly inverse |

**Tuning tips:**
- Social science: MODERATE_POSITIVE at r > 0.3 (lower sample sizes)
- Physics/engineering: STRONG_POSITIVE might be r > 0.95
- Marketing analytics: consider r > 0.5 as actionable correlation

## Creating Custom Analyzers

You can create domain-specific analyzers by wrapping the core functions:

```python
from semantic_frame.core.analyzers import classify_trend, classify_volatility
from semantic_frame.core.enums import TrendState, VolatilityState
import numpy as np

def classify_financial_trend(slope: float) -> TrendState:
    """Financial-specific trend classification with tighter bounds."""
    # Financial data: smaller movements are significant
    if slope > 0.3:  # Default: 0.5
        return TrendState.RISING_SHARP
    if slope > 0.05:  # Default: 0.1
        return TrendState.RISING_STEADY
    if slope < -0.3:
        return TrendState.FALLING_SHARP
    if slope < -0.05:
        return TrendState.FALLING_STEADY
    return TrendState.FLAT

def classify_iot_volatility(cv: float) -> VolatilityState:
    """IoT sensor volatility with adjusted thresholds."""
    # Sensors: tighter tolerances expected
    if cv < 0.02:  # Default: 0.05
        return VolatilityState.COMPRESSED
    if cv < 0.10:  # Default: 0.15
        return VolatilityState.STABLE
    if cv < 0.20:  # Default: 0.30
        return VolatilityState.MODERATE
    if cv < 0.35:  # Default: 0.50
        return VolatilityState.EXPANDING
    return VolatilityState.EXTREME
```

## Domain-Specific Presets

Here are recommended threshold adjustments for common domains:

### Financial Markets

```python
FINANCIAL_THRESHOLDS = {
    "trend": {
        "sharp": 0.3,    # More sensitive to rapid moves
        "steady": 0.05,  # Smaller trends matter
    },
    "volatility": {
        "stable": 0.20,  # Markets naturally volatile
        "extreme": 0.60,
    },
    "anomaly_z": 2.5,    # Lower threshold for risk
    "correlation": 0.6,  # Lower for market relationships
}
```

### IoT / Sensor Data

```python
IOT_THRESHOLDS = {
    "trend": {
        "sharp": 0.6,    # Less sensitive (sensor drift)
        "steady": 0.15,
    },
    "volatility": {
        "stable": 0.10,  # Expect tight tolerance
        "extreme": 0.40,
    },
    "anomaly_z": 3.5,    # Higher to reduce false alarms
    "data_quality": {
        "sparse": 0.10,  # Lower tolerance for missing data
    },
}
```

### Social Media / Engagement

```python
SOCIAL_THRESHOLDS = {
    "trend": {
        "sharp": 0.8,    # Viral requires big moves
        "steady": 0.2,
    },
    "volatility": {
        "stable": 0.30,  # Engagement naturally varies
        "extreme": 0.80,
    },
    "anomaly_z": 3.0,    # Standard
    "seasonality": {
        "strong": 0.6,   # Weekly patterns common
    },
}
```

## Validating Custom Thresholds

Before deploying custom thresholds:

1. **Baseline testing**: Run on historical data with known patterns
2. **Compare outputs**: Check that classifications match domain expert expectations
3. **Edge case testing**: Verify behavior near threshold boundaries
4. **A/B testing**: Compare default vs custom on new data

```python
import numpy as np
from semantic_frame import describe_series

# Test data with known characteristics
test_cases = [
    ("steady_growth", np.linspace(100, 150, 100)),
    ("volatile", np.random.normal(100, 30, 100)),
    ("seasonal", 100 + 20 * np.sin(np.linspace(0, 4*np.pi, 100))),
]

for name, data in test_cases:
    result = describe_series(data, output="full")
    print(f"{name}: trend={result.trend}, volatility={result.volatility}")
    # Verify matches expectations
```

## Future: Configuration File Support

A future version may support YAML configuration:

```yaml
# semantic_frame.yaml (proposed)
thresholds:
  trend:
    rising_sharp: 0.3
    rising_steady: 0.05
    falling_steady: -0.05
    falling_sharp: -0.3

  volatility:
    compressed: 0.02
    stable: 0.10
    moderate: 0.20
    expanding: 0.35
    extreme: 0.50

  anomaly:
    z_score: 2.5
    iqr_multiplier: 1.5

  seasonality:
    weak: 0.25
    moderate: 0.45
    strong: 0.65
```

Until configuration file support is added, use the custom analyzer approach described above.

## See Also

- [Universal Dictionary](universal-dictionary.md) - Complete enum reference
- [API Reference](api.md) - Core function documentation
- [Integrations](integrations.md) - Framework-specific usage

# The Universal Dictionary

To ensure AI Agents can reliably "understand" data, we map mathematical properties to a standardized vocabulary. This is the **Universal Dictionary**.

Instead of raw numbers (Slope = 0.05), the Agent receives a semantic concept (`RISING_STEADY`).

## Trend

Calculated using Linear Regression (Slope).

| Enum | Description | Math Logic |
|------|-------------|------------|
| `RISING_SHARP` | Rapid growth | Slope > 0.5 (Normalized) |
| `RISING_STEADY` | Consistent growth | 0.1 < Slope <= 0.5 |
| `FLAT` | No significant trend | -0.1 <= Slope <= 0.1 |
| `FALLING_STEADY` | Consistent decline | -0.5 <= Slope < -0.1 |
| `FALLING_SHARP` | Rapid decline | Slope < -0.5 |

## Volatility

Calculated using Coefficient of Variation (CV = StdDev / Mean).

| Enum | Description | Math Logic |
|------|-------------|------------|
| `COMPRESSED` | Extremely tight range | CV < 0.05 |
| `STABLE` | Normal variation | 0.05 <= CV < 0.15 |
| `MODERATE` | Noticeable fluctuation | 0.15 <= CV < 0.30 |
| `EXPANDING` | High volatility | 0.30 <= CV < 0.50 |
| `EXTREME` | Chaotic / Unpredictable | CV >= 0.50 |

## Anomalies

Calculated using an adaptive approach:

- **Z-Score** (for normal distributions): Threshold > 3.0
- **IQR** (for skewed distributions): Threshold > 1.5 * IQR

| Enum | Description |
|------|-------------|
| `NONE` | No outliers detected |
| `MINOR` | 1-2 outliers |
| `SIGNIFICANT` | 3-5 outliers (requires attention) |
| `EXTREME` | >5 outliers or any with z-score > 5 |

## Seasonality

Calculated using Autocorrelation (ACF).

| Enum | Description | Math Logic |
|------|-------------|------------|
| `NONE` | No cyclic pattern | Peak autocorrelation < 0.3 |
| `WEAK` | Faint pattern detected | 0.3 <= Peak < 0.5 |
| `MODERATE` | Clear cyclic behavior | 0.5 <= Peak < 0.7 |
| `STRONG` | Highly predictable cycles | Peak >= 0.7 |

## Data Quality

Classification of data completeness based on missing values.

| Enum | Description | Math Logic |
|------|-------------|------------|
| `PRISTINE` | High quality data | < 1% missing |
| `GOOD` | Good quality data | 1-5% missing |
| `SPARSE` | Sparse data | 5-20% missing |
| `FRAGMENTED` | Fragmented data | > 20% missing |

## Distribution Shape

Based on skewness and kurtosis analysis.

| Enum | Description | Math Logic |
|------|-------------|------------|
| `UNIFORM` | Uniformly distributed | Kurtosis < -1.2 and \|Skewness\| < 0.3 |
| `BIMODAL` | Bimodal distribution | Kurtosis < -1 (flat-topped) |
| `NORMAL` | Normally distributed | \|Skewness\| < 0.5 |
| `LEFT_SKEWED` | Left-skewed | Skewness < -0.5 |
| `RIGHT_SKEWED` | Right-skewed | Skewness > 0.5 |

*Note: BIMODAL detection is heuristic based on kurtosis; true bimodality requires more sophisticated analysis (e.g., Hartigan's dip test).*

## Correlation

Classification of correlation strength between two variables (Pearson r).

| Enum | Description | Math Logic |
|------|-------------|------------|
| `STRONG_POSITIVE` | Strongly correlated | r > 0.7 |
| `MODERATE_POSITIVE` | Moderately correlated | 0.4 < r <= 0.7 |
| `WEAK` | Weakly related | \|r\| <= 0.4 |
| `MODERATE_NEGATIVE` | Inversely related | -0.7 <= r < -0.4 |
| `STRONG_NEGATIVE` | Strongly inverse | r < -0.7 |

## Acceleration

Classification of rate of change in trend (second derivative). Measures whether the trend is speeding up, slowing down, or constant.

Calculated using the second derivative of a polynomial fit, normalized by data range.

| Enum | Description | Math Logic |
|------|-------------|------------|
| `ACCELERATING_SHARPLY` | Rapidly increasing rate of change | second_deriv > 0.3 |
| `ACCELERATING` | Increasing rate of change | 0.1 < second_deriv ≤ 0.3 |
| `STEADY` | Constant rate of change | -0.1 ≤ second_deriv ≤ 0.1 |
| `DECELERATING` | Decreasing rate of change | -0.3 ≤ second_deriv < -0.1 |
| `DECELERATING_SHARPLY` | Rapidly decreasing rate of change | second_deriv < -0.3 |

**Use cases:**
- Detecting trend reversals ("rising but slowing down")
- Identifying momentum changes in financial data
- Predicting when a growth pattern will plateau

## Structural Change

Classification of structural baseline shifts in time series data, detected using step change analysis.

| Enum | Description |
|------|-------------|
| `NONE` | No significant baseline shift |
| `STEP_UP` | Sudden increase in baseline mean |
| `STEP_DOWN` | Sudden decrease in baseline mean |

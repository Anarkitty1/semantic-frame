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
| `STABLE` | Normal variation | 0.05 <= CV < 0.2 |
| `MODERATE` | Noticeable fluctuation | 0.2 <= CV < 0.5 |
| `EXPANDING` | High volatility | 0.5 <= CV < 1.0 |
| `EXTREME` | Chaotic / Unpredictable | CV >= 1.0 |

## Anomalies

Calculated using an adaptive approach:
*   **Z-Score** (for normal distributions): Threshold > 3.0
*   **IQR** (for skewed distributions): Threshold > 1.5 * IQR

| Enum | Description |
|------|-------------|
| `NONE` | No outliers detected |
| `MINOR` | 1-2 outliers |
| `SIGNIFICANT` | 3-5 outliers (requires attention) |
| `EXTREME` | >5 outliers (data quality issue or crisis) |

## Seasonality

Calculated using Autocorrelation (ACF).

| Enum | Description |
|------|-------------|
| `NONE` | No cyclic pattern |
| `WEAK` | Faint pattern detected |
| `MODERATE` | Clear cyclic behavior |
| `STRONG` | Highly predictable cycles |

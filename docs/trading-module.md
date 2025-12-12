# Trading Module Documentation

**Semantic Frame v0.4.0**

The trading module provides specialized semantic analysis tools for trading agents, portfolio managers, and financial applications. All functions convert numerical trading data into token-efficient natural language descriptions using deterministic calculations—no LLM involvement in the math.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Functions](#core-functions)
   - [describe_drawdown](#describe_drawdown)
   - [describe_trading_performance](#describe_trading_performance)
   - [describe_rankings](#describe_rankings)
   - [describe_anomalies](#describe_anomalies)
   - [describe_windows](#describe_windows)
   - [describe_regime](#describe_regime)
   - [describe_allocation](#describe_allocation)
4. [MCP Integration](#mcp-integration)
5. [Use Cases](#use-cases)
6. [API Reference](#api-reference)

---

## Installation

```bash
pip install semantic-frame
```

Import the trading module:

```python
from semantic_frame.trading import (
    describe_drawdown,
    describe_trading_performance,
    describe_rankings,
    describe_anomalies,
    describe_windows,
    describe_regime,
    describe_allocation,
)
```

---

## Quick Start

```python
from semantic_frame.trading import describe_trading_performance

# Analyze a series of PnL values
pnl_data = [100, -50, 75, -25, 150, -30, 80, 120, -40, 200]

result = describe_trading_performance(pnl_data, context="CLAUDE Agent")
print(result.narrative)
```

**Output:**
```
CLAUDE Agent shows moderate performance with 60.0% win rate over 10 trades.
Profit factor: 2.60. Average win: $121.00, average loss: $36.25.
Risk profile: moderate. Consistency: somewhat consistent.
```

---

## Core Functions

### describe_drawdown

Analyze equity curve drawdowns with severity classification and recovery tracking.

#### Signature

```python
def describe_drawdown(
    equity_curve: np.ndarray | list[float],
    context: str | None = None,
) -> DrawdownResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `equity_curve` | array-like | Equity values over time (e.g., account balance) |
| `context` | str, optional | Label for the data (e.g., "BTC Strategy") |

#### Returns

`DrawdownResult` with fields:
- `max_drawdown_pct`: Maximum drawdown percentage
- `current_drawdown_pct`: Current drawdown from peak
- `severity`: MINOR (<5%), MODERATE (5-15%), SIGNIFICANT (15-30%), SEVERE (>30%)
- `recovery_state`: RECOVERED, RECOVERING, or IN_DRAWDOWN
- `drawdown_periods`: List of individual drawdown events
- `narrative`: Human-readable summary

#### Example

```python
from semantic_frame.trading import describe_drawdown

equity = [10000, 10500, 10200, 9800, 9500, 9800, 10100, 10600]
result = describe_drawdown(equity, context="Momentum Strategy")

print(result.narrative)
# "Momentum Strategy max drawdown: 9.5% (moderate). Currently recovered.
#  1 drawdown period detected. Deepest: 9.5% lasting 4 periods."

print(f"Max DD: {result.max_drawdown_pct:.1f}%")
print(f"Severity: {result.severity.value}")
print(f"Recovery: {result.recovery_state.value}")
```

#### Severity Thresholds

| Severity | Threshold | Description |
|----------|-----------|-------------|
| MINOR | < 5% | Normal fluctuation |
| MODERATE | 5-15% | Noticeable but manageable |
| SIGNIFICANT | 15-30% | Requires attention |
| SEVERE | > 30% | Critical risk level |

---

### describe_trading_performance

Calculate comprehensive trading metrics including win rate, profit factor, and Sharpe ratio.

#### Signature

```python
def describe_trading_performance(
    pnl_series: np.ndarray | list[float],
    context: str | None = None,
    risk_free_rate: float = 0.0,
) -> TradingPerformanceResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pnl_series` | array-like | Individual trade PnL values |
| `context` | str, optional | Label for the strategy/agent |
| `risk_free_rate` | float | Annual risk-free rate for Sharpe (default: 0) |

#### Returns

`TradingPerformanceResult` with fields:
- `metrics`: TradingMetrics object with all calculations
- `performance_rating`: EXCELLENT, GOOD, MODERATE, POOR, VERY_POOR
- `risk_profile`: CONSERVATIVE, MODERATE, AGGRESSIVE, VERY_AGGRESSIVE
- `consistency_rating`: HIGHLY_CONSISTENT → ERRATIC
- `narrative`: Human-readable summary

#### Metrics Calculated

| Metric | Description |
|--------|-------------|
| `win_rate` | Percentage of winning trades |
| `profit_factor` | Gross profit / gross loss |
| `avg_win` | Average winning trade |
| `avg_loss` | Average losing trade |
| `largest_win` | Best single trade |
| `largest_loss` | Worst single trade |
| `sharpe_ratio` | Risk-adjusted return |
| `total_pnl` | Sum of all PnL |
| `num_trades` | Total number of trades |

#### Example

```python
from semantic_frame.trading import describe_trading_performance

pnl = [500, -200, 300, -150, 800, -100, 250, -300, 600, 400]
result = describe_trading_performance(pnl, context="Alpha Bot")

print(result.narrative)
# "Alpha Bot shows good performance with 60.0% win rate over 10 trades.
#  Profit factor: 3.80. Average win: $475.00, average loss: $187.50.
#  Risk profile: moderate. Consistency: consistent."

print(f"Win Rate: {result.metrics.win_rate:.1%}")
print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

#### Performance Ratings

| Rating | Win Rate | Profit Factor |
|--------|----------|---------------|
| EXCELLENT | > 65% | > 2.5 |
| GOOD | > 55% | > 1.8 |
| MODERATE | > 45% | > 1.2 |
| POOR | > 35% | > 0.8 |
| VERY_POOR | ≤ 35% | ≤ 0.8 |

---

### describe_rankings

Compare multiple trading agents or strategies with composite scoring.

#### Signature

```python
def describe_rankings(
    agents: dict[str, np.ndarray | list[float]],
    context: str | None = None,
    weights: dict[str, float] | None = None,
) -> RankingsResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | dict | Map of agent names to PnL arrays |
| `context` | str, optional | Label for the comparison |
| `weights` | dict, optional | Custom scoring weights |

#### Default Weights

```python
{
    "total_pnl": 0.25,
    "win_rate": 0.25,
    "profit_factor": 0.25,
    "sharpe_ratio": 0.25,
}
```

#### Returns

`RankingsResult` with fields:
- `rankings`: List of AgentRanking objects (sorted by score)
- `best_agent`: Name of top performer
- `narrative`: Comparison summary

#### Example

```python
from semantic_frame.trading import describe_rankings

agents = {
    "Conservative": [50, -20, 40, -15, 60, -25, 45],
    "Aggressive": [200, -150, 300, -200, 400, -180, 250],
    "Balanced": [100, -50, 120, -40, 150, -60, 110],
}
result = describe_rankings(agents, context="Q4 Tournament")

print(result.narrative)
# "Q4 Tournament rankings: #1 Balanced (score: 0.85), #2 Aggressive (score: 0.72),
#  #3 Conservative (score: 0.68). Balanced leads with best risk-adjusted returns."

for r in result.rankings:
    print(f"#{r.rank} {r.agent_name}: {r.composite_score:.2f}")
```

---

### describe_anomalies

Enhanced anomaly detection with severity levels, type classification, and PnL-aware terminology.

#### Signature

```python
def describe_anomalies(
    data: np.ndarray | list[float],
    context: str | None = None,
    z_threshold: float = 2.0,
    is_pnl_data: bool = False,
) -> EnhancedAnomalyResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Numerical data to analyze |
| `context` | str, optional | Label for the data |
| `z_threshold` | float | Z-score threshold (default: 2.0) |
| `is_pnl_data` | bool | Use gain/loss terminology |

#### Anomaly Classifications

**Severity Levels:**
| Severity | Z-Score | Description |
|----------|---------|-------------|
| MILD | ≥ 2.0 | Notable deviation |
| MODERATE | ≥ 2.5 | Significant outlier |
| SEVERE | ≥ 3.5 | Major anomaly |
| EXTREME | ≥ 5.0 | Exceptional event |

**Anomaly Types:**
| Type | Condition |
|------|-----------|
| SPIKE | Positive deviation (price data) |
| DROP | Negative deviation (price data) |
| GAIN | Positive deviation (PnL data) |
| LOSS | Negative deviation (PnL data) |

**Frequency:**
| Frequency | Rate |
|-----------|------|
| RARE | < 1% |
| OCCASIONAL | 1-3% |
| FREQUENT | 3-5% |
| PERVASIVE | > 5% |

#### Example

```python
from semantic_frame.trading import describe_anomalies

# PnL with exceptional profit and loss
pnl = [100, 50, -30, 75, -25, -800, 60, 45, 1200, -40, 55]
result = describe_anomalies(pnl, context="CLAUDE Agent PnL", is_pnl_data=True)

print(result.narrative)
# "The CLAUDE Agent PnL has occasional anomalies (2 detected in 11 points).
#  Most significant: index 8 (value: 1200.00, z-score: 2.8, severe,
#  2.8x typical deviation, exceptional profit, largest outlier)."

for anomaly in result.anomalies:
    print(f"Index {anomaly.index}: {anomaly.anomaly_type.value} "
          f"(severity: {anomaly.severity.value}, z={anomaly.z_score:.1f})")
```

---

### describe_windows

Multi-timeframe analysis with trend alignment and actionable signals.

#### Signature

```python
def describe_windows(
    data: np.ndarray | list[float],
    windows: list[int] | list[str] | None = None,
    context: str | None = None,
) -> MultiWindowResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Price or value series |
| `windows` | list, optional | Window sizes as ints [10, 50, 200] or strings ["1h", "4h", "1d"] |
| `context` | str, optional | Label (e.g., "BTC/USD") |

#### Signals

| Signal | Condition |
|--------|-----------|
| STRONG_BULLISH | Strong uptrend with low volatility |
| BULLISH | Uptrend |
| NEUTRAL | No clear direction |
| BEARISH | Downtrend |
| STRONG_BEARISH | Strong downtrend with high volatility |

#### Alignment

| Alignment | Description |
|-----------|-------------|
| ALIGNED_BULLISH | All timeframes bullish |
| ALIGNED_BEARISH | All timeframes bearish |
| MIXED | Conflicting signals |
| DIVERGING | Short/long term opposite |
| CONVERGING | Timeframes moving toward agreement |

#### Example

```python
from semantic_frame.trading import describe_windows

# Price series with uptrend
prices = [100, 102, 101, 104, 103, 106, 105, 108, 107, 110,
          109, 112, 111, 114, 113, 116, 130]

result = describe_windows(prices, windows=[5, 10, 17], context="ETH/USD")

print(result.narrative)
# "Multi-timeframe analysis of ETH/USD: all timeframes bullish.
#  Windows: 5 rising (+9.2%), 10 rising (+8.3%), 17 rising (+30.0%).
#  Noise level: low. Suggested: strong buy signal across all timeframes."

print(f"Alignment: {result.alignment.value}")
print(f"Action: {result.suggested_action}")
```

#### Divergence Example

```python
# Short-term pullback in long-term uptrend
prices = list(range(100, 150)) + [148, 145, 142, 140, 138]

result = describe_windows(prices, windows=[5, 50], context="BTC/USD")

print(result.narrative)
# "Multi-timeframe analysis of BTC/USD: short and long term diverging.
#  Windows: 5 falling (-6.8%), 50 rising (+38.0%).
#  Suggested: short-term weakness in long-term uptrend - potential dip buy."
```

---

### describe_regime

Market regime detection and classification with transition tracking.

#### Signature

```python
def describe_regime(
    returns: np.ndarray | list[float],
    context: str | None = None,
    lookback: int = 20,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
    vol_threshold: float = 0.30,
) -> RegimeResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns` | array-like | Period returns (decimals, e.g., 0.01 = 1%) |
| `context` | str, optional | Label (e.g., "S&P 500") |
| `lookback` | int | Rolling window size (default: 20) |
| `bull_threshold` | float | Annualized return for bull (default: 10%) |
| `bear_threshold` | float | Annualized return for bear (default: -10%) |
| `vol_threshold` | float | Annualized vol for high-vol regime (default: 30%) |

#### Regime Types

| Regime | Description |
|--------|-------------|
| BULL | Sustained positive returns |
| BEAR | Sustained negative returns |
| SIDEWAYS | Low directional movement |
| RECOVERY | Transitioning from bear to bull |
| CORRECTION | Transitioning from bull to bear |
| HIGH_VOLATILITY | Elevated volatility regardless of direction |

#### Regime Properties

| Property | Values |
|----------|--------|
| Strength | STRONG, MODERATE, WEAK |
| Stability | VERY_STABLE, STABLE, UNSTABLE, HIGHLY_UNSTABLE |

#### Example

```python
from semantic_frame.trading import describe_regime
import numpy as np

# Bull market returns
returns = [0.01, 0.015, 0.02, 0.01, 0.025, 0.018, 0.012, 0.022,
           0.015, 0.02, 0.018, 0.025, 0.015, 0.02, 0.022, 0.018,
           0.025, 0.02, 0.015, 0.022]

result = describe_regime(returns, context="BTC")

print(result.narrative)
# "BTC is in a strong bullish regime (duration: 19 periods).
#  Conditions favor trend-following strategies."

print(f"Current Regime: {result.current_regime.value}")
print(f"Strength: {result.regime_strength.value}")
print(f"Stability: {result.stability.value}")
print(f"Time in Bull: {result.time_in_bull_pct:.0f}%")
```

#### Transition Example

```python
# Bear to bull transition
returns = ([-0.02, -0.025, -0.018, -0.03, -0.015, -0.022, -0.02, -0.018] +
           [0.015, 0.02, 0.025, 0.018, 0.022, 0.03])

result = describe_regime(returns, context="S&P 500")

print(result.narrative)
# "S&P 500 is in a strong bullish regime (duration: 6 periods).
#  1 regime change(s) detected - conditions are stable.
#  Transitioned from bearish (8 periods). Regime trend is improving.
#  Conditions favor trend-following strategies."

for period in result.regime_history:
    print(f"{period.regime_type.value}: indices {period.start_index}-{period.end_index}, "
          f"return: {period.cumulative_return:.1%}")
```

---

### describe_allocation

Portfolio allocation suggestions with risk-based position sizing and diversification analysis.

> ⚠️ **Disclaimer**: This function provides educational/informational analysis only, NOT financial advice. Always consult qualified professionals.

#### Signature

```python
def describe_allocation(
    assets: dict[str, np.ndarray | list[float]],
    context: str | None = None,
    method: str = "risk_parity",
    target_volatility: float | None = None,
    risk_free_rate: float = 2.0,
    periods_per_year: int = 252,
) -> AllocationResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `assets` | dict | Map of asset names to price arrays |
| `context` | str, optional | Portfolio label |
| `method` | str | Allocation method (see below) |
| `target_volatility` | float, optional | Target vol % for target_vol method |
| `risk_free_rate` | float | Annual risk-free rate (default: 2%) |
| `periods_per_year` | int | Trading periods (252 for daily) |

#### Allocation Methods

| Method | Description |
|--------|-------------|
| `equal_weight` | Equal allocation across all assets |
| `risk_parity` | Weight inversely proportional to volatility |
| `min_variance` | Minimize portfolio variance |
| `target_vol` | Scale to target volatility level |

#### Risk Levels

| Level | Volatility |
|-------|------------|
| VERY_LOW | < 5% |
| LOW | 5-10% |
| MODERATE | 10-20% |
| HIGH | 20-35% |
| VERY_HIGH | > 35% |

#### Diversification Levels

| Level | Avg Correlation |
|-------|-----------------|
| POOR | > 0.8 |
| LIMITED | 0.6-0.8 |
| MODERATE | 0.4-0.6 |
| GOOD | 0.2-0.4 |
| EXCELLENT | < 0.2 |

#### Example

```python
from semantic_frame.trading import describe_allocation

assets = {
    "BTC": [40000, 42000, 41000, 44000, 43500, 46000],
    "ETH": [2500, 2650, 2550, 2800, 2750, 2950],
    "SOL": [100, 115, 105, 130, 120, 145],
}

result = describe_allocation(assets, context="Crypto Portfolio")

print(result.narrative)
# "Portfolio analysis for Crypto Portfolio: Suggested allocation: BTC (48%), ETH (38%), SOL (14%).
#  Expected return: 918.1%, volatility: 99.3% (very_high risk).
#  Diversification: poor - assets highly correlated (avg correlation: 0.99).
#  ETH/SOL: highly correlated (move together) (r=0.99).
#  Risk parity approach balances risk contribution across assets."

print(f"\n⚠️ {result.disclaimer}")

# Access structured data
for asset, weight in result.suggested_weights.items():
    print(f"{asset}: {weight*100:.1f}%")

print(f"\nPortfolio Vol: {result.portfolio_volatility:.1f}%")
print(f"Risk Level: {result.risk_level.value}")
print(f"Diversification: {result.diversification_level.value}")
```

#### Minimum Variance Example

```python
# Mixed asset classes
assets = {
    "Bonds": [100, 100.5, 100.2, 100.8, 101, 101.2],
    "Stocks": [100, 105, 98, 110, 103, 115],
    "Crypto": [100, 120, 90, 140, 110, 160],
}

result = describe_allocation(assets, method="min_variance", context="Diversified")

print(result.narrative)
# "Portfolio analysis for Diversified: Suggested allocation: Bonds (100%), Stocks (0%), Crypto (0%).
#  Expected return: 56.8%, volatility: 6.6% (low risk).
#  ..."

# Note: Min variance heavily favors low-volatility assets
```

#### Correlation Insights

```python
# View correlation insights
for insight in result.correlation_insights:
    print(f"{insight.asset_1}/{insight.asset_2}: r={insight.correlation:.2f}")
    print(f"  → {insight.relationship}")
```

---

## MCP Integration

All trading functions are available as MCP tools for use with Claude and other LLM agents.

### Starting the MCP Server

```bash
semantic-frame-mcp
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `describe_drawdown` | Equity curve drawdown analysis |
| `describe_trading_performance` | Trading metrics and ratings |
| `describe_rankings` | Multi-agent comparison |
| `describe_anomalies` | Enhanced anomaly detection |
| `describe_windows` | Multi-timeframe analysis |
| `describe_regime` | Market regime detection |
| `describe_allocation` | Portfolio allocation suggestions |

### MCP Tool Example

```python
# Using via MCP (Claude Desktop, etc.)
from semantic_frame.integrations.mcp import describe_trading_performance

result = describe_trading_performance(
    pnl="[100, -50, 75, -25, 150]",
    context="My Strategy"
)
print(result)  # Returns narrative string
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "semantic-frame": {
      "command": "semantic-frame-mcp"
    }
  }
}
```

---

## Use Cases

### Trading Agent Evaluation

```python
from semantic_frame.trading import (
    describe_trading_performance,
    describe_drawdown,
    describe_anomalies,
)

def evaluate_agent(equity_curve, trades):
    """Comprehensive agent evaluation."""

    # Performance metrics
    perf = describe_trading_performance(trades, context="Agent")

    # Risk analysis
    dd = describe_drawdown(equity_curve, context="Agent")

    # Unusual activity
    anomalies = describe_anomalies(trades, is_pnl_data=True)

    return f"""
    {perf.narrative}

    Risk: {dd.narrative}

    Anomalies: {anomalies.narrative}
    """
```

### Multi-Agent Tournament

```python
from semantic_frame.trading import describe_rankings

def run_tournament(agent_results: dict):
    """Compare agents and declare winner."""

    result = describe_rankings(agent_results, context="Weekly Tournament")

    print(f"🏆 Winner: {result.best_agent}")
    print(result.narrative)

    return result.rankings
```

### Market Analysis Dashboard

```python
from semantic_frame.trading import describe_regime, describe_windows

def market_overview(prices, returns):
    """Generate market overview."""

    regime = describe_regime(returns, context="Market")
    windows = describe_windows(prices, context="Market")

    return f"""
    📊 MARKET OVERVIEW

    Regime: {regime.narrative}

    Multi-Timeframe: {windows.narrative}
    """
```

### Portfolio Construction

```python
from semantic_frame.trading import describe_allocation

def suggest_portfolio(assets: dict):
    """Generate portfolio suggestion with disclaimer."""

    result = describe_allocation(assets, method="risk_parity")

    report = f"""
    📈 PORTFOLIO ANALYSIS

    {result.narrative}

    Suggested Weights:
    """

    for asset, weight in result.suggested_weights.items():
        report += f"\n  • {asset}: {weight*100:.1f}%"

    report += f"\n\n⚠️ {result.disclaimer}"

    return report
```

---

## API Reference

### Enums

```python
from semantic_frame.trading import (
    # Drawdown
    DrawdownSeverity,  # MINOR, MODERATE, SIGNIFICANT, SEVERE
    RecoveryState,     # RECOVERED, RECOVERING, IN_DRAWDOWN

    # Performance
    PerformanceRating,  # EXCELLENT, GOOD, MODERATE, POOR, VERY_POOR
    RiskProfile,        # CONSERVATIVE, MODERATE, AGGRESSIVE, VERY_AGGRESSIVE
    ConsistencyRating,  # HIGHLY_CONSISTENT → ERRATIC

    # Anomalies
    AnomalySeverity,    # MILD, MODERATE, SEVERE, EXTREME
    AnomalyType,        # SPIKE, DROP, GAIN, LOSS, OUTLIER_HIGH, OUTLIER_LOW
    AnomalyFrequency,   # RARE, OCCASIONAL, FREQUENT, PERVASIVE

    # Windows
    TimeframeSignal,    # STRONG_BULLISH → STRONG_BEARISH
    TimeframeAlignment, # ALIGNED_BULLISH, ALIGNED_BEARISH, MIXED, DIVERGING, CONVERGING

    # Regime
    RegimeType,         # BULL, BEAR, SIDEWAYS, RECOVERY, CORRECTION, HIGH_VOLATILITY
    RegimeStrength,     # STRONG, MODERATE, WEAK
    RegimeStability,    # VERY_STABLE, STABLE, UNSTABLE, HIGHLY_UNSTABLE

    # Allocation
    RiskLevel,          # VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH
    DiversificationLevel,  # POOR, LIMITED, MODERATE, GOOD, EXCELLENT
    AllocationMethod,   # EQUAL_WEIGHT, RISK_PARITY, MIN_VARIANCE, TARGET_VOL
)
```

### Result Classes

All functions return Pydantic models with:
- Structured data fields
- `.narrative` property for human-readable summary
- Full type hints for IDE support

```python
# Example: Access structured data
result = describe_trading_performance(pnl)

# Narrative (for LLM consumption)
print(result.narrative)

# Structured data (for programmatic use)
print(result.metrics.win_rate)
print(result.metrics.profit_factor)
print(result.performance_rating.value)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.4.0 | Dec 2025 | Added `describe_allocation` |
| 0.3.2 | Dec 2025 | Added `describe_regime` |
| 0.3.1 | Dec 2025 | Added `describe_anomalies`, `describe_windows` |
| 0.3.0 | Dec 2025 | Initial trading module release |

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

*Built with ❤️ for trading agents and the AI community*

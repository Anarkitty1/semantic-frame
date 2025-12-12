# Trading Module Quick Reference

**Semantic Frame v0.4.0** - All 7 trading functions at a glance.

---

## Import

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

## Function Summary

| Function | Input | Key Output | Use Case |
|----------|-------|------------|----------|
| `describe_drawdown` | Equity curve | Max DD%, severity | Risk monitoring |
| `describe_trading_performance` | PnL trades | Win rate, Sharpe | Strategy evaluation |
| `describe_rankings` | {agent: pnl} | Ranked scores | Tournament/comparison |
| `describe_anomalies` | Any series | Anomaly list | Unusual activity |
| `describe_windows` | Prices | Multi-TF signals | Entry/exit timing |
| `describe_regime` | Returns | Bull/bear/sideways | Market conditions |
| `describe_allocation` | {asset: prices} | Weights, risk | Portfolio construction |

---

## Quick Examples

### Drawdown
```python
result = describe_drawdown([10000, 10500, 9800, 9500, 10200], context="Strategy")
# → "Strategy max drawdown: 9.5% (moderate). Currently recovering."
```

### Trading Performance
```python
result = describe_trading_performance([100, -50, 75, -25, 150], context="Bot")
# → "Bot shows good performance with 60.0% win rate... Profit factor: 2.60"
```

### Rankings
```python
result = describe_rankings({"A": [100, -50], "B": [80, -30]}, context="Tournament")
# → "Tournament rankings: #1 B (score: 0.85), #2 A (score: 0.72)"
```

### Anomalies
```python
result = describe_anomalies([10, 12, 11, 100, 13], is_pnl_data=True)
# → "occasional anomalies (1 detected). Index 3: exceptional gain, z=3.2"
```

### Windows
```python
result = describe_windows(prices, windows=[5, 20, 50], context="BTC/USD")
# → "all timeframes bullish. Suggested: strong buy signal"
```

### Regime
```python
result = describe_regime(returns, context="Market")
# → "Market is in a strong bullish regime (duration: 15 periods)"
```

### Allocation
```python
result = describe_allocation({"BTC": [...], "ETH": [...]}, method="risk_parity")
# → "Suggested allocation: BTC (55%), ETH (45%). Risk: high. ⚠️ Not financial advice"
```

---

## Severity/Rating Scales

### Drawdown Severity
| Level | Threshold |
|-------|-----------|
| MINOR | < 5% |
| MODERATE | 5-15% |
| SIGNIFICANT | 15-30% |
| SEVERE | > 30% |

### Performance Rating
| Rating | Win Rate | Profit Factor |
|--------|----------|---------------|
| EXCELLENT | > 65% | > 2.5 |
| GOOD | > 55% | > 1.8 |
| MODERATE | > 45% | > 1.2 |
| POOR | ≤ 45% | ≤ 1.2 |

### Anomaly Severity
| Level | Z-Score |
|-------|---------|
| MILD | ≥ 2.0 |
| MODERATE | ≥ 2.5 |
| SEVERE | ≥ 3.5 |
| EXTREME | ≥ 5.0 |

### Regime Types
- `BULL` - Sustained uptrend
- `BEAR` - Sustained downtrend
- `SIDEWAYS` - Range-bound
- `RECOVERY` - Bear → Bull transition
- `CORRECTION` - Bull → Bear transition

### Risk Levels (Allocation)
| Level | Volatility |
|-------|------------|
| VERY_LOW | < 5% |
| LOW | 5-10% |
| MODERATE | 10-20% |
| HIGH | 20-35% |
| VERY_HIGH | > 35% |

---

## Allocation Methods

| Method | Description |
|--------|-------------|
| `equal_weight` | 1/N allocation |
| `risk_parity` | Inverse volatility weighted |
| `min_variance` | Minimize portfolio variance |
| `target_vol` | Scale to target volatility |

---

## MCP Tools

All functions available via MCP server:

```bash
semantic-frame-mcp
```

Tools accept JSON string inputs:
```python
describe_trading_performance(pnl="[100, -50, 75]", context="Bot")
describe_allocation(assets='{"BTC": [100, 105], "ETH": [50, 52]}')
```

---

## Common Patterns

### Full Agent Evaluation
```python
perf = describe_trading_performance(trades)
dd = describe_drawdown(equity)
anomalies = describe_anomalies(trades, is_pnl_data=True)

report = f"{perf.narrative}\n{dd.narrative}\n{anomalies.narrative}"
```

### Market Dashboard
```python
regime = describe_regime(returns)
windows = describe_windows(prices)

status = f"Regime: {regime.current_regime.value}\n"
status += f"Signal: {windows.alignment.value}"
```

### Tournament Leaderboard
```python
result = describe_rankings(agent_pnls)
for r in result.rankings:
    print(f"#{r.rank} {r.agent_name}: {r.composite_score:.2f}")
```

---

## Tips

1. **Always include context** - Makes narratives more readable
2. **Use `is_pnl_data=True`** - For gain/loss terminology in anomalies
3. **Check `.narrative`** - For LLM consumption
4. **Access structured data** - For programmatic use (`.metrics`, `.suggested_weights`, etc.)
5. **Disclaimer always present** - In allocation outputs

---

[Full Documentation →](trading-module.md)

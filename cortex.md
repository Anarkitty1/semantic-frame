Product Requirements Document (PRD): Project Cortex

Built - /Users/mini_kitty/Projects/cortex-api

**Version:** 1.0 (MVP)
**Type:** Infrastructure / API
**Tagline:** The Semantic Bridge for AI Agents.
**Core Philosophy:** Cognitive Offloading. We handle the math and safety; the AI handles the strategy.

---

## 1. Executive Summary

AI Agents (built on LLMs) are excellent at strategy and sentiment but terrible at arithmetic and resource management. Currently, agents waste valuable context windows processing raw price arrays and frequently hallucinate trends due to their probabilistic nature.

**Cortex** is a middleware API that translates raw, noisy crypto market data into concise, high-context text narratives. It acts as the "Calculator" and "Safety Switch" for the AI, ensuring agents trade based on mathematical reality rather than probabilistic guesses.

---

## 2. The Problem Statement

| Pain Point | Description | The Cortex Solution |
|---|---|---|
| **The Context Cost** | Sending 500 rows of OHLCV data to GPT-4 costs money and fills the context window. | **Compression:** We turn 1,500 tokens of raw JSON into a 30-token semantic summary. |
| **The Math Gap** | LLMs cannot reliably calculate a 200-day Moving Average; they "hallucinate" the result. | **Deterministic Math:** We use Python (Pandas) to calculate indicators with 100% accuracy. |
| **The "Fat Finger"** | Agents lack common sense and will trade into honeypots or high slippage pools. | **Guardrails:** A logic-based safety layer that simulates trades before execution. |
| **Liability Risk** | Giving an AI "black box" total control is a legal nightmare for institutions. | **The "Inspector" Model:** We provide factual observations, not financial advice. |

---

## 3. User Personas

1. **The Agent Developer:** Building on ElizaOS or LangChain. Needs a reliable tool so their bot doesn't crash or drain funds.

2. **The Prop Trading Firm:** Wants to deploy AI bots but needs a hard-coded "Kill Switch" and risk manager that the AI cannot override.

⠀
---

## 4. MVP Scope: The Two Core Services

### Service A: The Narrative Engine (The "Eyes")

**Goal:** Convert quantitative data into qualitative context.

* **Input:** Symbol (e.g., ETH/USDT), Timeframe (e.g., 4h).

* **Process:** Fetch data via CCXT

```
→→
```

* Calculate Indicators (RSI, EMA, Bollinger Bands)

```
→→
```

* Apply Logic string interpolation.

* **Output:** A semantic text string optimized for LLM ingestion.

### Service B: The Guardrail (The "Reflexes")

**Goal:** Prevent catastrophic execution errors.

* **Input:** Trade Intent (Token Address, Amount, Slippage Tolerance).

* **Process:** Simulate trade (or check pool liquidity logic).

* **Output:** Boolean SAFE / UNSAFE + Reason String.

---

## 5. Technical Architecture

### The Stack

* **Language:** Python 3.12+

* **Package Manager:** uv (fast, modern Python tooling).

* **Web Framework:** FastAPI (High performance, async).

* **Data Fetching:** CCXT (Unified exchange API).

* **Analytics:** Pandas-TA (Technical Analysis library).

* **Caching:** Redis (CRITICAL: To prevent hitting exchange rate limits).

* **Interface:** MCP (Model Context Protocol) Server.

### Data Flow Diagram

1. **AI Agent** requests: get_market_narrative("BTC")

2. **Cortex API** checks Redis Cache.

   * Hit: Return cached string (Latency: <10ms).

   * Miss: Call Binance API

```
→→
```

	* Calculate Pandas Metrics

```
→→
```

	* Generate String

```
→→
```

	* Cache

```
→→
```

	* Return (Latency: ~400ms).

⠀
---

## 6. API Specification (The Contract)

### Endpoint 1: /v1/narrative

* **Method:** GET

* **Parameters:** symbol (str), interval (str, default="1h")

* **Response (JSON):**

* code JSON

```
{
  "symbol": "BTC/USDT",
  "timestamp": "2025-10-24T14:30:00Z",
  "narrative": "MARKET STRUCTURE: Bullish Uptrend (Price is above 200 EMA). MOMENTUM: Neutral (RSI is 55). VOLATILITY: Low (Bandwidth tight). NOTE: Price is consolidating near Support ($92,000).",
  "data_freshness": "live"
}
```

* Note: The "narrative" field is designed to be injected directly into the LLM's system prompt.

### Endpoint 2: /v1/guardrail/check

* **Method:** POST

* **Body:**

* code JSON

```
{
  "slippage_setting": 0.05,
  "liquidity_depth_usd": 50000,
  "trade_size_usd": 10000
}
```

* **Response (JSON):**

* code JSON

```
{
  "status": "REJECTED",
  "risk_flag": "HIGH_SLIPPAGE",
  "message": "Projected slippage is 20% due to low liquidity. Safety threshold is 1%."
}
```

---

## 7. The "Secret Sauce" (Logic Definitions)

To avoid liability, we define strict mathematical rules for our language.

* **"Bullish Trend"** = Price > 200-period EMA.

* **"Bearish Trend"** = Price < 200-period EMA.

* **"Overbought"** = RSI > 70.

* **"Oversold"** = RSI < 30.

* **"Volatile"** = ATR (Average True Range) > 2x 14-day average.

---

## 8. Integration Strategy (The "Moat")

We will not just build a REST API; we will build a **Model Context Protocol (MCP) Server**.

* **Why?** This allows Cortex to run natively inside Claude Desktop, ElizaOS, and other agent frameworks without the developer writing custom fetch code.

* **Deliverable:** A cortex_mcp.py file that wraps the FastAPI endpoints into MCP tools.

---

## 9. Success Metrics (KPIs)

1. **Latency:** P95 response time under 500ms (requires effective Redis caching).

2. **Token Savings:** Average context tokens saved per query (Target: >95% reduction vs raw JSON).

3. **Accuracy:** Zero calculation errors (Deterministic Python vs Probabilistic LLM).

⠀
---

## 10. Development Roadmap

### Phase 1: The Engine (Days 1-3)

Set up Python environment with uv (`uv init`, `uv add` dependencies).

Build engine.py using CCXT and Pandas-TA.

Implement the "Logic-to-Text" translator.

**Milestone:** Run script locally, input "ETH", get text summary output.

### Phase 2: The API & Caching (Days 4-7)

Wrap engine in FastAPI.

Implement Redis caching (TTL = 60 seconds).

Add Rate Limiting.

**Milestone:** curl localhost:8000/v1/narrative returns JSON.

### Phase 3: The Agent Integration (Days 8-10)

Implement MCP Server wrapper.

Create a simple "Test Agent" using LangChain/Eliza to call the tool.

**Milestone:** Ask the Agent "How is BTC looking?" and have it reply using Cortex data.

### Phase 4: Production (Days 11-14)

Deploy to Railway/Fly.io.

Set up API Key authentication (Monetization prep).

Publish documentation.

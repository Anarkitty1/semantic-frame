# Claude Desktop & Claude Code Integration

Semantic Frame is available in the [MCP Registry](https://registry.modelcontextprotocol.io) for easy integration with Anthropic's AI tools.

## Claude Code (CLI)

The fastest way to add semantic-frame to Claude Code:

```bash
# Install semantic-frame with MCP support
pip install semantic-frame[mcp]

# Add to Claude Code (one-time setup)
claude mcp add semantic-frame -- python -m mcp run semantic_frame.integrations.mcp:mcp

# Verify connection
claude mcp list
# semantic-frame: ... - ✓ Connected
```

Now Claude Code can automatically use `describe_data` when you ask it to analyze numerical data!

**Example prompt:**
> "Analyze this server latency data: [100, 102, 99, 500, 101, 98, 100]"

Claude will call the `describe_data` tool and return:
> "The Server Latency data shows a flat/stationary pattern with stable variability. 1 anomaly detected at index 3 (value: 500.00)."

## Claude Desktop (macOS/Windows)

Add semantic-frame to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semantic-frame": {
      "command": "python",
      "args": ["-m", "mcp", "run", "semantic_frame.integrations.mcp:mcp"],
      "env": {}
    }
  }
}
```

Or with uv (no global install needed):

```json
{
  "mcpServers": {
    "semantic-frame": {
      "command": "uvx",
      "args": ["--with", "semantic-frame[mcp]", "mcp", "run", "semantic_frame.integrations.mcp:mcp"],
      "env": {}
    }
  }
}
```

Restart Claude Desktop after adding the configuration.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `describe_data` | Analyze a single data series (JSON array, CSV, or newline-separated) |
| `describe_batch` | Analyze multiple series at once (for DataFrames/multi-metric data) |
| `describe_json` | Return structured JSON output instead of narrative text |

## Why Use Semantic Frame with Claude?

1. **95%+ Token Reduction**: Send 10,000 data points as a 50-word summary
2. **Zero Hallucination**: Deterministic math via NumPy (no LLM guessing)
3. **Rich Analysis**: Trends, volatility, anomalies, seasonality, step changes
4. **Context Efficiency**: More room for your actual conversation

## Example: Financial Data Analysis

```
You: "Here's my portfolio returns for the last month: [0.02, 0.01, -0.03, 0.05, -0.08, 0.02, 0.01, -0.02, 0.03, -0.15, 0.02, 0.01]"

Claude (using describe_data): "The Portfolio Returns data shows a flat/stationary pattern with
extreme variability. 2 anomalies detected at indices 4 and 9 (values: -0.08, -0.15).
The negative outliers suggest significant drawdown events worth investigating."
```

## Troubleshooting

**"Tool not found" in Claude**
- Verify MCP server is running: `claude mcp list`
- Check Python has semantic-frame installed: `pip show semantic-frame`

**Connection errors**
- Restart Claude Desktop/Claude Code after configuration changes
- Ensure Python is in your PATH

**Rate limiting concerns**
- semantic-frame runs locally - no API calls during analysis
- Only Claude's own context window is used

# Integrations

Semantic Frame is designed to be the "Math Plugin" for your AI Agents.

## ElizaOS (via MCP)

Semantic Frame provides a **Model Context Protocol (MCP)** server, allowing ElizaOS (and Claude Desktop) to use it natively.

### Installation

```bash
pip install semantic-frame[mcp]
```

### Usage

Run the MCP server:

```bash
mcp run semantic_frame.integrations.mcp:mcp
```

Configure your ElizaOS character or Claude Desktop to connect to this server. The agent will now have access to the `describe_data` tool.

---

## Claude Code

Use Semantic Frame as a native tool in [Claude Code](https://claude.ai/code), Anthropic's CLI for Claude.

### Installation

```bash
pip install semantic-frame[mcp]
```

### Setup

Add the MCP server to Claude Code (run from your project directory):

```bash
claude mcp add semantic-frame -- uv run --project /path/to/semantic-frame mcp run /path/to/semantic-frame/semantic_frame/integrations/mcp.py
```

Restart Claude Code for the server to load.

### Usage

Once configured, Claude Code has access to the `mcp__semantic-frame__describe_data` tool. You can ask Claude to analyze data directly:

```
"Analyze this data: [10, 12, 15, 14, 18, 22, 25, 28, 35, 42]"
```

Claude will automatically use the semantic-frame tool and return:

```
The data shows a rapidly rising pattern with expanding variability.
A strong seasonality was detected. Baseline: 22.10 (range: 10.00-42.00).
```

### Verify Setup

Check the MCP server is connected:

```bash
claude mcp list
```

You should see:
```
semantic-frame: ... - ✓ Connected
```

---

## LangChain

We provide a native LangChain tool wrapper.

### Installation

```bash
pip install semantic-frame[langchain]
```

### Usage

```python
from semantic_frame.integrations.langchain import get_semantic_tool
from langchain.agents import create_openai_tools_agent

# Create the tool
tool = get_semantic_tool(context="Sales Data")

# Add to your agent
tools = [tool]
# ... initialize agent ...
```

---

## CrewAI

We provide a native CrewAI tool decorator.

### Installation

```bash
pip install semantic-frame[crewai]
```

### Usage

```python
from semantic_frame.integrations.crewai import get_crewai_tool
from crewai import Agent

# Create the tool
semantic_tool = get_crewai_tool()

# Add to your agent
analyst = Agent(
    role="Data Analyst",
    goal="Analyze market trends",
    tools=[semantic_tool],
    # ...
)
```

---

## Anthropic API (Native Tool Use)

For direct integration with the Anthropic Python SDK. Use this when building your own agentic applications that call the Claude API directly.

### Installation

```bash
pip install semantic-frame[anthropic]
```

### Basic Usage

```python
import anthropic
from semantic_frame.integrations.anthropic import (
    get_anthropic_tool,
    handle_tool_call,
    create_tool_result,
)

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Analyze this data: [100, 102, 99, 500, 101]"}]

# Make API call with semantic analysis tool
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[get_anthropic_tool()],
    messages=messages,
)

# Handle tool calls in response
for block in response.content:
    if block.type == "tool_use" and block.name == "semantic_analysis":
        result = handle_tool_call(block.input)
        tool_result = create_tool_result(block.id, result)

        # Continue conversation with tool result
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [tool_result]})
```

### Using the Helper Class

For cleaner code, use `AnthropicSemanticTool`:

```python
from semantic_frame.integrations.anthropic import AnthropicSemanticTool

semantic = AnthropicSemanticTool(context="Sensor Readings")

# Get tool definition
tool = semantic.get_tool()

# Handle tool calls
for block in response.content:
    if block.type == "tool_use" and block.name == "semantic_analysis":
        result = semantic.handle(block.input)
        tool_result = semantic.create_result(block.id, result)
```

### Advanced Tool Use (Beta)

Supports Anthropic's [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use) features:

- **Tool Use Examples**: Concrete examples for improved parameter accuracy
- **Deferred Loading**: For agents with many tools (50+)
- **Programmatic Tool Calling**: Batch analysis via code execution

```python
from semantic_frame.integrations.anthropic import get_advanced_tool

# Get tool with all advanced features
tool = get_advanced_tool()

response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-5-20250929",
    tools=[
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        {"type": "code_execution_20250825", "name": "code_execution"},
        tool,
    ],
    messages=[...],
)
```

### When to Use This vs MCP

| Use Case | Integration |
|----------|-------------|
| Claude Code CLI | MCP (`mcp.py`) |
| Claude Desktop | MCP (`mcp.py`) |
| ElizaOS | MCP (`mcp.py`) |
| Custom Python app calling Claude API | Anthropic (`anthropic.py`) |
| Building your own agentic loop | Anthropic (`anthropic.py`) |

---

## MCP Wrapper (Transform Existing Tools)

Use `mcp_wrapper` to add semantic transformation to **existing** functions or MCP tools without modifying their source code.

### Installation

No extra dependencies required—included in the base package.

### Use Cases

- Wrap an existing MCP tool that returns raw numbers
- Add semantic output to data-fetching functions
- Transform API responses into LLM-friendly narratives

### Decorator Usage

```python
from semantic_frame.integrations.mcp_wrapper import wrap_numeric_output

@wrap_numeric_output(context="CPU Usage %")
def get_cpu_readings():
    # Original function returns raw numbers
    return [45, 47, 46, 95, 44, 45]

# Now returns semantic narrative:
# "The CPU Usage % data shows a flat/stationary pattern with 1 anomaly..."
print(get_cpu_readings())
```

### Dynamic Context from Return Data

```python
@wrap_numeric_output(context_key="metric_name")
def get_metrics():
    return {
        "metric_name": "Server Latency (ms)",
        "values": [100, 102, 99, 500, 101],
    }

# Context is extracted from the return dict
print(get_metrics())
# Returns: {"metric_name": "...", "values": [...], "semantic_narrative": "The Server Latency..."}
```

### Standalone Function

For one-off transformations without decoration:

```python
from semantic_frame.integrations.mcp_wrapper import transform_to_semantic

raw_data = fetch_sensor_readings()  # Returns [100, 102, 99, 500, 101]
narrative = transform_to_semantic(raw_data, context="Temperature (C)")
print(narrative)
```

### Class-Based Wrapper

For configuring defaults across multiple functions:

```python
from semantic_frame.integrations.mcp_wrapper import SemanticMCPWrapper

wrapper = SemanticMCPWrapper(
    default_context="Sensor Data",
    default_format="text",
    passthrough_on_failure=True,
)

# Use as decorator
@wrapper.wrap(context="Temperature")
def get_temperature():
    return fetch_temp_readings()

# Or transform directly
narrative = wrapper.transform([1.5, 1.6, 1.4, 1.7], context="Humidity")
```

### JSON Output

```python
@wrap_numeric_output(context="Sales", output_format="json")
def get_sales():
    return [100, 200, 300, 400, 500]

# Returns structured JSON instead of narrative
result = get_sales()
# {"narrative": "...", "trend": "rising", "volatility": "stable", ...}
```

### Comparison: When to Use What

| Module | Purpose | Example |
|--------|---------|---------|
| `mcp.py` | Expose semantic-frame **as** an MCP tool | Claude Code uses `mcp__semantic-frame__describe_data` |
| `anthropic.py` | Expose semantic-frame **as** an Anthropic tool | Your app calls Claude API with tool |
| `mcp_wrapper.py` | Add semantic output **to** existing tools | Wrap `get_cpu_metrics()` to return narratives |

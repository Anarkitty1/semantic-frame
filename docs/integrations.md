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

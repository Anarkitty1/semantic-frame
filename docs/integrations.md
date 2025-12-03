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

"""Framework integrations for semantic-frame.

This package provides tool wrappers for popular AI agent frameworks.

Available integrations:
- anthropic: Native Anthropic Claude tool use
- langchain: LangChain BaseTool wrapper
- crewai: CrewAI @tool decorator wrapper
- mcp: Model Context Protocol server

Install optional dependencies:
    pip install semantic-frame[anthropic]  # For Anthropic SDK
    pip install semantic-frame[langchain]  # For LangChain
    pip install semantic-frame[crewai]     # For CrewAI
    pip install semantic-frame[mcp]        # For MCP server
    pip install semantic-frame[all]        # All integrations
"""

__all__: list[str] = []

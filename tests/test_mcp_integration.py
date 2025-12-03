"""Tests for MCP integration."""

import pytest

# Skip tests if mcp is not installed
try:
    from semantic_frame.integrations.mcp import describe_data, mcp

    mcp_available = True
except ImportError:
    mcp_available = False


@pytest.mark.skipif(not mcp_available, reason="mcp not installed")
class TestMCPIntegration:
    def test_server_initialization(self):
        """Test that the MCP server is initialized correctly."""
        assert mcp.name == "semantic-frame"

    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test that the describe_data tool is registered."""
        # FastMCP.list_tools() returns a list of Tool objects
        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "describe_data" in tool_names

    def test_describe_data_tool(self):
        """Test the describe_data tool logic directly."""
        data = "[10, 12, 11, 13, 12]"
        context = "Test Metrics"

        result = describe_data(data, context)

        # We expect a string result that mentions the context
        assert isinstance(result, str)
        assert "Test Metrics" in result
        assert "flat/stationary" in result.lower() or "stable" in result.lower()

    def test_describe_data_error_handling(self):
        """Test error handling for invalid input."""
        result = describe_data("invalid data", "Context")
        assert "Error analyzing data" in result

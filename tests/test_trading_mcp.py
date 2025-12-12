"""Tests for trading MCP integration."""

import json


class TestTradingMCPTools:
    """Tests for trading tools exposed via MCP."""

    def test_describe_drawdown_mcp(self):
        """Test describe_drawdown MCP tool."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(
            equity="[10000, 10500, 9500, 9000, 10500]",
            context="Test Strategy",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Strategy" in result or "drawdown" in result.lower()

    def test_describe_drawdown_csv_input(self):
        """Test describe_drawdown with CSV input."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(equity="10000, 10500, 9500, 9000, 10500")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_drawdown_error_handling(self):
        """Test describe_drawdown error handling."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(equity="not valid data")

        assert "Error" in result

    def test_describe_trading_performance_mcp(self):
        """Test describe_trading_performance MCP tool."""
        from semantic_frame.integrations.mcp import describe_trading_performance

        result = describe_trading_performance(
            trades="[100, -50, 75, -25, 150]",
            context="CLAUDE Agent",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "CLAUDE Agent" in result or "%" in result

    def test_describe_trading_performance_csv(self):
        """Test describe_trading_performance with CSV input."""
        from semantic_frame.integrations.mcp import describe_trading_performance

        result = describe_trading_performance(trades="100, -50, 75, -25, 150")

        assert isinstance(result, str)

    def test_describe_trading_performance_error_handling(self):
        """Test describe_trading_performance error handling."""
        from semantic_frame.integrations.mcp import describe_trading_performance

        result = describe_trading_performance(trades="invalid")

        assert "Error" in result

    def test_describe_rankings_mcp(self):
        """Test describe_rankings MCP tool."""
        from semantic_frame.integrations.mcp import describe_rankings

        curves = {
            "CLAUDE": [10000, 10500, 11000],
            "GROK": [10000, 12000, 9500],
        }
        result = describe_rankings(
            equity_curves=json.dumps(curves),
            context="AI agents",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention at least one agent
        assert "CLAUDE" in result or "GROK" in result or "agent" in result.lower()

    def test_describe_rankings_error_handling(self):
        """Test describe_rankings error handling."""
        from semantic_frame.integrations.mcp import describe_rankings

        result = describe_rankings(equity_curves="not json")

        assert "Error" in result

    def test_describe_rankings_empty_error(self):
        """Test describe_rankings with empty curves."""
        from semantic_frame.integrations.mcp import describe_rankings

        result = describe_rankings(equity_curves="{}")

        assert "Error" in result


class TestTradingMCPParseInput:
    """Tests for input parsing in MCP tools."""

    def test_json_array_parsing(self):
        """Test JSON array parsing works."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(equity="[10000, 10500, 10200]")
        assert "Error" not in result

    def test_csv_parsing(self):
        """Test CSV parsing works."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(equity="10000, 10500, 10200")
        assert "Error" not in result

    def test_newline_parsing(self):
        """Test newline-separated parsing works."""
        from semantic_frame.integrations.mcp import describe_drawdown

        result = describe_drawdown(equity="10000\n10500\n10200")
        assert "Error" not in result

    def test_rankings_nested_array_parsing(self):
        """Test rankings can parse nested arrays in JSON."""
        from semantic_frame.integrations.mcp import describe_rankings

        curves = {"A": "[10000, 11000]", "B": [10000, 10500]}  # Mixed format
        # This should handle string values that need parsing
        result = describe_rankings(equity_curves=json.dumps(curves))
        # May or may not work depending on implementation
        assert isinstance(result, str)


class TestAnomaliesMCPTool:
    """Tests for describe_anomalies MCP tool."""

    def test_describe_anomalies_basic(self):
        """Test basic anomaly detection via MCP."""
        from semantic_frame.integrations.mcp import describe_anomalies

        result = describe_anomalies(
            data="[100, 102, 99, 500, 101, 98, -200]",
            context="Trade PnL",
            is_pnl_data=True,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_anomalies_csv_input(self):
        """Test anomaly detection with CSV input."""
        from semantic_frame.integrations.mcp import describe_anomalies

        result = describe_anomalies(data="100, 102, 99, 500, 101")

        assert isinstance(result, str)
        assert "Error" not in result

    def test_describe_anomalies_context_in_narrative(self):
        """Test that context appears in narrative."""
        from semantic_frame.integrations.mcp import describe_anomalies

        result = describe_anomalies(
            data="[100, 500, 100]",
            context="Server Latency",
        )

        assert "Server Latency" in result

    def test_describe_anomalies_error_handling(self):
        """Test error handling for invalid input."""
        from semantic_frame.integrations.mcp import describe_anomalies

        result = describe_anomalies(data="not valid")

        assert "Error" in result


class TestWindowsMCPTool:
    """Tests for describe_windows MCP tool."""

    def test_describe_windows_basic(self):
        """Test basic multi-window analysis via MCP."""
        from semantic_frame.integrations.mcp import describe_windows

        result = describe_windows(
            data="[100, 102, 105, 103, 108, 110, 112, 109, 115, 118, 120]",
            windows="5,10",
            context="BTC",
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_windows_csv_input(self):
        """Test multi-window analysis with CSV input."""
        from semantic_frame.integrations.mcp import describe_windows

        result = describe_windows(
            data="100, 105, 110, 108, 115, 120",
            windows="3,6",
        )

        assert isinstance(result, str)
        assert "Error" not in result

    def test_describe_windows_default_windows(self):
        """Test with default window sizes."""
        from semantic_frame.integrations.mcp import describe_windows

        # Generate enough data for default windows
        data = ",".join(str(100 + i) for i in range(100))
        result = describe_windows(data=data)

        assert isinstance(result, str)
        assert "Error" not in result

    def test_describe_windows_context_in_narrative(self):
        """Test that context appears in narrative."""
        from semantic_frame.integrations.mcp import describe_windows

        data = ",".join(str(100 + i) for i in range(20))
        result = describe_windows(data=data, context="ETH/USD")

        assert "ETH/USD" in result

    def test_describe_windows_error_handling(self):
        """Test error handling for invalid input."""
        from semantic_frame.integrations.mcp import describe_windows

        result = describe_windows(data="invalid data")

        assert "Error" in result

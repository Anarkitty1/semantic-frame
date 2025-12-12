# MCP Registry Publishing Guide for Semantic Frame

## Overview
This guide walks you through publishing semantic-frame to the official MCP Registry,
making it discoverable in Claude Desktop, Claude Code, and other MCP clients.

## Prerequisites
✅ server.json created (done!)
✅ mcp-publisher CLI downloaded (done!)
✅ PyPI package published (semantic-frame 0.2.0 ✓)
✅ GitHub repository exists (Anarkitty1/semantic-frame)

## Step 1: Authenticate with GitHub

Run this command - it will open your browser for GitHub OAuth:

```bash
cd /Users/mini_kitty/Projects/semantic_serializer
./mcp-publisher login github
```

This will:
1. Open a browser window
2. Ask you to authorize the MCP Registry app
3. Verify you own the Anarkitty1/semantic-frame repository
4. Save tokens locally (in .mcpregistry_github_token)

## Step 2: Publish to Registry

Once authenticated, publish your server:

```bash
./mcp-publisher publish
```

Note: The registry is in preview mode with high traffic. You may need to retry
a few times if you get timeout errors.

## Step 3: Verify Publication

Check your server appears in the registry:

```bash
curl "https://registry.modelcontextprotocol.io/v0/servers?search=semantic-frame"
```

You should see your server metadata in the response.

## Step 4: Add to .gitignore

Add these files to keep auth tokens private:

```
.mcpregistry_github_token
.mcpregistry_registry_token
mcp-publisher
```

## After Publishing

Once published, users can discover semantic-frame through:

1. **Registry API**: Clients can query the registry for your server
2. **MCP Marketplaces**: Third-party directories will index your server
3. **Claude Desktop**: Will appear in server discovery (when enabled)

## Claude Code Integration

Users can add your MCP server to Claude Code with:

```bash
# Install semantic-frame
pip install semantic-frame[mcp]

# Add to Claude Code
claude mcp add semantic-frame -- python -m mcp run semantic_frame.integrations.mcp:mcp

# Or with uv
claude mcp add semantic-frame -- uv run --with "semantic-frame[mcp]" python -m mcp run semantic_frame.integrations.mcp:mcp
```

## Claude Desktop Integration

Users add this to their claude_desktop_config.json:

```json
{
  "mcpServers": {
    "semantic-frame": {
      "command": "python",
      "args": ["-m", "mcp", "run", "semantic_frame.integrations.mcp:mcp"]
    }
  }
}
```

## Updating Your Server

To publish updates after changing server.json or releasing a new version:

```bash
# Update version in server.json
# Then republish
./mcp-publisher publish
```

## Troubleshooting

**"Not authenticated" error**
Run `./mcp-publisher login github` again

**Timeout/503 errors during publish**
The registry has high traffic - retry a few times

**"Package not found" error**
Ensure semantic-frame 0.2.0 is live on PyPI: https://pypi.org/project/semantic-frame/

**GitHub auth fails**
Make sure you're logged into GitHub in your browser and have access to Anarkitty1/semantic-frame

## Support

- MCP Registry Issues: https://github.com/modelcontextprotocol/registry/issues
- Semantic Frame Issues: https://github.com/Anarkitty1/semantic-frame/issues

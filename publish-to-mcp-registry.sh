#!/bin/bash
# MCP Registry Publishing Script for Semantic Frame
# Run this from the project root: ./publish-to-mcp-registry.sh

set -e

echo "🚀 Publishing Semantic Frame to MCP Registry"
echo "============================================="
echo ""

# Check if mcp-publisher exists
if [ ! -f "./mcp-publisher" ]; then
    echo "❌ mcp-publisher not found. Downloading..."
    curl -L "https://github.com/modelcontextprotocol/registry/releases/download/v1.0.0/mcp-publisher_1.0.0_darwin_arm64.tar.gz" -o mcp-publisher.tar.gz
    tar xzf mcp-publisher.tar.gz
    rm mcp-publisher.tar.gz
    chmod +x mcp-publisher
    echo "✅ mcp-publisher downloaded"
fi

# Check if server.json exists
if [ ! -f "./server.json" ]; then
    echo "❌ server.json not found!"
    exit 1
fi

echo ""
echo "📋 Server configuration:"
echo "------------------------"
cat server.json | head -15
echo "..."
echo ""

# Step 1: Login
echo "Step 1: Authenticating with GitHub..."
echo "This will open your browser for OAuth authorization."
echo ""
read -p "Press Enter to continue (or Ctrl+C to cancel)..."
./mcp-publisher login github

echo ""
echo "✅ Authentication successful!"
echo ""

# Step 2: Publish
echo "Step 2: Publishing to registry..."
echo ""

# Retry logic for high-traffic periods
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ./mcp-publisher publish; then
        echo ""
        echo "🎉 SUCCESS! Semantic Frame published to MCP Registry!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo ""
            echo "⚠️  Publish failed (attempt $RETRY_COUNT/$MAX_RETRIES). Retrying in 5 seconds..."
            sleep 5
        else
            echo ""
            echo "❌ Publish failed after $MAX_RETRIES attempts."
            echo "The registry may be experiencing high traffic. Try again later with:"
            echo "  ./mcp-publisher publish"
            exit 1
        fi
    fi
done

echo ""
echo "Step 3: Verifying publication..."
sleep 2

echo ""
echo "Checking registry for semantic-frame..."
curl -s "https://registry.modelcontextprotocol.io/v0/servers?search=semantic-frame" | head -100

echo ""
echo ""
echo "============================================="
echo "🎉 DONE! Your server is now discoverable in the MCP Registry!"
echo ""
echo "Next steps:"
echo "  1. View your listing: https://registry.modelcontextprotocol.io"
echo "  2. Users can now add semantic-frame to Claude Desktop/Claude Code"
echo "  3. Consider adding a badge to your README"
echo ""

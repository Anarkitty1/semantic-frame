#!/bin/bash
# Release script for semantic-frame
# Run this from the project root

set -e

echo "🔍 Running tests..."
uv run pytest -q

echo "🔍 Running linter..."
uvx ruff check semantic_frame/

echo "📦 Building package..."
rm -rf dist/
uv build

echo "✅ Build complete!"
echo ""
echo "Files ready for upload:"
ls -la dist/

echo ""
echo "📤 To upload to PyPI, run:"
echo "   # Test PyPI first (recommended):"
echo "   uv run twine upload --repository testpypi dist/*"
echo ""
echo "   # Production PyPI:"
echo "   uv run twine upload dist/*"
echo ""
echo "💡 Make sure you have twine installed:"
echo "   uv pip install twine"

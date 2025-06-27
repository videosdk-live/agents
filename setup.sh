#!/bin/bash

set -e

echo "Setting up VideoSDK Agents..."

echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.13 -m venv venv
fi

source venv/bin/activate

echo "Fixing videosdk namespace..."
PYTHON_PATH=$(python -c "import videosdk; print(videosdk.__file__)" 2>/dev/null || echo "")
if [ -n "$PYTHON_PATH" ]; then
    VIDEOSDK_DIR=$(dirname "$PYTHON_PATH")
    if ! grep -q "extend_path" "$VIDEOSDK_DIR/__init__.py" 2>/dev/null; then
        echo "__path__ = __import__('pkgutil').extend_path(__path__, __name__)" >> "$VIDEOSDK_DIR/__init__.py"
    fi
fi

echo "Installing plugins..."
for plugin in videosdk-agents videosdk-plugins/*; do
    if [ -d "$plugin" ] && [ -f "$plugin/pyproject.toml" ]; then
        echo "  Installing $(basename "$plugin")..."
        pip install -e "$plugin"
    fi
done

echo "Setup complete!"

echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🛠️ Modifying videosdk __init__.py file..."
VIDEOSDK_INIT_FILE="venv/lib/$(python -c 'import sys; print("python{}.{}".format(*sys.version_info))')/site-packages/videosdk/__init__.py"

if [ -f "$VIDEOSDK_INIT_FILE" ]; then
  echo "__path__ = __import__('pkgutil').extend_path(__path__, __name__)" >> "$VIDEOSDK_INIT_FILE"
  echo "✅ __init__.py modified."
else
  echo "❌ videosdk __init__.py not found. Please check your installation."
  exit 1
fi

echo "📁 Installing videosdk plugins (google)..."
cd videosdk-plugins/videosdk-plugins-google
pip install -e .

echo "📁 Installing videosdk plugins (openai)..."
cd ../videosdk-plugins-openai
pip install -e .

echo "📁 Installing videosdk-agent..."
cd ../../videosdk-agents
pip install -e .

echo "🎉 Setup complete!"
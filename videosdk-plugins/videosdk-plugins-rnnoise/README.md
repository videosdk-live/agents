# VideoSDK RNNoise Plugin

Agent Framework plugin for de-noising with RNNoise.

## Installation

```bash
pip install videosdk-plugins-rnnoise
```

## Building RNNoise for your OS

To avoid OS security issues with prebuilt libraries, build RNNoise locally:
1. Ensure you have git and build tools (autoconf/make on Mac/Linux, Visual Studio with nmake on Windows).
2. Run `python build_rnnoise.py` in the project root.
3. This clones RNNoise, builds the library for your OS, and places it in videosdk/plugins/rnnoise/files/.
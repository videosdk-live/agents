import os

HG_MODEL = "latishab/turnsense"
ONNX_FILENAME = "model_quantized.onnx"
VIDEOSDK_MODEL_URL = "https://cdn.videosdk.live/models/turn-detection-v1/"
VIDEOSDK_MODEL_FILES = [
    "model.onnx",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "config.json",
]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "turn-detector-model")

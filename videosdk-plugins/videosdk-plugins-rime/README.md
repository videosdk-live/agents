# VideoSDK Rime AI TTS Plugin

A Text-to-Speech (TTS) plugin for VideoSDK that integrates with Rime AI's TTS service.

## Installation

```bash
pip install videosdk-plugins-rime
```

## Configuration

Set your Rime AI API key as an environment variable:

```bash
export RIME_API_KEY="your-api-key-here"
```

## Usage

```python
from videosdk.agents import Agent
from videosdk.plugins.rime import RimeTTS

# Initialize the TTS with default settings
tts = RimeTTS(
    api_key="your-api-key-here",  # Optional if RIME_API_KEY env var is set
    speaker="river",  # Default speaker (known to work)
    model_id="mist",  # or "mistv2" for the newest model
    lang="eng",  # Language code
    sampling_rate=16000,  # Sampling rate (4000-44100)
    speed_alpha=1.0,  # Speed adjustment (< 1.0 = faster, > 1.0 = slower)
    reduce_latency=False,  # Enable low-latency mode
)

# Create an agent with Rime TTS
agent = Agent(
    tts=tts,
    # ... other agent configuration
)
```

## Parameters

### Required Parameters

- `speaker` (str): The voice to use. Must be one of the available voices for your model and language.
- `text` (str): The text to synthesize (max 500 characters via API).

### Optional Parameters

- `model_id` (str): Choose between "mistv2" (fastest, most accurate) or "mist" (default).
- `lang` (str): Language code that must match the speaker's language (default: "eng").
- `sampling_rate` (int): Audio sampling rate between 4000 and 44100 (default: 16000).
- `speed_alpha` (float): Speed adjustment factor (default: 1.0).
- `reduce_latency` (bool): Reduces response latency at potential cost of pronunciation accuracy (default: False).
- `pause_between_brackets` (bool): Adds pauses for text in angle brackets, e.g., `<200>` = 200ms pause (default: False).
- `phonemize_between_brackets` (bool): Enables custom pronunciation for text in curly brackets (default: False).
- `inline_speed_alpha` (str): Comma-separated speed values for words in square brackets (e.g., "3, 0.5").

## Available Speakers

Based on testing, the following speakers are known to work with the "mist" and "mistv2" models:

- `river` (default)
- `storm`
- `brook`
- `ember`
- `iris`
- `pearl`

**Note**: Speaker availability may vary by model and language. The plugin will warn you if a speaker might not be available.

## Advanced Features

### Custom Pauses

Enable pause injection with angle brackets:

```python
tts = RimeTTS(
    pause_between_brackets=True
)

# Text with pauses
text = "Hi. <200> I'd love to have a conversation with you."
```

### Custom Pronunciation

Enable phoneme specification with curly brackets:

```python
tts = RimeTTS(
    phonemize_between_brackets=True
)

# Text with custom pronunciation
text = "{h'El.o} World"  # Pronounces "Hello" phonetically
```

### Inline Speed Control

Control speed for specific words using square brackets:

```python
tts = RimeTTS(
    inline_speed_alpha="3, 0.5"  # First bracket slow, second bracket fast
)

# Text with speed variations
text = "This is [slow] and [fast]"
```

## Error Handling

The plugin handles common errors:
- Invalid API key (401)
- Bad requests (400)
- Text exceeding 500 character limit
- Network timeouts
- Invalid speaker/model combinations

## Testing

The plugin includes several test scripts:

```bash
# Test basic functionality
python quick_test.py

# Test with explicit speaker
python explicit_test.py

# Test mistv2 model
python test_mistv2.py

# Comprehensive debug test
python debug_rime_tts.py
```

## License

MIT 
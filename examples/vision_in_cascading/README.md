# Image Analysis Agent

A voice-enabled AI agent built with VideoSDK that can analyze images and answer questions about them through real-time voice conversation via cascading pipeline and ImageContent Class.

## Features

- **Voice Interaction**: Full voice conversation capability using speech-to-text and text-to-speech
- **Image Analysis**: Analyze images using GPT-4o vision capabilities
- **Real-time Processing**: Processes speech and images in real-time
- **Web Interface**: Accessible via browser through VideoSDK playground
- **Document Analysis**: Specifically designed for analyzing documents like PAN cards, Aadhar cards, etc.

## Prerequisites

- Python 3.12+
- OpenAI API key
- ElevenLabs API key (for TTS)
- Deepgram API key (for STT)
- VideoSDK account

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test-cascading-vision
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
VIDEOSDK_AUTH_TOKEN=your_videosdk_token
ROOM_ID=your_room_id
```

## Usage

1. Place your image file in the project directory (e.g., `pan-sample.jpeg`)

2. Update the image filename in `main.py`:
```python
self.test_images = [    
    self._convert_file_to_data_uri("your-image-file.jpeg")  
]
```

3. Update the room ID in `main.py`:
```python
room_options = RoomOptions(
    room_id="your-room-id",  # Replace with your room ID
    name="Image Analysis Agent", 
    playground=True
)
```

4. Run the agent:
```bash
python3 main.py
```

5. Open the provided playground URL in your browser

6. Join the room and start talking to the agent!

## Generate A Room ID

Use your `VIDEOSDK_AUTH_TOKEN` to create a room via REST. Example with curl:

```bash
curl -X POST https://api.videosdk.live/v2/rooms \
  -H "Authorization: YOUR_VIDEOSDK_AUTH_TOKEN" \
  -H "Content-Type: application/json"
```
For more details on the Create Room API, refer to the [VideoSDK documentation](https://docs.videosdk.live/api-reference/realtime-communication/create-room).

## How It Works

### Architecture

The agent uses a cascading pipeline with the following components:

- **STT (Speech-to-Text)**: DeepgramSTT converts your voice to text
- **LLM (Large Language Model)**: OpenAI GPT-4o processes text and analyzes images
- **TTS (Text-to-Speech)**: ElevenLabsTTS converts responses back to voice
- **VAD (Voice Activity Detection)**: SileroVAD detects when you're speaking
- **Turn Detector**: Manages conversation flow and turn-taking

### Image Processing

1. **Local File Conversion**: Local images are converted to data URIs using base64 encoding
2. **High-Detail Analysis**: Images are processed with `inference_detail="high"` for maximum accuracy
3. **Context Integration**: Images are added to the chat context for the LLM to analyze

### Conversation Flow

1. **Startup**: Agent connects to VideoSDK room
2. **Image Analysis**: Automatically analyzes the provided image on startup
3. **Voice Interaction**: 
   - You speak → STT converts to text
   - LLM processes text and image → generates response
   - TTS converts response → agent speaks back
4. **Continuous Conversation**: Responds to questions about the analyzed image

## Example Usage

Once the agent is running, you can ask questions like:

- "What do you see in this image?"
- "Can you read the text in the document?"
- "What personal information is visible?"
- "Describe the document structure"
- "What type of document is this?"

## Configuration

### Room Options
```python
room_options = RoomOptions(
    room_id="your-room-id",
    name="Image Analysis Agent", 
    playground=True  # Enables web interface
)
```

Or read from environment variables (recommended):
```python
import os

room_options = RoomOptions(
    room_id=os.getenv("ROOM_ID"),
    name="Image Analysis Agent",
    playground=True
)
```

### Pipeline Components
```python
pipeline = CascadingPipeline(  
    stt=DeepgramSTT(),
    llm=OpenAILLM(model="gpt-4o"),
    tts=ElevenLabsTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector()
)
```

## Troubleshooting

### Agent Not Speaking
- Check that all API keys are correctly set
- Ensure the image file exists and is readable
- Verify browser permissions for microphone and speakers
- Check console logs for error messages

### Image Analysis Issues
- Ensure the image file is in the correct format (JPEG, PNG)
- Check that the file path in `main.py` is correct
- Verify the image is not corrupted

### Connection Issues
- Check your internet connection
- Verify VideoSDK credentials
- Ensure the room ID is valid

## File Structure

```
test-cascading-vision/
├── main.py                 # Main agent implementation
├── README.md              # This file
├── .env                   # Environment variables (create from .env.example)
├── venv/                  # Virtual environment
└── your-image-file.jpeg   # Image to analyze
```

## API Requirements

- **OpenAI**: GPT-4o model access for text and image processing
- **ElevenLabs**: Text-to-speech conversion
- **Deepgram**: Speech-to-text conversion
- **VideoSDK**: Real-time communication platform

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Review VideoSDK documentation
- Check API provider documentation for key setup

## Frontend Meeting With Chat: Send Images The Agent Can Analyze

You can build a VideoSDK web meeting UI (or your own) with chat input that sends an image to the agent. The agent will analyze images if you send them as either an HTTPS URL or a data URI in the chat payload.

The agent expects a message content format equivalent to OpenAI's vision shape. Example JSON your frontend can post to your backend that forwards to the agent:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Please analyze this image" },
    {
      "type": "image_url",
      "image_url": { "url": "<https-or-data-uri>" }
    }
  ]
}
```

### Option 1: HTTPS Image URL
- Host the image on a public URL (S3/CDN/drive that allows public access)
- Send the URL in `image_url.url`

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "What do you see in this document?" },
    { "type": "image_url", "image_url": { "url": "https://example.com/path/to/document.jpg" } }
  ]
}
```

### Option 2: Data URI (No Hosting Required)
- Convert the local file to base64 in the browser and send a data URI

```js
// Build a data URI from a File input (browser)
const toDataUri = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result); // data:<mime>;base64,<data>
  reader.onerror = reject;
  reader.readAsDataURL(file);
});

async function sendImageMessage(file) {
  const dataUri = await toDataUri(file);
  const payload = {
    role: "user",
    content: [
      { type: "text", text: "Please analyze this image in detail." },
      { type: "image_url", image_url: { url: dataUri } }
    ]
  };
  // Post this to your backend that forwards to the agent
  await fetch("/api/agent/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
}
```

### Backend Forwarder (Minimal Example)
Your backend should forward the above payload into the agent's conversation (same shape the agent already uses in `main.py` via `ImageContent`).

```python
from videosdk.agents import ChatRole, ImageContent

async def forward_to_agent(session, payload):
    content = []
    for item in payload["content"]:
        if item["type"] == "text":
            content.append(item["text"])
        elif item["type"] == "image_url":
            content.append(ImageContent(image=item["image_url"]["url"], inference_detail="high"))

    session.agent.chat_context.add_message(role=ChatRole.USER, content=content)
    # Optionally trigger a response depending on your flow
```

Notes:
- Data URIs avoid hosting and work well for quick demos.
- Resize/compress large images on the client to reduce latency.
- If using the VideoSDK Web SDK chat, map incoming chat messages to the above payload and forward to the agent.

# Local Avatar Server Example

Demonstrates the full local avatar setup using the VideoSDK Agents framework. The agent streams TTS audio to a separate Avatar Server process via data channel. The Avatar Server renders a real-time waterfall spectrogram and publishes audio + video tracks back to the room.

## How It Works

```
Agent  ──(data channel)──►  Avatar Dispatcher  ──►  Avatar Service
                            (videosdk_avatar_launcher.py)   (videosdk_avatar_service.py)
                                                             renders waterfall viz
                                                             publishes audio + video
```

## Files

| File | Role |
|---|---|
| `videosdk_avatar_launcher.py` | Dispatcher — HTTP server that spawns one Avatar Service per room |
| `videosdk_avatar_service.py` | Avatar Service — joins room, receives audio, renders and publishes |
| `waterfall_viz.py` | Waterfall spectrogram renderer (FFT heat-map, scrolls upward) |
| `videosdk_avatar_agent.py` | Agent using cascade (STT → LLM → TTS) |

## Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
pip install videosdk-agents
```

**Set environment variables** (create a `.env` file):

```bash
VIDEOSDK_API_KEY=your_api_key
VIDEOSDK_SECRET_KEY=your_secret_key
VIDEOSDK_ROOM_ID=your_room_id
VIDEOSDK_AUTH_TOKEN=your_auth_token
DEEPGRAM_API_KEY=your_deepgram_key
GOOGLE_API_KEY=your_google_key
```

## Running

Start the dispatcher first, then the agent in a separate terminal.

```bash
# Terminal 1 — start the dispatcher
python videosdk_avatar_launcher.py

# Terminal 2 — start the agent
python videosdk_avatar_agent.py
```

The dispatcher listens on `http://localhost:8089`. When the agent connects, it calls `POST /launch` and the dispatcher spawns the Avatar Service for that room automatically.

## Notes

- The dispatcher manages one Avatar Service process per room. If the agent reconnects, the existing service is replaced automatically.
- The waterfall visualizer uses FFT bands rendered as a heat-map that scrolls upward in sync with the agent's speech.
- To use a custom renderer, implement `AvatarRenderer` in place of `waterfall_viz.py` and wire it into `videosdk_avatar_service.py`.

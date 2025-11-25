# Audio Circular Example

This directory contains examples for running voice agents in both cascading and real-time modes, as well as an avatar launcher.

## Getting Started

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Avatar Launcher

To start the avatar launcher, run:

```bash
python avatar_launcher.py
```

## Running the Cascading Voice Agent

To start the cascading voice agent, run:

```bash
python agent_worker_cascading.py
```

## Running the Real-time Voice Agent

To start the real-time voice agent, run:

```bash
python agent_worker_realtime.py
```

## Configuration

Before running the examples, ensure you have a `.env` file in the root directory of this project with the following environment variables:

```
VIDEOSDK_API_KEY="YOUR_API_KEY"
VIDEOSDK_SECRET_KEY="YOUR_SECRET_KEY"
VIDEOSDK_AUTH_TOKEN="YOUR_VIDEOSDK_AUTH_TPKEN"
VIDEOSDK_ROOM_ID="YOUR_ROOM_ID"
```

- **VIDEOSDK_API_KEY** and **VIDEOSDK_SECRET_KEY**: These are required for authentication. You will use these to generate a VideoSDK Auth Token. For more information on generating an authentication token, refer to the [VideoSDK Authentication and Tokens documentation](https://docs.videosdk.live/ai_agents/authentication-and-token).
- **VIDEOSDK_ROOM_ID**: This is the ID of the room your agents will join. You can create a room using the [VideoSDK Create a Room API](https://docs.videosdk.live/api-reference/realtime-communication/create-room).


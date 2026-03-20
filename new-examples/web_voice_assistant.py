"""
Use Case: Website product assistant (Buildfast) — same agent logic across VideoSDK, WebRTC, and WebSocket transports.
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: All three transport modes in a single file — select by uncommenting the desired make_context block.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import logging
from videosdk.agents import (
    Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions,
    WebRTCConfig, WebSocketConfig,
)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class BuildfastAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the product assistant for Buildfast, a no-code website builder.
            Help website visitors understand Buildfast's features, pricing, and integrations.
            Keep answers under three sentences — visitors are browsing and want quick answers.
            Pricing tiers: Free (3 pages), Pro ($19/mo, unlimited pages), Business ($49/mo, custom domain + analytics).
            If a visitor asks to speak with sales, direct them to sales@buildfast.io.
            Never make up feature names or pricing not listed above.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi! I'm the Buildfast assistant. "
            "I can tell you about our features, pricing, and how to get started. "
            "What would you like to know?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thanks for stopping by Buildfast. "
            "Start building for free at buildfast.io — no credit card needed!"
        )


async def entrypoint(ctx: JobContext):
    agent = BuildfastAssistant()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


# --- Transport Mode 1: VideoSDK Room (default) ---
# Requires a room_id from the VideoSDK dashboard.
def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Buildfast Product Assistant",
        playground=True,
    )
    return JobContext(room_options=room_options)


# --- Transport Mode 2: WebRTC P2P ---
# Use with examples/browser_transports/webrtc/webrtc_mode.html
# 1. Start a signaling server (e.g., node signaling_server.js)
# 2. Open webrtc_mode.html in your browser and click "Connect"
# 3. Run this script
# def make_context() -> JobContext:
#     room_options = RoomOptions(
#         transport_mode="webrtc",
#         webrtc=WebRTCConfig(
#             signaling_url="ws://localhost:8081",
#             signaling_type="websocket",
#             ice_servers=[{"urls": "stun:stun.l.google.com:19302"}],
#         ),
#     )
#     return JobContext(room_options=room_options)


# --- Transport Mode 3: WebSocket (raw PCM audio) ---
# Use with examples/browser_transports/websocket/websocket_mode.html
# 1. Open websocket_mode.html in your browser
# 2. Run this script
# 3. Click "Connect" in the browser
# def make_context() -> JobContext:
#     room_options = RoomOptions(
#         transport_mode="websocket",
#         websocket=WebSocketConfig(port=8080, path="/ws"),
#     )
#     return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

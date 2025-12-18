import logging
import aiohttp
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob,ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks and help with horoscopes.",
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
    @function_tool
    async def get_horoscope(self, sign: str) -> dict:
        """Get today's horoscope for a given zodiac sign.

        Args:
            sign: The zodiac sign (e.g., Aries, Taurus, Gemini, etc.)
        """
        horoscopes = {
            "Aries": "Today is your lucky day!",
            "Taurus": "Focus on your goals today.",
            "Gemini": "Communication will be important today.",
        }
        return {
            "sign": sign,
            "horoscope": horoscopes.get(sign, "The stars are aligned for you today!"),
        }

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(), 
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    # --- Example 1: Default VideoSDK Room (Existing) ---
    room_options = RoomOptions(
       room_id="<room_id>", 
       name="Connection Mode Test Agent", 
       playground=True
    )

    # --- Example 2: WebRTC Connection (P2P) ---
    # Used with examples/client_examples/webrtc/client.html & signaling_server.js
    # Ensure you start the signaling server first: `node examples/client_examples/webrtc/signaling_server.js`
    
    # Basic configuration (uses default Google STUN server)
    # room_options = RoomOptions(
    #    connection_mode="webrtc",
    #    webrtc_signaling_url="ws://localhost:8081",
    #    webrtc_signaling_type="websocket",
    #    webrtc_ice_servers=[
    #        {"urls": "stun:stun.l.google.com:19302"},
    #    ],
    # )

    # --- Example 3: WebSocket Connection (Raw PCM) ---
    # Used with examples/client_examples/websocket/client.html
    # room_options = RoomOptions(
    #    connection_mode="websocket",
    #    websocket_port=8080,
    #    websocket_path="/ws"
    # )
    
    # Advanced configuration with custom ICE servers (STUN + TURN)
    # Uncomment to use custom servers:
    # room_options = RoomOptions(
    #    connection_mode="webrtc",
    #    webrtc_signaling_url="ws://localhost:8081",
    #    webrtc_signaling_type="websocket",
    #    webrtc_ice_servers=[
    #        {"urls": "stun:stun.l.google.com:19302"},
    #        {"urls": "stun:stun1.l.google.com:19302"},
    #        {
    #            "urls": "turn:your-turn-server.com:3478",
    #            "username": "your-username",
    #            "credential": "your-password"
    #        }
    #    ]
    # )
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
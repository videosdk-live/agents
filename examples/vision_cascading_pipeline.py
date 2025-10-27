# This test script is used to test cascading pipeline.
import logging
import base64
import aiohttp
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, MCPServerStdio, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT, DeepgramTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.rnnoise import RNNoise
from videosdk.agents.llm.chat_context import ImageContent, ChatRole
from videosdk.agents.images import encode, EncodeOptions
from videosdk.agents.job import get_current_job_context

logging.basicConfig(
    level=logging.INFO,            # change to DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# logging.getLogger().setLevel(logging.CRITICAL)

pre_download_model()

def frame_to_data_uri(frame, format="JPEG", quality=80):
    """Convert a frame to a base64-encoded data URI for vision model input."""
    try:
        encoded = encode(frame, EncodeOptions(format=format, quality=quality))
        logger.info(f"Encoded frame size: {len(encoded)} bytes")

        b64_image = base64.b64encode(encoded).decode("utf-8")
        logger.info(f"Base64 image size: {len(b64_image)} characters")
        
        data_url = f"data:image/{format.lower()};base64,{b64_image}"
        logger.info(f"Data URL size: {len(data_url)} characters")
        logger.info(f"Data URL preview: {data_url[:100]}...")
        
        if not data_url or not data_url.startswith("data:image/") or len(b64_image) < 100:
            raise ValueError("Encoded image is too small or invalid")
        
        return data_url
    
    except Exception as e:
        logger.error(f"Frame encoding failed: {e}")
        return None


@function_tool
async def capture_and_process_frame():
    """
    Capture and analyze the latest frame from the user's live video feed.
    """
    ctx = get_current_job_context()
    room = getattr(ctx, "room", None)
    frame = getattr(room, "_last_video_frame", None)

    try:
        if frame is not None:
            logger.info("Type of captured frame: %s", type(frame))
            image_data = frame_to_data_uri(frame)
            logger.info("Processing live video frame for analysis")
        else:
            logger.warning("No frame available for analysis")
            return {"ok": False, "reason": "No frame available"}

        if not image_data.startswith("http") and not image_data.startswith("data:image/"):
            logger.error(f"Invalid image data format: {image_data[:40]}...")
            return {"ok": False, "reason": "Invalid image format"}

        image_part = ImageContent(image=image_data, inference_detail="auto")

        agent = getattr(room.pipeline, "agent", None)
        if agent and hasattr(agent, "chat_context"):
            agent.chat_context.add_message(
                role=ChatRole.USER,
                content=["Here is the latest image for analysis", image_part],
            )
            
            logger.info("✅ Image successfully added to chat context")
            return {"ok": True, "detail": "Image added to context"}
        else:
            logger.error("No agent chat context available")
            return {"ok": False, "reason": "No agent chat context available"}

    except Exception as e:
        logger.exception(f"capture_and_process_frame failed: {e}")
        return {"ok": False, "reason": str(e)}

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly voice assistant that can see through the user's camera, listen, and speak naturally. "
                "Your role is to understand the user's intent and respond helpfully — describing what's visible, performing actions when needed, or providing detailed explanations when asked. "

                "Speak clearly and naturally using simple, descriptive language without formatting. Keep replies short unless the user asks for more detail. "

                "When the user refers to something visible or says things like 'look', 'analyze', or 'describe this', 'object holding?', 'what do you see?' etc ,capture a fresh frame and explain what you see — objects, colors, text, people, or expressions. "
                "If unclear, politely ask the user to adjust the view. "

                "Only capture a frame when the intent involves visual context; otherwise, answer using general knowledge conversationally. "
                "Never assume unseen details or reuse old frames. "

                "Act based on intent — for example, describe, analyze, extract details, or explain what's happening. "
            ),
            tools=[capture_and_process_frame],
            
        )

        
    async def on_enter(self) -> None:
        await self.session.say("Hello, I am vision AI assistant. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=OpenAILLM(),
        # llm=GoogleLLM(),
        # tts=ElevenLabsTTS(),
        tts=DeepgramTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        denoise=RNNoise()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="22z5-h5y3-ij49", 
        name="Sandbox Agent", 
        playground=True,
        vision=True
    )
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

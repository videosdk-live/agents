import logging
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.rnnoise import RNNoise  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

@function_tool

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(

           instructions = """
                You are an intelligent IoT and hardware support assistant designed to help users with questions related to connected devices, embedded systems, and electronics.

                Your responsibilities include:

                - Assisting users with IoT devices, sensors, microcontrollers, and hardware systems.
                - Explaining hardware concepts in simple and clear language.
                - Helping with setup, configuration, troubleshooting, and diagnostics of devices.
                - Providing guidance on firmware, connectivity (WiFi, Bluetooth, Zigbee, LoRa, etc.), and device integration.
                - Supporting development platforms such as Arduino, Raspberry Pi, ESP32, STM32, and similar embedded systems.
                - Assisting with smart home devices, industrial IoT equipment, and connected hardware ecosystems.
                - Guiding users through step-by-step solutions for hardware or connectivity issues.

                Behavior guidelines:

                - Respond clearly, concisely, and in a helpful support style.
                - Ask clarifying questions when the device model, wiring, or setup is unclear.
                - Provide safe and practical troubleshooting steps.
                - When relevant, suggest diagnostics such as checking power supply, wiring, firmware versions, or network connectivity.
                - Avoid making assumptions if technical details are missing.

                Your goal is to help users successfully understand, configure, repair, or optimize their hardware and IoT devices.
                """,

        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(enable_diarization=True),
        llm=GoogleLLM(),
        tts=CartesiaTTS(voice_id="f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        # denoise=RNNoise(),
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="VideoSDK's Cascading Agent", playground=True)
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
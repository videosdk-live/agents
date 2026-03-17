import logging
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions, MCPServerHTTP, Options, InterruptConfig
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.sarvamai import SarvamAISTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

INSTRUCTIONS = """
    You are a doctor appointment follow-up assistant.

    Your task is to confirm whether the customer will attend their scheduled appointment.

    When retrieving customer appointment details, call: get_customer_data

    Use the slot_id provided in the call metadata.

    Welcome the customer by his name and provide the appointment details to him

    Greet the customer "Hello [Customer Name] this is a reminder for your appointment with [Doctor Name] on [Appointment Date and Time]. Can you please confirm if you will attend?"

    If the customer confirms they will attend, update the appointment status to "Booked" using the tool "confirmed_booking" with the customer's Slot ID.

    If the customer says they will not attend, update the appointment status to "Cancelled" using the tool "cancel_booking" with the customer's Slot ID.

    Do not ask for the customer's name. Always use the Slot ID from the retrieved data.

    Keep responses short and clear.

    Always speak the date and time of the appointment in a natural, human-friendly way. (e.g. March 15 at nine fifteen PM instead of 15/03/2024 21:00)

    User can ask qustions related to his appointment date and time, doctor name, doctors speciality. 
"""


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=INSTRUCTIONS,
            mcp_servers=[
                MCPServerHTTP(
                    endpoint_url="MCP_PRODUCTION_URL_FROM_N8N_MCP_TRIGGER",
                )
            ]
        )
        
    async def on_enter(self) -> None:
        await self.session.say(
            message="This call is for appointment scheduling purposes",
            interruptible=False   
        )
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=SarvamAISTT(),
        llm=OpenAILLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        interrupt_config=InterruptConfig(
            interrupt_min_words=1
        )
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(name="Restaurant Agent", playground=True)
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    options = Options(
        agent_id="YOUR_AGENT_ID",
        max_processes=3, 
        register=True,
    )

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context, options=options)
    job.start()




import logging
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob ,ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class TravelAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a travel assistant. Help users with general travel inquiries and guide them to booking or travel support when needed.""",
        )

    async def on_enter(self) -> None:
        await self.session.reply(instructions="Greet the user and ask how you can help with their travel plans.")

    async def on_exit(self) -> None:
        await self.session.say("Safe travels!")

    @function_tool()
    async def transfer_to_booking(self) -> Agent:
        """Transfer the user to a booking specialist for reservations and scheduling."""
        return BookingAgent(inherit_context=True)

    @function_tool()
    async def transfer_to_travel_support(self) -> Agent:
        """Transfer the user to travel support for itinerary issues, disruptions, or requirements."""
        return TravelSupportAgent(inherit_context=True)


class BookingAgent(Agent):
    def __init__(self, inherit_context: bool = False):
        super().__init__(
            instructions="""You are a booking specialist. Help users book or modify flights, hotels, and travel reservations.""",
            inherit_context=inherit_context
        )

    async def on_enter(self) -> None:
        await self.session.say("I'm a booking specialist. What would you like to book or modify today?")

    async def on_exit(self) -> None:
        await self.session.say("Your booking request is complete. Have a great trip!")


class TravelSupportAgent(Agent):
    def __init__(self, inherit_context: bool = False):
        super().__init__(
            instructions="""You are a travel support specialist. Help users with travel disruptions, requirements, or itinerary assistance.""",
            inherit_context=inherit_context
        )

    async def on_enter(self) -> None:
        await self.session.say("You're now connected with travel support. What travel issue can I help you with?")

    async def on_exit(self) -> None:
        await self.session.say("Glad I could assist. Safe journey!")

async def entrypoint(ctx: JobContext):
    
    agent = TravelAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
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
    room_options = RoomOptions(name="Multi Agent Switch Agent", playground=True)
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

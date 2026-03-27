"""
Use Case: Clinic triage agent that routes patients to the appropriate specialist (MedPlus Clinic).
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: Multi-agent handoff — TriageAgent transfers to specialist agents with inherit_context=True.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class CardiologyAgent(Agent):
    def __init__(self, inherit_context: bool = False):
        super().__init__(
            instructions="""You are a cardiology intake specialist at MedPlus Clinic.
            Collect the patient's cardiac symptoms, severity on a scale of 1-10, and current medications.
            Ask about chest pain location, radiation, onset, and any associated symptoms (breathlessness, sweating).
            Inform the patient that a cardiologist will follow up within 2 hours.
            Be calm and reassuring — cardiac patients may be anxious.""",
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "You're now connected with the cardiology intake team at MedPlus Clinic. "
            "I'm here to gather some information before your appointment. "
            "Can you describe your symptoms in detail, starting with when they began?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you. A cardiologist will contact you within 2 hours. "
            "If your symptoms worsen, please call emergency services immediately."
        )


class GeneralMedicineAgent(Agent):
    def __init__(self, inherit_context: bool = False):
        super().__init__(
            instructions="""You are a general medicine intake specialist at MedPlus Clinic.
            Collect the patient's chief complaint, symptom duration, severity, and any current medications.
            Ask about fever, pain level (1-10), and whether symptoms are worsening.
            Inform the patient that a general practitioner will follow up within 4 hours.
            Be friendly, clear, and thorough.""",
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "You're now with the general medicine team at MedPlus Clinic. "
            "I'll collect a few details to prepare for your consultation. "
            "Can you tell me more about what brought you in today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for sharing that. A doctor will be in touch within 4 hours. Take care!"
        )


class TriageAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the triage nurse at MedPlus Clinic.
            Ask the patient about their chief complaint in one or two questions.
            Route them to the correct department based on their symptoms:
            - Cardiology: chest pain, heart palpitations, shortness of breath, arm/jaw pain
            - General Medicine: fever, cold, body ache, headache, stomach issues, and all other complaints
            Be efficient — triage should take under 2 minutes.
            Do not diagnose. Only route.""",
        )

    async def on_enter(self) -> None:
        await self.session.reply(
            instructions="Greet the patient warmly and ask for their chief complaint in one sentence."
        )

    async def on_exit(self) -> None:
        await self.session.say("Transferring you now. Please hold for a moment.")

    @function_tool()
    async def transfer_to_cardiology(self) -> Agent:
        """Transfer the patient to the cardiology intake specialist."""
        return CardiologyAgent(inherit_context=True)

    @function_tool()
    async def transfer_to_general_medicine(self) -> Agent:
        """Transfer the patient to the general medicine intake specialist."""
        return GeneralMedicineAgent(inherit_context=True)


async def entrypoint(ctx: JobContext):
    agent = TriageAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="MedPlus Clinic - Triage", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

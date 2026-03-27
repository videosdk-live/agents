"""
Use Case: Healthcare clinic front-desk receptionist for Sunrise Clinic.
Pipeline: P1 — DeepgramSTT + OpenAILLM + ElevenLabsTTS + SileroVAD + TurnDetector
Demonstrates: UtteranceHandle for sequential multi-step speech, graceful interruption during tool calls.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions, UtteranceHandle
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class ClinicReceptionistAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the virtual receptionist at Sunrise Clinic.
            Collect the patient's full name, preferred appointment date, and preferred time slot.
            Use check_availability to confirm the slot is open before booking.
            Use book_appointment to confirm the reservation.
            Speak slowly and clearly — patients may be elderly or anxious.
            Always confirm each detail before moving to the next step.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Good morning, thank you for calling Sunrise Clinic. "
            "I'm your virtual receptionist. I can help you schedule or reschedule an appointment. "
            "May I start with your full name please?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for choosing Sunrise Clinic. We look forward to seeing you. Goodbye!"
        )

    @function_tool
    async def check_availability(self, date: str, time: str) -> dict:
        """Check if a specific date and time slot is available for booking.

        Args:
            date: The requested appointment date (e.g., 'March 15' or '2026-03-15')
            time: The requested time slot (e.g., '10:00 AM' or '14:30')
        """
        # In production, replace with a real calendar/scheduling API call.
        unavailable_slots = [("March 15", "10:00 AM"), ("March 16", "9:00 AM")]
        is_available = (date, time) not in unavailable_slots
        return {
            "date": date,
            "time": time,
            "available": is_available,
            "message": f"Slot on {date} at {time} is {'available' if is_available else 'already booked'}.",
        }

    @function_tool
    async def book_appointment(self, patient_name: str, date: str, time: str) -> dict:
        """Book a confirmed appointment slot for the patient.

        Args:
            patient_name: Full name of the patient
            date: Appointment date (e.g., 'March 15')
            time: Appointment time (e.g., '10:30 AM')
        """
        utterance: UtteranceHandle | None = self.session.current_utterance

        confirmation_id = f"SC-{abs(hash(patient_name + date + time)) % 10000:04d}"

        # Announce each detail sequentially, checking for interruption between steps.
        handle1 = self.session.say(
            f"Perfect. I've booked your appointment, {patient_name.split()[0]}."
        )
        await handle1

        if utterance and utterance.interrupted:
            return {"status": "interrupted", "confirmation_id": confirmation_id}

        handle2 = self.session.say(
            f"Your appointment is confirmed for {date} at {time}."
        )
        await handle2

        if utterance and utterance.interrupted:
            return {"status": "interrupted", "confirmation_id": confirmation_id}

        handle3 = self.session.say(
            f"Your confirmation number is {confirmation_id}. "
            "Please arrive 10 minutes early. Is there anything else I can help you with?"
        )
        await handle3

        return {
            "confirmation_id": confirmation_id,
            "patient_name": patient_name,
            "date": date,
            "time": time,
            "clinic": "Sunrise Clinic",
            "status": "confirmed",
        }


async def entrypoint(ctx: JobContext):
    agent = ClinicReceptionistAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sunrise Clinic Receptionist", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

"""
Use Case: Emergency insurance hotline agent with full provider failover (SafeGuard Insurance).
Pipeline: P1 — FallbackSTT + FallbackLLM + FallbackTTS + SileroVAD + TurnDetector
Demonstrates: FallbackSTT/LLM/TTS with temporary_disable_sec and permanent_disable_after_attempts.
Env Vars: VIDEOSDK_AUTH_TOKEN, OPENAI_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY, CEREBRAS_API_KEY
"""

import logging
from videosdk.agents import (
    Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions,
    FallbackSTT, FallbackLLM, FallbackTTS,
)
from videosdk.plugins.openai import OpenAISTT, OpenAILLM, OpenAITTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.cerebras import CerebrasLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class EmergencyHotlineAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the emergency support agent for SafeGuard Insurance.
            You handle urgent insurance-related calls: accident claims, roadside assistance, and medical emergencies.
            Always stay calm, professional, and empathetic — callers may be in distress.
            Collect: caller name, policy number, type of emergency, and current location.
            For roadside assistance, use the dispatch_roadside_assistance tool.
            For injury claims, use create_emergency_claim tool and advise the caller to seek medical attention.
            Never refuse to help. Never put a caller on hold for more than 30 seconds.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "You've reached SafeGuard Insurance Emergency Support. "
            "I'm here 24/7 to help you. Are you safe? "
            "Please tell me your name and the nature of your emergency."
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Help is on the way. Stay calm and safe. SafeGuard Insurance is with you."
        )

    @function_tool
    async def dispatch_roadside_assistance(self, location: str, vehicle_issue: str) -> dict:
        """Dispatch roadside assistance to the caller's location.

        Args:
            location: Caller's current location or address
            vehicle_issue: Description of the vehicle problem (flat tire, dead battery, etc.)
        """
        dispatch_id = f"RDA-{abs(hash(location + vehicle_issue)) % 100000:05d}"
        return {
            "dispatch_id": dispatch_id,
            "status": "dispatched",
            "eta_minutes": 25,
            "location": location,
            "issue": vehicle_issue,
            "message": f"Roadside unit dispatched. ETA: 25 minutes. Reference: {dispatch_id}",
        }

    @function_tool
    async def create_emergency_claim(self, caller_name: str, policy_number: str, incident_description: str) -> dict:
        """Create an emergency insurance claim for the caller.

        Args:
            caller_name: Full name of the policyholder
            policy_number: Insurance policy number
            incident_description: Brief description of the incident
        """
        claim_id = f"CLM-{abs(hash(caller_name + incident_description)) % 100000:05d}"
        return {
            "claim_id": claim_id,
            "status": "opened",
            "policy_number": policy_number,
            "adjuster_eta": "2 hours",
            "message": f"Emergency claim {claim_id} created. An adjuster will contact you within 2 hours.",
        }


async def entrypoint(ctx: JobContext):
    agent = EmergencyHotlineAgent()

    # All three pipeline stages have failover — the system never goes down.
    stt = FallbackSTT(
        [OpenAISTT(api_key="dgsdfsdfs"), DeepgramSTT()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
    )
    llm = FallbackLLM(
        [OpenAILLM(model="gpt-4o-mini"), CerebrasLLM()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
    )
    tts = FallbackTTS(
        [OpenAITTS(voice="alloy"), CartesiaTTS()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
    )

    pipeline = Pipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="SafeGuard Emergency Hotline", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

"""
Use Case: Virtual bank loan advisor avatar (Rohan) for BankEasy — explains loan options and calculates EMI.
Pipeline: P2 — GeminiRealtime + SimliAvatar (pure realtime with visual avatar)
Demonstrates: avatar= param with realtime pipeline, function tool for EMI calculation.
Env Vars: VIDEOSDK_AUTH_TOKEN, GOOGLE_API_KEY, SIMLI_API_KEY

Alternative (Standard voice pipeline):
    Uncomment the P1 block below to use DeepgramSTT + OpenAILLM + ElevenLabsTTS with the Simli avatar.
"""

import os
import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.simli import SimliAvatar, SimliConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class LoanAdvisorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Rohan, a loan advisor at BankEasy, a digital banking platform.
            Help customers understand loan options:
            - Home Loan: up to ₹1 crore, 7.5% p.a., tenure 5-30 years
            - Personal Loan: up to ₹10 lakh, 12% p.a., tenure 1-5 years
            - Car Loan: up to ₹20 lakh, 9% p.a., tenure 1-7 years
            Use calculate_emi when the customer wants to know monthly payments.
            Help customers decide based on their income and credit score.
            Speak in simple, jargon-free English. Be warm, patient, and trustworthy.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome to BankEasy. I'm Rohan, your personal loan advisor. "
            "Whether you're thinking about a home loan, personal loan, or car loan, "
            "I'm here to walk you through your options. What can I help you explore today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for choosing BankEasy. "
            "Apply online at bankeast.in or visit your nearest branch. Have a great day!"
        )

    @function_tool
    async def calculate_emi(self, principal: float, annual_rate: float, tenure_years: int) -> dict:
        """Calculate the monthly EMI for a loan.

        Args:
            principal: Loan amount in INR
            annual_rate: Annual interest rate as a percentage (e.g., 7.5 for 7.5%)
            tenure_years: Loan tenure in years
        """
        monthly_rate = annual_rate / (12 * 100)
        n = tenure_years * 12

        if monthly_rate == 0:
            emi = principal / n
        else:
            emi = principal * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)

        total_payment = emi * n
        total_interest = total_payment - principal

        return {
            "principal": round(principal),
            "annual_rate_percent": annual_rate,
            "tenure_years": tenure_years,
            "monthly_emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
        }


async def entrypoint(ctx: JobContext):
    simli_avatar = SimliAvatar(
        api_key="086mueap451x7aw61kxsm",
        config=SimliConfig(
            faceId="cace3ef7-a4c4-425d-a8cf-a5358eb0c427",
            maxSessionLength=1800,
            maxIdleTime=600,
        ),
        is_trinity_avatar=True,
    )

    agent = LoanAdvisorAgent()

    # --- P2: Pure realtime pipeline with Simli avatar (primary) ---
    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"],
        ),
    )
    pipeline = Pipeline(llm=model, avatar=simli_avatar)

    # --- P1: Standard voice pipeline with Simli avatar (alternative) ---
    # from videosdk.plugins.deepgram import DeepgramSTT
    # from videosdk.plugins.openai import OpenAILLM
    # from videosdk.plugins.elevenlabs import ElevenLabsTTS
    # from videosdk.plugins.silero import SileroVAD
    # from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
    # pre_download_model()
    # pipeline = Pipeline(
    #     stt=DeepgramSTT(),
    #     llm=OpenAILLM(model="gpt-4o-mini"),
    #     tts=ElevenLabsTTS(),
    #     vad=SileroVAD(),
    #     turn_detector=TurnDetector(),
    #     avatar=simli_avatar,
    # )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="BankEasy Loan Advisor - Rohan", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

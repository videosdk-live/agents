"""
Use Case: Telecom customer care agent with human transfer capability (ConnectTel).
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: session.call_transfer() for live agent handoff, Options for agent registration.
Env Vars: VIDEOSDK_AUTH_TOKEN, VIDEOSDK_CALL_TRANSFER_TO, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import os
import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions, Options
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class ConnectTelAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a virtual customer care agent for ConnectTel, a telecom provider.
            Help customers with:
            - Billing queries: use get_bill_details tool to fetch account information
            - Plan upgrades: describe available plans (Basic $15/mo, Pro $30/mo, Ultimate $50/mo)
            - Outage reports: acknowledge the outage and provide the reference ticket number
            If the customer is frustrated, distressed, or the issue requires human judgement,
            immediately use transfer_to_human_agent without asking for confirmation.
            Be empathetic and efficient. Never put a frustrated customer on hold.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Thank you for calling ConnectTel customer care. "
            "I'm your virtual agent and I'm here to help with billing, plans, and outages. "
            "What can I assist you with today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for calling ConnectTel. We value your business. Goodbye!"
        )

    @function_tool
    async def get_bill_details(self, account_id: str) -> dict:
        """Retrieve billing information for a customer account.

        Args:
            account_id: Customer's account ID or phone number
        """
        # In production, replace with actual billing API call.
        mock_bills = {
            "ACC-001": {"plan": "Pro", "monthly_charge": 30.00, "balance_due": 30.00, "due_date": "March 15", "last_payment": "Feb 15"},
            "ACC-002": {"plan": "Basic", "monthly_charge": 15.00, "balance_due": 0.00, "due_date": "March 20", "last_payment": "Feb 20"},
        }
        account = mock_bills.get(account_id.upper())
        if not account:
            return {"found": False, "message": "Account not found. Please verify the account ID."}
        return {"found": True, "account_id": account_id.upper(), **account}

    @function_tool
    async def transfer_to_human_agent(self) -> dict:
        """Transfer the call to a live human agent for complex or escalated issues."""
        token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        transfer_to = os.getenv("VIDEOSDK_CALL_TRANSFER_TO")
        if not transfer_to:
            return {"status": "error", "message": "No transfer destination configured."}
        result = await self.session.call_transfer(token, transfer_to)
        return {"status": "transferred", "result": result}


async def entrypoint(ctx: JobContext):
    agent = ConnectTelAgent()

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
    room_options = RoomOptions(name="ConnectTel Customer Care", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(
        entrypoint=entrypoint,
        jobctx=make_context,
        options=Options(agent_id="YOUR_AGENT_ID", register=True, host="localhost", port=8081),
    )
    job.start()

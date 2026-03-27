"""
Use Case: E-commerce order support agent for ShopEasy.
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: Function tools with realistic data shapes, structured on_enter greeting.
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


class CustomerSupportAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Aria, a customer support agent for ShopEasy, an online retail store.
            Help customers with order status, returns, refunds, and general inquiries.
            Always use the lookup_order_status tool before providing any order details.
            If you cannot resolve an issue, use create_support_ticket to escalate it.
            Be concise, professional, and empathetic. Never make up order information.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, you've reached ShopEasy customer support. I'm Aria. "
            "Can I have your order number to get started?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thank you for contacting ShopEasy. Have a great day!")

    @function_tool
    async def lookup_order_status(self, order_id: str) -> dict:
        """Look up the current status of a customer order.

        Args:
            order_id: The customer's order ID (e.g., ORD-12345)
        """
        # In production, replace with an actual database or API call.
        mock_orders = {
            "ORD-12345": {"status": "shipped", "carrier": "FedEx", "eta": "2 days", "tracking": "FX789012345"},
            "ORD-67890": {"status": "processing", "eta": "3-5 business days"},
            "ORD-11111": {"status": "delivered", "delivered_on": "2 days ago"},
        }
        order = mock_orders.get(order_id.upper())
        if not order:
            return {"found": False, "message": "Order not found. Please verify the order ID."}
        return {"found": True, "order_id": order_id.upper(), **order}

    @function_tool
    async def create_support_ticket(self, issue: str, order_id: str = None) -> dict:
        """Create a support ticket for issues that cannot be resolved immediately.

        Args:
            issue: Description of the customer's issue
            order_id: Optional order ID related to the issue
        """
        ticket_id = f"TKT-{abs(hash(issue)) % 100000:05d}"
        return {
            "ticket_id": ticket_id,
            "status": "created",
            "resolution_eta": "24 hours",
            "message": f"Ticket {ticket_id} created. Our team will follow up within 24 hours.",
            "order_id": order_id,
        }


async def entrypoint(ctx: JobContext):
    agent = CustomerSupportAgent()

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
    room_options = RoomOptions(room_id="<room_id>", name="ShopEasy Customer Support", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

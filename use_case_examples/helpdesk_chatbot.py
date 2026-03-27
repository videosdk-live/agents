"""
Use Case: Text-based IT helpdesk bot for Acme Corp — resolves employee tech issues via chat.
Pipeline: P5 — GoogleLLM only (no STT/TTS; text in via pubsub, text out via pubsub)
Demonstrates: LLM-only pipeline, PubSub CHAT topic input, HELPDESK_RESPONSE topic output.
Env Vars: VIDEOSDK_AUTH_TOKEN, GOOGLE_API_KEY
"""

import asyncio
import logging
from videosdk import PubSubSubscribeConfig, PubSubPublishConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class HelpdeskAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="""You are the IT Helpdesk bot for Acme Corp.
            Help employees with password resets, VPN setup, software installation, and hardware issues.
            Always ask for the employee's ID before resolving any account-related issue.
            For issues you cannot resolve, create a Tier 2 escalation and provide a ticket number.
            Keep responses concise — employees need quick answers to get back to work.
            Do not discuss topics unrelated to IT support.""",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self._publish("IT Helpdesk ready. Send your issue to topic 'CHAT' and I'll assist you. Please include your employee ID.")

    async def on_exit(self) -> None:
        await self._publish("IT Helpdesk session ended. For urgent issues, call ext. 5555.")

    async def _publish(self, message: str) -> None:
        try:
            publish_config = PubSubPublishConfig(topic="HELPDESK_RESPONSE", message=message)
            await self.ctx.room.publish_to_pubsub(publish_config)
        except Exception as e:
            logging.error("Failed to publish helpdesk response: %s", e)


async def entrypoint(ctx: JobContext):
    agent = HelpdeskAgent(ctx)

    pipeline = Pipeline(llm=GoogleLLM())

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_chat_message(message):
        text = message.get("message", "") if isinstance(message, dict) else str(message)
        if text.strip():
            logging.info("Employee message: %s", text)
            await pipeline.process_text(text)

    def on_chat_message_wrapper(message):
        asyncio.create_task(on_chat_message(message))

    @pipeline.on("llm")
    async def on_llm(data: dict):
        text = data.get("text", "")
        if text.strip():
            logging.info("Helpdesk response: %s", text)
            try:
                publish_config = PubSubPublishConfig(topic="HELPDESK_RESPONSE", message=text)
                await ctx.room.publish_to_pubsub(publish_config)
            except Exception as e:
                logging.error("Failed to publish response: %s", e)

    async def cleanup_session():
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        await ctx.room.wait_for_participant()

        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_chat_message_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Acme IT Helpdesk", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

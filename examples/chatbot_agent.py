# Chatbot agent: LLM only. Text in/out via pubsub.

import asyncio
import logging
from videosdk import PubSubSubscribeConfig, PubSubPublishConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class ChatbotAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="You are a helpful assistant. Respond naturally and concisely to the user.",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.send_text_response("Hello! Send me a text message on topic 'CHAT' and I'll respond with text.")
        print("Chatbot ready. Send text via pubsub topic 'CHAT'.")

    async def on_exit(self) -> None:
        await self.send_text_response("Goodbye!")

    async def send_text_response(self, message: str) -> None:
        """Send text response via pubsub on AGENT_RESPONSE (so we don't re-read it on CHAT)."""
        try:
            publish_config = PubSubPublishConfig(topic="AGENT_RESPONSE", message=message)
            await self.ctx.room.publish_to_pubsub(publish_config)
        except Exception as e:
            logging.error("Failed to publish text response: %s", e)


async def entrypoint(ctx: JobContext):
    agent = ChatbotAgent(ctx)

    pipeline = Pipeline(llm=GoogleLLM())

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_pubsub_message(message):
        """Handle incoming user text from pubsub topic CHAT (agent responses go to AGENT_RESPONSE)."""
        text = message.get("message", "") if isinstance(message, dict) else str(message)
        print("Received text:", text)
        if text.strip():
            await pipeline.process_text(text)

    def on_pubsub_message_wrapper(message):
        asyncio.create_task(on_pubsub_message(message))

    @pipeline.on("content_generated")
    async def on_content_generated(data: dict):
        """Send agent text response to AGENT_RESPONSE topic (avoids re-processing on CHAT)."""
        text = data.get("text", "")
        if text.strip():
            print("Agent response:", text)
            try:
                publish_config = PubSubPublishConfig(topic="AGENT_RESPONSE", message=text)
                await ctx.room.publish_to_pubsub(publish_config)
            except Exception as e:
                logging.error("Failed to publish response: %s", e)

    async def cleanup_session():
        print("Cleaning up session...")
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        print("Session ended:", reason)
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)

        print("Waiting for participant...")
        await ctx.room.wait_for_participant()

        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_pubsub_message_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="VideoSDK Chatbot Agent (LLM only)",
        playground=True,
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

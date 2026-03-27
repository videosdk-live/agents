"""
Use Case: Real-time Spanish language tutor (Sofia) — ultra-low-latency voice interaction.
Pipeline: P2 — OpenAIRealtime (no separate STT/TTS; realtime model handles audio end-to-end)
Demonstrates: Pure realtime pipeline, server-side VAD, OpenAI Realtime model.
Env Vars: VIDEOSDK_AUTH_TOKEN, OPENAI_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import TurnDetection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class SpanishTutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Sofia, a warm and patient Spanish language tutor.
            The student is a beginner learning conversational Spanish.
            For each exchange:
            - Speak a short phrase slowly in Spanish first
            - Then immediately repeat it in English
            - Gently correct grammar or pronunciation mistakes without discouraging the student
            - Keep each topic focused: greetings, numbers, colors, food, or directions
            - Celebrate small wins with encouragement like 'Muy bien!' or '¡Perfecto!'
            Never overwhelm the student with long explanations. Keep lessons short and interactive.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "¡Hola! I'm Sofia, your Spanish tutor. "
            "We'll start with something simple. Ready? Let's begin with greetings. "
            "Repeat after me: Hola, ¿cómo estás? — Hello, how are you?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "¡Hasta luego! Great work today. Keep practicing — practice makes perfect!"
        )


async def entrypoint(ctx: JobContext):
    model = OpenAIRealtime(
        model="gpt-realtime-2025-08-28",
        config=OpenAIRealtimeConfig(
            voice="shimmer",
            modalities=["audio"],
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
            ),
            tool_choice="auto",
        ),
    )

    pipeline = Pipeline(llm=model)

    agent = SpanishTutorAgent()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Spanish Tutor - Sofia", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

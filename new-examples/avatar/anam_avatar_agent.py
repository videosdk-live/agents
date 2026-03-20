"""
Use Case: AI-powered HR interviewer avatar (Maya) for TalentFirst — conducts structured behavioral interviews.
Pipeline: P1 — DeepgramSTT + OpenAILLM + ElevenLabsTTS + SileroVAD + TurnDetector + AnamAvatar
Demonstrates: avatar= param in unified Pipeline, function tool for logging interview scores per question.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, ANAM_API_KEY, ANAM_AVATAR_ID

Alternative (Realtime mode):
    Uncomment the P2 block below and comment out the P1 block to use OpenAI Realtime with the Anam avatar.
"""

import os
import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.anam import AnamAvatar

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()

INTERVIEW_QUESTIONS = [
    "Tell me about a time you led a project under a tight deadline.",
    "Describe a situation where you had to resolve a conflict within a team.",
    "Give me an example of when you had to adapt quickly to a major change.",
    "Tell me about a time you failed and what you learned from it.",
    "Describe your proudest professional achievement and your specific contribution.",
]


class HRInterviewerAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""You are Maya, an AI HR interviewer for TalentFirst, a talent acquisition platform.
            Conduct a structured behavioral interview using the STAR method (Situation, Task, Action, Result).
            Ask the following 5 questions, one at a time, in order:
            {chr(10).join(f"{i+1}. {q}" for i, q in enumerate(INTERVIEW_QUESTIONS))}
            After each answer, probe with one follow-up if the STAR elements are incomplete.
            After all 5 questions, use score_answer for each response, then give a brief summary.
            Be professional, encouraging, and neutral. Do not reveal scores to the candidate.""",
        )
        self.scores: dict = {}

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome to your TalentFirst behavioral interview. I'm Maya, your interviewer today. "
            "This interview consists of 5 questions, each focusing on a real situation from your experience. "
            "Take your time — there are no trick questions. Shall we begin?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for your time today. Your responses have been recorded. "
            "The TalentFirst team will be in touch within 2 business days. Best of luck!"
        )

    @function_tool
    async def score_answer(self, question_number: int, question: str, answer_summary: str, score: int) -> dict:
        """Log the interviewer's internal score for a candidate's answer (not revealed to candidate).

        Args:
            question_number: Question number (1-5)
            question: The interview question asked
            answer_summary: A 1-2 sentence summary of the candidate's answer
            score: Score from 1 (poor) to 5 (excellent) based on STAR completeness and relevance
        """
        self.scores[question_number] = {
            "question": question,
            "summary": answer_summary,
            "score": score,
        }
        logging.info("[SCORE] Q%d: %d/5 — %s", question_number, score, answer_summary[:60])
        return {
            "question_number": question_number,
            "score": score,
            "logged": True,
            "total_scored": len(self.scores),
        }


async def entrypoint(ctx: JobContext):
    anam_avatar = AnamAvatar(
        api_key=os.getenv("ANAM_API_KEY"),
        avatar_id=os.getenv("ANAM_AVATAR_ID"),
    )

    agent = HRInterviewerAgent()

    # --- P1: Standard voice pipeline with Anam avatar (primary) ---
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-3", language="en"),
        llm=OpenAILLM(model="gpt-4o-mini"),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8),
        avatar=anam_avatar,
    )

    # --- P2: Realtime pipeline with Anam avatar (alternative — lower latency) ---
    # from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
    # from openai.types.beta.realtime.session import TurnDetection
    # model = OpenAIRealtime(
    #     model="gpt-realtime-2025-08-28",
    #     config=OpenAIRealtimeConfig(
    #         voice="shimmer",
    #         modalities=["audio"],
    #         turn_detection=TurnDetection(type="server_vad", threshold=0.5),
    #     ),
    # )
    # pipeline = Pipeline(llm=model, avatar=anam_avatar)

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="TalentFirst - HR Interview", playground=False)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

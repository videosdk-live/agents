"""
Use Case: Voice-driven blog writer powered by a sequential LangGraph pipeline.
Pipeline: P1 — DeepgramSTT + LangGraphLLM + CartesiaTTS + SileroVAD + TurnDetector

Demonstrates: LangGraphLLM with a 3-question gathering phase before writing,
sequential section writing (one LLM call per section — observable in logs),
and a title-based filename.

Graph layout:
  START → coordinator_node   (extracts topic / audience / tone from conversation)
        ↓ all 3 gathered?
        → planner_node       (structured output: 4 BlogSection objects)
        → write_sections_node (4 sequential LLM calls — one per section, logged)
        → compiler_node      (join → save {title-slug}_{datetime}.md)
        → synthesizer_node  ← OUTPUT NODE (spoken announcement)
        ↓ info still missing?
        → synthesizer_node   (asks the next gathering question)
  END

All nodes use Gemini. output_node="synthesizer_node" ensures only the final
spoken summary (or gathering question) reaches TTS.

Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Install:  pip install -r requirements.txt
"""

import logging
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.langchain import LangGraphLLM

from dotenv import load_dotenv
load_dotenv(override=True)

pre_download_model()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

class GatheringInfo(BaseModel):
    """What we know about the blog so far from the conversation."""
    topic: str = Field(default="", description="The blog topic. Empty string if not yet stated.")
    audience: str = Field(default="", description="Target audience (e.g. 'developers', 'students'). Empty if not yet stated.")
    tone: str = Field(default="", description="Writing tone (e.g. 'professional', 'casual', 'technical'). Empty if not yet stated.")

class BlogSection(BaseModel):
    name: str
    description: str

class BlogSections(BaseModel):
    sections: list[BlogSection]


class BlogState(MessagesState):
    topic: str
    audience: str
    tone: str
    sections: list[BlogSection]
    completed_sections: list[str]   
    filename: str
    blog_done: bool          


_MODEL = "gemini-2.5-flash"

_coordinator_llm = ChatGoogleGenerativeAI(model=_MODEL).with_structured_output(GatheringInfo)
_planner_llm     = ChatGoogleGenerativeAI(model=_MODEL).with_structured_output(BlogSections)
_writer_llm      = ChatGoogleGenerativeAI(model=_MODEL, streaming=True)
_synth_llm       = ChatGoogleGenerativeAI(model=_MODEL, streaming=True)

_COORDINATOR_PROMPT = SystemMessage(content=(
    "Extract blog writing preferences from the conversation. "
    "Return exactly what the user has explicitly stated — do not guess or infer:\n"
    "  topic    — the specific blog topic (empty string if not yet mentioned)\n"
    "  audience — who the blog is for (empty string if not yet mentioned)\n"
    "  tone     — writing style such as professional, casual, technical (empty string if not yet mentioned)"
))

_PLANNER_PROMPT = SystemMessage(content=(
    "You are a blog outline specialist. "
    "Plan exactly 4 distinct sections for the blog. "
    "Each section should have a clear name and one-sentence description. "
    "Tailor the sections to the stated audience and tone."
))

_WRITER_PROMPT = SystemMessage(content=(
    "You are a professional blog writer. "
    "Write a complete blog section with a ## heading and 2–3 informative paragraphs. "
    "Use concrete examples. Match the requested tone and audience."
))

_SYNTH_PROMPT = SystemMessage(content=(
    "You are Aria, a friendly AI writing assistant on a phone call. "
    "Keep all responses under 2 sentences. Be warm and conversational.\n\n"
    "You will receive a context line. Respond as follows:\n"
    "- BLOG_SAVED: announce the blog was written and saved.\n"
    "- MISSING_TOPIC: ask what topic they'd like to write about.\n"
    "- MISSING_AUDIENCE: you already have the topic, ask who the blog is for.\n"
    "- MISSING_TONE: you have topic and audience, ask about the writing tone or style.\n"
    "- CHAT:<msg>: respond naturally and conversationally — the blog is already done, "
    "just have a friendly exchange."
))


def _title_slug(topic: str) -> str:
    """Convert a topic string to a filename-safe slug (max 45 chars)."""
    slug = re.sub(r"[^\w\s]", "", topic.lower()).strip()
    slug = re.sub(r"\s+", "-", slug)
    return slug[:45].rstrip("-")


def _blog_completed_in_history(messages: list) -> bool:
    """Return True if an assistant message already confirms a blog was saved this session."""
    for m in messages:
        if isinstance(m, AIMessage):
            text = m.content if isinstance(m.content, str) else ""
            if re.search(r"\.md\b", text) or (
                "saved" in text.lower() and "blog" in text.lower()
            ):
                return True
    return False


def coordinator_node(state: BlogState) -> dict:
    """Extract topic/audience/tone, or detect that the blog is already done."""
    if _blog_completed_in_history(state["messages"]):
        logger.info("[coordinator] blog already done — skipping pipeline")
        return {"blog_done": True}

    info: GatheringInfo = _coordinator_llm.invoke(
        [_COORDINATOR_PROMPT] + list(state["messages"])
    )
    logger.info(
        "[coordinator] topic=%r  audience=%r  tone=%r",
        info.topic, info.audience, info.tone,
    )
    return {"topic": info.topic, "audience": info.audience, "tone": info.tone, "blog_done": False}


def planner_node(state: BlogState) -> dict:
    """Plan 4 blog sections tailored to the gathered topic, audience, and tone."""
    result: BlogSections = _planner_llm.invoke([
        _PLANNER_PROMPT,
        HumanMessage(content=(
            f"Topic: {state.get('topic')}\n"
            f"Audience: {state.get('audience')}\n"
            f"Tone: {state.get('tone')}"
        )),
    ])
    logger.info("[planner] planned %d sections: %s", len(result.sections), [s.name for s in result.sections])
    return {"sections": result.sections}


def write_sections_node(state: BlogState) -> dict:
    """Write each section sequentially — one Gemini call per section, logged individually."""
    sections = state.get("sections") or []
    completed: list[str] = []

    for i, section in enumerate(sections, 1):
        logger.info("[writer] section %d/%d — %s", i, len(sections), section.name)
        result = _writer_llm.invoke([
            _WRITER_PROMPT,
            HumanMessage(content=(
                f"Write the section '{section.name}' for a blog about '{state.get('topic')}' "
                f"targeting {state.get('audience')} in a {state.get('tone')} tone. "
                f"Focus: {section.description}"
            )),
        ])
        completed.append(result.content)
        logger.info("[writer] section %d done (%d chars)", i, len(result.content))

    return {"completed_sections": completed}


def compiler_node(state: BlogState) -> dict:
    """Join sections and save the blog to a title-based markdown file."""
    content = "\n\n".join(state.get("completed_sections") or [])
    slug = _title_slug(state.get("topic", "blog"))
    filename = f"{slug}.md"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(f"# {state.get('topic', 'Blog')}\n\n")
        fh.write(f"*Audience: {state.get('audience')} | Tone: {state.get('tone')}*\n\n")
        fh.write(content + "\n")
    logger.info("[compiler] saved → %s", filename)
    return {"blog_content": content, "filename": filename}


def synthesizer_node(state: BlogState) -> dict:
    """Announce the completed blog, chat after completion, or ask the next gathering question."""
    if state.get("blog_done"):
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        context = f"CHAT:{last_human.content if last_human else 'general follow-up'}"
    elif state.get("filename"):
        context = f"BLOG_SAVED: topic='{state['topic']}', filename={state['filename']}, sections={len(state.get('sections', []))}."
    elif not state.get("topic"):
        context = "MISSING_TOPIC"
    elif not state.get("audience"):
        context = f"MISSING_AUDIENCE: topic='{state['topic']}' is already provided."
    else:
        context = f"MISSING_TONE: topic='{state['topic']}', audience='{state['audience']}' are already provided."

    result = _synth_llm.invoke([_SYNTH_PROMPT, HumanMessage(content=context)])
    return {"messages": [result]}

def route_after_coordinator(state: BlogState) -> str:
    """Route: skip pipeline if blog done, write if all gathered, else ask next question."""
    if state.get("blog_done"):
        return "synthesizer_node"
    if state.get("topic") and state.get("audience") and state.get("tone"):
        return "planner"
    return "synthesizer_node"


builder = StateGraph(BlogState)
builder.add_node("coordinator",      coordinator_node)
builder.add_node("planner",          planner_node)
builder.add_node("write_sections",   write_sections_node)
builder.add_node("compiler",         compiler_node)
builder.add_node("synthesizer_node", synthesizer_node)

builder.add_edge(START, "coordinator")
builder.add_conditional_edges(
    "coordinator", route_after_coordinator,
    {"planner": "planner", "synthesizer_node": "synthesizer_node"},
)
builder.add_edge("planner",          "write_sections")
builder.add_edge("write_sections",   "compiler")
builder.add_edge("compiler",         "synthesizer_node")
builder.add_edge("synthesizer_node", END)

blog_graph = builder.compile()


class BlogWriterAgent(Agent):
    """Voice agent that writes blog posts through a 3-question gathering flow + Gemini pipeline."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are Aria, a friendly AI writing assistant. "
                "The LangGraph pipeline handles all gathering questions, planning, writing, and saving. "
                "Just greet the user warmly."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi! I'm Aria, your AI writing assistant powered by Gemini. "
            "I'll ask you a couple of quick questions before I start writing. "
            "What topic would you like me to write a blog about?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thanks for using the blog writer. Happy writing!")


async def entrypoint(ctx: JobContext):
    agent = BlogWriterAgent()

    langgraph_llm = LangGraphLLM(
        graph=blog_graph,
        output_node="synthesizer_node",
    )

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=langgraph_llm,
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Blog Writer", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
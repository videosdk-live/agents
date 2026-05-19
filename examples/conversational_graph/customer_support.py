"""
customer_support.py
===================
A simple customer-support chatbot using LLM + PubSub (text in/out, no voice).

Uses:
  - ConversationalGraph for structured conversation flow
  - GoogleLLM for natural language understanding & generation
  - PubSub for text messaging (topic CHAT → in, AGENT_RESPONSE → out)

Flow
----
START → welcome → identify_issue ──┬─ billing   → resolve_billing   ─┐
                                   ├─ technical → resolve_technical  ├→ feedback → END
                                   ├─ account   → resolve_account    ┘
                                   └─ other     → escalate_to_human ──→ END
"""

import asyncio
import logging
import re
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from videosdk.conversational_graph import (
    ConversationalGraph,
    GraphState,
    Context,
    Route,
    END,
    START,
    Interrupt,
    GraphConfig,
)

from videosdk import PubSubSubscribeConfig, PubSubPublishConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM

from pydantic import Field, field_validator
from typing import Optional


#  State 
VALID_ISSUE_TYPES = {"billing", "technical", "account", "other"}
VALID_SATISFACTION = {"happy", "neutral", "unhappy"}


class SupportState(GraphState):
    customer_name: Optional[str] = Field(None, description="Customer's name")
    issue_type: Optional[str] = Field(
        None, description="Issue category: billing | technical | account | other"
    )
    issue_description: Optional[str] = Field(None, description="Brief description of the issue")
    resolved: Optional[bool] = Field(None, description="Whether the issue was resolved")
    ticket_id: Optional[str] = Field(None, description="Support ticket ID if escalated")
    satisfaction: Optional[str] = Field(
        None, description="Customer satisfaction: happy | neutral | unhappy"
    )

    @field_validator("issue_type")
    @classmethod
    def _validate_issue_type(cls, v):
        if v and v.lower().strip() not in VALID_ISSUE_TYPES:
            raise ValueError(f"Must be one of {VALID_ISSUE_TYPES}")
        return v.lower().strip() if v else v

    @field_validator("satisfaction")
    @classmethod
    def _validate_satisfaction(cls, v):
        if v and v.lower().strip() not in VALID_SATISFACTION:
            raise ValueError(f"Must be one of {VALID_SATISFACTION}")
        return v.lower().strip() if v else v


#  Extraction schemas 
class NameSchema(GraphState):
    customer_name: str = Field(..., description="Customer's name")


class IssueSchema(GraphState):
    issue_description: str = Field(..., description="Brief description of the customer's issue")


#  Helpers
_NAME_PATTERNS = [
    re.compile(r"(?:my name is|i'm|i am|this is|call me|it's)\s+(\w+)", re.IGNORECASE),
]


def extract_name(text: str) -> str | None:
    """Extract a customer name from explicit name-giving patterns.

    Matches phrases like "my name is X", "I'm X", "call me X".
    Returns None for bare words — those are handled by LLM extraction via NameSchema.
    """
    if not text or not text.strip():
        return None
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text.strip())
        if match:
            return match.group(1)
    return None


#  Graph config 
config = GraphConfig(name="customer_support", debug=True, stream=True)
graph = ConversationalGraph(SupportState, config)


#  Nodes 
async def welcome(state: SupportState, ctx: Context):
    """Greet, collect customer name, then ask about their issue."""
    if state.customer_name and (state.issue_description or state.issue_type):
        return Route("identify_issue")

    if state.customer_name:
        await ctx.extractor.collect(schema=IssueSchema)
        return Interrupt(
            say=f"Nice to meet you, {state.customer_name}! How can I help you today? I can assist with billing, technical issues, or account problems.",
            id="ask_issue",
        )

    name = extract_name(ctx.last_user_message)
    if name:
        ctx.set_state("customer_name", name)
        await ctx.extractor.collect(schema=IssueSchema)
        return Interrupt(
            say=f"Nice to meet you, {name}! How can I help you today? I can assist with billing, technical issues, or account problems.",
            id="ask_issue",
        )

    await ctx.extractor.collect(schema=NameSchema)
    return Interrupt(
        say="Hello! Welcome to customer support. Could you please tell me your name?",
        id="ask_name",
    )


async def identify_issue(state: SupportState, ctx: Context):
    """Classify the customer's issue into a category."""
    if not state.issue_description and not state.issue_type:
        return Interrupt(
            say=f"Could you describe your issue, {state.customer_name}? "
                "I can help with billing, technical problems, or account questions.",
            id="ask_issue",
        )

    if state.issue_type and state.issue_type in VALID_ISSUE_TYPES:
        if not state.issue_description:
            msg = (ctx.last_user_message or "").strip()
            if msg:
                ctx.set_state("issue_description", msg)
        return Route("route_issue")

    intent = await ctx.extractor.match_intent(
        intents={
            "billing": "billing, charge, invoice, payment, refund, overcharged",
            "technical": "technical issue, bug, error, not working, crash, slow, broken, unable to open, device not responding, freeze,hang, stuck",
            "account": "account, login, password, locked out, update profile, settings",
        }
    )

    if intent and intent != "unknown":
        ctx.set_state("issue_type", intent)
    else:
        ctx.set_state("issue_type", "other")

    return Route("route_issue")


async def route_issue(state: SupportState, ctx: Context):
    """Pure routing node — conditional transition handles direction based on issue_type."""
    pass


async def resolve_billing(state: SupportState, ctx: Context):
    """Handle billing issues with LLM assistance."""
    if state.resolved is not None:
        return Route("feedback")

    await ctx.ask(
        f"The customer {state.customer_name} has a billing issue: "
        f"'{state.issue_description or 'not specified'}'. "
        "Acknowledge the issue empathetically, explain that you've reviewed their account, "
        "and offer to process a refund or adjustment if applicable. "
        "Ask if this resolves their concern.",
        interruptible=True,
    )
    ctx.set_state("resolved", True)
    return Interrupt(say="", id="await_response")


async def resolve_technical(state: SupportState, ctx: Context):
    """Handle technical issues with LLM assistance."""
    if state.resolved is not None:
        return Route("feedback")

    await ctx.ask(
        f"The customer {state.customer_name} has a technical issue: "
        f"'{state.issue_description or 'not specified'}'. "
        "Provide helpful troubleshooting steps: try clearing cache, restarting, "
        "or checking their internet connection. Be specific and helpful. "
        "Ask if the issue is resolved.",
        interruptible=True,
    )
    ctx.set_state("resolved", True)
    return Interrupt(say="", id="await_response")


async def resolve_account(state: SupportState, ctx: Context):
    """Handle account issues with LLM assistance."""
    if state.resolved is not None:
        return Route("feedback")

    await ctx.ask(
        f"The customer {state.customer_name} has an account issue: "
        f"'{state.issue_description or 'not specified'}'. "
        "Explain the steps to resolve the issue (password reset link, account unlock, "
        "profile update instructions). Be clear and reassuring. "
        "Ask if they need further help.",
        interruptible=True,
    )
    ctx.set_state("resolved", True)
    return Interrupt(say="", id="await_response")


async def escalate_to_human(state: SupportState, ctx: Context):
    """Escalate unclassified issues to a human agent."""
    import random, string
    ticket_id = "SUP-" + "".join(random.choices(string.digits, k=5))
    ctx.set_state("ticket_id", ticket_id)

    await ctx.say(
        f"I've created support ticket {ticket_id} for you, {state.customer_name}. "
        "A specialist will reach out to you within 2 hours. "
        "Thank you for your patience!"
    )
    return END


async def feedback(state: SupportState, ctx: Context):
    """Collect customer satisfaction feedback."""
    if state.satisfaction:
        return END

    intent = await ctx.extractor.match_intent(
        intents={
            "happy": "satisfied, great, good, thanks, helpful, resolved, awesome, excellent",
            "neutral": "okay, fine, average, it's alright, so-so, acceptable",
            "unhappy": "not satisfied, bad, unhappy, terrible, didn't help, still broken, angry",
        }
    )

    if intent and intent != "unknown":
        ctx.set_state("satisfaction", intent)
        if intent == "unhappy":
            await ctx.say(
                f"I'm sorry to hear that, {state.customer_name}. "
                "Let me escalate this to a senior agent who can help further. "
                "Thank you for your feedback.",
                interruptible=True,
            )
        else:
            await ctx.say(
                f"Thank you for your feedback, {state.customer_name}! "
                "Is there anything else I can help with? Have a great day!",
                interruptible=True,
            )
        return END

    return Interrupt(
        say="I hope that helped! Could you rate your experience? "
            "Was the support helpful, okay, or did it not resolve your issue?",
        id="ask_feedback",
    )


#  Graph wiring 

graph.add_start_node("welcome", welcome)
graph.add_node("identify_issue", identify_issue)
graph.add_node("route_issue", route_issue)
graph.add_node("resolve_billing", resolve_billing)
graph.add_node("resolve_technical", resolve_technical)
graph.add_node("resolve_account", resolve_account)
graph.add_node("escalate_to_human", escalate_to_human)
graph.add_node("feedback", feedback)

# Main backbone
graph.add_transition(
    START,
    "welcome",
    "identify_issue",
    "route_issue",
)

# Resolution paths all lead to feedback
graph.add_transition("resolve_billing", "feedback", END)
graph.add_transition("resolve_technical", "feedback", END)
graph.add_transition("resolve_account", "feedback", END)
graph.add_transition("escalate_to_human", END)
graph.add_transition("feedback", END)

# Conditional routing based on issue type
graph.add_conditional_transition(
    "route_issue",
    mapping={
        "billing": "resolve_billing",
        "technical": "resolve_technical",
        "account": "resolve_account",
        "other": "escalate_to_human",
    },
    decide=lambda state: state.issue_type if state.issue_type in VALID_ISSUE_TYPES else "other",
)

logger.info("[GRAPH] Customer Support Topology:\n%s", graph.get_graph_status())


#  Agent 
class SupportAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="You are a customer support agent which helps user with billing, technical and account queries.",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        logger.info("[AGENT] Customer support session started")

    async def on_exit(self) -> None:
        logger.info("[AGENT] Session ended")
        logger.info("[AGENT] Final state:\n%s", graph.debug_state())


#  Entrypoint 

async def entrypoint(ctx: JobContext):
    agent = SupportAgent(ctx)

    pipeline = Pipeline(
        llm=GoogleLLM(),
        conversational_graph=graph,
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    # Handle incoming user text from PubSub topic "CHAT"
    async def on_pubsub_message(message):
        text = message.get("message", "") if isinstance(message, dict) else str(message)
        logger.info("[PUBSUB] Received: %s", text)
        if text.strip():
            await pipeline.orchestrator.process_text(text)

    def on_pubsub_message_wrapper(message):
        asyncio.create_task(on_pubsub_message(message))

    # Send LLM responses back via PubSub
    @pipeline.on("llm")
    async def on_llm(data: dict):
        text = data.get("text", "")
        if text.strip():
            logger.info("[LLM] Response: %s", text)
            try:
                publish_config = PubSubPublishConfig(topic="AGENT_RESPONSE", message=text)
                await ctx.room.publish_to_pubsub(publish_config)
            except Exception as e:
                logger.error("Failed to publish LLM response: %s", e)

        if graph.is_ended:
            logger.info("[AGENT] Graph ended, hanging up after %.0fs", config.hangup_delay)
            await asyncio.sleep(config.hangup_delay)
            await agent.hangup()

    async def cleanup_session():
        logger.info("Cleaning up session...")
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        logger.info("Session ended: %s", reason)
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)

        logger.info("Waiting for participant...")
        await ctx.room.wait_for_participant()

        # Compile graph and start session first
        await graph.compile()
        await session.start()

        # Subscribe to incoming user messages on "CHAT" topic
        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_pubsub_message_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await shutdown_event.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    return JobContext(room_options=RoomOptions(
        name="Customer Support Chatbot",
        playground=True,
    ))


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

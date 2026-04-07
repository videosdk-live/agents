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
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from conversational_graph import (
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


#  Graph config 

config = GraphConfig(name="customer_support", debug=True, stream=True)
graph = ConversationalGraph(SupportState, config)


#  Nodes 

async def welcome(state: SupportState, ctx: Context):
    """Greet and collect the customer's name."""
    if state.customer_name:
        return Route("identify_issue")

    result = await ctx.extractor.collect(schema=NameSchema)
    if result.extracted.customer_name:
        return Route("identify_issue")

    return Interrupt(
        say="Hello! Welcome to customer support. May I have your name please? and could you share your issue?",
        id="ask_name",
    )


async def identify_issue(state: SupportState, ctx: Context):
    """Classify the customer's issue into a category."""
    if state.issue_type and state.issue_type in VALID_ISSUE_TYPES:
        return Route("route_issue")

    intent = await ctx.extractor.match_intent(
        intents={
            "billing": "billing problem, charge, invoice, payment, refund, overcharged",
            "technical": "technical issue, bug, error, not working, crash, slow, broken",
            "account": "account problem, login, password, locked out, update profile, settings",
            "other": "something else, general question, none of the above, other",
            "unknown": "something else, general question, none of the above, other",
        }
    )

    if intent:
        ctx.set_state("issue_type", intent)
        result = await ctx.extractor.collect(schema=IssueSchema)
        if result.extracted.issue_description:
            pass
        else:
            msg = (ctx.last_user_message or "").strip()
            if msg:
                ctx.set_state("issue_description", msg)

        return Route("route_issue")

    await ctx.say(
        f"Thanks {state.customer_name}! How can I help you today? "
        "I can assist with billing, technical issues, or account problems."
    )
    return Interrupt(say="", id="ask_issue")


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
        "Ask if this resolves their concern."
    )
    ctx.set_state("resolved", True)
    return Route("feedback")


async def resolve_technical(state: SupportState, ctx: Context):
    """Handle technical issues with LLM assistance."""
    if state.resolved is not None:
        return Route("feedback")

    await ctx.ask(
        f"The customer {state.customer_name} has a technical issue: "
        f"'{state.issue_description or 'not specified'}'. "
        "Provide helpful troubleshooting steps: try clearing cache, restarting, "
        "or checking their internet connection. Be specific and helpful. "
        "Ask if the issue is resolved."
    )
    ctx.set_state("resolved", True)
    return Route("feedback")


async def resolve_account(state: SupportState, ctx: Context):
    """Handle account issues with LLM assistance."""
    if state.resolved is not None:
        return Route("feedback")

    await ctx.ask(
        f"The customer {state.customer_name} has an account issue: "
        f"'{state.issue_description or 'not specified'}'. "
        "Explain the steps to resolve the issue (password reset link, account unlock, "
        "profile update instructions). Be clear and reassuring. "
        "Ask if they need further help."
    )
    ctx.set_state("resolved", True)
    return Route("feedback")


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

    if intent:
        ctx.set_state("satisfaction", intent)
        if intent == "unhappy":
            await ctx.say(
                f"I'm sorry to hear that, {state.customer_name}. "
                "Let me escalate this to a senior agent who can help further. "
                "Thank you for your feedback."
            )
        else:
            await ctx.say(
                f"Thank you for your feedback, {state.customer_name}! "
                "Is there anything else I can help with? Have a great day!"
            )
        return END

    await ctx.say(
        "I hope that helped! Could you rate your experience? "
        "Was the support helpful, okay, or did it not resolve your issue?"
    )
    return Interrupt(say="", id="ask_feedback")


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
        "unknown": "escalate_to_human",
    },
    decide=lambda state: state.issue_type if state.issue_type in VALID_ISSUE_TYPES else "other",
)

logger.info("[GRAPH] Customer Support Topology:\n%s", graph.get_graph_status())


#  Agent 
class SupportAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions=graph.get_system_instructions(),
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.send_text("Hello! Welcome to customer support. How can I help you today?")
        logger.info("[AGENT] Customer support session started")

    async def on_exit(self) -> None:
        logger.info("[AGENT] Session ended")
        logger.info("[AGENT] Final state:\n%s", graph.debug_state())

    async def send_text(self, message: str) -> None:
        """Publish agent response via PubSub."""
        try:
            publish_config = PubSubPublishConfig(topic="AGENT_RESPONSE", message=message)
            await self.ctx.room.publish_to_pubsub(publish_config)
        except Exception as e:
            logger.error("Failed to publish response: %s", e)


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
            await pipeline.orchestrator.process_graph_text(text)

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

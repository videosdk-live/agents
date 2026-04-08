"""
pizza_order_voice.py
====================
A pizza ordering voice bot — NO LLM required.

Uses only local graph methods:
  - ctx.say()                  → speak directly via TTS (bypasses LLM)
  - ctx.extractor.match_intent → classify intent via local ONNX embeddings
  - ctx.extractor.collect      → extract enum values via semantic matching
  - ctx.last_user_message      → raw STT text for free-text fields

Flow
----
START → welcome → collect_name → collect_phone → select_size
      → select_crust → select_toppings → confirm_order → END

Total: 8 nodes
"""

import logging
import random
import string

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

from videosdk.agents import Agent, Pipeline, AgentSession, JobContext, WorkerJob, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS

from pydantic import Field, field_validator
from typing import Optional


# Constants

VALID_SIZES = {"small", "medium", "large"}
VALID_CRUSTS = {"thin", "regular", "stuffed", "cheese_burst"}

PRICES = {"small": 199, "medium": 349, "large": 499}
CRUST_EXTRA = {"thin": 0, "regular": 0, "stuffed": 60, "cheese_burst": 80}
TOPPING_PRICE = 30

_GREETINGS = {"hi", "hello", "hey", "start", "begin", "yes", "ok", "okay", "sure", "yeah"}


# State

class PizzaState(GraphState):
    customer_name: Optional[str] = Field(None, description="Customer name")
    phone:         Optional[str] = Field(None, description="Phone number (10+ digits)")
    size:          Optional[str] = Field(None, description="small, medium, or large")
    crust:         Optional[str] = Field(None, description="thin, regular, stuffed, or cheese_burst")
    toppings:      Optional[str] = Field(None, description="Comma-separated toppings or 'none'")
    confirmed:     Optional[bool] = Field(None, description="Order confirmed")
    order_id:      Optional[str] = Field(None, description="Generated order ID")
    total:         Optional[int] = Field(None, description="Total price in INR")

    @field_validator("phone")
    @classmethod
    def _v_phone(cls, v):
        if v:
            digits = "".join(filter(str.isdigit, v))
            if len(digits) < 10:
                raise ValueError("Phone must be at least 10 digits")
            return digits
        return v

    @field_validator("size")
    @classmethod
    def _v_size(cls, v):
        if v and v.lower().strip() not in VALID_SIZES:
            raise ValueError(f"Must be one of: {', '.join(VALID_SIZES)}")
        return v.lower().strip() if v else v

    @field_validator("crust")
    @classmethod
    def _v_crust(cls, v):
        if not v:
            return v
        normalized = v.lower().strip().replace(" ", "_")
        aliases = {"cheese": "cheese_burst", "cheesy": "cheese_burst", "thick": "regular"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in VALID_CRUSTS:
            raise ValueError(f"Must be one of: {', '.join(VALID_CRUSTS)}")
        return normalized


# Extraction schemas

class NameSchema(GraphState):
    customer_name: str = Field(..., description="Customer name")

class PhoneSchema(GraphState):
    phone: str = Field(..., description="Phone number (10+ digits)")

class SizeSchema(GraphState):
    size: str = Field(..., description="small, medium, or large")

class CrustSchema(GraphState):
    crust: str = Field(..., description="thin, regular, stuffed, or cheese_burst")

class ToppingsSchema(GraphState):
    toppings: str = Field(..., description="mushroom, onion, capsicum, olive, jalapeno, corn, chicken, pepperoni, or none")


# Helpers

def _gen_order_id() -> str:
    return "PZA-" + "".join(random.choices(string.digits, k=4))


def _calc_total(size: str, crust: str, toppings: str) -> int:
    base = PRICES.get(size, 349)
    crust_cost = CRUST_EXTRA.get(crust, 0)
    topping_list = [
        t.strip() for t in (toppings or "").split(",")
        if t.strip() and t.strip().lower() != "none"
    ]
    return base + crust_cost + (len(topping_list) * TOPPING_PRICE)


# Graph config

config = GraphConfig(name="pizza_order_voice", graph_id="pizza_order_voice", debug=True, stream=True)
graph = ConversationalGraph(PizzaState, config)

async def welcome(state: PizzaState, ctx: Context):
    """Entry point — just routes to collect_name."""
    return Route("collect_name")


async def collect_name(state: PizzaState, ctx: Context):
    """Collect customer name"""
    if state.customer_name:
        return Route("collect_phone")

    result = await ctx.extractor.collect(schema=NameSchema)
    if result.extracted.customer_name:
        await ctx.say(
            f"Nice to meet you, {state.customer_name}! "
            f"Can you share your 10 digit phone number for the order?"
        )
        return Route("collect_phone")

    msg = (ctx.last_user_message or "").strip()
    if msg and msg.lower() not in _GREETINGS and len(msg) >= 2:
        ctx.set_state("customer_name", msg)
        await ctx.say(
            f"Nice to meet you, {msg}! "
            f"Can you share your 10 digit phone number for the order?"
        )
        return Route("collect_phone")

    # Ask
    await ctx.say("What's your name?")
    return Interrupt(say="", id="ask_name")


async def collect_phone(state: PizzaState, ctx: Context):
    """Collect phone number"""
    if state.phone:
        return Route("select_size")

    # Try extractor
    result = await ctx.extractor.collect(schema=PhoneSchema)
    if result.extracted.phone:
        await ctx.say(
            f"Got it, phone ending in {state.phone[-4:]}. "
            f"Now, what size pizza would you like? "
            f"We have small at 199, medium at 349, and large at 499 rupees."
        )
        return Route("select_size")

    # Fallback: parse digits
    msg = (ctx.last_user_message or "").strip()
    digits = "".join(filter(str.isdigit, msg))
    if len(digits) >= 10:
        ctx.set_state("phone", digits)
        await ctx.say(
            f"Got it, phone ending in {digits[-4:]}. "
            f"Now, what size pizza would you like? "
            f"We have small at 199, medium at 349, and large at 499 rupees."
        )
        return Route("select_size")

    # Ask
    await ctx.say("Can you share your 10 digit phone number?")
    return Interrupt(say="", id="ask_phone")


async def select_size(state: PizzaState, ctx: Context):
    """Select pizza size via semantic extraction."""
    if state.size:
        return Route("select_crust")

    result = await ctx.extractor.collect(schema=SizeSchema)
    if result.extracted.size:
        price = PRICES.get(state.size, 349)
        await ctx.say(
            f"{state.size} size, that's {price} rupees. Nice choice! "
            f"And what crust would you prefer? "
            f"We have thin, regular, stuffed, and cheese burst."
        )
        return Route("select_crust")

    # Ask
    sizes_text = ", ".join(f"{s} at {p} rupees" for s, p in PRICES.items())
    await ctx.say(f"What size pizza would you like? We have {sizes_text}.")
    return Interrupt(say="", id="ask_size")


async def select_crust(state: PizzaState, ctx: Context):
    """Select crust via semantic extraction."""
    if state.crust:
        return Route("select_toppings")

    result = await ctx.extractor.collect(schema=CrustSchema)
    if result.extracted.crust:
        crust_display = (state.crust or "regular").replace("_", " ")
        await ctx.say(
            f"{crust_display} crust, great pick! "
            f"Any toppings? We have mushroom, onion, capsicum, olive, jalapeno, "
            f"corn, chicken, and pepperoni. 30 rupees each. Or say none for plain cheese."
        )
        return Route("select_toppings")

    # Ask
    await ctx.say(
        "What crust would you like? "
        "We have thin, regular, stuffed, and cheese burst. "
        "Stuffed and cheese burst have a small extra charge."
    )
    return Interrupt(say="", id="ask_crust")


async def select_toppings(state: PizzaState, ctx: Context):
    """Collect toppings via extractor, fallback to raw STT text."""
    if state.toppings:
        return Route("confirm_order")

    # Try extractor
    result = await ctx.extractor.collect(schema=ToppingsSchema)
    if result.extracted.toppings:
        total = _calc_total(state.size or "medium", state.crust or "regular", state.toppings or "none")
        crust_display = (state.crust or "regular").replace("_", " ")
        await ctx.say(
            f"Got it, {state.toppings}! "
            f"So here's your order. "
            f"A {state.size} pizza with {crust_display} crust, "
            f"toppings {state.toppings}, "
            f"total comes to {total} rupees. "
            f"Shall I place the order? Say yes or no."
        )
        return Route("confirm_order")

    # Fallback: accept raw message
    msg = (ctx.last_user_message or "").strip()
    if msg:
        ctx.set_state("toppings", msg.lower().strip())
        total = _calc_total(state.size or "medium", state.crust or "regular", state.toppings or "none")
        crust_display = (state.crust or "regular").replace("_", " ")
        await ctx.say(
            f"Got it, {state.toppings}! "
            f"So here's your order. "
            f"A {state.size} pizza with {crust_display} crust, "
            f"toppings {state.toppings}, "
            f"total comes to {total} rupees. "
            f"Shall I place the order? Say yes or no."
        )
        return Route("confirm_order")

    # Ask
    await ctx.say(
        "Any toppings? We have mushroom, onion, capsicum, olive, jalapeno, "
        "corn, chicken, and pepperoni. 30 rupees each. Or say none for plain cheese."
    )
    return Interrupt(say="", id="ask_toppings")


async def confirm_order(state: PizzaState, ctx: Context):
    """Confirm order via yes/no intent matching."""
    if state.confirmed:
        return END

    total = _calc_total(state.size or "medium", state.crust or "regular", state.toppings or "none")
    crust_display = (state.crust or "regular").replace("_", " ")

    # Keyword pre-check — short words like "yes"/"no"
    _yes_words = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "place", "go ahead", "do it"}
    _no_words = {"no", "nah", "nope", "cancel", "don't", "stop", "never mind"}
    msg = (ctx.last_user_message or "").strip().lower().rstrip(".")
    if msg in _yes_words:
        intent = "yes"
    elif msg in _no_words:
        intent = "no"
    else:
        intent = await ctx.extractor.match_intent({
            "yes": "User confirms, agrees, says yes, place the order, go ahead, sure, definitely",
            "no":  "User declines, cancels, says no, don't want it, never mind, nah",
        })

    if intent == "yes":
        order_id = _gen_order_id()
        ctx.update_states({"order_id": order_id, "total": total, "confirmed": True})
        await ctx.say(
            f"Awesome, {state.customer_name}! Your order is placed. "
            f"Order ID is {order_id}, total {total} rupees. "
            f"Delivery in about 30 minutes to phone ending {(state.phone or '0000')[-4:]}. "
            f"Enjoy your meal!"
        )
        return END

    if intent == "no":
        ctx.set_state("confirmed", False)
        await ctx.say(
            f"No worries, {state.customer_name}! "
            f"Your order is cancelled. Come back anytime you're hungry!"
        )
        return END

    # Intent unclear — read summary and ask again
    await ctx.say(
        f"Here's your order, {state.customer_name}. "
        f"A {state.size} pizza with {crust_display} crust, "
        f"toppings {state.toppings}, "
        f"total {total} rupees. "
        f"Shall I place the order? Say yes or no."
    )
    return Interrupt(say="", id="ask_confirm")


# Graph wiring

graph.add_start_node("welcome",        welcome)
graph.add_node("collect_name",         collect_name)
graph.add_node("collect_phone",        collect_phone)
graph.add_node("select_size",          select_size)
graph.add_node("select_crust",         select_crust)
graph.add_node("select_toppings",      select_toppings)
graph.add_node("confirm_order",        confirm_order)

graph.add_transition(
    START,
    "welcome",
    "collect_name",
    "collect_phone",
    "select_size",
    "select_crust",
    "select_toppings",
    "confirm_order",
    END,
)

logger.info("[GRAPH] Pizza Order Voice Topology:\n%s", graph.get_graph_status())


# Agent + Pipeline (STT + TTS, no LLM)

class PizzaAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly pizza order assistant at Pizza Planet. "
                "Be casual, warm, and keep responses short. "
                "You are speaking to the customer over a voice call.\n\n"
            ),
        )

    async def on_enter(self) -> None:
        logger.info("[AGENT] Pizza order voice session started")
        await self.session.say(
            "Hey! Welcome to Pizza Planet! "
            "I'll help you place your order. What's your name?"
        )

    async def on_exit(self) -> None:
        logger.info("[AGENT] Pizza order voice session ended")


async def entrypoint(ctx: JobContext) -> None:
    agent = PizzaAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en-IN"),
        tts=CartesiaTTS(model="sonic-3"),
        conversational_graph=graph,
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await graph.compile()

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(room_options=RoomOptions(
        name="Pizza Planet Voice Agent",
        playground=True,
    ))


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

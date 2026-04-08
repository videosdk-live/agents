import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Framework
from videosdk.conversational_graph import (
    ConversationalGraph,
    GraphState,
    Context,
    Route,
    END,
    Interrupt,
    HumanInLoop,
    GraphConfig,
    MongoDBSaver,
)

# VideoSDK
from videosdk.agents import Agent, Pipeline, AgentSession, JobContext, WorkerJob, RoomOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS

from pydantic import Field, field_validator
from typing import Optional


# State

class LoanState(GraphState):
    name:               Optional[str]  = Field(None, description="Applicant full name (first + last)")
    employment_status:  Optional[str]  = Field(
        None,
        description="Employment status: employed | unemployed | student | self-employed",
    )
    income:             Optional[int]  = Field(None, ge=0,        description="Annual income in INR")
    loan_amount:        Optional[int]  = Field(None,              description="Requested loan amount in INR")
    credit_score:       Optional[int]  = Field(None, ge=300, le=850, description="Credit score 300–850")
    documents_verified: Optional[bool] = Field(None,              description="All required docs uploaded")
    decision:           Optional[str]  = Field(None,              description="approved | rejected | pending_docs | review")

    @field_validator("employment_status")
    @classmethod
    def _validate_employment(cls, v):
        valid = {"employed", "unemployed", "student", "self-employed"}
        if v and v.lower() not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v.lower() if v else v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v):
        if v and len(v.strip()) < 2:
            raise ValueError("Name too short")
        return v.strip().title() if v else v


class NameSchema(GraphState):
    name: str= Field(..., description="Applicant full name (first + last)")


# Graph setup

mongo_saver = MongoDBSaver(
    uri="mongodb://localhost:27017",
    db_name="conversations",
    collection_name="checkpoints",
)

config = GraphConfig(
    name="loan_assistant",
    debug=True,
    checkpointer=mongo_saver,
)

graph = ConversationalGraph(LoanState, config)


# Nodes

async def welcome_node(state: LoanState, ctx: Context):
    intent = await ctx.extractor.match_intent(
        intents={
            "apply":  "User wants to apply for a loan or start an application",
            "status": "User wants to check existing application status",
            "exit":   "User wants to hang up, stop, call later, or say goodbye",
        }
    )
    logger.info("[NODE] welcome → intent=%r", intent)

    if intent == "exit":
        return END

    if intent == "status":
        return Route("check_status")

    return Route("collect_name")


async def check_status_node(state: LoanState, ctx: Context):
    await ctx.say(
        "To check your loan application status, you can visit loans.example.com "
        "or call 1800-XXX-XXXX. Is there anything else I can help you with?"
    )
    return END


async def collect_name_node(state: LoanState, ctx: Context):
    result = await ctx.extractor.collect(
        schema=NameSchema,
        prompt="Ask the user for their full name to begin the loan application",
    )

    if not result.extracted.name:
        return Interrupt(
            say="Could you please tell me your full name?",
            id="retry_name",
        )

    return Route("collect_employment")


async def collect_employment_node(state: LoanState, ctx: Context):
    result = await ctx.extractor.collect(
        fields=["employment_status"],
        prompt=(
            f"Ask {state.name or 'the applicant'} about their current employment status. "
            "Options: employed, self-employed, student, or unemployed."
        ),
    )

    if not result.extracted.employment_status:
        return Interrupt(
            say=(
                "Could you clarify your employment status? "
                "Are you employed, self-employed, a student, or unemployed?"
            ),
            id="retry_employment",
        )

    return Route("collect_income")


async def collect_income_node(state: LoanState, ctx: Context):
    if state.employment_status == "unemployed":
        return Route("reject_unemployed", update={"income": 0})

    result = await ctx.extractor.collect(
        fields=["income"],
        prompt=(
            f"Ask {state.name or 'the applicant'} for their annual income in INR. "
            "Ask them to give a number, for example '6 lakh' or '600000'."
        ),
    )

    if result.extracted.income is None:
        return Interrupt(
            say="I need your annual income to continue. Could you share that figure?",
            id="retry_income",
        )

    return Route("collect_loan")


async def collect_loan_node(state: LoanState, ctx: Context):
    income = state.income or 1000
    max_eligible = income * 5

    result = await ctx.extractor.collect(
        fields=["loan_amount"],
        prompt=(
            f"Ask {state.name or 'the applicant'} how much loan they would like to apply for. "
            f"Based on their income of ₹{income:,}, the maximum eligible amount is ₹{max_eligible:,}."
        ),
    )

    if result.extracted.loan_amount is None:
        return Interrupt(
            say="How much would you like to borrow? Please give me a number.",
            id="retry_loan",
        )


async def validate_loan_node(state: LoanState, ctx: Context):
    """ runs after loan_amount is extracted. Clears and re-asks if over limit."""
    income = state.income or 1000
    max_eligible = income * 5
    loan = state.loan_amount or 0

    if loan > max_eligible:
        ctx.clear_frame("loan_amount")
        await ctx.say(
            f"I'm sorry, ₹{loan:,} exceeds your maximum eligibility of ₹{max_eligible:,}. "
            "Please provide a lower loan amount."
        )
        return Route("collect_loan")

    return Route("credit_check")


async def credit_check_node(state: LoanState, ctx: Context):
    result = await ctx.extractor.collect(
        fields=["credit_score"],
        prompt="Ask the applicant for their credit score (a number between 300 and 850).",
    )

    if result.extracted.credit_score is None:
        return Interrupt(
            say="Could you tell me your credit score? It is a number between 300 and 850.",
            id="retry_credit",
        )

    return Route("credit_decision")


async def credit_decision_node(state: LoanState, ctx: Context):
    """Pure logic node — sets `decision` in states, StateMachine conditional edge picks next node."""
    score = state.credit_score or 0
    logger.info("[NODE] credit_decision | score=%d", score)

    if score >= 700:
        return {"decision": "pending_docs"}
    elif score >= 600:
        return {"decision": "review"}
    else:
        return {"decision": "rejected"}


async def document_check_node(state: LoanState, ctx: Context):
    result = await ctx.extractor.collect(
        fields=["documents_verified"],
        prompt=(
            "Ask the applicant if they have already uploaded all required documents "
            "(identity proof, income proof, address proof). A simple yes or no."
        ),
    )

    if result.extracted.documents_verified is None:
        return Interrupt(
            say="Have you uploaded all required documents? Please say yes or no.",
            id="retry_docs",
        )
    # Routing to approve/pending_docs handled by the StateMachine conditional transition


async def manual_review_node(state: LoanState, ctx: Context):
    """Pause the graph and wait for an underwriter to review the case."""
    return HumanInLoop(
        reason=f"Marginal credit score: {state.credit_score}",
        say=(
            "Your application is currently under manual review by our underwriting team. "
            "We will contact you within 24 hours with a decision."
        ),
    )


async def reject_unemployed_node(state: LoanState, ctx: Context):
    name = state.name or "there"
    await ctx.ask(
        f"Politely inform {name} that employment is a mandatory requirement for a loan. "
        "Express empathy and encourage them to apply again once they are employed. "
        "Close the conversation warmly."
    )
    return END


async def approve_node(state: LoanState, ctx: Context):
    name   = state.name or "the applicant"
    amount = f"₹{state.loan_amount:,}" if state.loan_amount else "the requested amount"
    ctx.update_states({"decision": "approved"})
    await ctx.ask(
        f"Congratulate {name} warmly on their loan approval for {amount}. "
        "Inform them that a loan officer will contact them within 2 business days "
        "with the final paperwork. Thank them for choosing our service."
    )
    return END


async def reject_node(state: LoanState, ctx: Context):
    name = state.name or "the applicant"
    await ctx.ask(
        f"Politely inform {name} that their application cannot be approved at this time "
        "due to the minimum credit score requirement of 600. Suggest steps to improve "
        "credit score and thank them for applying."
    )
    return END


async def pending_docs_node(state: LoanState, ctx: Context):
    name = state.name or "the applicant"
    await ctx.ask(
        f"Tell {name} that their application is pending document verification. "
        "They will receive an email with upload instructions within 24 hours. "
        "Thank them for applying."
    )
    return END


# Graph wiring

graph.add_start_node("welcome",            welcome_node)
graph.add_node("check_status",             check_status_node)
graph.add_node("collect_name",             collect_name_node)
graph.add_node("collect_employment",       collect_employment_node)
graph.add_node("collect_income",           collect_income_node)
graph.add_node("collect_loan",             collect_loan_node)
graph.add_node("validate_loan",            validate_loan_node)
graph.add_node("credit_check",             credit_check_node)
graph.add_node("credit_decision",          credit_decision_node)
graph.add_node("document_check",           document_check_node)
graph.add_node("manual_review",            manual_review_node)
graph.add_node("approve",                  approve_node)
graph.add_node("reject",                   reject_node)
graph.add_node("reject_unemployed",        reject_unemployed_node)
graph.add_node("pending_docs",             pending_docs_node)

# Linear backbone
graph.add_transition(
    "welcome",
    "collect_name",
    "collect_employment",
    "collect_income",
    "collect_loan",
    "validate_loan",
    "credit_check",
    "credit_decision",
)

# Status check ends immediately
graph.add_transition("check_status", END)

# Routing after credit decision
graph.add_conditional_transition(
    "credit_decision",
    mapping={
        "pending_docs": "document_check",
        "review":       "manual_review",
        "rejected":     "reject",
    },
    decide=lambda state: state.decision,
)

# Routing after document check: approve if docs verified, else pending
graph.add_conditional_transition(
    "document_check",
    mapping={
        "approved": "approve",
        "pending":  "pending_docs",
    },
    decide=lambda state: "approved" if state.documents_verified else "pending",
)

logger.info("[GRAPH] Loan Application Topology:\n%s", graph.get_graph_status())


# Agent

class LoanAgent(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=graph.get_system_instructions())

    async def on_enter(self) -> None:
        logger.info("[AGENT] Session started")
        await self.session.say("Hello! I am your loan assistant. How can I help you today?")

    async def on_exit(self) -> None:
        logger.info("[AGENT] Session ended")
        logger.info("[AGENT] Final state:\n%s", graph.debug_state())


# Entrypoint 

async def entrypoint(ctx: JobContext):
    agent = LoanAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en-IN"),
        llm=GoogleLLM(model="gemini-2.5-flash"),
        tts=CartesiaTTS(model="sonic-3"),
        conversational_graph=graph,
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await graph.compile(user_config={"user_id":"test_user"})
    graph.visualize()
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context():
    room_options = RoomOptions(
        name="Loan Application Agent",
        playground=True,
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

"""
Use Case: SaaS product onboarding agent for new users (Taskflow project management tool).
Pipeline: P1 — DeepgramSTT + GoogleLLM + GoogleTTS + SileroVAD + TurnDetector + ConversationalGraph
Demonstrates: ConversationalGraph state machine for structured step-by-step data collection.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY
"""

import logging
from pydantic import Field
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM, GoogleTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

try:
    from conversational_graph import ConversationalGraph, ConversationalDataModel
except ImportError:
    raise ImportError(
        "Missing dependency: videosdk-conversational-graph\n"
        "Install with: pip install videosdk-conversational-graph"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class OnboardingData(ConversationalDataModel):
    company_name: str = Field(None, description="Name of the user's company or team")
    team_size: str = Field(None, description="Number of people on the team (e.g., 1-5, 6-20, 20+)")
    primary_use_case: str = Field(None, description="Main reason for using Taskflow (e.g., engineering, marketing, design)")
    invite_teammates: str = Field(None, description="Whether the user wants to invite teammates now (yes/no)")


onboarding_flow = ConversationalGraph(
    name="Taskflow Onboarding",
    DataModel=OnboardingData,
    off_topic_threshold=3,
)

# Step 1: Welcome
step_welcome = onboarding_flow.state(
    name="Welcome",
    instruction="Welcome the user to Taskflow and ask for their company or team name.",
)

# Step 2: Team size
step_team_size = onboarding_flow.state(
    name="Team Size",
    instruction="Ask how many people are on the team. Offer options: Just me, 2-10 people, 11-50 people, or 50+.",
)

# Step 3: Primary use case
step_use_case = onboarding_flow.state(
    name="Primary Use Case",
    instruction="Ask what they primarily plan to use Taskflow for. Options: Engineering, Marketing, Design, Operations, or Other.",
)

# Step 4: Invite teammates
step_invite = onboarding_flow.state(
    name="Invite Teammates",
    instruction="Ask if they'd like to invite their teammates now. They can do this later too.",
)

# Step 5: Complete
step_complete = onboarding_flow.state(
    name="Complete",
    instruction="Confirm setup is complete. Tell the user their workspace is ready. Wish them well with Taskflow.",
)

# Off-topic handler
step_off_topic = onboarding_flow.state(
    name="Off-Topic Handler",
    instruction="Acknowledge the off-topic message politely and guide the user back to the onboarding flow.",
    master=True,
)

# Transitions
onboarding_flow.transition(from_state=step_welcome, to_state=step_team_size, condition="Company name provided")
onboarding_flow.transition(from_state=step_team_size, to_state=step_use_case, condition="Team size selected")
onboarding_flow.transition(from_state=step_use_case, to_state=step_invite, condition="Primary use case selected")
onboarding_flow.transition(from_state=step_invite, to_state=step_complete, condition="User answered about inviting teammates")


class OnboardingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the onboarding specialist for Taskflow, a project management tool.
            Walk new users through 4 quick setup steps: company name, team size, primary use case, and teammate invites.
            Confirm each answer before proceeding to the next step.
            Be friendly, encouraging, and efficient — onboarding should take under 3 minutes.
            At the end, use complete_onboarding to save the user's profile.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome to Taskflow! I'm here to get your workspace set up in just a few minutes. "
            "I'll ask you four quick questions. Let's start — what's your company or team name?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Your Taskflow workspace is all set! "
            "Check your inbox for a link to get started. Happy building!"
        )

    @function_tool
    async def complete_onboarding(
        self, company_name: str, team_size: str, primary_use_case: str, invite_teammates: str
    ) -> dict:
        """Save the completed onboarding profile for the new user.

        Args:
            company_name: Name of the user's company or team
            team_size: Selected team size range
            primary_use_case: Primary use case for Taskflow
            invite_teammates: Whether to send invites now (yes/no)
        """
        workspace_id = f"TF-{abs(hash(company_name)) % 100000:05d}"
        return {
            "workspace_id": workspace_id,
            "company_name": company_name,
            "team_size": team_size,
            "primary_use_case": primary_use_case,
            "invite_sent": invite_teammates.lower() == "yes",
            "status": "onboarding_complete",
        }


async def entrypoint(ctx: JobContext):
    agent = OnboardingAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        conversational_graph=onboarding_flow,
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Taskflow Onboarding", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

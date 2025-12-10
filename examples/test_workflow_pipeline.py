
import logging
import aiohttp
from pydantic import BaseModel,Field
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM, GoogleTTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from conversational_graph import ConversationGraph,PreDefinedTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

pre_download_model()



get_weather = PreDefinedTool().http_tool(
    name="get_weather",
    description="Called when the user asks about the weather. This function will return the weather for the given location. When given a location, please estimate the latitude and longitude of the location and do not ask the user for them.\n\nArgs:\n    latitude: The latitude of the location\n    longitude: The longitude of the location",
    url="https://api.open-meteo.com/v1/forecast?latitude=21.1702&longitude=81.2402&current=temperature_2m",
    method="GET",
    auth={"enabled": False, "type": "none"},
    params={}
)

class LoanMemory(BaseModel):
    loan_type: str = Field(None, description="Type of loan: personal, home, car")
    annual_income: int = Field(None, description="Annual income of the applicant in INR")
    credit_score: int = Field(None, description="Credit score of the applicant. Must be between 300 and 850")
    property_value: int = Field(None, description="Value of the property for home loan in INR")
    vehicle_price: int = Field(None, description="Price of the vehicle for car loan in INR")
    loan_amount: int = Field(None, description="Desired loan amount in INR. MUST be greater than ₹11 lakh for approval")
    location: str = Field(None, description="User's location for weather information")

loan_workflow = ConversationGraph(
        name="Loan Application Workflow",
        DataModel= LoanMemory,
        off_topic_threshold=5
    )
    
# Start Greeting
q0 = loan_workflow.state(
    name="Greeting",
    instruction="Welcome user and start the conversation and ask user's location and the tell weather of that location",
    tool=get_weather
)

# Loan Type Selection
q1 = loan_workflow.state(
    name="Loan Type Selection",
    instruction="Ask user to select loan type. We only offer personal loan, home loan, and car loan at the moment.",
)

# Personal Loan Details
q1a = loan_workflow.state(
    name="Personal Loan Details",
    instruction="Collect annual income and credit score for personal loan. Annual income must be > ₹5 lakh and credit score > 500."
)

# Home Loan Details
q1b = loan_workflow.state(
    name="Home Loan Details",
    instruction="Collect annual income, credit score, and property value for home loan. Annual income must be > ₹6 lakh, credit score > 650, and property value must be reasonable."
)

# Car Loan Details
q1c = loan_workflow.state(
    name="Car Loan Details",
    instruction="Collect annual income, credit score, and vehicle price for car loan. Annual income must be > ₹10 lakh and credit score > 600."
)

# Merged State → Loan Amount Collection
q2 = loan_workflow.state(
    name="Loan Amount Collection",
    instruction="Collect the desired loan amount. Minimum loan amount is ₹11 lakh for approval."
)


q2a = loan_workflow.state(
    name="Loan Amount Rejection",
    instruction="Inform user that the requested loan amount is below the minimum requirement of ₹11 lakh and politely end the application process."
)   

q2b = loan_workflow.state(
    name="Loan Amount Acceptance",
    instruction="Proceed to the next step of the application process as the requested loan amount meets the minimum requirement."
)

# Review & Confirm
q3 = loan_workflow.state(
    name="Review and Confirm",
    instruction="Review all loan details with the user and get confirmation.",
    # tool=save_loan_details
)

# Completion State
q4 = loan_workflow.state(
    name="Complete",
    instruction="Inform user that the application has been submitted successfully and provide next steps"
)

# Master / Off-topic handler
q_master = loan_workflow.state(
    name="Off-topic Handler",
    instruction="Handle off-topic or inappropriate inputs respectfully and end the call politely",
    master=True
)


# ---------------------------
# Define Transitions
# ---------------------------

# Greeting → Loan Type Selection
loan_workflow.transition(
    from_state=q0,
    to_state=q1,
    condition="User ready to apply for loan"
)

# Branch from Loan Type Selection
loan_workflow.transition(
    from_state=q1,
    to_state=q1a,
    condition="User wants personal loan"
)

loan_workflow.transition(
    from_state=q1,
    to_state=q1b,
    condition="User wants home loan"
)

loan_workflow.transition(
    from_state=q1,
    to_state=q1c,
    condition="User wants car loan"
)

# Merge all branches → Loan Amount Collection
loan_workflow.transition(
    from_state=q1a,
    to_state=q2,
    condition="Personal loan details collected and verified"
)

loan_workflow.transition(
    from_state=q1b,
    to_state=q2,
    condition="Home loan details collected and verified"
)

loan_workflow.transition(
    from_state=q1c,
    to_state=q2,
    condition="Car loan details collected and verified"
)

# Loan Amount → Review and Confirm
loan_workflow.transition(
    from_state=q2,
    to_state=q2a,
    condition="Loan application rejected due to insufficient amount (< ₹11 lakh)"
)

loan_workflow.transition(
    from_state=q2,
    to_state=q2b,
    condition="Loan amount meets minimum requirement (≥ ₹11 lakh)"
)

loan_workflow.transition(
    from_state=q2b,
    to_state=q3,
    condition="Proceed to review after loan amount acceptance"
)

# Review → Complete
loan_workflow.transition(
    from_state=q3,
    to_state=q4,
    condition="User confirms submission of loan application"
)



class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can assist you with your loan applications"

        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, I am here to help with your loan application. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
  
async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=GoogleLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        conversational_graph = loan_workflow
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="fxyg-gp30-pwqb", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

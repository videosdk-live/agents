
import logging
from pydantic import Field
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleTTS
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from conversational_graph import ConversationalGraph,PreDefinedTool,HttpToolRequest,ConversationalDataModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

pre_download_model()


get_weather = PreDefinedTool().http_tool(HttpToolRequest(
        name="get_weather",
        description="Called when the user asks about the weather. This function will return the weather for the given location.",
        url="https://api.open-meteo.com/v1/forecast?latitude=21.1702&longitude=81.2402&current=temperature_2m",
        method="GET"
    )
)
 
class LoanFlow(ConversationalDataModel):
    loan_type: str = Field(None, description="Type of loan: personal, home, car")
    annual_income: int = Field(None, description="Annual income of the applicant in INR")
    credit_score: int = Field(None, description="Credit score of the applicant. Must be between 300 and 850")
    property_value: int = Field(None, description="Value of the property for home loan in INR")
    vehicle_price: int = Field(None, description="Price of the vehicle for car loan in INR")
    loan_amount: int = Field(None, description="Desired loan amount in INR. MUST be greater than ₹11 lakh for approval")

loan_application = ConversationalGraph(
        name="Loan Application",
        DataModel= LoanFlow,
        off_topic_threshold=5
    )
    
# Start Greeting
q0 = loan_application.state(
    name="Greeting",
    instruction="Welcome user and start the conversation and tell the weather",
    tool=get_weather
)

# Loan Type Selection
q1 = loan_application.state(
    name="Loan Type Selection",
    instruction="Ask user to select loan type. We only offer personal loan, home loan, and car loan at the moment.",
)

# Personal Loan Details
q1a = loan_application.state(
    name="Personal Loan Details",
    instruction="Collect annual income and credit score for personal loan. Annual income must be > ₹5 lakh and credit score > 500."
)

# Home Loan Details
q1b = loan_application.state(
    name="Home Loan Details",
    instruction="Collect annual income, credit score, and property value for home loan. Annual income must be > ₹6 lakh, credit score > 650, and property value must be reasonable."
)

# Car Loan Details
q1c = loan_application.state(
    name="Car Loan Details",
    instruction="Collect annual income, credit score, and vehicle price for car loan. Annual income must be > ₹10 lakh and credit score > 600."
)

# Merged State → Loan Amount Collection
q2 = loan_application.state(
    name="Loan Amount Collection",
    instruction="Collect the desired loan amount. Minimum loan amount is ₹11 lakh for approval."
)


q2a = loan_application.state(
    name="Loan Amount Rejection",
    instruction="Inform user that the requested loan amount is below the minimum requirement of ₹11 lakh and politely end the application process."
)   

q2b = loan_application.state(
    name="Loan Amount Acceptance",
    instruction="Proceed to the next step of the application process as the requested loan amount meets the minimum requirement."
)

# Review & Confirm
q3 = loan_application.state(
    name="Review and Confirm",
    instruction="Review all loan details with the user and get confirmation."
)

# Completion State
q4 = loan_application.state(
    name="Complete",
    instruction="Inform user that the application has been submitted successfully and provide next steps"
)

# Master / Off-topic handler
q_master = loan_application.state(
    name="Off-topic Handler",
    instruction="Handle off-topic or inappropriate inputs respectfully and end the call politely",
    master=True
)


# Define Transitions

# Greeting → Loan Type Selection
loan_application.transition(
    from_state=q0,
    to_state=q1,
    condition="User ready to apply for loan"
)

# Branch from Loan Type Selection
loan_application.transition(
    from_state=q1,
    to_state=q1a,
    condition="User wants personal loan"
)

loan_application.transition(
    from_state=q1,
    to_state=q1b,
    condition="User wants home loan"
)

loan_application.transition(
    from_state=q1,
    to_state=q1c,
    condition="User wants car loan"
)

# Merge all branches → Loan Amount Collection
loan_application.transition(
    from_state=q1a,
    to_state=q2,
    condition="Personal loan details collected and verified"
)

loan_application.transition(
    from_state=q1b,
    to_state=q2,
    condition="Home loan details collected and verified"
)

loan_application.transition(
    from_state=q1c,
    to_state=q2,
    condition="Car loan details collected and verified"
)

# Loan Amount → Review and Confirm
loan_application.transition(
    from_state=q2,
    to_state=q2a,
    condition="Loan application rejected due to insufficient amount (< ₹11 lakh)"
)

loan_application.transition(
    from_state=q2,
    to_state=q2b,
    condition="Loan amount meets minimum requirement (≥ ₹11 lakh)"
)

loan_application.transition(
    from_state=q2b,
    to_state=q3,
    condition="Proceed to review after loan amount acceptance"
)

# Review → Complete
loan_application.transition(
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
        llm=OpenAILLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        conversational_graph = loan_application
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()

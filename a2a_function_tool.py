# example_usage.py
import asyncio
from videosdk.agents.a2a.card import AgentCard
from videosdk.agents.a2a.protocol import A2AMessage
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import  TurnDetection
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool
from typing import Dict, Any
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig

class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="customer_service_1",
            instructions=(
                "You are a helpful bank customer service agent that can assist customers with general banking queries. "
                "You can handle questions about account balances, transactions, general banking services, and basic banking information. "
                "You have access to a function tool `forward_to_specialist(query: str, domain: str)`. "
                "If the customer asks about loans, interest rates, loan eligibility, or any loan-related queries, "
                "do NOT answer directly—instead, call the function `forward_to_specialist` with the full user query "
                "and domain set to \"loan\", and return only that function call."
            )
        )
        

    @function_tool
    async def forward_to_specialist(self, query: str, domain: str) -> Dict[str, Any]:
        """Forward a query to a specialist agent"""
        print("Forwarding query to specialist::",query, domain)

        # Find appropriate specialist
        specialists = self.a2a.registry.find_agents_by_domain(domain)
        print("specialist::",specialists)

        # (you could round-robin, pick the first, broadcast to all, etc.)
        id_of_target_agent = specialists[0] if specialists else None
        if not id_of_target_agent:
            return {"error": "no specialist found for domain " + domain}

        await self.a2a.send_message(
            to_agent=id_of_target_agent,
            message_type="specialist_query",
            content={"query": query}
        )
        
        return {"status": "forwarded", "specialist": "ahmed"}

    async def handle_specialist_response(self, message: A2AMessage) -> None:
        """Handle response from specialist agent"""
        response = message.content.get("response")
        if response:
            print("Response from specialist:", response)
            await self.session.pipeline.model.interrupt()
            await asyncio.sleep(1)
            await self.session.say(f"I've received the following information: car loans bad")

    async def on_exit(self) -> None:
        print("Customer agent Left the meeting")

    async def on_enter(self) -> None:
        # Register the agent for A2A communication
        card = AgentCard(
            id="customer_service_1",
            name="Customer Service Agent",
            domain="customer_service",
            capabilities=["query_handling", "specialist_coordination"],
            description="Handles customer queries and coordinates with specialists"
        )
        await self.register_a2a(card)
        
        await self.session.say("Hello! i am customer service agent")
                # Register message handlers
        self.a2a.on_message("model_response", self.handle_specialist_response)


class LoanAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="specialist_1", 
            instructions=(
                "You are a specialized loan agent that handles all loan-related queries. "
                "You can provide information about different types of loans, interest rates, "
                "eligibility criteria, and loan application processes. "
                # "You have access to a function tool `get_loan_details` that can provide specific loan information. "
                "When customers ask about loan details, give a hypothetical answer."
            ),
        )

    # @function_tool
    # async def get_loan_details(self, loan_type: str) -> Dict[str, Any]:
    #     """Get details about different types of loans.
        
    #     Args:
    #         loan_type: The type of loan (e.g., "personal", "home", "car", "business")
    #     """
    #     # Simulated loan data
    #     loan_data = {
    #         "personal": {
    #             "interest_rate": "8.5%",
    #             "term": "1-5 years",
    #             "min_amount": "5000",
    #             "max_amount": "50000"
    #         },
    #         "home": {
    #             "interest_rate": "6.2%",
    #             "term": "15-30 years",
    #             "min_amount": "100000",
    #             "max_amount": "2000000"
    #         },
    #         "car": {
    #             "interest_rate": "7.8%",
    #             "term": "3-7 years",
    #             "min_amount": "10000",
    #             "max_amount": "100000"
    #         },
    #         "business": {
    #             "interest_rate": "9.2%",
    #             "term": "1-10 years",
    #             "min_amount": "25000",
    #             "max_amount": "1000000"
    #         }
    #     }
        
    #     return loan_data.get(loan_type.lower(), {
    #         "error": f"No information available for {loan_type} loans"
    #     })
        

    async def on_enter(self) -> None:
        print("LoanAgent agent join the meeting")

        
    async def on_exit(self) -> None:
        print("LoanAgent agent Left the meeting")

# Usage 
async def main():
    # Initialize the customer service agent with WebRTC
    customer_model = OpenAIRealtime(
        model="gpt-4o-realtime-preview",
        config=OpenAIRealtimeConfig(
            voice="alloy",
            modalities=["text", "audio"],
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto",
            
        ),
    )

    technical_model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        api_key="AIzaSyACeIOnCtJvfrLOe-js6VBlic-y2BgstHA",
        config=GeminiLiveConfig(
            response_modalities=["TEXT"]
        )
    )
    # technical_model = OpenAIRealtime(
    #     model="gpt-4o-realtime-preview",
    #     config=OpenAIRealtimeConfig(
    #         modalities=["text"],
    #     ),
    # )

    customer_pipeline = RealTimePipeline(model=customer_model)
    technical_pipeline = RealTimePipeline(model=technical_model)
    
    # Create and start the customer service agent
    customer_agent = CustomerServiceAgent()
    customer_session = AgentSession(
        agent=customer_agent,
        pipeline=customer_pipeline,
        context={
            "name": "Customer Service Assistant",
            "meetingId": "pbow-6vec-vahn",
            "join_meeting":True,
            "videosdk_auth": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcGlrZXkiOiI0N2M3ZTJlYy01NzY5LTQ3OWQtYjdjNS0zYjU5MDcxYzhhMDkiLCJwZXJtaXNzaW9ucyI6WyJhbGxvd19qb2luIl0sImlhdCI6MTY3MjgwOTcxMywiZXhwIjoxODMwNTk3NzEzfQ.KeXr1cxORdq6X7-sxBLLV7MsUnwuJGLaG8_VTyTFBig"
        }
    )

    
    # Create the specialist agent (no WebRTC needed)
    specialist_agent = LoanAgent()

    print("specialist_agent::",specialist_agent)
    print("customer_agent::",customer_agent)

    specialist_session = AgentSession(
        agent=specialist_agent,
        pipeline=technical_pipeline,
        context={
            "join_meeting":False
        }
    )

    agentCard = AgentCard(
            id="specialist_1",
            name="Technical Specialist",
            domain="loan",
            capabilities=["technical_support", "problem_solving"],
            description="Handles technical queries and problems"
    )


    await specialist_agent.register_a2a(agentCard)
    
    try:
        await customer_session.start()
        await specialist_session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await customer_session.close()
        await customer_agent.unregister_a2a()
        await specialist_agent.unregister_a2a()

if __name__ == "__main__":
    asyncio.run(main())


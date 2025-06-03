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
                "You are a helpful bank customer service agent. "
                "For general banking queries (account balances, transactions, basic services), answer directly. "
                "For ANY loan-related queries, questions, or follow-ups, ALWAYS use the forward_to_specialist function "
                "with domain set to 'loan'. This includes initial loan questions AND all follow-up questions about loans. "
                "Do NOT attempt to answer loan questions yourself - always forward them to the specialist. "
                "After forwarding a loan query, stay engaged and automatically relay any response you receive from the specialist. "
                "Do not wait for the customer to ask if you received a response - automatically provide it when you get it. "
                "When you receive responses from specialists, immediately relay them naturally to the customer."
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
        
        return {"status": "forwarded", "specialist": id_of_target_agent, "message": "Let me get that information for you from our loan specialist..."}

    async def handle_specialist_response(self, message: A2AMessage) -> None:
        """Handle response from specialist agent"""
        response = message.content.get("response")
        if response:
            print("Response from specialist:", response)
            try:
                # Wait a moment for any current speaking to finish
                await asyncio.sleep(0.5)
                print("About to speak the specialist response...")
                
                # Use send_text_message to trigger a natural response from the model
                prompt = f"The loan specialist has responded. Please provide this information to the customer: {response}"
                await self.session.pipeline.send_text_message(prompt)
                print("Successfully sent specialist response via pipeline send_text_message!")
                
            except Exception as e:
                print(f"Error while trying to send specialist response via pipeline: {e}")
                # Try alternative approach with model send_message
                try:
                    print("Trying alternative approach with model send_message...")
                    await self.session.pipeline.model.send_message(response)
                    print("Successfully sent via model send_message!")
                except Exception as e2:
                    print(f"Model send_message also failed: {e2}")
                    # Last resort - try session.say
                    try:
                        print("Trying last resort with session.say...")
                        await self.session.say(response)
                        print("Successfully sent via session.say!")
                    except Exception as e3:
                        print(f"All methods failed: {e3}")

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
        
        await self.session.say("Hello! I am customer service agent. How can I help you today?")
        
        # Register message handlers
        self.a2a.on_message("specialist_response", self.handle_specialist_response)


class LoanAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="specialist_1", 
            instructions=(
                "You are a specialized loan expert at a bank. "
                "Provide detailed, helpful information about loans including interest rates, terms, and requirements. "
                "Give complete answers with specific details when possible. "
                "You can discuss personal loans, car loans, home loans, and business loans. "
                "Provide helpful guidance and next steps for loan applications. "
                "Be friendly and professional in your responses."
            ),
        )

    async def handle_specialist_query(self, message: A2AMessage) -> None:
        """Handle incoming queries from other agents"""
        query = message.content.get("query")
        print(f"LoanAgent received query: {query}")
        
        if query:
            # Process the query with the model
            await self.session.pipeline.send_text_message(query)

    async def handle_model_response(self, message: A2AMessage) -> None:
        """Handle model responses and send them back to the requesting agent"""
        response = message.content.get("response")
        if response and message.to_agent:
            print(f"LoanAgent sending response back to {message.to_agent}: {response}")
            
            # Send response back to the requesting agent
            await self.a2a.send_message(
                to_agent=message.to_agent,
                message_type="specialist_response",
                content={"response": response}
            )

    async def on_enter(self) -> None:
        print("LoanAgent agent join the meeting")
        
        # Register the agent for A2A communication
        card = AgentCard(
            id="specialist_1",
            name="Loan Specialist Agent",
            domain="loan",
            capabilities=["loan_consultation", "loan_information", "interest_rates"],
            description="Specialized agent for handling loan-related queries and providing loan information"
        )
        await self.register_a2a(card)
        
        # Register message handlers
        self.a2a.on_message("specialist_query", self.handle_specialist_query)
        self.a2a.on_message("model_response", self.handle_model_response)
        
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

    # customer_model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     config=GeminiLiveConfig(
    #         response_modalities=["AUDIO"]
    #     )
    # )

    # technical_model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     api_key="AIzaSyACeIOnCtJvfrLOe-js6VBlic-y2BgstHA",
    #     config=GeminiLiveConfig(
    #         response_modalities=["TEXT"]
    #     )
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
            "meetingId": "re2o-30kc-tbqt",
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

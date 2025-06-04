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
            error_msg = "no specialist found for domain " + domain
            print(f" Error: {error_msg}")
            return {"error": error_msg}

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
            
            # Wait a moment for any current speaking to finish
            await asyncio.sleep(0.5)
            
            prompt = f"The loan specialist has responded. Please provide this information to the customer: {response}"
            
            methods_to_try = [
                (self.session.pipeline.send_text_message, prompt),
                (self.session.pipeline.model.send_message, response),
                (self.session.say, response)
            ]
            
            sent_successfully = False
            for method, arg in methods_to_try:
                try:
                    print(f"Attempting to send specialist response via {method.__name__}...")
                    await method(arg)
                    print(f"Successfully sent specialist response via {method.__name__}!")
                    sent_successfully = True
                    break
                except Exception as e:
                    print(f"Error sending specialist response via {method.__name__}: {e}")
            
            if not sent_successfully:
                print("All methods to send specialist response failed.")

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
                "And make sure all of this will cover within 5-7 lines and short and understandable response"
            ),
        )

    async def handle_specialist_query(self, message: A2AMessage) -> None:
        """Handle incoming queries from other agents"""
        query = message.content.get("query")
        
        if query:
            # Process the query with the model
            await self.session.pipeline.send_text_message(query)

    async def handle_model_response(self, message: A2AMessage) -> None:
        """Handle model responses and send them back to the requesting agent"""
        response = message.content.get("response")
        requesting_agent = message.to_agent
        
        
        if response and requesting_agent:
            
            # Send response back to the requesting agent
            await self.a2a.send_message(
                to_agent=requesting_agent,
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
    print(" Starting main function")
    

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


    # technical_model = OpenAIRealtime(
    #         model="gpt-4o-realtime-preview",
    #         config=OpenAIRealtimeConfig(
    #             modalities=["text"],
    #             tool_choice="auto"
    #         )
    # )


    # customer_model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     config=GeminiLiveConfig(
    #         voice="Leda",
    #         response_modalities=["AUDIO"]
    #     )
    # )

    technical_model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        config=GeminiLiveConfig(
            response_modalities=["TEXT"]
        )
    )

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


    specialist_session = AgentSession(
        agent=specialist_agent,
        pipeline=technical_pipeline,
        context={
            "join_meeting":False
        }
    )

    try:
        print(" Starting agent sessions...")
        await customer_session.start()
        await specialist_session.start()
        print(" Both agent sessions started successfully.")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n Shutting down...")
    except Exception as e:
        print(f" Error in main: {e}")
    finally:
        await customer_session.close()
        await specialist_session.close()
        await customer_agent.unregister_a2a()
        await specialist_agent.unregister_a2a()

if __name__ == "__main__":
    asyncio.run(main())

from videosdk.agents import Agent, AgentCard, A2AMessage, function_tool
import asyncio
from typing import Dict, Any


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
        specialists = self.a2a.registry.find_agents_by_domain(domain)
        id_of_target_agent = specialists[0] if specialists else None
        if not id_of_target_agent:
            return {"error": f"no specialist found for domain {domain}"}

        await self.a2a.send_message(
            to_agent=id_of_target_agent,
            message_type="specialist_query",
            content={"query": query}
        )
        return {
            "status": "forwarded",
            "specialist": id_of_target_agent,
            "message": "Let me get that information for you from our loan specialist..."
        }

    async def handle_specialist_response(self, message: A2AMessage) -> None:
        response = message.content.get("response")
        if response:
            await asyncio.sleep(0.5)
            prompt = f"The loan specialist has responded: {response}"
            methods_to_try = [
                (self.session.pipeline.send_text_message, prompt),# While using Cascading as main agent, comment this
                (self.session.pipeline.model.send_message, response),# While using Cascading as main agent, comment this
                (self.session.say, response)
            ]
            for method, arg in methods_to_try:
                try:
                    await method(arg)
                    break
                except Exception as e:
                    print(f"Error with {method.__name__}: {e}")

    async def on_enter(self):
        print("CustomerAgent joined the meeting")
        await self.register_a2a(AgentCard(
            id="customer_service_1",
            name="Customer Service Agent",
            domain="customer_service",
            capabilities=["query_handling", "specialist_coordination"],
            description="Handles customer queries and coordinates with specialists"
        ))
        await self.session.say("Hello! I am your customer service agent. How can I help you?")
        self.a2a.on_message("specialist_response", self.handle_specialist_response)

    async def on_exit(self):
        print("Customer agent Left the meeting")
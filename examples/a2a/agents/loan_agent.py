from videosdk.agents import Agent, AgentCard, A2AMessage


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
            )
        )

    async def handle_specialist_query(self, message: A2AMessage):
        query = message.content.get("query")
        if query:
            await self.session.pipeline.send_text_message(query)

    async def handle_model_response(self, message: A2AMessage):
        response = message.content.get("response")
        requesting_agent = message.to_agent
        if response and requesting_agent:
            await self.a2a.send_message(
                to_agent=requesting_agent,
                message_type="specialist_response",
                content={"response": response}
            )


    async def on_enter(self):
        await self.register_a2a(AgentCard(
            id="specialist_1",
            name="Loan Specialist Agent",
            domain="loan",
            capabilities=["loan_consultation", "loan_information", "interest_rates"],
            description="Handles loan queries"
        ))
        self.a2a.on_message("specialist_query", self.handle_specialist_query)
        self.a2a.on_message("model_response", self.handle_model_response)

    async def on_exit(self):
        print("LoanAgent Left")
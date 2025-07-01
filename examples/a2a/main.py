import asyncio
from agents.customer_agent import CustomerServiceAgent
from agents.loan_agent import LoanAgent
from session_manager import create_pipeline, create_session

async def main():
    customer_agent = CustomerServiceAgent()
    specialist_agent = LoanAgent()

    customer_pipeline = create_pipeline("customer")
    specialist_pipeline = create_pipeline("specialist")

    customer_session = create_session(customer_agent, customer_pipeline, {
        "name": "Customer Service Assistant",
        "meetingId": "YOUR_MEETING_ID",
        "join_meeting": True
    })

    specialist_session = create_session(specialist_agent, specialist_pipeline, {
        "join_meeting": False
    })

    try:
        await customer_session.start()
        await specialist_session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await customer_session.close()
        await specialist_session.close()
        await customer_agent.unregister_a2a()
        await specialist_agent.unregister_a2a()


if __name__ == "__main__":
    asyncio.run(main())
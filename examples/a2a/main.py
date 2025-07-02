import asyncio
from agents.customer_agent import CustomerServiceAgent
from agents.loan_agent import LoanAgent
from session_manager import create_pipeline, create_session
from videosdk.agents import JobContext, RoomOptions, WorkerJob

async def run_specialist_agent(ctx: JobContext):
    specialist_agent = LoanAgent()
    specialist_pipeline = create_pipeline("specialist")
    specialist_session = create_session(specialist_agent, specialist_pipeline)

    try:
        await specialist_session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await specialist_session.close()
        await specialist_agent.unregister_a2a()

async def run_customer_agent(ctx: JobContext):
    customer_agent = CustomerServiceAgent()
    customer_pipeline = create_pipeline("customer")
    customer_session = create_session(customer_agent, customer_pipeline)

    try:
        await ctx.connect()
        await customer_session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await customer_session.close()
        await customer_agent.unregister_a2a()
        await ctx.shutdown()

def customer_agent_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Customer Service Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )
    
def specialist_agent_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Specialist Service Agent", playground=True, join_meeting=False)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    customer_job = WorkerJob(entrypoint=run_customer_agent, jobctx=customer_agent_context)
    customer_job.start()
    
    specialist_job = WorkerJob(entrypoint=run_specialist_agent, jobctx=specialist_agent_context)
    specialist_job.start()
import asyncio
from contextlib import suppress
from agents.customer_agent import CustomerServiceAgent
from agents.loan_agent import LoanAgent
from session_manager import create_pipeline, create_session
from videosdk.agents import JobContext, RoomOptions, WorkerJob

async def main(ctx: JobContext):
    specialist_agent = LoanAgent()
    specialist_pipeline = create_pipeline("specialist")
    specialist_session = create_session(specialist_agent, specialist_pipeline)

    customer_agent = CustomerServiceAgent()
    customer_pipeline = create_pipeline("customer")
    customer_session = create_session(customer_agent, customer_pipeline)

    specialist_task = asyncio.create_task(specialist_session.start())

    try:
        await ctx.connect()
        await customer_session.start()
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Shutting down...")
    finally:
        specialist_task.cancel()
        with suppress(asyncio.CancelledError):
            await specialist_task

        await specialist_session.close()
        await customer_session.close()
        
        await specialist_agent.unregister_a2a()
        await customer_agent.unregister_a2a()
        await ctx.shutdown()


def customer_agent_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Customer Service Agent", playground=True)
    
    return JobContext(
        room_options=room_options
    )


if __name__ == "__main__":
    job = WorkerJob(entrypoint=main, jobctx=customer_agent_context)
    job.start()
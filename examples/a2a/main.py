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

    shutdown_event = asyncio.Event()
    specialist_task = None 
    
    async def cleanup_sessions():
        print("Cleaning up agent sessions...")
        if specialist_task and not specialist_task.done():
            specialist_task.cancel()
            with suppress(asyncio.CancelledError):
                await specialist_task

        await specialist_session.close()
        await customer_session.close()
        
        await specialist_agent.unregister_a2a()
        await customer_agent.unregister_a2a()
        
        shutdown_event.set()
    
    ctx.add_shutdown_callback(cleanup_sessions)
    
    def on_session_end(reason: str):
        print(f"Session ended: {reason}")
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        specialist_task = asyncio.create_task(specialist_session.start())
        await customer_session.start()
        await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Shutting down...")
    finally:
        if specialist_task and not specialist_task.done():
            specialist_task.cancel()
            with suppress(asyncio.CancelledError):
                await specialist_task
        await specialist_session.close()
        await customer_session.close()
        await specialist_agent.unregister_a2a()
        await customer_agent.unregister_a2a()
        await ctx.shutdown()

def customer_agent_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Customer Service Agent", playground=True)
    
    return JobContext(
        room_options=room_options
    )


if __name__ == "__main__":
    job = WorkerJob(entrypoint=main, jobctx=customer_agent_context)
    job.start()
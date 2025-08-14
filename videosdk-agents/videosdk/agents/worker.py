import multiprocessing
import functools
import signal
import asyncio
import logging
logger = logging.getLogger(__name__)

def _job_runner(entrypoint, job_ctx_factory):
    """
    Creates the job context in the new process and runs the job function.
    """
    try:
        job_ctx = job_ctx_factory() if callable(job_ctx_factory) else job_ctx_factory
        
        from .job import _set_current_job_context, _reset_current_job_context
        token = _set_current_job_context(job_ctx)
        
        try:
            loop = job_ctx._loop

            async def driver():
                try:
                    await entrypoint(job_ctx)
                except asyncio.CancelledError:
                    pass
                finally:
                    try:
                        await job_ctx.shutdown()
                    except Exception as e:
                        logger.error(f"Error during job shutdown: {e}")

            task = loop.create_task(driver())

            shutting_down = False

            def _handle_signal(signum, frame):
                nonlocal shutting_down
                if shutting_down:
                    return
                shutting_down = True
                logger.info(f"Received signal {signum}. Shutting down...")
                try:
                    task.cancel()
                except Exception:
                    pass

            try:
                signal.signal(signal.SIGINT, _handle_signal)
                signal.signal(signal.SIGTERM, _handle_signal)
            except Exception:
                pass

            loop.run_until_complete(task)
        finally:
            _reset_current_job_context(token)
            
    except Exception as e:
        logger.error(f"Error in job runner: {e}")
        import traceback
        traceback.print_exc()


class Worker:
    def __init__(self, job):
        self.job = job
        self.processes = []

    def run(self):
        job_context = functools.partial(self.job.jobctx)
        entrypoint = functools.partial(self.job.entrypoint)
        p = multiprocessing.Process(
            target=_job_runner, args=(entrypoint, job_context)
        )
        p.start()
        self.processes.append((p.pid, p))
        logger.info(f"Started job in PID {p.pid}")

    def _terminate_all_processes(self):
        self._cleanup_processes()
        if not self.processes:
            logger.info("No active processes to terminate.")
            return

        logger.info("Terminating all running processes...")
        for pid, proc in self.processes:
            if proc.is_alive():
                try:
                    proc.terminate()
                    proc.join()
                    logger.info(f"Terminated PID {pid}")
                except Exception as e:
                    logger.error(f"Failed to terminate PID {pid}: {e}")
        self._cleanup_processes()

    def _cleanup_processes(self):
        self.processes = [
            (pid, proc) for pid, proc in self.processes if proc.is_alive()
        ]

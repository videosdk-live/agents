import multiprocessing
import functools

def _job_runner(entrypoint, job_ctx_factory):
    """
    Creates the job context in the new process and runs the job function.
    """
    try:
        job_ctx = job_ctx_factory() if callable(job_ctx_factory) else job_ctx_factory
        
        from .job import _set_current_job_context, _reset_current_job_context
        token = _set_current_job_context(job_ctx)
        
        try:
            job_ctx._loop.run_until_complete(entrypoint(job_ctx))
        finally:
            _reset_current_job_context(token)
            
    except Exception as e:
        print(f"Error in job runner: {e}")
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
        print(f"Started job in PID {p.pid}")

    def _terminate_all_processes(self):
        self._cleanup_processes()
        if not self.processes:
            print("No active processes to terminate.")
            return

        print("Terminating all running processes...")
        for pid, proc in self.processes:
            if proc.is_alive():
                try:
                    proc.terminate()
                    proc.join()
                    print(f"Terminated PID {pid}")
                except Exception as e:
                    print(f"Failed to terminate PID {pid}: {e}")
        self._cleanup_processes()

    def _cleanup_processes(self):
        self.processes = [
            (pid, proc) for pid, proc in self.processes if proc.is_alive()
        ]

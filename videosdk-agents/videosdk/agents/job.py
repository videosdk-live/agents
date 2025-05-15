class WorkerJob:
    def __init__(self, job_func, jobctx=None):
        """
        :param job_func: A function accepting one argument: jobctx
        :param jobctx: A static object or a callable that returns a context per job
        """
        self.job_func = job_func
        self.jobctx = jobctx

    def get_job_context(self):
        return self.jobctx() if callable(self.jobctx) else self.jobctx

    def start(self):
        from .worker import Worker

        worker = Worker(self)
        worker.run()

import multiprocessing


class Worker:
    def __init__(self, job):
        self.job = job  # instance of WorkerJob
        self.processes = []  # List of (pid, Process)

    def run(self):
        jobctx = self.job.get_job_context()
        p = multiprocessing.Process(target=self.job.job_func, args=(jobctx,))
        p.start()
        self.processes.append((p.pid, p))
        print(f"Started job in PID {p.pid}")

    def _terminate_process(self, pid):
        for i, (stored_pid, proc) in enumerate(self.processes):
            if stored_pid == pid:
                if proc.is_alive():
                    try:
                        proc.terminate()
                        proc.join()
                        print(f"Terminated process {pid}")
                    except Exception as e:
                        print(f"Error terminating process {pid}: {e}")
                else:
                    print(f"Process {pid} is already not running.")
                break
        else:
            print(f"No process found with PID {pid}.")

        self._cleanup_processes()

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

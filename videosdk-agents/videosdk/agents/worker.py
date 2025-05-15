import multiprocessing


class Worker:
    def __init__(self, job):
        self.job = job  # instance of WorkerJob
        self.processes = []  # List of (pid, Process)

    def run(self):
        print(
            "Worker CLI started.\n"
            "- 'n' + Enter: start a new worker process\n"
            "- 'l' + Enter: list running worker processes\n"
            "- 'q <pid>' + Enter: stop a specific process\n"
            "- 'q' + Enter: stop all running processes\n"
            "- 'x' + Enter: exit CLI"
        )

        while True:
            cmd = input("> ").strip()

            if cmd == "n":
                jobctx = self.job.get_job_context()

                p = multiprocessing.Process(target=self.job.job_func, args=(jobctx,))
                p.start()
                self.processes.append((p.pid, p))
                print(f"Started job in PID {p.pid}")

            elif cmd == "l":
                self._cleanup_processes()
                if not self.processes:
                    print("No running worker processes.")
                else:
                    for pid, proc in self.processes:
                        print(f"PID {pid} - Alive: {proc.is_alive()}")

            elif cmd == "q":
                self._terminate_all_processes()

            elif cmd.startswith("q "):
                parts = cmd.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("Invalid command. Usage: q <pid> or q to kill all")
                    continue
                pid = int(parts[1])
                self._terminate_process(pid)

            elif cmd == "x":
                print("Exiting CLI.")
                break

            else:
                print("Unknown command. Use 'n', 'l', 'q <pid>', 'q', or 'x'.")

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

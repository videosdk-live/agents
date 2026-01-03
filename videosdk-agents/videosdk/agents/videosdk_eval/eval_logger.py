"""
Custom logging utility for evaluation pipeline.
All logs are prefixed with 'Eval:' to distinguish from regular application logs.
"""
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Colors:
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    BRIGHT_CYAN = "\x1b[96m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"
    GREY = "\x1b[38;20m"

class CustomFormatter(logging.Formatter):
    format = "%(levelname)s - %(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: Colors.GREY + format + Colors.RESET,
        logging.INFO: Colors.GREY + format + Colors.RESET,
        logging.WARNING: Colors.YELLOW + format + Colors.RESET,
        logging.ERROR: Colors.RED + format + Colors.RESET,
        logging.CRITICAL: Colors.BOLD + Colors.RED + format + Colors.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class EvalLogger:
    """Custom logger for evaluation with 'Eval:' prefix and component tracking."""
    
    def __init__(self):
        self.component_start_times = {}
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(CustomFormatter())
            logger.addHandler(ch)
            logger.propagate = False

    def colorize(self, text: str, color: str) -> str:
        return f"{color}{text}{Colors.RESET}"
    
    def log(self, message: str):
        """Log a message with 'Eval:' prefix."""
        logger.info(f"{self.colorize('Eval:', Colors.BRIGHT_CYAN)} {message}")
    
    def debug(self, message: str):
        """Log a debug message (without Eval prefix for detailed logs)."""
        logger.debug(message)
    
    def component_start(self, component: str):
        """Log component start and track start time."""
        self.component_start_times[component] = time.perf_counter()
        color = Colors.CYAN
        if "STT" in component: color = Colors.YELLOW
        elif "LLM" in component: color = Colors.MAGENTA
        elif "TTS" in component: color = Colors.BLUE
        
        comp_str = self.colorize(f"[{component}]", color)
        self.log(f"{comp_str} Starting...")
    
    def component_end(self, component: str, latency_ms: Optional[float] = None):
        """Log component completion with latency."""
        if latency_ms is None and component in self.component_start_times:
            # Calculate latency from tracked start time
            elapsed = time.perf_counter() - self.component_start_times[component]
            latency_ms = elapsed * 1000
            del self.component_start_times[component]
        
        color = Colors.CYAN
        if "STT" in component: color = Colors.YELLOW
        elif "LLM" in component: color = Colors.MAGENTA
        elif "TTS" in component: color = Colors.BLUE
        
        comp_str = self.colorize(f"[{component}]", color)
        if latency_ms is not None:
            latency_str = self.colorize(f"{latency_ms:.2f}ms", Colors.GREEN)
            self.log(f"{comp_str} Completed in {latency_str}")
        else:
            self.log(f"{comp_str} Completed")
    
    def turn_start(self, turn_num: int, total: int, turn_id: str):
        """Log turn start."""
        sep = self.colorize('=' * 50, Colors.BRIGHT_CYAN)
        self.log(sep)
        self.log(f"{Colors.BOLD}Turn {turn_num}/{total}{Colors.RESET} (ID: {turn_id})")
        self.log(sep)
    
    def turn_end(self, turn_num: int, total_latency_ms: Optional[float] = None):
        """Log turn completion."""
        if total_latency_ms is not None:
            latency_str = self.colorize(f"{total_latency_ms:.2f}ms", Colors.GREEN)
            self.log(f"Turn {turn_num} completed - Total latency: {latency_str}")
        else:
            self.log(f"Turn {turn_num} completed")
        self.log("")
    
    def evaluation_start(self, name: str, turn_count: int):
        """Log evaluation start."""
        header = self.colorize(f"Starting evaluation: {name} with {turn_count} turns", Colors.BOLD + Colors.MAGENTA)
        self.log(header)
        self.log("")
    
    def evaluation_end(self):
        """Log evaluation completion."""
        self.log(self.colorize("Evaluation completed!", Colors.BOLD + Colors.GREEN))
    
    def display_turn_logs(self, turn_index: int, turn_id: str, stt_transcript: str, llm_response: str, tts_audio_file: str, llm_input: str = ""):
        """Display formatted logs for a single turn."""
        sep = self.colorize('=' * 40, Colors.YELLOW)
        self.log(sep)
        self.log(f"{Colors.BOLD}TURN {turn_index + 1} RESULTS{Colors.RESET} (ID: {turn_id})")
        self.log(sep)

        if stt_transcript:
            self.log(f"\n{self.colorize('STT Transcript:', Colors.YELLOW)}\n  {stt_transcript}")
        else:
            self.log(f"\n{self.colorize('STT Transcript:', Colors.YELLOW)}\n  N/A")
        
        if llm_input and llm_input != stt_transcript and llm_input != "N/A":
            self.log(f"\n{self.colorize('LLM Input (Overridden):', Colors.CYAN)}\n  {llm_input}")
        
        if llm_response:
            self.log(f"\n{self.colorize('LLM Response:', Colors.MAGENTA)}\n  {llm_response}")
        else:
            self.log(f"\n{self.colorize('LLM Response:', Colors.MAGENTA)}\n  N/A")
        
        if tts_audio_file:
            self.log(f"\n{self.colorize('TTS Audio File:', Colors.BLUE)}\n  → {tts_audio_file}")
        else:
            self.log(f"\n{self.colorize('TTS Audio File:', Colors.BLUE)}\n  N/A")

    def display_judge_results(self, results: list):
        """Display judge results in a formatted table."""
        sep = self.colorize('=' * 40, Colors.MAGENTA)
        self.log(sep)
        self.log(f"{Colors.BOLD}LLM-AS-JUDGE RESULTS{Colors.RESET}")
        self.log(sep)
        
        for res in results:
            judge = res.get("judge", {})
            turn_id = res.get("turn_id", "Unknown")
            
            self.log(f"Turn ID: {turn_id}")
            if judge.get("status") == "skipped":
                status_str = self.colorize("SKIPPED", Colors.YELLOW)
                self.log(f"Judge Status: {status_str} ({judge.get('reason', 'N/A')})")
            else:
                passed = judge.get("passed", False)
                status_str = self.colorize("PASSED", Colors.GREEN) if passed else self.colorize("FAILED", Colors.RED)
                self.log(f"Judge Status: {status_str}")
                self.log(f"{Colors.BOLD}Evaluation Details:{Colors.RESET}\n{judge.get('evaluation', 'N/A')}")
            self.log(self.colorize("-" * 20, Colors.GREY))
        self.log(sep + "\n")

    def display_latency_table(self, results: list, metrics_filter: list = None):
        """Display latencies in a formatted table based on specified metrics."""

        sep = self.colorize('=' * 40, Colors.GREEN)
        self.log(sep)
        self.log(f"{Colors.BOLD}LATENCY SUMMARY{Colors.RESET}")
        self.log(sep)
        
        METRIC_CONFIG = {
            "stt_latency": {"label": "STT (ms)", "key": "stt_latency", "color": Colors.YELLOW},
            "llm_ttft": {"label": "LLM (ms)", "key": "llm_ttft", "color": Colors.MAGENTA},
            "ttfb": {"label": "TTS (ms)", "key": "ttfb", "color": Colors.BLUE},
            "e2e_latency": {"label": "E2E (ms)", "key": "e2e_latency", "color": Colors.GREEN},
        }
        
        if metrics_filter:
            metric_keys = [m.value for m in metrics_filter]
            display_metrics = {k: v for k, v in METRIC_CONFIG.items() if k in metric_keys}
        else:
            display_metrics = METRIC_CONFIG

        col_width = 15
        header_parts = [f"{'Turn':<8}"]
        for metric_key in display_metrics:
            label = display_metrics[metric_key]["label"]
            header_parts.append(f"{label:<{col_width}}")
        header = "".join(header_parts)
        separator_length = len(header)
        
        self.log(self.colorize(header, Colors.BOLD))
        self.log(self.colorize("-" * separator_length, Colors.GREY))
        
        # Table rows
        for res in results:
            turn_idx = res.get("turn_index", 0)
            metrics_data = res.get("metrics", {})
            
            row_parts = [f"{turn_idx + 1:<8}"]
            
            for metric_key in display_metrics:
                key = display_metrics[metric_key]["key"]
                color = display_metrics[metric_key]["color"]
                latency = metrics_data.get(key, 0)
                
                if latency:
                    latency_str = self.colorize(f"{latency:.2f}", color)
                else:
                    latency_str = "N/A"
                padding = " " * (col_width - (len(f"{latency:.2f}") if latency else 3))
                row_parts.append(f"{latency_str}{padding}")
            
            row = "".join(row_parts)
            self.log(row)
        
        self.log("")


# Global eval logger instance
eval_logger = EvalLogger()

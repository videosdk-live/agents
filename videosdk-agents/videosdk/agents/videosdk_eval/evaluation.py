from typing import List, Optional
import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from videosdk.agents.videosdk_eval.metrics import Metric
from videosdk.agents.videosdk_eval.turn import Turn
from videosdk.agents.videosdk_eval.agent_wrapper import EvalAgent
from videosdk.agents.videosdk_eval.cascading_flow import CascadingConversationFlow
from videosdk.agents.videosdk_eval.factory import create_stt, create_llm, create_tts
from videosdk.agents.videosdk_eval.audio_track import MockAudioTrack
from videosdk.agents.videosdk_eval.providers.llm import LLM as LLMFactory
import uuid
import csv

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import json
import csv
import uuid
from .eval_logger import eval_logger


@dataclass
class EvaluationResult:
    results: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, output_dir: str) -> None:
        run_id = uuid.uuid4()
        os.makedirs(output_dir, exist_ok=True)

        self._save_json(output_dir, run_id)
        self._save_metrics_csv(output_dir, run_id)
        self._save_transcripts(output_dir, run_id)
        self._save_component_metrics(output_dir, run_id)
        self._save_judge_results(output_dir, run_id)

    # JSON
    def _save_json(self, output_dir: str, run_id: uuid.UUID) -> None:
        path = os.path.join(output_dir, f"results_{run_id}.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=4)
        eval_logger.log(f"Results saved to {path}")

    # Metrics CSV (TTFW, latency, etc.)

    def _save_metrics_csv(self, output_dir: str, run_id: uuid.UUID) -> None:
        rows = []
        fieldnames = {"turn_index", "turn_id"}

        for res in self.results:
            metrics = res.get("metrics", {})
            if not metrics:
                continue

            row = {
                "turn_index": res.get("turn_index"),
                "turn_id": res.get("turn_id"),
            }

            for key, value in metrics.items():
                if (
                    key.startswith(("stt_latency", "llm_latency", "tts_latency"))
                    or "time_to" in key
                ):
                    fieldnames.add(key)
                    row[key] = value

            if len(row) > 2:
                rows.append(row)

        if not rows:
            return

        path = os.path.join(output_dir, f"metrics_{run_id}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

        eval_logger.log(f"Metrics saved to {path}")

    # Transcripts TXT
    def _save_transcripts(self, output_dir: str, run_id: uuid.UUID) -> None:
        path = os.path.join(output_dir, f"transcripts_{run_id}.txt")

        try:
            with open(path, "w") as f:
                for res in self.results:
                    metrics = res.get("metrics", {})
                    transcripts = metrics.get("collected_transcripts", [])
                    if not transcripts:
                        continue

                    f.write(f"Turn ID: {res.get('turn_id', 'Unknown')}\n")
                    f.write(f"Transcript: {' '.join(transcripts)}\n")
                    f.write("-" * 40 + "\n")

            eval_logger.log(f"Transcripts saved to {path}")
        except Exception as e:
            eval_logger.log(f"Failed to save transcripts: {e}")

    # Component-level Metrics CSV
    def _save_component_metrics(self, output_dir: str, run_id: uuid.UUID) -> None:
        path = os.path.join(output_dir, f"component_metrics_{run_id}.csv")
        fieldnames = ["turn_index", "turn_id", "component", "latency_ms", "input", "output"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.results:
                metrics = res.get("metrics", {})
                turn_index = res.get("turn_index")
                turn_id = res.get("turn_id", "Unknown")

                if "stt_latency" in metrics:
                    writer.writerow({
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "component": "STT",
                        "latency_ms": round(metrics.get("stt_latency") or 0, 2),
                        "input": "Audio",
                        "output": " ".join(metrics.get("collected_transcripts", [])),
                    })

                if "llm_latency" in metrics:
                    writer.writerow({
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "component": "LLM",
                        "latency_ms": round(metrics.get("llm_latency") or 0, 2),
                        "input": " ".join(metrics.get("collected_transcripts", [])),
                        "output": metrics.get("agent_response", ""),
                    })

                if "tts_latency" in metrics:
                    writer.writerow({
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "component": "TTS",
                        "latency_ms": round(metrics.get("tts_latency") or 0, 2),
                        "input": metrics.get("agent_response", ""),
                        "output": metrics.get("tts_audio_file", ""),
                    })

        eval_logger.log(f"Component metrics saved to {path}")

    # Judge Results TXT
    def _save_judge_results(self, output_dir: str, run_id: uuid.UUID) -> None:
        path = os.path.join(output_dir, f"judge_results_{run_id}.txt")

        try:
            with open(path, "w") as f:
                for res in self.results:
                    judge = res.get("judge", {})
                    f.write(f"Turn ID: {res.get('turn_id', 'Unknown')}\n")
                    f.write(f"Judge Passed: {judge.get('passed', False)}\n")

                    if "reason" in judge:
                        f.write(f"Reason: {judge['reason']}\n")
                    if "evaluation" in judge:
                        f.write(f"Evaluation: {judge['evaluation']}\n")

                    f.write("-" * 40 + "\n")

            eval_logger.log(f"Judge results saved to {path}")
        except Exception as e:
            eval_logger.log(f"Failed to save judge results: {e}")


class Evaluation:
    def __init__(
        self,
        name: str,
        metrics: List[Metric]
    ):
        self.name = name
        self.metrics = metrics
        self.turns: List[Turn] = []

    def add_turn(self, turn: Turn):
        self.turns.append(turn)

    def run(self) -> EvaluationResult:
        try:
            return asyncio.run(self._run_async())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._run_async())

    async def _run_async(self) -> EvaluationResult:
        eval_logger.evaluation_start(self.name, len(self.turns))
        results = []
        
        for i, turn in enumerate(self.turns):
            eval_logger.turn_start(i+1, len(self.turns), str(turn.id or 'N/A'))
            turn_start_time = time.perf_counter()
            turn_result = await self._process_turn(turn, i)
            turn_elapsed = (time.perf_counter() - turn_start_time) * 1000
            eval_logger.turn_end(i+1, turn_elapsed)
            results.append(turn_result)
        
        # Display formatted logs at the end
        eval_logger.log("\n" + "=" * 60)
        eval_logger.log("EVALUATION RESULTS")
        eval_logger.log("=" * 60)
        
        for res in results:
            turn_idx = res.get("turn_index", 0)
            turn_id = str(res.get("turn_id", "N/A"))
            metrics = res.get("metrics", {})
            
            # Get STT transcript
            stt_transcripts = metrics.get("collected_transcripts", [])
            stt_transcript = " ".join(stt_transcripts) if stt_transcripts else ""
            
            # Get LLM response
            llm_response = res.get("response_text", "")
            
            # Get TTS audio file
            tts_audio_file = metrics.get("tts_audio_file", "")
            
            # Display turn logs
            eval_logger.display_turn_logs(turn_idx, turn_id, stt_transcript, llm_response, tts_audio_file)
        
        # Display latency table
        eval_logger.display_latency_table(results)
        
        eval_logger.evaluation_end()
        return EvaluationResult(results=results)

    async def tts_audio_file(self, audio_track: MockAudioTrack, index: int):
        if audio_track:
            audio_bytes = audio_track.get_audio_bytes()
            if audio_bytes:
                import uuid
                unique_id = uuid.uuid4()
                tts_filename = f"tts_output_{unique_id}.wav"
                os.makedirs("./reports", exist_ok=True)
                tts_filepath = os.path.join("./reports", tts_filename)
                import wave
                with wave.open(tts_filepath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_bytes)
                
                eval_logger.log(f"TTS Audio saved to {tts_filepath}")
                return tts_filepath

    async def _process_turn(self, turn: Turn, index: int) -> dict:
        stt_instance = None
        llm_instance = None
        tts_instance = None

        if turn.stt:
            provider, config = turn.stt
            stt_instance = create_stt(provider, config)
        
        agent_tools = []
        agent_instructions = "You are a helpful assistant."
        if turn.llm:
            provider, config = turn.llm
            llm_instance = create_llm(provider, config)
            system_prompt = config.get("system_prompt")
            # Extract tools if available in the config
            agent_tools = config.get("tools", [])
            if system_prompt:
                agent_instructions += "\n\n" + system_prompt

        if turn.tts:
            provider, config = turn.tts
            tts_instance = create_tts(provider, config)

        agent = EvalAgent(instructions=agent_instructions, tools=agent_tools)
        flow = CascadingConversationFlow(
            agent=agent,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance
        )
        
        computed_metrics = {}
        response_text = ""

        try:
            await flow.start()
            if turn.stt:
                from videosdk.agents.metrics import cascading_metrics_collector
                cascading_metrics_collector.on_stt_start()
            
            # --- Config Extraction ---
            stt_file_path = None
            stt_chunk_size = 24000
            if turn.stt:
                _, stt_config = turn.stt
                stt_file_path = stt_config.get("file_path")
                stt_chunk_size = stt_config.get("chunk_size", 24000)

            use_stt_output = True
            llm_mock_input = None
            if turn.llm:
                _, llm_config = turn.llm
                use_stt_output = llm_config.get("use_stt_output", True)
                llm_mock_input = llm_config.get("mock_input")

            use_llm_output = True
            tts_mock_input = None
            if turn.tts:
                _, tts_config = turn.tts
                use_llm_output = tts_config.get("use_llm_output", True)
                tts_mock_input = tts_config.get("mock_input")

            # --- Flow Configuration ---
            # Configure TTS behavior BEFORE generation starts
            if turn.tts and not use_llm_output and tts_mock_input:
                # Enable TTS but swap LLM output with mock input
                flow.enable_tts_processing = True
                flow.tts_mock_input = tts_mock_input
                eval_logger.log(f"Configured Mock TTS Input (will override LLM output): {tts_mock_input}")
            else:
                # Standard behavior: use LLM output for TTS if configured
                flow.enable_tts_processing = use_llm_output
                flow.tts_mock_input = None

            # --- STT / Input Phase ---
            if stt_file_path and os.path.exists(stt_file_path):
                eval_logger.component_start("STT")
                should_suppress_llm_during_stt = False
                if not use_stt_output and llm_mock_input:
                    should_suppress_llm_during_stt = True
                
                if should_suppress_llm_during_stt:
                    flow.enable_llm_processing = False
                    eval_logger.log("Suppressing LLM for STT streaming (Waiting for transcript)...")

                with open(stt_file_path, "rb") as f:
                    data = f.read()
                    for j in range(0, len(data), stt_chunk_size):
                        chunk = data[j:j+stt_chunk_size]
                        await flow.send_audio_delta(chunk)
                        await asyncio.sleep(0.075)  # Simulate real-time streaming
                
                if should_suppress_llm_during_stt:
                    eval_logger.log("Waiting for STT events to settle (draining buffer)...")
                    await asyncio.sleep(8.0)  # Give STT time to process all audio
                     
                    if flow.agent and flow.agent.chat_context:
                        from videosdk.agents.llm.chat_context import ChatRole
                        ctx = flow.agent.chat_context
                        ctx._items = [item for item in ctx._items if getattr(item, 'role', None) == ChatRole.SYSTEM]
                        eval_logger.log("Cleared STT transcripts from ChatContext for pure Mock LLM input.")
                    
                    flow.allowed_mock_input = llm_mock_input
                    eval_logger.log(f"STT skipped for LLM. Processing mock input: {llm_mock_input}")
                    flow.generation_done_event.clear()
                    await flow.process_text_input(llm_mock_input)
            else:
                if turn.llm and llm_mock_input:
                    eval_logger.log(f"No STT file. Processing mock input directly: {llm_mock_input}")
                    flow.allowed_mock_input = llm_mock_input
                    flow.generation_done_event.clear()
                    await flow.process_text_input(llm_mock_input)
            
            # Wait for generation if LLM is configured 
            if turn.llm:
                try:
                    await asyncio.wait_for(flow.generation_done_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    eval_logger.log("Timed out waiting for LLM generation.")
            
            # For TTS-only turns (no LLM), manually synthesize
            if turn.tts and not turn.llm and tts_mock_input:
                eval_logger.log(f"TTS-only turn. Synthesizing: {tts_mock_input}")
                from videosdk.agents.utterance_handle import UtteranceHandle
                from videosdk.agents.utils import AgentState
                import uuid

                if flow.agent and flow.agent.session:
                    mock_handle_id = f"mock_tts_{uuid.uuid4()}" 
                    mock_handle = UtteranceHandle(utterance_id=mock_handle_id, interruptible=False)
                    flow.agent.session.current_utterance = mock_handle
                    flow.agent.session._emit_agent_state(AgentState.SPEAKING)
                
                async def text_iterator():
                    yield tts_mock_input
                
                await flow._synthesize_with_tts(text_iterator())
    
    
            computed_metrics = flow.metrics
            
            # Prefer LLM response from Chat Context if available
            response_text = ""
            try:
                if flow.agent and flow.agent.chat_context:
                    from videosdk.agents.llm.chat_context import ChatRole
                    # Get the last assistant message
                    assistant_msgs = [m for m in flow.agent.chat_context._items if getattr(m, 'role', None) == ChatRole.ASSISTANT]
                    if assistant_msgs:
                        response_text = assistant_msgs[-1].content
            except Exception as e:
                eval_logger.log(f"Error extracting from ChatContext: {e}")
            if not response_text:
                response_text = computed_metrics.get("agent_response", "")

            # Fallback: If no LLM response (e.g. LLM skipped or muted) and we used TTS mock input,
            # consider the TTS mock input as the agent's response for judging.
            if not response_text and turn.tts and not use_llm_output and tts_mock_input:
                response_text = tts_mock_input

        except Exception as e:
            eval_logger.log(f"Error executing turn: {e}")
            response_text = f"Error: {str(e)}"
            import traceback
            traceback.eval_logger.log_exc()

        finally:
            await flow.cleanup()
            if stt_instance and hasattr(stt_instance, 'aclose'): await stt_instance.aclose()
            if llm_instance and hasattr(llm_instance, 'aclose'): await llm_instance.aclose()
            if tts_instance and hasattr(tts_instance, 'aclose'): await tts_instance.aclose()
            await agent.cleanup()

        # 3. Judge
        judge_result = {"passed": False, "reason": "No judge configured"}
        if turn.judge:
            provider, config = turn.judge
            from .providers.llm import LLMEvalConfig
            judge_llm_config = LLMEvalConfig(model=config.model, api_key=os.getenv("GOOGLE_API_KEY_LLM"))
            judge_llm = create_llm(provider, judge_llm_config)
            # Determine user input for judge: prefer STT transcript if available, otherwise LLM mock input
            if turn.stt:
                # Use collected transcripts from metrics if present
                stt_transcripts = computed_metrics.get('collected_transcripts', [])
                if stt_transcripts:
                    if turn.llm and not use_stt_output:
                        # User actually spoke, but we used mock input for LLM
                        # Extract the LLM config properly
                        _, llm_cfg = turn.llm
                        user_input = llm_cfg.get('mock_input', 'Audio')
                    else:
                        # User spoke and we used their speech
                        user_input = " ".join(stt_transcripts)
                else:
                    user_input = "Audio"
            elif turn.llm:
                _, llm_config = turn.llm
                user_input = llm_config.get('mock_input', 'Audio')
            else:
                user_input = 'Audio'

            prompt = f"{config.prompt}\n\nUser Input: {user_input}\nAgent Output: {response_text}\n\nEvaluate based on: {config.checks}"
            # Simple judge chat
            from videosdk.agents.llm.chat_context import ChatContext, ChatRole
            judge_context = ChatContext()
            if judge_context:
                judge_context.add_message(ChatRole.USER, prompt)
            
            judge_response = ""
            try:
                async for chunk in judge_llm.chat(judge_context):
                    judge_response += chunk.content
                judge_result = {"passed": True, "score": "N/A", "evaluation": judge_response}
            except Exception as e:
                eval_logger.log(f"Judge evaluation failed: {e}")
                judge_result = {"passed": False, "reason": str(e)}
            if judge_llm and hasattr(judge_llm, 'aclose'):
                await judge_llm.aclose()

        if turn.tts and flow.audio_track:
            audio_bytes = flow.audio_track.get_audio_bytes()
            if audio_bytes:
                import uuid
                tts_filename = f"tts_output_{uuid.uuid4()}.wav"
                os.makedirs("./reports", exist_ok=True)
                tts_filepath = os.path.join("./reports", tts_filename)
                import wave
                with wave.open(tts_filepath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_bytes)
                
                eval_logger.log(f"TTS Audio saved to {tts_filepath}")
                computed_metrics["tts_audio_file"] = tts_filepath

        return {
            "turn_index": index,
            "turn_id": turn.id,
            "metrics": computed_metrics,
            "response_text": response_text,
            "judge": judge_result
        }

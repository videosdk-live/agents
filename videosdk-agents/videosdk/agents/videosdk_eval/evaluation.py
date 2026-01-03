# Standard library
import asyncio
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List
from videosdk.agents.llm import ChatContext, ChatRole
from videosdk.agents.metrics import cascading_metrics_collector

# VideoSDK eval framework
from videosdk.agents.videosdk_eval.agent_wrapper import EvalAgent
from videosdk.agents.videosdk_eval.audio_track import MockAudioTrack
from videosdk.agents.videosdk_eval.cascading_flow import EvalConversationFlow
from videosdk.agents.videosdk_eval.factory import (
    create_llm,
    create_stt,
    create_tts,
)
from videosdk.agents.videosdk_eval.metrics import EvalMetric
from videosdk.agents.videosdk_eval.turn import EvalTurn

# Local imports
from .eval_logger import eval_logger


@dataclass
class EvaluationResult:
    results: List[Dict[str, Any]] = field(default_factory=list)
    metrics_filter: List = field(default_factory=lambda: None)
    output_dir: str = field(default="./eval_reports")

    def save(self) -> None:
        run_id = uuid.uuid4()
        os.makedirs(self.output_dir, exist_ok=True)

        self._save_metrics_csv(run_id)
        self._save_transcripts(run_id)
        self._save_judge_results(run_id)

    def _save_metrics_csv(self, run_id: uuid.UUID) -> None:
        rows = []
        fieldnames = {"turn_index", "turn_id"}
        if self.metrics_filter:
            allowed_metrics = {m.value for m in self.metrics_filter}
        else:
            allowed_metrics = None

        for res in self.results:
            metrics = res.get("metrics", {})

            row = {
                "turn_index": res.get("turn_index"),
                "turn_id": res.get("turn_id"),
            }

            for key, value in metrics.items():
                is_latency_metric = (
                    key.startswith(("stt_latency", "llm_ttft", "ttfb", "e2e_latency"))
                    or "time_to" in key
                )
                if is_latency_metric:
                    if allowed_metrics is None or key in allowed_metrics:
                        fieldnames.add(key)
                        row[key] = value

            if len(row) > 2:
                rows.append(row)

        if not rows:
            return

        path = os.path.join(self.output_dir, f"metrics_{run_id}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

        eval_logger.log(f"Metrics saved to {path}")

    def _save_transcripts(self, run_id: uuid.UUID) -> None:
        path = os.path.join(self.output_dir, f"transcripts_{run_id}.txt")

        try:
            with open(path, "w") as f:
                for res in self.results:
                    metrics = res.get("metrics", {})
                    user_transcripts = metrics.get("collected_transcripts", [])
                    agent_response = res.get("response_text", "")
                    
                    if not user_transcripts and not agent_response:
                        continue

                    f.write(f"Turn ID: {res.get('turn_id', 'Unknown')}\n")
                    stt_text = ' '.join(user_transcripts) if user_transcripts else 'N/A'
                    f.write(f"User STT Transcript: {stt_text}\n")
                    
                    llm_input = metrics.get("llm_input")
                    if llm_input and llm_input != stt_text:
                        f.write(f"LLM Input (Overridden): {llm_input}\n")

                    f.write(f"Agent: {agent_response if agent_response else 'N/A'}\n")
                    f.write("-" * 40 + "\n")

            eval_logger.log(f"Transcripts saved to {path}")
        except Exception as e:
            eval_logger.log(f"Failed to save transcripts: {e}")

    def _save_judge_results(self, run_id: uuid.UUID) -> None:
        path = os.path.join(self.output_dir, f"judge_results_{run_id}.txt")

        try:
            with open(path, "w") as f:
                for res in self.results:
                    judge = res.get("judge", {})
                    f.write(f"Turn ID: {res.get('turn_id', 'Unknown')}\n")
                    f.write(f"Judge Passed: {judge.get('passed', False)}\n")
                    f.write(f"LLM-as-Judge Response: {judge}\n")
                    f.write("-" * 40 + "\n")

            eval_logger.log(f"Judge results saved to {path}")
        except Exception as e:
            eval_logger.log(f"Failed to save judge results: {e}")

class Evaluation:
    def __init__(
        self,
        name: str,
        metrics: List[EvalMetric] = None,
        include_context: bool = False,
        output_dir:str = "./eval_reports"
    ):
        self.name = name
        self.metrics = metrics
        self.include_context = include_context
        self.output_dir = output_dir
        self.turns: List[EvalTurn] = []

    def add_turn(self, turn: EvalTurn):
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
        
        agent_instructions = "You are a helpful assistant."
        agent_tools = []
        if self.turns:
            for turn in self.turns:
                if turn.llm:
                    _, config = turn.llm
                    if config.get("system_prompt"):
                        agent_instructions = config.get("system_prompt")
                    break

        agent = EvalAgent(instructions=agent_instructions, tools=agent_tools)
        
        try:
            for i, turn in enumerate(self.turns):
                eval_logger.turn_start(i+1, len(self.turns), str(turn.id or 'N/A'))
                turn_start_time = time.perf_counter()
                turn_result = await self._process_turn(turn, i, agent,self.output_dir)
                turn_elapsed = (time.perf_counter() - turn_start_time) * 1000
                eval_logger.turn_end(i+1, turn_elapsed)
                results.append(turn_result)
        finally:
            await agent.cleanup()
        
        # Display formatted logs at the end
        eval_logger.log("=" * 40)
        eval_logger.log("EVALUATION RESULTS")
        eval_logger.log("=" * 40)
        
        for res in results:
            turn_idx = res.get("turn_index", 0)
            turn_id = str(res.get("turn_id", "N/A"))
            metrics = res.get("metrics", {})
            
            # Get LLM input (what was actually sent to LLM)
            llm_input = metrics.get("llm_input") or ""
            
            # Get STT transcript
            stt_transcripts = metrics.get("collected_transcripts", [])
            stt_transcript = ""
            if stt_transcripts:
                stt_transcript = " ".join(stt_transcripts)
            
            # Fallback to user_speech ONLY if it's not the same as llm_input
            if not stt_transcript:
                user_speech = metrics.get("user_speech")
                if user_speech and user_speech != llm_input:
                    stt_transcript = user_speech
                else:
                    stt_transcript = "N/A"
            
            # Get LLM response
            llm_response = res.get("response_text", "") or "N/A"
            
            # Get TTS audio file
            tts_audio_file = metrics.get("tts_audio_file", "")
            
            # Display turn logs
            eval_logger.display_turn_logs(turn_index=turn_idx, turn_id=turn_id, stt_transcript=stt_transcript, 
                                        llm_response=llm_response, tts_audio_file=tts_audio_file, llm_input=llm_input)

        eval_logger.display_judge_results(results)
        eval_logger.display_latency_table(results, self.metrics)
        eval_logger.evaluation_end()
        return EvaluationResult(results=results, metrics_filter=self.metrics,output_dir=self.output_dir)

    async def tts_audio_file(self, audio_track: MockAudioTrack,output_dir:str):
        if audio_track:
            audio_bytes = audio_track.get_audio_bytes()
            if audio_bytes:
                import uuid
                unique_id = uuid.uuid4()
                tts_filename = f"tts_output_{unique_id}.wav"
                os.makedirs(output_dir, exist_ok=True)
                tts_filepath = os.path.join(output_dir, tts_filename)
                import wave
                with wave.open(tts_filepath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_bytes)
                
                eval_logger.log(f"TTS Audio saved to {tts_filepath}")
                return tts_filepath

    async def _process_turn(self, turn: EvalTurn, index: int, agent: EvalAgent,output_dir:str) -> dict:
        stt_instance = None
        llm_instance = None
        tts_instance = None

        if turn.stt:
            provider, config = turn.stt
            stt_instance = create_stt(provider, config)
        
        if turn.llm:
            provider, config = turn.llm
            llm_instance = create_llm(provider, config)
            system_prompt = config.get("system_prompt")
            if system_prompt:
                agent.instructions = system_prompt
            if config.get("tools"):
                agent.update_tools(config.get("tools"))

        if turn.tts:
            provider, config = turn.tts
            tts_instance = create_tts(provider, config)

        flow = EvalConversationFlow(
            agent=agent,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance
        )
        flow.enable_stt_processing = turn.stt is not None
        flow.enable_llm_processing = turn.llm is not None
        flow.enable_tts_processing = turn.tts is not None
        
        computed_metrics = {}

        try:
            # Initialize metrics for this turn
            cascading_metrics_collector.start_new_interaction()
            
            await flow.start()
            if turn.stt:
                cascading_metrics_collector.on_stt_start()
            
            # Config Extraction
            stt_filepath = None
            stt_chunk_size = 96000
            if turn.stt:
                _, stt_config = turn.stt
                stt_filepath = stt_config.get("file_path")
                stt_chunk_size = stt_config.get("chunk_size", 96000)

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

            # Flow Configuration
            if turn.tts and not use_llm_output and tts_mock_input:
                flow.enable_tts_processing = True
                flow.tts_mock_input = tts_mock_input
                eval_logger.log(f"Configured Mock TTS Input (will override LLM output): {tts_mock_input}")
            else:
                # Standard behavior: use LLM output for TTS if configured
                flow.enable_tts_processing = use_llm_output
                flow.tts_mock_input = None

            # STT
            if stt_filepath and os.path.exists(stt_filepath):
                eval_logger.component_start("STT")
                should_suppress_llm_during_stt = False
                if not use_stt_output and llm_mock_input:
                    should_suppress_llm_during_stt = True
                
                if should_suppress_llm_during_stt:
                    flow.enable_llm_processing = False
                    eval_logger.log("Suppressing LLM for STT streaming (Waiting for transcript)...")

                with open(stt_filepath, "rb") as f:
                    data = f.read()
                    for j in range(0, len(data),stt_chunk_size):
                        chunk = data[j:j+stt_chunk_size]
                        await flow.send_audio_delta(chunk)
                        await asyncio.sleep(0.075)
                
                if hasattr(flow.stt, 'flush'):
                    await flow.stt.flush()
                
                cascading_metrics_collector.on_user_speech_end()
                
                if should_suppress_llm_during_stt:
                    eval_logger.log("Waiting for STT events to settle (draining buffer)...")
                    await asyncio.sleep(15.0)

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
            elif turn.stt:
                try:
                    await asyncio.wait_for(flow.stt_done_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    eval_logger.log("Timed out waiting for STT transcription.")
            
            # For TTS-only turns (no LLM), manually synthesize
            if turn.tts and not turn.llm and tts_mock_input:
                eval_logger.log(f"TTS-only turn. Synthesizing: {tts_mock_input}")
                from videosdk.agents.utterance_handle import UtteranceHandle
                from videosdk.agents.utils import AgentState

                if flow.agent and flow.agent.session:
                    mock_handle_id = f"mock_tts_{uuid.uuid4()}" 
                    mock_handle = UtteranceHandle(utterance_id=mock_handle_id, interruptible=False)
                    flow.agent.session.current_utterance = mock_handle
                    flow.agent.session._emit_agent_state(AgentState.SPEAKING)
                
                async def text_iterator():
                    yield tts_mock_input
                
                await flow._synthesize_with_tts(text_iterator())
                cascading_metrics_collector.on_agent_speech_end()

            computed_metrics = flow.metrics
            metrics_data = flow.metrics
            actual_llm_response = metrics_data.get("agent_speech", "") or "N/A"
            spoken_response = actual_llm_response
            
            # For TTS-override turns, keep track of what was actually spoken vs generated
            if turn.tts and not use_llm_output and tts_mock_input:
                spoken_response = tts_mock_input
                if not actual_llm_response or actual_llm_response == "N/A":
                    if flow.agent and flow.agent.chat_context:
                        flow.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=tts_mock_input)
            cascading_metrics_collector.complete_current_turn()
            computed_metrics = flow.metrics

        except Exception as e:
            eval_logger.log(f"Error executing turn: {e}")
            actual_llm_response = f"Error: {str(e)}"
            spoken_response = f"Error: {str(e)}"
            import traceback
            eval_logger.log(traceback.format_exc())

        finally:
            await flow.cleanup()
            if stt_instance and hasattr(stt_instance, 'aclose'): await stt_instance.aclose()
            if llm_instance and hasattr(llm_instance, 'aclose'): await llm_instance.aclose()
            if tts_instance and hasattr(tts_instance, 'aclose'): await tts_instance.aclose()

        # 3. Judge
        judge_result = {"status": "skipped", "reason": "No judge configured"}
        if turn.judge:
            provider, config = turn.judge
            from .components.llm import LLMEvalConfig
            judge_llm_config = LLMEvalConfig(model=config.model, api_key=os.getenv("GOOGLE_API_KEY_LLM"))
            judge_llm = create_llm(provider, judge_llm_config)
            
            # Construct Judge History from agent's chat context
            history_str = ""
            if agent.chat_context and self.include_context:
                for msg in agent.chat_context._items:
                    role = "User" if msg.role == ChatRole.USER else "Agent" if msg.role == ChatRole.ASSISTANT else "System"
                    history_str += f"{role}: {msg.content}\n"

            current_user_input = "Audio/Unknown"
            
            if turn.llm:
                _, llm_cfg = turn.llm
                if llm_cfg.get('mock_input'):
                    current_user_input = llm_cfg.get('mock_input')
                elif turn.stt:
                    stt_transcripts = computed_metrics.get('collected_transcripts', [])
                    if stt_transcripts:
                        current_user_input = " ".join(stt_transcripts)
            elif turn.stt:
                stt_transcripts = computed_metrics.get('collected_transcripts', [])
                if stt_transcripts:
                    current_user_input = " ".join(stt_transcripts)

            prompt = f"{config.prompt}\n\nFull Conversation History:\n{history_str}\n\nFocus more on the following Current Turn Details:\nUser Input: {current_user_input}\nAgent LLM Response (Actual Output): {actual_llm_response}\nAgent Spoken Output (TTS): {spoken_response}\n\n. Evaluate based on: {config.checks}. It should follow the output format. That will have all keys from checks and one conclusion key that will be the summary of it "
            
            judge_response_text = ""
            try:
                judge_context = ChatContext()
                judge_context.add_message(role=ChatRole.USER, content=prompt)
                
                async for chunk in judge_llm.chat(judge_context):
                    if chunk.content:
                        judge_response_text += chunk.content
                    if chunk.metadata:
                        if not judge_response_text:
                            judge_response_text = json.dumps(chunk.metadata, indent=2)
                
                judge_result = {"passed": True, "evaluation": judge_response_text}
                try:
                    cleaned_text = judge_response_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()
                    
                    eval_data = json.loads(cleaned_text)
                    if isinstance(eval_data, dict):
                        score = eval_data.get("score")
                        if score is not None:
                            try:
                                if float(score) >= 3:
                                    judge_result["passed"] = True
                                else:
                                    judge_result["passed"] = False
                            except:
                                pass
                        
                        if "passed" in eval_data:
                            judge_result["passed"] = bool(eval_data["passed"])
                        elif "conclusion" in eval_data:
                            pass
                except:
                    if "score: 1" in judge_response_text or "score: 2" in judge_response_text:
                        judge_result["passed"] = False
                    elif "score:" in judge_response_text:
                        judge_result["passed"] = True
            except Exception as e:
                eval_logger.log(f"Judge evaluation failed: {e}")
                judge_result = {"passed": False, "reason": str(e)}
            if judge_llm and hasattr(judge_llm, 'aclose'):
                await judge_llm.aclose()

        if turn.tts and flow.audio_track:
            tts_filepath = await self.tts_audio_file(flow.audio_track,output_dir)
            computed_metrics["tts_audio_file"] = tts_filepath

        return {
            "turn_index": index,
            "turn_id": turn.id,
            "metrics": computed_metrics,
            "response_text": actual_llm_response,
            "judge": judge_result
        }

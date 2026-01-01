import asyncio
from dataclasses import asdict
from videosdk.agents import SpeechEventType, ConversationFlow
from videosdk.agents.metrics import cascading_metrics_collector
from .eval_logger import eval_logger


class EvalConversationFlow(ConversationFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_preemptive_generation = False
        from .audio_track import MockAudioTrack
        self.audio_track = MockAudioTrack()
        if self.tts:
            self.tts.audio_track = self.audio_track
            self.tts.loop = asyncio.get_event_loop()
                
        self.enable_stt_processing = True
        self.enable_llm_processing = True
        self.enable_tts_processing = True       
        self.stt_done_event = asyncio.Event()
        self.llm_done_event = asyncio.Event()
        self.generation_done_event = asyncio.Event()
        self.allowed_mock_input = None
        self.tts_mock_input = None
        self.collected_transcripts = []
        
        # Disable interruptions for evaluation to avoid Sports.wav cutting off responses
        self.interrupt_mode = "NONE"
        self.interrupt_min_words = 1000
        
    async def on_stt_transcript(self, stt_response) -> None:
        if not self.enable_stt_processing:
            return
        if stt_response.event_type == SpeechEventType.FINAL:
            eval_logger.component_end("STT")
            eval_logger.log(f"FINAL STT transcript: '{stt_response.data.text}'")
        
        await super().on_stt_transcript(stt_response)

    async def process_text_input(self, text: str) -> None:
        await self._process_final_transcript(text)

    async def _process_final_transcript(self, user_text: str) -> None:
        if not self.agent:
            return

        if user_text != self.allowed_mock_input:
            self.collected_transcripts.append(user_text)

        self.stt_done_event.set()
        await super()._process_final_transcript(user_text)

    async def cleanup(self) -> None:
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        
        if hasattr(self, '_interruption_check_task') and self._interruption_check_task and not self._interruption_check_task.done():
            self._interruption_check_task.cancel()
            

    async def _generate_and_synthesize_response(self, user_text: str, handle, wait_for_authorization: bool = False) -> None:
        is_allowed_mock = False
        if self.allowed_mock_input and user_text == self.allowed_mock_input:
            is_allowed_mock = True
        self.generation_done_event.clear()

        if self.enable_llm_processing or is_allowed_mock:
            if cascading_metrics_collector.data.current_turn:
                cascading_metrics_collector.data.current_turn.llm_input = user_text
            eval_logger.component_start("LLM")
            await super()._generate_and_synthesize_response(user_text, handle, wait_for_authorization)
        else:
            eval_logger.debug(f"DEBUG: Suppressing LLM generation for: '{user_text}'")
            if not handle.done():
                handle._mark_done()
        
        if not self.allowed_mock_input or is_allowed_mock or self.enable_llm_processing:
             self.generation_done_event.set()
    async def _synthesize_with_tts(self, response_iterator) -> None:
        self.llm_done_event.set()
        eval_logger.component_end("LLM")

        if self.enable_tts_processing:
            if self.tts_mock_input:
                 if hasattr(response_iterator, '__aiter__'):
                     async for _ in response_iterator: pass
                 elif hasattr(response_iterator, '__iter__'):
                      for _ in response_iterator: pass
                 
                 async def mock_iterator():
                     yield self.tts_mock_input
                 
                 await super()._synthesize_with_tts(mock_iterator())
            else:
                await super()._synthesize_with_tts(response_iterator)
            eval_logger.component_end("TTS")
        else:
            if hasattr(response_iterator, '__aiter__'):
                async for _ in response_iterator:
                    pass
            elif hasattr(response_iterator, '__iter__'):
                 for _ in response_iterator:
                    pass

    @property
    def metrics(self) -> dict:
        """Get metrics for the current or last completed interaction"""
        
        data = {}
        if cascading_metrics_collector.data.current_turn:
             data = asdict(cascading_metrics_collector.data.current_turn)
        elif cascading_metrics_collector.data.turns:
            data = asdict(cascading_metrics_collector.data.turns[-1])
            
        data['collected_transcripts'] = self.collected_transcripts
        return data

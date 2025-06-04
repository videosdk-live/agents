from __future__ import annotations

import asyncio
import numpy as np
from typing import Any, AsyncIterator, Literal
import time

from .onnx_runtime import OnnxModel, new_inference_session, SUPPORTED_SAMPLE_RATES
from videosdk.agents.vad import VAD as BaseVAD, VADResponse, VADEventType, VADData

class VAD(BaseVAD):
    """Silero Voice Activity Detection implementation using ONNX runtime"""
    
    def __init__(
        self,
        sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.45,
        max_buffered_speech: float = 60.0,
        force_cpu: bool = True,
        window_size_ms: int = 96
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Sample rate {sample_rate} not supported. Must be one of {SUPPORTED_SAMPLE_RATES}")
            
        super().__init__(
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )
        
        self._session = new_inference_session(force_cpu)
        self._model = OnnxModel(onnx_session=self._session, sample_rate=sample_rate)
        
        self._speech_start = 0.0
        self._silence_start = 0.0
        self._is_speech = False
        self._exp_filter = 0.0
        
        self._window_size_samples = int(sample_rate * window_size_ms / 1000)
        self._max_buffer_samples = int(max_buffered_speech * sample_rate)
        self._speech_buffer = np.zeros(self._max_buffer_samples, dtype=np.float32)
        self._buffer_index = 0
        self._buffer_full = False
        
    async def process_audio(
        self, 
        audio_frames: bytes,
        **kwargs: Any
    ) -> AsyncIterator[VADResponse]:
        window_size = self._model.window_size_samples
        frame_duration = window_size / self._sample_rate
        
        samples = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        
        pre_emphasis = 0.97
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
        
        timestamp = 0.0
        speech_duration = 0.0
        silence_duration = 0.0
        
        i = 0
        buffer = np.zeros(window_size, dtype=np.float32)
        buffer_idx = 0
        
        while i < len(samples):
            remaining = window_size - buffer_idx
            chunk_size = min(remaining, len(samples) - i)
            buffer[buffer_idx:buffer_idx + chunk_size] = samples[i:i + chunk_size]
            buffer_idx += chunk_size
            i += chunk_size
            
            if buffer_idx == window_size:
                start_time = time.perf_counter()
                
                prob = self._model(buffer)
                
                alpha = 0.15
                self._exp_filter = alpha * prob + (1 - alpha) * self._exp_filter
                
                if self._buffer_index + window_size <= self._max_buffer_samples:
                    self._speech_buffer[self._buffer_index:self._buffer_index + window_size] = buffer
                    self._buffer_index += window_size
                else:
                    if not self._buffer_full:
                        self._buffer_full = True
                    
                    shift_size = window_size
                    self._speech_buffer[:-shift_size] = self._speech_buffer[shift_size:]
                    self._speech_buffer[-shift_size:] = buffer
                
                is_speech = self._exp_filter > self._threshold
                
                if is_speech:
                    if not self._is_speech:
                        if speech_duration >= self._min_speech_duration:
                            self._is_speech = True
                            yield VADResponse(
                                event_type=VADEventType.START,
                                data=VADData(
                                    is_speech=True,
                                    confidence=self._exp_filter,
                                    timestamp=timestamp,
                                    speech_duration=speech_duration,
                                    silence_duration=silence_duration
                                )
                            )
                    speech_duration += frame_duration
                    silence_duration = 0
                else:
                    if self._is_speech:
                        if silence_duration >= self._min_silence_duration:
                            self._is_speech = False
                            yield VADResponse(
                                event_type=VADEventType.END,
                                data=VADData(
                                    is_speech=False,
                                    confidence=self._exp_filter,
                                    timestamp=timestamp,
                                    speech_duration=speech_duration,
                                    silence_duration=silence_duration
                                )
                            )
                            speech_duration = 0
                            
                            self._buffer_index = 0
                            self._buffer_full = False
                    silence_duration += frame_duration
                
                yield VADResponse(
                    event_type=VADEventType.SPEECH if is_speech else VADEventType.SILENCE,
                    data=VADData(
                        is_speech=is_speech,
                        confidence=self._exp_filter,
                        timestamp=timestamp,
                        speech_duration=speech_duration,
                        silence_duration=silence_duration
                    )
                )
                
                inference_duration = time.perf_counter() - start_time
                if inference_duration > 0.2:
                    print(f"Warning: Slow inference detected ({inference_duration:.3f}s)")
                
                timestamp += frame_duration
                buffer_idx = 0
                buffer.fill(0)
        
    async def aclose(self) -> None:
        """Cleanup resources"""
        self._speech_buffer = None
        self._buffer_index = 0
        self._buffer_full = False

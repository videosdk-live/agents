from __future__ import annotations

import numpy as np
from typing import Any, Literal
import time
from scipy import signal
import logging
from .onnx_runtime import VadModelWrapper, SAMPLE_RATES
from videosdk.agents.vad import VAD as BaseVAD, VADResponse, VADEventType, VADData

logger = logging.getLogger(__name__)
class SileroVAD(BaseVAD):
    """Silero Voice Activity Detection implementation using ONNX runtime"""
    
    def __init__(
        self,
        input_sample_rate: int = 48000,
        model_sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.3,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.75,
        max_buffered_speech: float = 60.0,
        force_cpu: bool = True,
        prefix_padding_duration: float = 0.3,
    ) -> None:
        
        if model_sample_rate not in SAMPLE_RATES:
            self.emit("error", f"Invalid model sample rate {model_sample_rate}: must be one of {SAMPLE_RATES}")
            raise ValueError(f"Model sample rate {model_sample_rate} not supported. Must be one of {SAMPLE_RATES}")
            
        super().__init__(
            sample_rate=model_sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )
        
        self._input_sample_rate = input_sample_rate
        self._model_sample_rate = model_sample_rate
        self._needs_resampling = input_sample_rate != model_sample_rate
        self._prefix_padding_duration = prefix_padding_duration
        
        try:
            self._session = VadModelWrapper.create_inference_session(force_cpu)
            self._model = VadModelWrapper(session=self._session, rate=model_sample_rate)
        except Exception as e:
            self.emit("error", f"Failed to initialize VAD model: {str(e)}")
            raise
        
        self._exp_filter = 0.0
        
        self._speech_threshold_duration = 0.0
        self._silence_threshold_duration = 0.0
        
        self._pub_speaking = False
        self._pub_speech_duration = 0.0
        self._pub_silence_duration = 0.0
        self._pub_timestamp = 0.0
        
        self._prefix_padding_samples = int(prefix_padding_duration * input_sample_rate)
        self._speech_buffer = np.empty(
            int(max_buffered_speech * input_sample_rate) + self._prefix_padding_samples,
            dtype=np.int16
        )
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        
        self._input_copy_remaining_fract = 0.0
        
        self._input_accumulator = np.array([], dtype=np.int16)
        self._inference_accumulator = np.array([], dtype=np.float32)
        
        self._frame_count = 0
        self._inference_count = 0
        
        self._consecutive_low_confidence_count = 0
        self._error_emission_threshold = 10
        
    async def process_audio(
        self, 
        audio_frames: bytes,
        **kwargs: Any
    ) -> None:
        try:
            input_frame_data = np.frombuffer(audio_frames, dtype=np.int16)
            self._frame_count += 1
            
            self._input_accumulator = np.concatenate([self._input_accumulator, input_frame_data])
            
            if self._needs_resampling:
                input_float = input_frame_data.astype(np.float32) / 32768.0
                target_length = int(len(input_float) * self._model_sample_rate / self._input_sample_rate)
                
                if target_length > 0:
                    resampled_float = signal.resample(input_float, target_length)
                    self._inference_accumulator = np.concatenate([
                        self._inference_accumulator, 
                        resampled_float.astype(np.float32)
                    ])
            else:
                input_float = input_frame_data.astype(np.float32) / 32768.0
                self._inference_accumulator = np.concatenate([self._inference_accumulator, input_float])
            
            while len(self._inference_accumulator) >= self._model.frame_size:
                inference_window = self._inference_accumulator[:self._model.frame_size]
                self._inference_count += 1
                
                start_time = time.perf_counter()
                
                try:
                    raw_prob = self._model.process(inference_window)
                    if raw_prob < self._threshold:
                        self._consecutive_low_confidence_count += 1
                        if self._consecutive_low_confidence_count >= self._error_emission_threshold:
                            self._consecutive_low_confidence_count = 0
                    else:
                        self._consecutive_low_confidence_count = 0
                except Exception as e:
                    self.emit("error", f"Error processing audio frame in VAD model: {str(e)}")
                    raw_prob = 0.0  # Default to no speech detection on error
                
                alpha = 0.6
                self._exp_filter = alpha * raw_prob + (1 - alpha) * self._exp_filter
                
                window_duration = self._model.frame_size / self._model_sample_rate
                self._pub_timestamp += window_duration
                
                resampling_ratio = self._input_sample_rate / self._model_sample_rate
                input_samples_for_window = int(self._model.frame_size * resampling_ratio + self._input_copy_remaining_fract)
                self._input_copy_remaining_fract = (self._model.frame_size * resampling_ratio + self._input_copy_remaining_fract) - input_samples_for_window
                
                if len(self._input_accumulator) >= input_samples_for_window:
                    input_window = self._input_accumulator[:input_samples_for_window]
                    
                    available_space = len(self._speech_buffer) - self._speech_buffer_index
                    to_copy_buffer = min(len(input_window), available_space)
                    
                    if to_copy_buffer > 0:
                        self._speech_buffer[self._speech_buffer_index:self._speech_buffer_index + to_copy_buffer] = input_window[:to_copy_buffer]
                        self._speech_buffer_index += to_copy_buffer
                    elif not self._speech_buffer_max_reached:
                        self._speech_buffer_max_reached = True
                        logger.warning(f"Warning: max_buffered_speech reached")
                    
                    self._input_accumulator = self._input_accumulator[input_samples_for_window:]
                
                if self._exp_filter >= self._threshold:
                    self._speech_threshold_duration += window_duration
                    self._silence_threshold_duration = 0.0
                else:
                    self._silence_threshold_duration += window_duration
                    self._speech_threshold_duration = 0.0
                
                if self._pub_speaking:
                    self._pub_speech_duration += window_duration
                else:
                    self._pub_silence_duration += window_duration
                
                if self._exp_filter >= self._threshold:
                    if not self._pub_speaking:
                        if self._speech_threshold_duration >= self._min_speech_duration:
                            self._pub_speaking = True
                            self._pub_silence_duration = 0.0
                            self._pub_speech_duration = self._speech_threshold_duration
                            
                            response = VADResponse(
                                event_type=VADEventType.START_OF_SPEECH,
                                data=VADData(
                                    is_speech=True,
                                    confidence=self._exp_filter,
                                    timestamp=self._pub_timestamp,
                                    speech_duration=self._pub_speech_duration,
                                    silence_duration=0.0
                                )
                            )
                            if self._vad_callback:
                                await self._vad_callback(response)
                else:
                    if not self._pub_speaking:
                        self._reset_speech_buffer()
                    
                    if self._pub_speaking and self._silence_threshold_duration >= self._min_silence_duration:
                        self._pub_speaking = False
                        self._pub_speech_duration = 0.0
                        self._pub_silence_duration = self._silence_threshold_duration
                        
                        response = VADResponse(
                            event_type=VADEventType.END_OF_SPEECH,
                            data=VADData(
                                is_speech=False,
                                confidence=self._exp_filter,
                                timestamp=self._pub_timestamp,
                                speech_duration=0.0,
                                silence_duration=self._pub_silence_duration
                            )
                        )
                        if self._vad_callback:
                            await self._vad_callback(response)
                        
                        self._reset_speech_buffer()
                
                inference_duration = time.perf_counter() - start_time
                if inference_duration > 0.1:
                    logger.warning(f"Warning: Slow inference detected ({inference_duration:.3f}s)")
                
                self._inference_accumulator = self._inference_accumulator[self._model.frame_size:]
        except Exception as e:
            self.emit("error", f"Unexpected error in VAD audio processing: {str(e)}")
                
            
    def _reset_speech_buffer(self) -> None:
        if self._speech_buffer_index <= self._prefix_padding_samples:
            return
        
        padding_data = self._speech_buffer[
            self._speech_buffer_index - self._prefix_padding_samples:self._speech_buffer_index
        ]
        
        self._speech_buffer_max_reached = False
        self._speech_buffer[:self._prefix_padding_samples] = padding_data
        self._speech_buffer_index = self._prefix_padding_samples
        
    async def aclose(self) -> None:
        """Cleanup resources"""
        try:
            self._speech_buffer = None
            self._speech_buffer_index = 0
            self._speech_buffer_max_reached = False
            self._input_accumulator = np.array([], dtype=np.int16)
            self._inference_accumulator = np.array([], dtype=np.float32)
        except Exception as e:
            self.emit("error", f"Error during VAD cleanup: {str(e)}")

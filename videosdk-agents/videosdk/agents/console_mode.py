from __future__ import annotations
import asyncio
import queue
from typing import Any, Optional, Callable

try:
    import aec_audio_processing  as apm
except ImportError:
    raise ImportError("aec-audio-processing is required for echo cancellation. Install with: pip install aec-audio-processing")

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False

import numpy as np
from .room.audio_stream import TeeCustomAudioStreamTrack
import logging

logger = logging.getLogger(__name__)

class LocalAudioPlayer:
    def __init__(self, samplerate: int = 24000, channels: int = 1, output_device: Optional[int] = None,
                 apm_processor: Optional[apm.AudioProcessor] = None,
                 reverse_stream_queue: Optional[queue.Queue] = None):
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is required for voice console. Install with: pip install sounddevice numpy")
        import threading
        self.samplerate = samplerate
        self.channels = channels
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.apm = apm_processor
        self.reverse_queue = reverse_stream_queue
        self.apm_frame_size = 480

        def _callback(outdata, frames, time_info, status):
            try:
                bytes_needed = frames * self.channels * 2
                with self._lock:
                    if len(self._buffer) >= bytes_needed:
                        chunk = self._buffer[:bytes_needed]
                        del self._buffer[:bytes_needed]
                    else:
                        chunk = bytes(self._buffer)
                        self._buffer.clear()

                if chunk:
                    arr = np.frombuffer(chunk, dtype=np.int16)
                    if arr.size % self.channels != 0:
                        trim = arr.size - (arr.size // self.channels) * self.channels
                        if trim > 0: arr = arr[:-trim]
                    if arr.size == 0:
                        outdata.fill(0)
                        return
                    arr = arr.reshape(-1, self.channels)
                    if arr.shape[0] < frames:
                        padded = np.zeros((frames, self.channels), dtype=np.int16)
                        padded[:arr.shape[0], :] = arr
                        outdata[:] = padded
                    else:
                        outdata[:] = arr[:frames, :]
                else:
                    outdata.fill(0)
            except Exception:
                outdata.fill(0)

        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='int16',
            blocksize=int(0.02 * self.samplerate),
            callback=_callback,
            device=output_device or sd.default.device[1],
        )
        self.stream.start()

    async def handle_audio_input(self, audio_bytes: bytes):
        if self.reverse_queue:
            try:
                samples_24k = np.frombuffer(audio_bytes, dtype=np.int16)

                samples_48k = np.repeat(samples_24k, 2)
                
                for i in range(0, len(samples_48k), self.apm_frame_size):
                    chunk = samples_48k[i:i + self.apm_frame_size]
                    if len(chunk) < self.apm_frame_size:
                        padded_chunk = np.zeros(self.apm_frame_size, dtype=np.int16)
                        padded_chunk[:len(chunk)] = chunk
                        self.reverse_queue.put(padded_chunk)
                    else:
                        self.reverse_queue.put(chunk)
            except Exception as e:
                logger.error(f"[AEC Error] Failed to queue reverse stream audio: {e}")

        with self._lock:
            self._buffer.extend(audio_bytes)

    def close(self):
        try:
            self.stream.stop()
        finally:
            self.stream.close()


class MicrophoneStreamer:
    def __init__(self, samplerate: int = 48000, channels: int = 1, block_ms: int = 20, input_device: Optional[int] = None,
                 meter: bool = True, idle_dbfs: float = -42.0,
                 apm_processor: Optional[apm.AudioProcessor] = None,
                 reverse_stream_queue: Optional[queue.Queue] = None):

        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is required for voice console. Install with: pip install sounddevice numpy and if your linux the make sure you do 'sudo apt-get install libasound2-dev'")
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = int(self.samplerate * block_ms / 1000) 
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._in_stream = None
        self._running = False
        self._micro_db: float = -60.0
        self._meter_task: asyncio.Task | None = None
        self._input_device_name: str = "Mic"
        self._meter_visible: bool = False
        self._last_bar: int = -1
        self._last_db: float = -60.0
        self._meter = meter
        self._idle_threshold = idle_dbfs
        self._input_device = input_device

        self.apm = apm_processor
        self.reverse_queue = reverse_stream_queue
        self.frame_size = self.blocksize // 2 

    def _callback(self, indata, frames, time_info, status):
        try:
            if self.apm and self.reverse_queue:
                mic_chunk_1 = indata[0:self.frame_size]
                mic_chunk_2 = indata[self.frame_size:self.blocksize]
                
                cleaned_bytes = bytearray()

                try:
                    rev_frame = self.reverse_queue.get_nowait()
                    self.apm.process_reverse_stream(rev_frame.tobytes())
                    cleaned_chunk_1 = self.apm.process_stream(mic_chunk_1.tobytes())
                    cleaned_bytes.extend(cleaned_chunk_1)
                except queue.Empty:
                    cleaned_bytes.extend(self.apm.process_stream(mic_chunk_1.tobytes()))

                try:
                    rev_frame = self.reverse_queue.get_nowait()
                    self.apm.process_reverse_stream(rev_frame.tobytes())
                    cleaned_chunk_2 = self.apm.process_stream(mic_chunk_2.tobytes())
                    cleaned_bytes.extend(cleaned_chunk_2)
                except queue.Empty:
                    cleaned_bytes.extend(self.apm.process_stream(mic_chunk_2.tobytes()))
                
                data_bytes = bytes(cleaned_bytes)

            else:
                data_bytes = bytes(indata)
            
            self.queue.put_nowait(data_bytes)
            
            samples = np.frombuffer(indata, dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2) + 1e-12))
                dbfs = 20.0 * np.log10(rms / 32768.0 + 1e-12)
                if dbfs < -120.0: dbfs = -120.0
                if dbfs > 0.0: dbfs = 0.0
                self._micro_db = dbfs
        except asyncio.QueueFull:
            pass
        except Exception as e:
            logger.error(f"Error in mic callback: {e}")


    def start(self):
        self._running = True
        self._in_stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.blocksize,
            callback=self._callback,
            device=self._input_device or sd.default.device[0],
        )
        self._in_stream.start()
        try:
            dev_index = self._in_stream.device
            dev_info = sd.query_devices(dev_index)
            self._input_device_name = dev_info.get('name', 'Mic')
        except Exception:
            self._input_device_name = 'Mic'
        if self._meter:
            self._meter_task = asyncio.create_task(self._meter_loop())

    async def consume_to(self, on_bytes: Callable[[bytes], Any]):
        while self._running:
            data = await self.queue.get()
            try:
                await on_bytes(data)
            except Exception:
                pass

    async def stop(self):
        self._running = False
        if self._meter_task and not self._meter_task.done():
            self._meter_task.cancel()
            try:
                await self._meter_task
            except asyncio.CancelledError:
                pass
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None

    async def _meter_loop(self):
        MAX_AUDIO_BAR = 40
        def _esc(code: int) -> str: return f"\x1b[{code}m"
        def _normalize_db(db: float, db_min: float = -60.0, db_max: float = 0.0) -> float:
            if db <= db_min: return 0.0
            if db >= db_max: return 1.0
            return (db - db_min) / (db_max - db_min)
        try:
            while self._running:
                amplitude = _normalize_db(self._micro_db)
                idle = self._micro_db <= self._idle_threshold
                if idle:
                    if self._meter_visible:
                        import sys as _sys
                        _sys.stdout.write("\r\x1b[2K")
                        _sys.stdout.flush()
                        self._meter_visible = False
                    await asyncio.sleep(0.3)
                    continue
                nb_bar = round(amplitude * MAX_AUDIO_BAR)
                if nb_bar == self._last_bar and abs(self._micro_db - self._last_db) < 1.5:
                    await asyncio.sleep(0.12)
                    continue
                color_code = 31 if amplitude > 0.75 else 33 if amplitude > 0.5 else 32
                bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)
                import sys as _sys
                _sys.stdout.write("\r\x1b[2K")
                _sys.stdout.write(f"[Audio] {self._input_device_name[-20:]} [{self._micro_db:6.2f} dBFS] {_esc(color_code)}[{bar}]{_esc(0)}")
                _sys.stdout.flush()
                self._meter_visible = True
                self._last_bar = nb_bar
                self._last_db = self._micro_db
                await asyncio.sleep(0.12)
        finally:
            try:
                import sys as _sys
                _sys.stdout.write("\r\x1b[2K")
                _sys.stdout.flush()
            except Exception: pass


class ConsoleMode:
    def __init__(self, *, audio_track: TeeCustomAudioStreamTrack, loop: asyncio.AbstractEventLoop) -> None:
        self.audio_track = audio_track
        self.agent_audio_track = None
        self.loop = loop
        self.vision = False
        self._pubsub_subs: dict[str, list[Callable[[Any], None]]] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging handlers to display metrics logs in console with colored log levels and shortened names."""
        logger_cascading = logging.getLogger('videosdk.agents.metrics.cascading_metrics_collector')
        logger_realtime = logging.getLogger('videosdk.agents.metrics.realtime_metrics_collector')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                message = record.getMessage()
                asctime = self.formatTime(record, self.datefmt)

                if levelname == 'INFO':
                    levelname = f'\033[32m{levelname}\033[0m' 
                    message = f'\033[36m{message}\033[0m'      
                elif levelname == 'ERROR':
                    levelname = f'\033[31m{levelname}\033[0m'  
                    message = f'\033[91m{message}\033[0m'     
                elif levelname == 'WARNING':
                    levelname = f'\033[33m{levelname}\033[0m'  
                    message = f'\033[93m{message}\033[0m'      

                asctime = f'\033[90m{asctime}\033[0m'
                return f'{asctime} - {levelname} - {message}'
        
        formatter = ColoredFormatter()
        console_handler.setFormatter(formatter)
        
        logger_cascading.addHandler(console_handler)
        logger_realtime.addHandler(console_handler)
        
        logger_cascading.setLevel(logging.INFO)
        logger_realtime.setLevel(logging.INFO)

        logger_cascading.propagate = False
        logger_realtime.propagate = False

    def init_meeting(self) -> None: return
    async def join(self) -> None: return
    def leave(self) -> None: return
    async def cleanup(self) -> None: return
    async def wait_for_participant(self, participant_id: str | None = None) -> str: return participant_id or "console-user"
    async def subscribe_to_pubsub(self, pubsub_config: Any):
        topic = getattr(pubsub_config, 'topic', None)
        cb = getattr(pubsub_config, 'cb', None)
        if not topic or not cb: return []
        self._pubsub_subs.setdefault(topic, []).append(cb)
        return []
    async def publish_to_pubsub(self, pubsub_config: Any):
        topic = getattr(pubsub_config, 'topic', None)
        message = getattr(pubsub_config, 'message', None)
        if not topic: return
        for cb in self._pubsub_subs.get(topic, []):
            try: cb(message)
            except Exception: pass


async def setup_console_voice_for_ctx(
    ctx: Any,
    *,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    meter: bool = True,
    idle_dbfs: float = -42.0,
) -> Callable[[], Any]:
    """
    Sets up a console voice environment, automatically detecting if the STT
    plugin requires stereo audio and adapting the stream accordingly.
    """
    
    print(f"\033[90m{'='*100}\033[0m")
    print(f"\033[96m                             Videosdk's AI Agent Console Mode\033[0m")
    print(f"\033[90m{'='*100}\033[0m")

    loop = ctx._loop
    if ctx._pipeline is None:
        raise RuntimeError("Pipeline must be constructed before ctx.connect() in console mode")

    SAMPLE_RATE = 48000
    NUM_CHANNELS = 1 

    try:
        processor = apm.AudioProcessor(
            enable_aec=True, enable_ns=True, ns_level=1,
            enable_agc=True, agc_mode=2, enable_vad=False
        )
    except TypeError:
        processor = apm.AudioProcessor(enable_aec=True, enable_ns=True, enable_agc=True)

    processor.set_stream_format(SAMPLE_RATE, NUM_CHANNELS)
    processor.set_reverse_stream_format(SAMPLE_RATE, NUM_CHANNELS)
    processor.set_stream_delay(30)
    system_audio_queue = queue.Queue()

    speaker = LocalAudioPlayer(samplerate=24000, channels=1, output_device=output_device,
                             apm_processor=processor, reverse_stream_queue=system_audio_queue)

    audio_track = TeeCustomAudioStreamTrack(loop=loop, sinks=[speaker])

    ctx.room = ConsoleMode(audio_track=audio_track, loop=loop)

    if hasattr(ctx._pipeline, '_set_loop_and_audio_track'):
        ctx._pipeline._set_loop_and_audio_track(loop, audio_track)

    mic = MicrophoneStreamer(
        samplerate=SAMPLE_RATE,
        channels=NUM_CHANNELS,
        block_ms=20,
        input_device=input_device,
        meter=meter,
        idle_dbfs=idle_dbfs,
        apm_processor=processor,
        reverse_stream_queue=system_audio_queue
    )
    mic.start()

    logger.info(f"Using microphone: {mic._input_device_name}")

    stt_agent = getattr(ctx._pipeline, 'stt', None)
    needs_stereo = False

    if stt_agent and type(stt_agent).__name__ == 'GoogleSTT' or type(stt_agent).__name__ == 'DeepgramSTT' or type(stt_agent).__name__ == 'SarvamAISTT':
        needs_stereo = True
    
    if needs_stereo:
        async def mono_to_stereo_adapter(mono_bytes: bytes):
            """
            Receives 1-channel audio, converts to 2-channel, and forwards to the pipeline.
            """
            mono_samples = np.frombuffer(mono_bytes, dtype=np.int16)
            stereo_samples = np.repeat(mono_samples, 2)
            stereo_bytes = stereo_samples.tobytes()
            await ctx._pipeline.on_audio_delta(stereo_bytes)

        consumer_task = asyncio.create_task(mic.consume_to(mono_to_stereo_adapter))
    else:
        consumer_task = asyncio.create_task(mic.consume_to(ctx._pipeline.on_audio_delta))


    async def _cleanup() -> None:
        try:
            await mic.stop()
        finally:
            speaker.close()
            if not consumer_task.done():
                consumer_task.cancel()
                import contextlib
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer_task

    return _cleanup
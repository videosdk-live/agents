from __future__ import annotations
import asyncio
from typing import Any, Optional, Callable
try:
    import aec_audio_processing as apm
    _APM_AVAILABLE = True
except ImportError:
    apm = None
    _APM_AVAILABLE = False

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False


def _require_console_deps() -> None:
    """Raise a single, actionable error if the voice-console extras are missing."""
    missing = []
    if not _APM_AVAILABLE:
        missing.append("aec-audio-processing")
    if not _SD_AVAILABLE:
        missing.append("sounddevice")
    if missing:
        raise ImportError(
            f"Voice console mode requires {' and '.join(missing)}, which are optional. "
            f"Install the console extra with: pip install \"videosdk-agents[console]\" "
            f"(on Linux also run: sudo apt-get install libasound2-dev)"
        )

import numpy as np
from fractions import Fraction
from av import AudioResampler
from .room.output_stream import CustomAudioStreamTrack, AUDIO_PTIME
import logging

logger = logging.getLogger(__name__)


def _setup_metrics_console_logging():
    """Attach a colored console handler to the metrics logger so the latency/
    metrics logs are visible in the terminal during a console session."""
    logger_metrics = logging.getLogger('videosdk.agents.metrics.metrics_collector')

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

    console_handler.setFormatter(ColoredFormatter())

    logger_metrics.addHandler(console_handler)
    logger_metrics.setLevel(logging.INFO)
    logger_metrics.propagate = False


class DuplexAudioIO:
    """Full-duplex sounddevice stream: plays the agent's audio and captures the
    mic in one callback, running echo cancellation on the aligned render/capture
    pair. Cleaned mic audio goes to ``consume_to`` (for publishing); agent audio
    is queued via ``enqueue_playback``.
    """

    def __init__(self, *, samplerate: int = 48000, channels: int = 1, block_ms: int = 10,
                 input_device: Optional[int] = None, output_device: Optional[int] = None,
                 apm_processor: Optional[apm.AudioProcessor] = None,
                 meter: bool = True, idle_dbfs: float = -42.0):
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is required for voice console. Install with: pip install sounddevice numpy "
                "(on Linux also run: sudo apt-get install libasound2-dev)"
            )
        import threading
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = int(samplerate * block_ms / 1000)
        self.apm = apm_processor

        self._play_buf = bytearray()
        self._lock = threading.Lock()
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        self._input_device = input_device
        self._output_device = output_device
        self._stream = None
        self._running = False

        self._meter = meter
        self._idle_threshold = idle_dbfs
        self._micro_db: float = -60.0
        self._meter_task: asyncio.Task | None = None
        self._meter_visible: bool = False
        self._last_bar: int = -1
        self._last_db: float = -60.0
        self._input_device_name: str = "Mic"

    def enqueue_playback(self, audio_bytes: bytes) -> None:
        """Queue agent audio (mono int16 at this stream's samplerate) for playback."""
        with self._lock:
            self._play_buf.extend(audio_bytes)

    def _callback(self, indata, outdata, frames, time_info, status):
        try:
            bytes_needed = frames * self.channels * 2
            with self._lock:
                if len(self._play_buf) >= bytes_needed:
                    chunk = self._play_buf[:bytes_needed]
                    del self._play_buf[:bytes_needed]
                else:
                    chunk = bytes(self._play_buf)
                    self._play_buf.clear()

            if chunk:
                arr = np.frombuffer(chunk, dtype=np.int16)
                need = frames * self.channels
                if arr.size < need:
                    padded = np.zeros(need, dtype=np.int16)
                    padded[:arr.size] = arr
                    arr = padded
                outdata[:] = arr[:need].reshape(frames, self.channels)
            else:
                outdata.fill(0)

            if self.apm is not None:
                self.apm.process_reverse_stream(outdata.tobytes())
                mic_bytes = self.apm.process_stream(indata.tobytes())
            else:
                mic_bytes = bytes(indata)

            try:
                self.queue.put_nowait(bytes(mic_bytes))
            except asyncio.QueueFull:
                pass

            samples = np.frombuffer(indata, dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2) + 1e-12))
                dbfs = 20.0 * np.log10(rms / 32768.0 + 1e-12)
                self._micro_db = min(0.0, max(-120.0, dbfs))
        except Exception as e:
            logger.error(f"Error in duplex audio callback: {e}")
            try:
                outdata.fill(0)
            except Exception:
                pass

    def start(self):
        self._running = True
        in_dev = self._input_device if self._input_device is not None else sd.default.device[0]
        out_dev = self._output_device if self._output_device is not None else sd.default.device[1]
        self._stream = sd.Stream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.blocksize,
            callback=self._callback,
            device=(in_dev, out_dev),
        )
        self._stream.start()
        try:
            self._input_device_name = sd.query_devices(self._stream.device[0]).get('name', 'Mic')
        except Exception:
            self._input_device_name = 'Mic'

        if self.apm is not None:
            try:
                lat = self._stream.latency
                lat_s = (float(lat[0]) + float(lat[1])) if isinstance(lat, (tuple, list)) else float(lat)
                self.apm.set_stream_delay(max(0, int(lat_s * 1000)))
            except Exception:
                try:
                    self.apm.set_stream_delay(40)
                except Exception:
                    pass
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
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None

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


async def setup_console_room_client_for_ctx(
    ctx: Any,
    *,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    meter: bool = True,
    idle_dbfs: float = -42.0,
) -> Callable[[], Any]:
    """
    Join the agent's VideoSDK room from the terminal as a second, *real* human
    participant: publish the local microphone as a room track and play the
    agent's audio on local speakers — no browser/ sdk client or playground
    URL needed.
    """
    _require_console_deps()

    from videosdk import VideoSDK
    from .room.meeting_event_handler import MeetingHandler
    from .room.participant_event_handler import ParticipantHandler

    loop = ctx._loop
    room_id = ctx.room_options.room_id
    if not room_id:
        raise RuntimeError("console mode requires a room_id (agent must be connected first)")

    print(f"\033[90m{'='*100}\033[0m")
    print(f"\033[96m                             Videosdk's AI Agent Console Mode\033[0m")
    print(f"\033[90m{'='*100}\033[0m")

    _setup_metrics_console_logging()

    try:
        vad = getattr(ctx._pipeline, "vad", None)
        if (
            vad is not None
            and type(vad).__name__ == "SileroVAD"
            and getattr(vad, "_executor", None) is None
        ):
            import concurrent.futures
            vad._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vad-inference"
            )
            logger.info("[console] Offloaded SileroVAD inference to keep the audio loop real-time")
    except Exception as e:
        logger.warning(f"[console] could not enable VAD inference offload: {e}")

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

    audio_io = DuplexAudioIO(
        samplerate=SAMPLE_RATE, channels=NUM_CHANNELS, block_ms=10,
        input_device=input_device, output_device=output_device,
        apm_processor=processor, meter=meter, idle_dbfs=idle_dbfs,
    )

    mic_track = CustomAudioStreamTrack(loop=loop)
    mic_track.sample_rate = SAMPLE_RATE
    mic_track.channels = NUM_CHANNELS
    mic_track.samples = int(AUDIO_PTIME * SAMPLE_RATE)
    mic_track.chunk_size = mic_track.samples * mic_track.channels * mic_track.sample_width
    mic_track.time_base_fraction = Fraction(1, SAMPLE_RATE)

    agent_audio_tasks: list[asyncio.Task] = []
    subscribed_streams: set[str] = set()

    def _start_agent_audio_listener(stream: Any):
        if stream is None or getattr(stream, "kind", None) != "audio":
            return
        stream_id = getattr(stream, "id", id(stream))
        if stream_id in subscribed_streams:
            return
        subscribed_streams.add(stream_id)

        async def _agent_audio_loop():

            resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            while True:
                try:
                    frame = await stream.track.recv()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[console] agent audio recv error: {e}")
                    break
                try:
                    out = resampler.resample(frame)
                    frames = out if isinstance(out, list) else [out]
                    for rf in frames:
                        if rf is None:
                            continue
                        pcm = rf.to_ndarray().flatten().astype(np.int16).tobytes()
                        audio_io.enqueue_playback(pcm)
                except Exception as e:
                    logger.error(f"[console] error playing agent audio: {e}")

        agent_audio_tasks.append(asyncio.create_task(_agent_audio_loop()))

    def _attach_to_agent(participant: Any):
        """Subscribe to a remote agent participant's audio (existing + future streams)."""
        try:
            streams = getattr(participant, "streams", None)
            if streams:
                for stream in list(streams.values()):
                    _start_agent_audio_listener(stream)
        except Exception as e:
            logger.error(f"[console] error scanning agent streams: {e}")

        def on_stream_enabled(stream: Any):
            _start_agent_audio_listener(stream)

        def on_stream_disabled(stream: Any):
            return

        try:
            participant.add_event_listener(
                ParticipantHandler(
                    participant_id=participant.id,
                    on_stream_enabled=on_stream_enabled,
                    on_stream_disabled=on_stream_disabled,
                )
            )
        except Exception as e:
            logger.error(f"[console] error attaching to agent participant: {e}")

    meeting_config = {
        "name": "Terminal User",
        "meeting_id": room_id,
        "token": ctx.videosdk_auth,
        "mic_enabled": True,
        "webcam_enabled": False,
        "custom_microphone_audio_track": mic_track,
        "peer_type": "normal",
        "meta_data": {"terminal_client": True},
        "loop": loop,
    }
    if ctx.room_options.signaling_base_url is not None:
        meeting_config["signaling_base_url"] = ctx.room_options.signaling_base_url

    meeting = VideoSDK.init_meeting(**meeting_config)
    meeting.add_event_listener(
        MeetingHandler(
            on_meeting_joined=lambda data: logger.info(f"[console] Terminal joined room {room_id}"),
            on_meeting_left=lambda data: logger.info("[console] Terminal left room"),
            on_participant_joined=_attach_to_agent,
            on_participant_left=lambda p: None,
            on_error=lambda data: logger.error(f"[console] room error: {data}"),
            on_agent_joined=_attach_to_agent,
            on_agent_left=lambda a: None,
        )
    )

    audio_io.start()
    logger.info(f"Using microphone: {audio_io._input_device_name}")

    MAX_BUFFERED_MIC_FRAMES = 5 

    async def _feed_mic(mono_bytes: bytes):
        await mic_track.add_new_bytes(mono_bytes)
        fb = mic_track.frame_buffer
        if len(fb) > MAX_BUFFERED_MIC_FRAMES:
            del fb[:-MAX_BUFFERED_MIC_FRAMES]

    consumer_task = asyncio.create_task(audio_io.consume_to(_feed_mic))

    await meeting.async_join()

    print(f"\033[1;36mTerminal participant connected to room {room_id}\033[0m")
    print("\033[1;75mSpeak into your microphone — the agent is in the room. Press Ctrl+C to exit.\033[0m")

    async def _cleanup() -> None:
        import contextlib
        for task in agent_audio_tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        try:
            await audio_io.stop()
        finally:
            if not consumer_task.done():
                consumer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer_task

            try:
                rc = getattr(meeting, "_Meeting__room_client", None)
                if rc is not None and hasattr(rc, "async_leave"):
                    await asyncio.wait_for(rc.async_leave(), timeout=5.0)
                else:
                    meeting.leave()
                    await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"[console] error leaving terminal meeting: {e}")

    return _cleanup
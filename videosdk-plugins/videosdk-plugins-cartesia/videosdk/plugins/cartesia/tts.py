from __future__ import annotations

from typing import Any, AsyncIterator, Optional
import os
import asyncio
from cartesia import Cartesia

from videosdk.agents import TTS

CARTESIA_SAMPLE_RATE = 24000
CARTESIA_CHANNELS = 1
DEFAULT_MODEL = "sonic-2"
DEFAULT_VOICE_ID = "794f9389-aac1-45b6-b726-9d9369183238"

class CartesiaTTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE_ID,
        api_key: str | None = None,
        language: str = "en",
    ) -> None:
        super().__init__(sample_rate=CARTESIA_SAMPLE_RATE, num_channels=CARTESIA_CHANNELS)

        self.model = model
        self.voice_id = voice_id
        self.language = language
        self.audio_track = None
        self.loop = None

        api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError("Cartesia API key must be provided either through api_key parameter or CARTESIA_API_KEY environment variable")

        self.cartesia_client = Cartesia(api_key=api_key)
        self._voice_embedding = None

    async def _get_voice_embedding(self):
        """Get voice embedding for the specified voice ID"""
        if self._voice_embedding is None:
            try:
                voice = self.cartesia_client.voices.get(self.voice_id)
                if hasattr(voice, 'embedding'):
                    self._voice_embedding = voice.embedding
                else:
                    raise ValueError(f"Voice {self.voice_id} does not have an embedding")
                    
            except Exception as e:
                self.emit("error", f"Failed to get voice embedding: {str(e)}")
                return None
        return self._voice_embedding

    async def _generate_audio_chunks(self, text: str) -> list[bytes]:
        """Generate audio chunks using Cartesia TTS"""
        try:

            voice_embedding = await self._get_voice_embedding()
            if voice_embedding is None:
                return []

            ws = self.cartesia_client.tts.websocket()
            audio_chunks = []
            total_bytes = 0

            for output in ws.send(
                model_id=self.model,
                transcript=text,
                voice={
                    "mode": "embedding",
                    "embedding": voice_embedding,
                },
                stream=True,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
            ):
                if hasattr(output, 'audio') and output.audio:
                    audio_chunks.append(output.audio)
                    total_bytes += len(output.audio)
            
            return audio_chunks
            
        except Exception as e:
            self.emit("error", f"Audio generation failed: {str(e)}")
            return []

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:            
            if isinstance(text, AsyncIterator):
                full_text = ""
                async for chunk in text:
                    if full_text and not full_text.endswith(' ') and not chunk.startswith(' ') and chunk not in '.,!?;:':
                        full_text += ' '
                    full_text += chunk
            else:
                full_text = text

            if not full_text.strip():
                return

            MAX_TEXT_LENGTH = 200  
            if len(full_text) > MAX_TEXT_LENGTH:
                truncated = full_text[:MAX_TEXT_LENGTH]
                last_sentence_end = max(
                    truncated.rfind('.'),
                    truncated.rfind('!'),
                    truncated.rfind('?')
                )
                if last_sentence_end > MAX_TEXT_LENGTH - 50: 
                    full_text = truncated[:last_sentence_end + 1]
                else:
                    last_space = truncated.rfind(' ')
                    if last_space > MAX_TEXT_LENGTH - 30:
                        full_text = truncated[:last_space] + "."
                    else:
                        full_text = truncated + "."

            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            if voice_id and voice_id != self.voice_id:
                self.voice_id = voice_id
                self._voice_embedding = None

            audio_chunks = await self._generate_audio_chunks(full_text)
            if not audio_chunks:
                return

            await self._stream_to_audio_track(audio_chunks)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _stream_to_audio_track(self, audio_chunks: list[bytes]) -> None:
        """Stream audio chunks to the audio track with optimized buffer management"""
        try:
            CHUNK_SIZE = 960   
            BATCH_SIZE = 20      
            BATCH_DELAY = 0.05   
            total_sent = 0

            all_audio_data = b''.join(audio_chunks)
            total_chunks = len(all_audio_data) // CHUNK_SIZE
            if len(all_audio_data) % CHUNK_SIZE > 0:
                total_chunks += 1
                            
            chunk_buffer = []
            for i in range(0, len(all_audio_data), CHUNK_SIZE):
                chunk = all_audio_data[i:i + CHUNK_SIZE]
                
                if len(chunk) < CHUNK_SIZE:
                    chunk += b'\x00' * (CHUNK_SIZE - len(chunk))
                
                chunk_buffer.append(chunk)
                
                if len(chunk_buffer) >= BATCH_SIZE or i + CHUNK_SIZE >= len(all_audio_data):
                    for batch_chunk in chunk_buffer:
                        if self.audio_track and self.loop:
                            try:
                                await self.audio_track.add_new_bytes(batch_chunk)
                                total_sent += 1
                            except Exception as e:
                                return
                    
                    chunk_buffer = []
                    
                    if total_sent % 50 == 0 or total_sent == total_chunks:
                        progress = (total_sent / total_chunks) * 100
                    
                    if total_sent < total_chunks:
                        await asyncio.sleep(BATCH_DELAY)
                        
        except Exception as e:
            self.emit("error", f"Audio streaming failed: {str(e)}")

    async def aclose(self) -> None:
        """Cleanup resources"""
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()

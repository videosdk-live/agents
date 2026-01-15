class MockAudioTrack:
    def __init__(self):
        self.audio_buffer = bytearray()
        self._interrupted = False

    async def add_new_bytes(self, data: bytes):
        if not self._interrupted:
            self.audio_buffer.extend(data)

    def interrupt(self):
        self._interrupted = True

    def get_audio_bytes(self) -> bytes:
        return bytes(self.audio_buffer)

    def clear(self):
        self.audio_buffer = bytearray()
        self._interrupted = False
    
    def on_last_audio_byte(self, callback):
        pass

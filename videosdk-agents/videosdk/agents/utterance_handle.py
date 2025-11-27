import asyncio
from typing import Generator, Any

class UtteranceHandle:
    """Manages the lifecycle of a single agent utterance."""
    def __init__(self, utterance_id: str, interruptible: bool = True):
        self._id = utterance_id
        self._done_fut = asyncio.Future()
        self._interrupt_fut = asyncio.Future()
        self._interruptible = interruptible

    @property
    def id(self) -> str:
        return self._id

    @property
    def is_interruptible(self) -> bool: 
        return self._interruptible

    def done(self) -> bool:
        """Returns True if the utterance is complete (played out or interrupted)."""
        return self._done_fut.done()

    @property
    def interrupted(self) -> bool:
        """Returns True if the utterance was interrupted."""
        return self._interrupt_fut.done()

    def interrupt(self, *, force: bool = False) -> None: 
        """Marks the utterance as interrupted."""
        if not force and not self.is_interruptible:
            raise RuntimeError("This utterance does not allow interruptions.")

        if not self._interrupt_fut.done():
            self._interrupt_fut.set_result(None)
        self._mark_done() 
        
    def _mark_done(self) -> None:
        """Internal method to mark the utterance as complete."""
        if not self._done_fut.done():
            self._done_fut.set_result(None)

    def __await__(self) -> Generator[Any, None, None]:
        """Allows the handle to be awaited."""
        return self._done_fut.__await__()
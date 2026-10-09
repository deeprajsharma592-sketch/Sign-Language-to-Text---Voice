"""Thread-safe background Text-To-Speech worker using pyttsx3.

Features:
- Non-blocking operation via dedicated daemon thread
- Task queue to serialize spoken speech
- Stale queue drop policy: drops oldest items when queue exceeds max_size
- Graceful shutdown
"""

import queue
import threading
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class TTSWorker:
    """Manages offline speech synthesis on a dedicated background thread."""

    def __init__(self, max_queue_size: int = 4, rate: int = 160, enabled: bool = True):
        self.enabled = enabled and (pyttsx3 is not None)
        self.max_queue_size = max_queue_size
        self.rate = rate
        self._queue = queue.Queue(maxsize=max_queue_size * 2)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TTS-Thread")
            self._thread.start()
        else:
            if pyttsx3 is None:
                logger.warning("pyttsx3 is not installed; TTS will be disabled.")

    def _run_loop(self):
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            self.enabled = False
            return

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Error speaking text '{text}': {e}")
            finally:
                self._queue.task_done()

        # Clean up
        try:
            engine.stop()
        except Exception:
            pass

    def speak(self, text: str):
        """Enqueue a word/phrase to speak. Drops oldest phrase if queue is backed up."""
        if not self.enabled or not text:
            return

        # If queue is full or backing up, drop older phrases so spoken output remains fresh
        while self._queue.qsize() >= self.max_queue_size:
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.debug(f"Dropped stale TTS item: '{dropped}'")
            except queue.Empty:
                break

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self):
        """Stop background worker and wait for thread to join."""
        if not self.enabled or self._thread is None:
            return
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

"""Postprocessing module for real-time sign language prediction stream.

Handles:
- Minimum confidence thresholding
- Debouncing: winning prediction must repeat K times consecutively
- Cooldown: prevents identical word from repeating within a cooldown window
- Idle suppression: 'idle' predictions never produce emitted words
- Running sentence builder: maintains history of emitted words, supports clearing
"""

from typing import Optional, List, Dict, Any


class SignPostprocessor:
    """Stateful postprocessing engine for streaming predictions."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        debounce_frames: int = 3,
        cooldown_frames: int = 20,
        idle_label: str = "idle",
        max_sentence_words: int = 10,
    ):
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames
        self.cooldown_frames = cooldown_frames
        self.idle_label = idle_label
        self.max_sentence_words = max_sentence_words

        # Internal state
        self._consecutive_label: Optional[str] = None
        self._consecutive_count: int = 0
        self._last_emitted_label: Optional[str] = None
        self._frames_since_last_emission: int = cooldown_frames + 1
        self._sentence: List[str] = []

    def reset_tracking(self):
        """Reset consecutive detection counters when tracking is lost."""
        self._consecutive_label = None
        self._consecutive_count = 0

    def process(self, predicted_label: str, confidence: float) -> Optional[str]:
        """Process a single frame's prediction.

        Returns:
            The emitted word string if a sign was confirmed, else None.
        """
        self._frames_since_last_emission += 1

        # Check confidence
        if confidence < self.confidence_threshold:
            self.reset_tracking()
            return None

        # Check consecutive streak (debounce)
        if predicted_label == self._consecutive_label:
            self._consecutive_count += 1
        else:
            self._consecutive_label = predicted_label
            self._consecutive_count = 1

        # If debounce condition met
        if self._consecutive_count == self.debounce_frames:
            # Idle class never produces output
            if predicted_label == self.idle_label:
                return None

            # Cooldown check: if same as last emitted word and still in cooldown window
            if (
                predicted_label == self._last_emitted_label
                and self._frames_since_last_emission < self.cooldown_frames
            ):
                return None

            # Emit word!
            self._last_emitted_label = predicted_label
            self._frames_since_last_emission = 0
            self._sentence.append(predicted_label)
            if len(self._sentence) > self.max_sentence_words:
                self._sentence.pop(0)

            return predicted_label

        return None

    def get_sentence(self) -> str:
        """Return formatted running sentence string."""
        return " ".join(self._sentence)

    def clear_sentence(self):
        """Clear the running sentence history."""
        self._sentence.clear()

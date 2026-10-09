"""Unit tests for the streaming postprocessing engine and TTS queue."""

import pytest
from src.postprocess import SignPostprocessor
from src.tts import TTSWorker


def test_confidence_threshold_filtering():
    post = SignPostprocessor(confidence_threshold=0.75, debounce_frames=2)
    # Low confidence predictions should not trigger emission
    assert post.process("hello", 0.60) is None
    assert post.process("hello", 0.70) is None
    # Reset streak if confidence drops
    assert post._consecutive_count == 0


def test_debounce_mechanism():
    post = SignPostprocessor(confidence_threshold=0.70, debounce_frames=3, cooldown_frames=10)

    # Frame 1: first sighting
    assert post.process("hello", 0.85) is None
    assert post._consecutive_count == 1

    # Frame 2: second sighting
    assert post.process("hello", 0.90) is None
    assert post._consecutive_count == 2

    # Frame 3: reaches debounce count of 3 -> emits word
    assert post.process("hello", 0.88) == "hello"

    # Frame 4: continued high confidence sign does NOT repeat emission until new sequence
    assert post.process("hello", 0.88) is None


def test_idle_suppression():
    post = SignPostprocessor(confidence_threshold=0.70, debounce_frames=2, idle_label="idle")
    assert post.process("idle", 0.95) is None
    # Even after debounce frames met, idle should return None
    assert post.process("idle", 0.99) is None
    assert post.get_sentence() == ""


def test_cooldown_prevention():
    post = SignPostprocessor(
        confidence_threshold=0.70,
        debounce_frames=2,
        cooldown_frames=5,
        idle_label="idle"
    )

    # First emission of 'help'
    post.process("help", 0.80)
    assert post.process("help", 0.85) == "help"

    # Reset tracking
    post.reset_tracking()

    # Immediate second detection of 'help' within cooldown window (only 2 frames later)
    post.process("help", 0.80)
    assert post.process("help", 0.85) is None  # Blocked by cooldown!

    # Advance beyond cooldown (5 frames)
    for _ in range(6):
        post.process("idle", 0.50)  # low conf frames to tick clock

    # Now 'help' can be emitted again
    post.process("help", 0.85)
    assert post.process("help", 0.85) == "help"


def test_running_sentence_accumulation():
    post = SignPostprocessor(confidence_threshold=0.70, debounce_frames=2, cooldown_frames=0)

    post.process("hello", 0.80)
    post.process("hello", 0.85)

    post.reset_tracking()
    post.process("thank_you", 0.80)
    post.process("thank_you", 0.85)

    assert post.get_sentence() == "hello thank_you"

    post.clear_sentence()
    assert post.get_sentence() == ""


def test_tts_worker_lifecycle():
    worker = TTSWorker(enabled=False)
    worker.speak("test")
    worker.stop()
    assert True

"""Real-time webcam inference loop for sign language recognition.

Features:
- Rolling window of N frames (default 30)
- Graceful camera failure handling
- Hand tracking verification
- Model inference with postprocessing debounce & cooldown
- Non-blocking TTS speech
- On-screen HUD with landmarks, running sentence, confidence, and FPS
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np
import yaml
import torch

from src.features import LandmarkExtractor, compute_sequence_features, TOTAL_RAW_DIM
from src.model import build_model
from src.postprocess import SignPostprocessor
from src.tts import TTSWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/labels.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Train a model first via `python -m src.train`.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "gru")
    input_dim = checkpoint.get("input_dim", TOTAL_RAW_DIM)
    num_classes = checkpoint.get("num_classes", len(checkpoint.get("vocabulary", [])))
    vocab = checkpoint.get("vocabulary", [])

    model = build_model(model_type, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded {model_type.upper()} model with {num_classes} classes from {checkpoint_path}")
    return model, vocab, input_dim


def run_inference(
    config_path: str = "config/labels.yaml",
    checkpoint_path: str = "models/best_model.pt",
    no_tts: bool = False,
    mock_frames: Optional[int] = None,
):
    cfg = load_config(config_path)
    seq_len = cfg["sequence"]["sequence_length"]
    include_velocity = cfg["sequence"].get("include_velocity", False)
    conf_thresh = cfg["inference"].get("confidence_threshold", 0.70)
    debounce = cfg["inference"].get("debounce_frames", 3)
    cooldown = cfg["inference"].get("cooldown_frames", 20)
    cam_idx = cfg["inference"].get("camera_index", 0)

    device = torch.device("cpu")
    model, vocab, input_dim = load_model(checkpoint_path, device)

    postprocessor = SignPostprocessor(
        confidence_threshold=conf_thresh,
        debounce_frames=debounce,
        cooldown_frames=cooldown,
        idle_label="idle"
    )

    tts = TTSWorker(enabled=(not no_tts))
    extractor = LandmarkExtractor()
    buffer = deque(maxlen=seq_len)

    logger.info("Initializing inference pipeline...")
    logger.info(f"Vocabulary: {vocab} | Confidence Thresh: {conf_thresh} | Window: {seq_len}")

    if mock_frames is not None:
        logger.info(f"Running automated mock inference for {mock_frames} frames...")
        for i in range(mock_frames):
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            features, has_hands, _ = extractor.process_frame(dummy_frame)
            buffer.append(features)

            if len(buffer) == seq_len:
                seq_arr = np.array(buffer, dtype=np.float32)
                seq_feat = compute_sequence_features(seq_arr, include_velocity=include_velocity)
                x_tensor = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    top_idx = probs.argmax().item()
                    conf = probs[top_idx].item()
                    pred_label = vocab[top_idx]

                emitted = postprocessor.process(pred_label, conf)
                if emitted:
                    logger.info(f"[EMITTED] '{emitted}' (Sentence: '{postprocessor.get_sentence()}')")
                    tts.speak(emitted)

        tts.stop()
        extractor.close()
        logger.info("Mock inference test completed successfully.")
        return

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.error(f"Cannot access webcam at index {cam_idx}. Make sure your camera is connected.")
        tts.stop()
        extractor.close()
        return

    fps_tracker = deque(maxlen=30)
    current_prediction = "Waiting for hands..."
    current_conf = 0.0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            features, has_hands, debug_info = extractor.process_frame(frame, is_mirrored=True)

            if has_hands:
                buffer.append(features)
            else:
                # Reset tracking state if hands disappear
                postprocessor.reset_tracking()
                buffer.clear()
                current_prediction = "No Hands"
                current_conf = 0.0

            if len(buffer) == seq_len:
                seq_arr = np.array(buffer, dtype=np.float32)
                seq_feat = compute_sequence_features(seq_arr, include_velocity=include_velocity)
                x_tensor = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    top_idx = probs.argmax().item()
                    current_conf = probs[top_idx].item()
                    current_prediction = vocab[top_idx]

                emitted = postprocessor.process(current_prediction, current_conf)
                if emitted:
                    logger.info(f"[SPOKEN] '{emitted}' | Sentence: '{postprocessor.get_sentence()}'")
                    tts.speak(emitted)

            # Calculate FPS
            dt = time.time() - t0
            if dt > 0:
                fps_tracker.append(1.0 / dt)
            fps = np.mean(fps_tracker) if fps_tracker else 0.0

            # Render HUD overlay
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 100), (20, 20, 20), -1)
            cv2.rectangle(hud_bg, (0, h - 60), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

            # Header info
            color = (0, 255, 0) if current_conf >= conf_thresh and current_prediction != "idle" else (0, 200, 255)
            cv2.putText(frame, f"Prediction: {current_prediction.upper()} ({current_conf * 100:.1f}%)",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | Buffer: {len(buffer)}/{seq_len}",
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Bottom Sentence bar
            sentence_text = postprocessor.get_sentence() or "[Sentence Empty]"
            cv2.putText(frame, f"Sentence: {sentence_text}",
                        (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("Sign Language Recognition (Sign2Text)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                postprocessor.clear_sentence()
                logger.info("Sentence cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        extractor.close()
        logger.info("Inference loop exited cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Inference Loop")
    parser.add_argument("--config", type=str, default="config/labels.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt", help="Path to checkpoint file")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech synthesis")
    parser.add_argument("--mock-frames", type=int, default=None, help="Run headless mock frames for test verification")
    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        no_tts=args.no_tts,
        mock_frames=args.mock_frames
    )

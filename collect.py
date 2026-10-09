"""Data collection tool for recording sign language sequences.

Features:
- Fixed-length window recording (30 frames default)
- Auto-rejection of bad samples (missing hands > max_missing_hand_ratio)
- Visual countdown and live recording indicator HUD
- Undo / redo last recorded sample key ('u')
- Session metadata tracking (person, lighting, timestamp)
- Dedicated idle mode ('--idle')
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import yaml

from src.features import LandmarkExtractor, TOTAL_RAW_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/labels.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_existing_sample_count(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return len(list(output_dir.glob("sample_*.npy")))


def save_sample(
    output_dir: Path,
    sequence: np.ndarray,
    metadata: Dict[str, Any]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_count = get_existing_sample_count(output_dir)
    sample_id = existing_count + 1
    npy_path = output_dir / f"sample_{sample_id:04d}.npy"
    meta_path = output_dir / f"sample_{sample_id:04d}.json"

    np.save(npy_path, sequence.astype(np.float32))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved sample {sample_id} to {npy_path}")
    return npy_path


def delete_last_sample(output_dir: Path) -> Optional[int]:
    samples = sorted(list(output_dir.glob("sample_*.npy")))
    if not samples:
        return None
    last_sample = samples[-1]
    last_id_str = last_sample.stem.split("_")[-1]
    last_meta = output_dir / f"{last_sample.stem}.json"

    try:
        last_sample.unlink(missing_ok=True)
        last_meta.unlink(missing_ok=True)
        return int(last_id_str)
    except Exception as e:
        logger.error(f"Failed to delete {last_sample}: {e}")
        return None


def run_collection(
    label: str,
    target_samples: int,
    session: str,
    is_idle: bool = False,
    config_path: str = "config/labels.yaml",
    mock_mode: bool = False
):
    cfg = load_config(config_path)
    seq_len = cfg["sequence"]["sequence_length"]
    max_missing_ratio = cfg["sequence"]["max_missing_hand_ratio"]
    cam_idx = cfg["inference"].get("camera_index", 0)

    label_to_record = "idle" if is_idle else label
    out_dir = Path(f"data/raw/{label_to_record}")

    logger.info(f"Target Label: {label_to_record} | Target Samples: {target_samples} | Session: {session}")
    current_count = get_existing_sample_count(out_dir)
    logger.info(f"Existing samples for '{label_to_record}': {current_count}")

    if mock_mode:
        logger.info("Running in mock mode (generating synthetic samples)...")
        for i in range(target_samples):
            # Generate dummy sequence: (seq_len, TOTAL_RAW_DIM)
            seq = np.random.randn(seq_len, TOTAL_RAW_DIM).astype(np.float32)
            meta = {
                "label": label_to_record,
                "session": session,
                "timestamp": time.time(),
                "sequence_length": seq_len,
                "is_mock": True
            }
            save_sample(out_dir, seq, meta)
        logger.info(f"Mock recording complete! Total samples: {get_existing_sample_count(out_dir)}")
        return

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam index {cam_idx}. Exiting.")
        return

    extractor = LandmarkExtractor()
    recorded_count = 0
    state = "READY"  # READY, COUNTDOWN, RECORDING
    countdown_start = 0.0
    recorded_frames = []
    missing_hand_count = 0
    message = "Press 'SPACE' to record, 'u' to undo, 'q' to quit"

    try:
        while recorded_count < target_samples:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            features, has_hands, debug_info = extractor.process_frame(frame, is_mirrored=True)

            current_time = time.time()
            total_saved = get_existing_sample_count(out_dir)

            # Draw HUD
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 80), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, f"Label: {label_to_record.upper()} ({total_saved}/{target_samples + current_count})",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Session: {session} | Hands: {'YES' if has_hands else 'NO'}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user.")
                break
            elif key == ord('u'):
                deleted = delete_last_sample(out_dir)
                if deleted:
                    message = f"Deleted sample {deleted}!"
                    if recorded_count > 0:
                        recorded_count -= 1
                else:
                    message = "No samples to undo."

            if state == "READY":
                cv2.putText(frame, message, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if key == 32:  # SPACE
                    state = "COUNTDOWN"
                    countdown_start = time.time()

            elif state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = 3.0 - elapsed
                if remaining > 0:
                    cv2.putText(frame, f"Starting in: {int(np.ceil(remaining))}",
                                (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)
                else:
                    state = "RECORDING"
                    recorded_frames = []
                    missing_hand_count = 0

            elif state == "RECORDING":
                cv2.putText(frame, f"RECORDING ({len(recorded_frames) + 1}/{seq_len})",
                            (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                recorded_frames.append(features)
                if not has_hands and not is_idle:
                    missing_hand_count += 1

                if len(recorded_frames) == seq_len:
                    # Evaluate quality
                    missing_ratio = missing_hand_count / seq_len
                    if not is_idle and missing_ratio > max_missing_ratio:
                        message = f"REJECTED: Hands missing in {int(missing_ratio * 100)}% frames! Please retry."
                        logger.warning(message)
                    else:
                        seq_arr = np.array(recorded_frames, dtype=np.float32)
                        meta = {
                            "label": label_to_record,
                            "session": session,
                            "timestamp": time.time(),
                            "sequence_length": seq_len,
                            "missing_hand_ratio": float(missing_ratio)
                        }
                        save_sample(out_dir, seq_arr, meta)
                        recorded_count += 1
                        message = f"Sample recorded successfully! ({recorded_count}/{target_samples})"

                    state = "READY"

            cv2.imshow("Sign Language Data Collector", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        logger.info(f"Collection finished. Final sample count in {out_dir}: {get_existing_sample_count(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dataset Collector")
    parser.add_argument("--label", type=str, default="hello", help="Label to record (e.g., hello, thank_you, etc.)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to record")
    parser.add_argument("--session", type=str, default="default_session", help="Session or person identifier tag")
    parser.add_argument("--idle", action="store_true", help="Record idle class (resting hands, fidgeting)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (headless/synthetic)")
    args = parser.parse_args()

    run_collection(
        label=args.label,
        target_samples=args.samples,
        session=args.session,
        is_idle=args.idle,
        mock_mode=args.mock
    )

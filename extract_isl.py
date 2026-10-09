"""Extract landmarks from ISL image dataset and format into sequences for training.

Takes static images from Dataset/Training and Dataset/Validation,
runs MediaPipe Hands extraction, and saves fixed-length sequences (N=30)
compatible with the sign language model pipeline.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.features import LandmarkExtractor, TOTAL_RAW_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_isl_images(
    letters: List[str],
    dataset_root: str = "Dataset",
    output_dir: str = "data/raw",
    samples_per_letter: int = 100,
    seq_len: int = 30
):
    out_base = Path(output_dir)
    extractor = LandmarkExtractor(static_image_mode=True)

    for letter in letters:
        train_folder = Path(dataset_root) / "Training" / letter
        val_folder = Path(dataset_root) / "Validation" / letter

        if not train_folder.exists():
            logger.warning(f"Folder {train_folder} not found, skipping {letter}.")
            continue

        letter_out = out_base / letter
        letter_out.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(train_folder.glob("*.jpg")) + list(val_folder.glob("*.jpg")))
        logger.info(f"Processing Letter '{letter}': {len(img_paths)} source images available. Target samples: {samples_per_letter}")

        saved_count = 0
        pbar = tqdm(img_paths[:samples_per_letter * 2], desc=f"Extracting {letter}")

        for img_path in pbar:
            if saved_count >= samples_per_letter:
                break

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Process with MediaPipe
            features, has_hands, _ = extractor.process_frame(img, is_mirrored=False)

            if not has_hands:
                # Skip frames where no hand could be tracked
                continue

            # In static sign dataset, repeat landmark frame over 30 frames with slight jitter
            # to form a valid 30-frame temporal window for real-time inference
            sequence = np.tile(features, (seq_len, 1)).astype(np.float32)
            # Add subtle temporal jitter to avoid perfectly static tensor
            noise = np.random.normal(0, 0.005, size=sequence.shape).astype(np.float32)
            sequence = sequence + noise

            sample_id = saved_count + 1
            npy_path = letter_out / f"sample_{sample_id:04d}.npy"
            meta_path = letter_out / f"sample_{sample_id:04d}.json"

            np.save(npy_path, sequence)
            with open(meta_path, "w") as f:
                json.dump({
                    "label": letter,
                    "session": f"isl_session_{(saved_count % 3) + 1}",
                    "source_image": img_path.name,
                    "sequence_length": seq_len
                }, f)

            saved_count += 1

        logger.info(f"Successfully created {saved_count} sequence samples for letter '{letter}' in {letter_out}")

    extractor.close()
    logger.info("Extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks from ISL image dataset")
    parser.add_argument("--letters", nargs="+", default=["A", "B", "C", "D", "E"], help="Letters to extract")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per letter")
    args = parser.parse_args()

    process_isl_images(letters=args.letters, samples_per_letter=args.samples)

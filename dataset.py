"""Dataset loading, session-stratified splitting, and augmentations.

Augmentation techniques (applied during training only):
- Gaussian jitter
- Random temporal shift / time warping
- Random spatial scaling
- In-plane 2D rotation of landmarks
- Random frame dropout (simulating occlusions / missed detections)
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.features import compute_sequence_features, TOTAL_RAW_DIM

logger = logging.getLogger(__name__)


def apply_jitter(seq: np.ndarray, std: float = 0.02) -> np.ndarray:
    noise = np.random.normal(0, std, size=seq.shape)
    return seq + noise


def apply_scaling(seq: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return seq * factor


def apply_time_shift(seq: np.ndarray, max_shift: int = 3) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq
    res = np.zeros_like(seq)
    if shift > 0:
        res[shift:] = seq[:-shift]
        res[:shift] = seq[0]  # pad start
    else:
        res[:shift] = seq[-shift:]
        res[shift:] = seq[-1]  # pad end
    return res


def apply_frame_dropout(seq: np.ndarray, p: float = 0.1) -> np.ndarray:
    seq_copy = seq.copy()
    mask = np.random.binomial(1, p, size=seq.shape[0]).astype(bool)
    seq_copy[mask] = 0.0  # drop out landmarks for randomly selected frames
    return seq_copy


def apply_rotation_2d(seq: np.ndarray, max_angle_deg: float = 10.0) -> np.ndarray:
    """Rotate (x, y) coordinates around the origin by a small random angle."""
    angle_rad = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    seq_copy = seq.copy()
    # Apply to each frame and each 3D coordinate (x, y)
    t, d = seq.shape
    # d is 126 (or 252). The first 126 are 42 landmarks * 3 coords.
    num_coords = min(d, TOTAL_RAW_DIM)
    reshaped = seq_copy[:, :num_coords].reshape(t, -1, 3)
    xy = reshaped[:, :, :2]  # (t, num_lm, 2)
    rotated_xy = np.matmul(xy, rot_mat.T)
    reshaped[:, :, :2] = rotated_xy
    seq_copy[:, :num_coords] = reshaped.reshape(t, num_coords)
    return seq_copy


class SignDataset(Dataset):
    """PyTorch Dataset with optional on-the-fly training augmentation."""

    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        augment: bool = False,
        include_velocity: bool = False,
    ):
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.include_velocity = include_velocity

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx].copy()

        if self.augment:
            if random.random() < 0.5:
                seq = apply_jitter(seq)
            if random.random() < 0.5:
                seq = apply_scaling(seq)
            if random.random() < 0.5:
                seq = apply_time_shift(seq)
            if random.random() < 0.3:
                seq = apply_frame_dropout(seq)
            if random.random() < 0.4:
                seq = apply_rotation_2d(seq)

            # Slot augmentation for 1-handed signs: allow single hand to appear in either left or right slot
            if random.random() < 0.5:
                # If only one hand has coordinates, swap slots
                l_act = np.any(seq[:, :63] != 0)
                r_act = np.any(seq[:, 63:126] != 0)
                if l_act and not r_act:
                    swapped = seq.copy()
                    swapped[:, 63:126] = seq[:, :63]
                    swapped[:, :63] = 0
                    seq = swapped
                elif r_act and not l_act:
                    swapped = seq.copy()
                    swapped[:, :63] = seq[:, 63:126]
                    swapped[:, 63:126] = 0
                    seq = swapped

        seq = compute_sequence_features(seq, include_velocity=self.include_velocity)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def load_dataset_from_disk(
    data_dir: str = "data/raw",
    vocabulary: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[int], List[str], Dict[str, int]]:
    """Scan raw directory, load .npy samples and corresponding session metadata.

    Returns:
        samples: list of (30, 126) arrays
        labels: list of class integer indices
        sessions: list of session/person string tags
        label_to_idx: map from class name to int
    """
    raw_path = Path(data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    if vocabulary is None:
        vocabulary = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])

    label_to_idx = {name: idx for idx, name in enumerate(vocabulary)}
    samples, labels, sessions = [], [], []

    for label_name in vocabulary:
        label_dir = raw_path / label_name
        if not label_dir.exists():
            continue

        npy_files = sorted(list(label_dir.glob("*.npy")))
        for npy_file in npy_files:
            try:
                arr = np.load(npy_file)
                meta_file = npy_file.with_suffix(".json")
                session_tag = "default"
                if meta_file.exists():
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                        session_tag = meta.get("session", "default")

                samples.append(arr)
                labels.append(label_to_idx[label_name])
                sessions.append(session_tag)
            except Exception as e:
                logger.warning(f"Error loading {npy_file}: {e}")

    return samples, labels, sessions, label_to_idx


def split_dataset(
    samples: List[np.ndarray],
    labels: List[int],
    sessions: List[str],
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split data indices into train and validation ensuring all classes are represented in both."""
    total_samples = len(samples)
    unique_labels = set(labels)

    # If sessions are available per class, do stratified session-split per class
    # Otherwise perform stratified split to ensure train and val have balanced classes
    train_idx, val_idx = [], []
    for lbl in unique_labels:
        lbl_indices = [i for i, y in enumerate(labels) if y == lbl]
        if len(lbl_indices) == 0:
            continue
        lbl_sessions = [sessions[i] for i in lbl_indices]
        unique_lbl_sessions = list(set(lbl_sessions))

        if len(unique_lbl_sessions) > 1:
            # Pick validation sessions for this class
            val_session_count = max(1, int(len(unique_lbl_sessions) * test_size))
            val_sess_set = set(unique_lbl_sessions[:val_session_count])
            for i in lbl_indices:
                if sessions[i] in val_sess_set:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
        else:
            # Fall back to stratified ratio for this class
            n_val = max(1, int(len(lbl_indices) * test_size))
            rng = np.random.RandomState(seed + lbl)
            shuffled = rng.permutation(lbl_indices)
            val_idx.extend(shuffled[:n_val])
            train_idx.extend(shuffled[n_val:])

    return train_idx, val_idx


def generate_synthetic_dataset(
    vocabulary: List[str],
    samples_per_class: int = 25,
    seq_len: int = 30,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[str], Dict[str, int]]:
    """Generate synthetic landmark sequences to test full training pipeline end-to-end.

    Creates distinct synthetic spatio-temporal trajectories for each class.
    """
    np.random.seed(seed)
    label_to_idx = {name: idx for idx, name in enumerate(vocabulary)}
    samples, labels, sessions = [], [], []

    for class_idx, name in enumerate(vocabulary):
        for s in range(samples_per_class):
            session_tag = f"synthetic_session_{(s % 3) + 1}"
            seq = np.zeros((seq_len, TOTAL_RAW_DIM), dtype=np.float32)

            # Assign class-specific base trajectory frequency
            freq = (class_idx + 1) * 0.3
            t_steps = np.linspace(0, np.pi * 2, seq_len)

            for t_idx, t_val in enumerate(t_steps):
                # Class signal on wrist-relative landmarks
                signal = np.sin(t_val * freq) + np.cos(t_val * (class_idx + 0.5))
                noise = np.random.normal(0, 0.05, size=TOTAL_RAW_DIM).astype(np.float32)
                seq[t_idx] = signal * 0.5 + noise

            samples.append(seq)
            labels.append(class_idx)
            sessions.append(session_tag)

    return samples, labels, sessions, label_to_idx

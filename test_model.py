"""Unit tests for models and dataset pipeline."""

import pytest
import torch
import numpy as np

from src.model import SignGRU, Sign1DCNN, build_model
from src.dataset import (
    apply_jitter,
    apply_scaling,
    apply_time_shift,
    apply_frame_dropout,
    apply_rotation_2d,
    generate_synthetic_dataset,
    SignDataset,
    split_dataset,
)
from src.features import TOTAL_RAW_DIM


def test_sign_gru_forward():
    batch_size = 4
    seq_len = 30
    model = SignGRU(input_dim=TOTAL_RAW_DIM, hidden_dim=32, num_layers=2, num_classes=6)
    x = torch.randn(batch_size, seq_len, TOTAL_RAW_DIM)
    out = model(x)
    assert out.shape == (batch_size, 6)
    # Check backward pass
    loss = out.sum()
    loss.backward()
    assert model.fc[1].weight.grad is not None


def test_sign_1dcnn_forward():
    batch_size = 4
    seq_len = 30
    model = Sign1DCNN(input_dim=TOTAL_RAW_DIM, num_classes=6)
    x = torch.randn(batch_size, seq_len, TOTAL_RAW_DIM)
    out = model(x)
    assert out.shape == (batch_size, 6)
    # Check backward pass
    loss = out.sum()
    loss.backward()
    assert model.classifier[1].weight.grad is not None


def test_build_model():
    m_gru = build_model("gru", input_dim=126, num_classes=5)
    assert isinstance(m_gru, SignGRU)

    m_cnn = build_model("cnn", input_dim=126, num_classes=5)
    assert isinstance(m_cnn, Sign1DCNN)

    with pytest.raises(ValueError):
        build_model("unknown_arch")


def test_dataset_augmentations():
    seq = np.ones((30, TOTAL_RAW_DIM), dtype=np.float32)

    # Jitter
    jittered = apply_jitter(seq, std=0.05)
    assert jittered.shape == seq.shape
    assert not np.allclose(jittered, seq)

    # Scaling
    scaled = apply_scaling(seq, scale_range=(1.5, 1.5))
    np.testing.assert_allclose(scaled, seq * 1.5)

    # Time shift
    shifted = apply_time_shift(seq, max_shift=2)
    assert shifted.shape == seq.shape

    # Frame dropout
    dropped = apply_frame_dropout(seq, p=0.5)
    assert dropped.shape == seq.shape
    assert (dropped == 0).any()

    # 2D rotation
    rotated = apply_rotation_2d(seq, max_angle_deg=10.0)
    assert rotated.shape == seq.shape


def test_synthetic_dataset_generation():
    vocab = ["idle", "hello", "thank_you"]
    samples, labels, sessions, label_to_idx = generate_synthetic_dataset(vocab, samples_per_class=10)
    assert len(samples) == 30
    assert len(labels) == 30
    assert len(sessions) == 30
    assert samples[0].shape == (30, TOTAL_RAW_DIM)
    assert set(labels) == {0, 1, 2}

    train_idx, val_idx = split_dataset(samples, labels, sessions, test_size=0.2)
    assert len(train_idx) + len(val_idx) == 30

"""Unit tests for feature extraction and landmark normalization."""

import numpy as np
import pytest
from src.features import (
    normalize_hand_landmarks,
    assemble_frame_features,
    compute_sequence_features,
    WRIST_INDEX,
    MIDDLE_MCP_INDEX,
    NUM_LANDMARKS_PER_HAND,
    COORDS_PER_LANDMARK,
    HAND_FEATURE_DIM,
    TOTAL_RAW_DIM,
)


def create_dummy_hand(offset_x: float = 0.0, offset_y: float = 0.0, scale_factor: float = 1.0) -> np.ndarray:
    """Generate a synthetic 21x3 hand landmark array with a known wrist and MCP distance."""
    landmarks = np.zeros((NUM_LANDMARKS_PER_HAND, COORDS_PER_LANDMARK), dtype=np.float32)
    # Set up some landmark offsets
    for i in range(NUM_LANDMARKS_PER_HAND):
        landmarks[i] = [i * 0.05 * scale_factor + offset_x, i * 0.03 * scale_factor + offset_y, i * 0.01 * scale_factor]

    # Ensure wrist is specifically at the offset
    landmarks[WRIST_INDEX] = [offset_x, offset_y, 0.0]
    # Ensure middle MCP is at a controlled distance
    landmarks[MIDDLE_MCP_INDEX] = [offset_x + 0.2 * scale_factor, offset_y + 0.0, 0.0]
    return landmarks


def test_normalization_shape():
    hand = create_dummy_hand()
    normalized = normalize_hand_landmarks(hand)
    assert normalized.shape == (NUM_LANDMARKS_PER_HAND, COORDS_PER_LANDMARK)
    assert normalized.dtype == np.float32


def test_translation_invariance():
    """Moving the hand in camera frame shouldn't change the normalized features."""
    hand1 = create_dummy_hand(offset_x=0.0, offset_y=0.0, scale_factor=1.0)
    hand2 = create_dummy_hand(offset_x=10.5, offset_y=-5.2, scale_factor=1.0)

    norm1 = normalize_hand_landmarks(hand1)
    norm2 = normalize_hand_landmarks(hand2)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)
    # Wrist should always be at origin (0, 0, 0)
    np.testing.assert_allclose(norm1[WRIST_INDEX], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(norm2[WRIST_INDEX], np.zeros(3), atol=1e-6)


def test_scale_invariance():
    """Moving closer or further from the camera shouldn't change the normalized features."""
    hand_near = create_dummy_hand(scale_factor=2.5)
    hand_far = create_dummy_hand(scale_factor=0.8)

    norm_near = normalize_hand_landmarks(hand_near)
    norm_far = normalize_hand_landmarks(hand_far)

    np.testing.assert_allclose(norm_near, norm_far, atol=1e-5)


def test_assemble_frame_features_both_hands():
    left = create_dummy_hand(offset_x=0.1)
    right = create_dummy_hand(offset_x=0.5)

    feat = assemble_frame_features(left, right)
    assert feat.shape == (TOTAL_RAW_DIM,)
    assert feat.dtype == np.float32

    # Left hand is in first 63 dims
    norm_left = normalize_hand_landmarks(left).flatten()
    np.testing.assert_allclose(feat[:HAND_FEATURE_DIM], norm_left, atol=1e-6)

    # Right hand is in next 63 dims
    norm_right = normalize_hand_landmarks(right).flatten()
    np.testing.assert_allclose(feat[HAND_FEATURE_DIM:], norm_right, atol=1e-6)


def test_assemble_frame_features_missing_hands():
    """Missing hand should result in all zeros for its 63 dimensions."""
    left = create_dummy_hand()
    feat_left_only = assemble_frame_features(left, None)
    assert feat_left_only.shape == (TOTAL_RAW_DIM,)
    np.testing.assert_allclose(feat_left_only[HAND_FEATURE_DIM:], np.zeros(HAND_FEATURE_DIM), atol=1e-6)

    feat_right_only = assemble_frame_features(None, left)
    assert feat_right_only.shape == (TOTAL_RAW_DIM,)
    np.testing.assert_allclose(feat_right_only[:HAND_FEATURE_DIM], np.zeros(HAND_FEATURE_DIM), atol=1e-6)

    feat_no_hands = assemble_frame_features(None, None)
    np.testing.assert_allclose(feat_no_hands, np.zeros(TOTAL_RAW_DIM), atol=1e-6)


def test_compute_sequence_features():
    T = 30
    seq = np.random.randn(T, TOTAL_RAW_DIM).astype(np.float32)

    # Without velocity
    out_no_vel = compute_sequence_features(seq, include_velocity=False)
    assert out_no_vel.shape == (T, TOTAL_RAW_DIM)
    np.testing.assert_allclose(out_no_vel, seq)

    # With velocity
    out_vel = compute_sequence_features(seq, include_velocity=True)
    assert out_vel.shape == (T, TOTAL_RAW_DIM * 2)
    # First frame delta is 0
    np.testing.assert_allclose(out_vel[0, TOTAL_RAW_DIM:], np.zeros(TOTAL_RAW_DIM), atol=1e-6)
    # Second frame delta is seq[1] - seq[0]
    expected_delta_1 = seq[1] - seq[0]
    np.testing.assert_allclose(out_vel[1, TOTAL_RAW_DIM:], expected_delta_1, atol=1e-6)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        normalize_hand_landmarks(np.zeros((20, 3)))
    with pytest.raises(ValueError):
        compute_sequence_features(np.zeros((10, 50)))

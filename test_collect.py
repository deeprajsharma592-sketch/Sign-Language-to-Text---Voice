"""Unit tests for the data collection pipeline."""

import json
import shutil
from pathlib import Path
import numpy as np
import pytest

from src.collect import (
    save_sample,
    delete_last_sample,
    get_existing_sample_count,
    TOTAL_RAW_DIM,
)

TEST_DATA_DIR = Path("data/test_raw/hello")


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    yield
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)


def test_save_and_count_sample():
    seq = np.random.randn(30, TOTAL_RAW_DIM).astype(np.float32)
    meta = {
        "label": "hello",
        "session": "test_session_1",
        "timestamp": 1234567.89,
        "sequence_length": 30,
        "missing_hand_ratio": 0.0
    }

    assert get_existing_sample_count(TEST_DATA_DIR) == 0
    saved_path = save_sample(TEST_DATA_DIR, seq, meta)
    assert saved_path.exists()
    assert get_existing_sample_count(TEST_DATA_DIR) == 1

    # Verify loaded array
    loaded_seq = np.load(saved_path)
    assert loaded_seq.shape == (30, TOTAL_RAW_DIM)
    np.testing.assert_allclose(loaded_seq, seq)

    # Verify metadata JSON
    meta_path = saved_path.with_suffix(".json")
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        loaded_meta = json.load(f)
    assert loaded_meta["label"] == "hello"
    assert loaded_meta["session"] == "test_session_1"


def test_delete_last_sample_undo():
    for i in range(3):
        seq = np.random.randn(30, TOTAL_RAW_DIM).astype(np.float32)
        meta = {"label": "hello", "session": "s1", "sample_num": i}
        save_sample(TEST_DATA_DIR, seq, meta)

    assert get_existing_sample_count(TEST_DATA_DIR) == 3

    deleted_id = delete_last_sample(TEST_DATA_DIR)
    assert deleted_id == 3
    assert get_existing_sample_count(TEST_DATA_DIR) == 2

    # Verify the remaining files are sample_0001 and sample_0002
    assert (TEST_DATA_DIR / "sample_0001.npy").exists()
    assert (TEST_DATA_DIR / "sample_0002.npy").exists()
    assert not (TEST_DATA_DIR / "sample_0003.npy").exists()
    assert not (TEST_DATA_DIR / "sample_0003.json").exists()

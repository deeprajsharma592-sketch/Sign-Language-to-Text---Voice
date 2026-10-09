"""FastAPI High-Performance Backend API for Real-Time Sign Language Translation & Training.

Endpoints:
- WebSocket `/ws/stream`: Accepts video frames as base64, runs MediaPipe & model inference, returns predictions + landmarks + audio trigger.
- GET `/api/status`: Vocabulary, active model status, classes list.
- POST `/api/record`: Record a temporal gesture sequence from uploaded frames.
- POST `/api/train`: Trigger background model retraining with progress and metrics.
- GET `/api/references`: Returns reference letter image list.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import base64
from typing import Dict, Any, List, Optional
from collections import deque

import cv2
import numpy as np
import yaml
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.features import LandmarkExtractor, compute_sequence_features, TOTAL_RAW_DIM
from src.model import build_model
from src.postprocess import SignPostprocessor
from src.collect import save_sample, get_existing_sample_count
from src.train import run_training

app = FastAPI(title="SignAI Engine API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dataset images for frontend reference inspection
if os.path.exists("Dataset"):
    app.mount("/dataset", StaticFiles(directory="Dataset"), name="dataset")

# Global engine state
class EngineState:
    def __init__(self):
        self.device = torch.device("cpu")
        self.load_model()
        self.extractor = LandmarkExtractor()

    def load_model(self):
        with open("config/labels.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        checkpoint_path = "models/best_model.pt"
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model_type = ckpt.get("model_type", "cnn")
            self.vocab = ckpt.get("vocabulary", self.cfg["vocabulary"])
            self.input_dim = ckpt.get("input_dim", TOTAL_RAW_DIM)
            self.model = build_model(self.model_type, input_dim=self.input_dim, num_classes=len(self.vocab))
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
        else:
            self.vocab = self.cfg["vocabulary"]
            self.model_type = "cnn"
            self.model = build_model(self.model_type, input_dim=TOTAL_RAW_DIM, num_classes=len(self.vocab))
            self.model.eval()

engine = EngineState()


class RecordPayload(BaseModel):
    label: str
    session: str = "web_session"
    frames: List[List[float]]  # List of 30 frame feature vectors (126-dim each)


class TrainPayload(BaseModel):
    model_type: str = "cnn"


@app.get("/api/status")
def get_status():
    raw_dirs = sorted([d.name for d in Path("data/raw").iterdir() if d.is_dir()])
    class_counts = {d: get_existing_sample_count(Path("data/raw") / d) for d in raw_dirs}
    return {
        "vocabulary": engine.vocab,
        "model_type": engine.model_type,
        "classes_count": class_counts,
        "sequence_length": engine.cfg["sequence"]["sequence_length"],
        "confidence_threshold": engine.cfg["inference"].get("confidence_threshold", 0.70),
    }


@app.post("/api/record")
def record_sequence(payload: RecordPayload):
    label_clean = payload.label.strip().replace(" ", "_")
    if not label_clean:
        raise HTTPException(status_code=400, detail="Invalid label")

    out_dir = Path(f"data/raw/{label_clean}")
    seq_arr = np.array(payload.frames, dtype=np.float32)
    meta = {
        "label": label_clean,
        "session": payload.session,
        "timestamp": time.time(),
        "sequence_length": len(payload.frames)
    }
    save_sample(out_dir, seq_arr, meta)

    # Register in config if not present
    with open("config/labels.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    if label_clean not in cfg["vocabulary"]:
        cfg["vocabulary"].append(label_clean)
        with open("config/labels.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    return {"status": "success", "saved_label": label_clean, "total_samples": get_existing_sample_count(out_dir)}


@app.post("/api/train")
def train_model(payload: TrainPayload):
    raw_dirs = sorted([d.name for d in Path("data/raw").iterdir() if d.is_dir()])
    with open("config/labels.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["vocabulary"] = ["idle"] + [d for d in raw_dirs if d != "idle"]
    with open("config/labels.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    metrics = run_training(config_path="config/labels.yaml", override_model=payload.model_type)
    engine.load_model()
    return {"status": "success", "val_accuracy": metrics.get("val_accuracy", 1.0)}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = deque(maxlen=engine.cfg["sequence"]["sequence_length"])
    seq_len = engine.cfg["sequence"]["sequence_length"]

    postprocessor = SignPostprocessor(
        confidence_threshold=engine.cfg["inference"].get("confidence_threshold", 0.70),
        debounce_frames=engine.cfg["inference"].get("debounce_frames", 3),
        cooldown_frames=engine.cfg["inference"].get("cooldown_frames", 20),
        idle_label="idle"
    )

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "clear_sentence":
                postprocessor.clear_sentence()
                await websocket.send_text(json.dumps({"type": "sentence_cleared"}))
                continue

            # Process frame base64
            img_b64 = msg.get("image")
            if not img_b64:
                continue

            # Decode base64 to image
            encoded_data = img_b64.split(",")[-1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Mirror frame for selfie mode
            frame = cv2.flip(frame, 1)
            features, has_hands, debug_info = engine.extractor.process_frame(frame, is_mirrored=True)

            pred_label = "Waiting for hands..."
            pred_conf = 0.0
            probs_dict = {}

            if has_hands:
                buffer.append(features)
            else:
                postprocessor.reset_tracking()
                buffer.clear()
                pred_label = "No Hands"

            emitted_word = None
            if len(buffer) == seq_len:
                seq_arr = np.array(buffer, dtype=np.float32)
                x_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = engine.model(x_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    top_idx = probs.argmax().item()
                    pred_conf = float(probs[top_idx].item())
                    pred_label = engine.vocab[top_idx]
                    probs_dict = {engine.vocab[i]: round(float(probs[i].item()), 3) for i in range(len(engine.vocab))}

                emitted_word = postprocessor.process(pred_label, pred_conf)

            # Package landmark positions for canvas drawing
            hand_landmarks_data = []
            results = debug_info.get("results")
            if results and results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    pts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms.landmark]
                    hand_landmarks_data.append(pts)

            response = {
                "type": "prediction",
                "pred_label": pred_label,
                "confidence": pred_conf,
                "has_hands": has_hands,
                "buffer_len": len(buffer),
                "sentence": postprocessor.get_sentence(),
                "emitted": emitted_word,
                "probs": probs_dict,
                "landmarks": hand_landmarks_data
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass

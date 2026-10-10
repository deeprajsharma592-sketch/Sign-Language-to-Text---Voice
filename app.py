"""Streamlit Web UI Frontend for Real-Time Sign Language Recognition & Interactive Data Collection.

Features:
- Tab 1: Live Real-Time Translation (ISL Letters + Words) with voice output
- Tab 2: Interactive "Record New Word" Tool with live camera feedback, countdown, and auto-save
- Tab 3: One-Click Model Retraining directly from the UI with live accuracy reporting
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
from collections import deque

import cv2
import numpy as np
import streamlit as st
import torch
import yaml

from src.features import LandmarkExtractor, compute_sequence_features, TOTAL_RAW_DIM
from src.model import build_model
from src.postprocess import SignPostprocessor
from src.tts import TTSWorker
from src.collect import save_sample, get_existing_sample_count
from src.train import run_training

st.set_page_config(
    page_title="Sign2Text - Real-Time Sign Language Translation & Training Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #6c757d;
        margin-bottom: 15px;
    }
    .sentence-box {
        background-color: #262730;
        color: #00FF66;
        font-family: 'Courier New', monospace;
        font-size: 1.6rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 8px;
        min-height: 60px;
        display: flex;
        align-items: center;
    }
    .record-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_app_resources(config_path="config/labels.yaml", checkpoint_path="models/best_model.pt"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type", "cnn")
    input_dim = checkpoint.get("input_dim", TOTAL_RAW_DIM)
    num_classes = checkpoint.get("num_classes", len(checkpoint.get("vocabulary", [])))
    vocab = checkpoint.get("vocabulary", [])

    model = build_model(model_type, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    extractor = LandmarkExtractor()
    return cfg, model, vocab, extractor


def main():
    st.markdown('<div class="main-title">🤟 Sign2Text Translation & Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Indian Sign Language (ISL) Recognition, Interactive Word Collector & Model Studio</div>', unsafe_allow_html=True)

    cfg, model, vocab, extractor = load_app_resources()

    # Sidebar controls
    st.sidebar.header("⚙️ Settings & Controls")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.4, 0.95, float(cfg["inference"].get("confidence_threshold", 0.70)), 0.05)
    debounce_val = st.sidebar.slider("Debounce Frames", 1, 6, int(cfg["inference"].get("debounce_frames", 3)))
    cooldown_val = st.sidebar.slider("Cooldown Frames", 5, 40, int(cfg["inference"].get("cooldown_frames", 20)))
    enable_tts = st.sidebar.checkbox("🔊 Voice Output (TTS)", value=True)
    cam_index = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Vocabulary")
    st.sidebar.write(", ".join([f"`{v}`" for v in vocab]))

    # Tabs for Translation vs Recording vs Retraining
    tab_live, tab_record, tab_train = st.tabs(["🎥 Live Translation", "➕ Record New Word / Gesture", "⚡ Retrain Model Studio"])

    # ==========================================
    # TAB 1: LIVE TRANSLATION
    # ==========================================
    with tab_live:
        if "sentence" not in st.session_state:
            st.session_state.sentence = ""
        if "is_running" not in st.session_state:
            st.session_state.is_running = False

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Live Camera Feed")
            frame_placeholder = st.empty()

            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                start_btn = st.button("▶️ Start Live Camera", key="btn_start_live", use_container_width=True)
            with col_btn2:
                stop_btn = st.button("⏹️ Stop Camera", key="btn_stop_live", use_container_width=True)
            with col_btn3:
                clear_btn = st.button("🗑️ Clear Sentence", key="btn_clear_live", use_container_width=True)

            if clear_btn:
                st.session_state.sentence = ""

        with col_right:
            st.subheader("Running Translation")
            sentence_placeholder = st.empty()
            sentence_placeholder.markdown(f'<div class="sentence-box">{st.session_state.sentence or "[No words translated yet]"}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Live Recognition Status")
            pred_placeholder = st.empty()
            conf_bar_placeholder = st.empty()
            fps_placeholder = st.empty()

            st.markdown("---")
            with st.expander("📖 View ISL Sign Guide (All 26 Letters)", expanded=True):
                # Allow selecting letter group to view
                guide_choice = st.radio("Letter Range", ["A - F", "G - L", "M - R", "S - Z"], horizontal=True)
                ranges = {
                    "A - F": ["A", "B", "C", "D", "E", "F"],
                    "G - L": ["G", "H", "I", "J", "K", "L"],
                    "M - R": ["M", "N", "O", "P", "Q", "R"],
                    "S - Z": ["S", "T", "U", "V", "W", "X", "Y", "Z"],
                }
                letters_to_show = ranges[guide_choice]
                cols = st.columns(len(letters_to_show))
                for i, char in enumerate(letters_to_show):
                    ref_path = Path(f"Dataset/Letters/{char}.jpg")
                    if ref_path.exists():
                        with cols[i]:
                            st.image(str(ref_path), caption=f"Sign '{char}'", use_container_width=True)

        if start_btn:
            st.session_state.is_running = True
        if stop_btn:
            st.session_state.is_running = False

        if st.session_state.is_running:
            postprocessor = SignPostprocessor(
                confidence_threshold=conf_thresh,
                debounce_frames=debounce_val,
                cooldown_frames=cooldown_val,
                idle_label="idle"
            )
            tts = TTSWorker(enabled=enable_tts)
            seq_len = cfg["sequence"]["sequence_length"]
            buffer = deque(maxlen=seq_len)

            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(f"Cannot access webcam device index {cam_index}.")
                st.session_state.is_running = False
                return

            fps_times = deque(maxlen=20)

            while st.session_state.is_running:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab video frame.")
                    break

                frame = cv2.flip(frame, 1)
                features, has_hands, debug_info = extractor.process_frame(frame, is_mirrored=True)

                pred_label = "Waiting for hands..."
                pred_conf = 0.0

                if has_hands:
                    buffer.append(features)
                else:
                    postprocessor.reset_tracking()
                    buffer.clear()
                    pred_label = "No Hands Detected"

                if len(buffer) == seq_len:
                    seq_arr = np.array(buffer, dtype=np.float32)
                    x_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=-1).squeeze(0)
                        top_idx = probs.argmax().item()
                        pred_conf = float(probs[top_idx].item())
                        pred_label = vocab[top_idx]

                    emitted = postprocessor.process(pred_label, pred_conf)
                    if emitted:
                        st.session_state.sentence = postprocessor.get_sentence()
                        tts.speak(emitted)

                # Draw skeleton landmarks on preview
                results = debug_info.get("results")
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        extractor.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            extractor.mp_hands.HAND_CONNECTIONS
                        )

                # Update FPS
                dt = time.time() - t0
                if dt > 0:
                    fps_times.append(1.0 / dt)
                current_fps = np.mean(fps_times) if fps_times else 0.0

                # Update UI components
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                sentence_placeholder.markdown(
                    f'<div class="sentence-box">{st.session_state.sentence or "[Waiting for sign...]"}</div>',
                    unsafe_allow_html=True
                )

                pred_placeholder.markdown(f"### Detected: **{pred_label}**")
                conf_bar_placeholder.progress(min(max(pred_conf, 0.0), 1.0), text=f"Confidence: {pred_conf * 100:.1f}%")
                if len(buffer) == seq_len:
                    prob_str = " | ".join([f"{v}: {float(probs[i])*100:.0f}%" for i, v in enumerate(vocab) if v != "idle"])
                    fps_placeholder.caption(f"⚡ {current_fps:.1f} FPS | Buffer: {len(buffer)}/{seq_len}\n\nProbabilities: `{prob_str}`")
                else:
                    fps_placeholder.caption(f"⚡ {current_fps:.1f} FPS | Filling Buffer ({len(buffer)}/{seq_len})")

                time.sleep(0.01)

            cap.release()
            tts.stop()

    # ==========================================
    # TAB 2: INTERACTIVE WORD / GESTURE COLLECTOR
    # ==========================================
    with tab_record:
        st.subheader("Record Words or Gestures via Webcam")
        st.write("Record temporal gesture sequences (e.g. `Hello`, `Thank You`, `Welcome`) directly into the training dataset.")

        col_rc1, col_rc2 = st.columns([2, 3])

        with col_rc1:
            st.markdown('<div class="record-card">', unsafe_allow_html=True)
            word_to_record = st.text_input("Word / Gesture Label", value="Hello", help="e.g. Hello, Thank_You, Welcome")
            word_label_clean = word_to_record.strip().replace(" ", "_")
            num_samples = st.slider("Target Number of Samples", 10, 50, 25, 5)
            session_name = st.text_input("Session Tag", value="session_user")

            out_dir = Path(f"data/raw/{word_label_clean}")
            existing_count = get_existing_sample_count(out_dir) if word_label_clean else 0
            st.info(f"Existing recorded samples for `{word_label_clean}`: **{existing_count}**")

            start_record_btn = st.button("🔴 Record Samples Now", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_rc2:
            record_cam_placeholder = st.empty()
            record_status_placeholder = st.empty()

        if start_record_btn:
            if not word_label_clean:
                st.error("Please enter a valid label name.")
            else:
                cap_rec = cv2.VideoCapture(int(cam_index))
                if not cap_rec.isOpened():
                    st.error(f"Cannot open webcam {cam_index}.")
                else:
                    seq_len = cfg["sequence"]["sequence_length"]
                    samples_captured = 0

                    record_status_placeholder.info(f"Starting recording for '{word_label_clean}'! Prepare your gesture...")
                    time.sleep(1.0)

                    while samples_captured < num_samples:
                        # 1. Countdown 2 seconds
                        t_count_start = time.time()
                        while time.time() - t_count_start < 2.0:
                            rem = 2.0 - (time.time() - t_count_start)
                            ret, frame = cap_rec.read()
                            if ret:
                                frame = cv2.flip(frame, 1)
                                cv2.putText(frame, f"Sample {samples_captured+1}/{num_samples} in {rem:.1f}s",
                                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                                record_cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                            time.sleep(0.02)

                        # 2. Capture 30 frames
                        captured_frames = []
                        while len(captured_frames) < seq_len:
                            ret, frame = cap_rec.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            features, has_hands, _ = extractor.process_frame(frame, is_mirrored=True)
                            captured_frames.append(features)

                            cv2.putText(frame, f"RECORDING! ({len(captured_frames)}/{seq_len})",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            record_cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                            time.sleep(0.015)

                        # 3. Save sequence
                        seq_arr = np.array(captured_frames, dtype=np.float32)
                        meta = {
                            "label": word_label_clean,
                            "session": session_name,
                            "timestamp": time.time(),
                            "sequence_length": seq_len
                        }
                        save_sample(out_dir, seq_arr, meta)
                        samples_captured += 1
                        record_status_placeholder.success(f"Captured sample {samples_captured}/{num_samples} for '{word_label_clean}'!")
                        time.sleep(0.5)

                    cap_rec.release()

                    # Add word to config/labels.yaml if not present
                    with open("config/labels.yaml", "r") as f:
                        cur_cfg = yaml.safe_load(f)

                    if word_label_clean not in cur_cfg["vocabulary"]:
                        cur_cfg["vocabulary"].append(word_label_clean)
                        with open("config/labels.yaml", "w") as f:
                            yaml.dump(cur_cfg, f, default_flow_style=False)
                        st.success(f"Added `{word_label_clean}` to vocabulary in `config/labels.yaml`!")

                    st.balloons()
                    st.success(f"Finished recording {num_samples} samples for '{word_label_clean}'! Go to the 'Retrain Model Studio' tab to train.")

    # ==========================================
    # TAB 3: ONE-CLICK MODEL RETRAINING STUDIO
    # ==========================================
    with tab_train:
        st.subheader("Model Retraining Studio")
        st.write("Train the neural network on all available dataset classes (Letters + Recorded Words).")

        raw_dirs = sorted([d.name for d in Path("data/raw").iterdir() if d.is_dir()])
        st.write("Classes currently available in `data/raw/`:")
        badge_cols = st.columns(len(raw_dirs) if raw_dirs else 1)
        for i, d in enumerate(raw_dirs):
            with badge_cols[i % len(badge_cols)]:
                count = get_existing_sample_count(Path("data/raw") / d)
                st.metric(label=f"Class: {d}", value=f"{count} samples")

        st.markdown("---")
        train_model_choice = st.selectbox("Model Architecture", ["cnn (Fast Temporal 1D-CNN)", "gru (Recurrent GRU)"])
        model_arch = "cnn" if "cnn" in train_model_choice else "gru"

        if st.button("🚀 Retrain Model Now", key="btn_retrain_studio", use_container_width=True):
            with st.spinner("Training model on CPU... Please wait ~10-20 seconds..."):
                # Update config/labels.yaml vocabulary to match all classes in data/raw
                with open("config/labels.yaml", "r") as f:
                    cur_cfg = yaml.safe_load(f)

                # Ensure idle is first
                all_classes = ["idle"] + [d for d in raw_dirs if d != "idle"]
                cur_cfg["vocabulary"] = all_classes
                with open("config/labels.yaml", "w") as f:
                    yaml.dump(cur_cfg, f, default_flow_style=False)

                # Execute training
                metrics = run_training(config_path="config/labels.yaml", override_model=model_arch)

                st.cache_resource.clear()
                st.success(f"Training Complete! New validation accuracy: **{metrics['val_accuracy']*100:.1f}%**")

                cm_path = Path(f"reports/confusion_matrix_{model_arch}.png")
                if cm_path.exists():
                    st.image(str(cm_path), caption="Updated Confusion Matrix", use_container_width=True)

                st.info("The new model is now active! Switch to 'Live Translation' tab and click 'Start Live Camera'.")


if __name__ == "__main__":
    main()

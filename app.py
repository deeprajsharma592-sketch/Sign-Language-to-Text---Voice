import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Sign Language to Text", layout="centered")
st.title("🤟 Real-Time Sign Language Recognition")
st.write("Allow webcam access and make gestures in front of the camera.")

# Load model if present in repo
model = None
model_path = "model.p"  # change if your pickled model has a different name
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict.get("model", model_dict)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

class SignDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            if model is not None and len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = str(prediction[0])
                cv2.putText(
                    img,
                    predicted_char,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

        return img

webrtc_streamer(key="sign-lang", video_transformer_factory=SignDetector)

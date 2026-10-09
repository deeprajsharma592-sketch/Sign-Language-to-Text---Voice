"""Feature extraction and landmark normalization for sign language recognition.

Handles:
- MediaPipe Hands extraction (up to 2 hands)
- Canonical left/right ordering
- Wrist-relative translation normalization
- Hand-scale invariance (wrist to middle-finger knuckle distance)
- Missing hand zero-padding (126 dimensions total)
- Optional velocity (frame-to-frame delta) calculation
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
HAND_FEATURE_DIM = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
TOTAL_RAW_DIM = HAND_FEATURE_DIM * 2  # 126


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a single hand's 21 landmarks (shape: [21, 3]).

    1. Translate wrist (landmark 0) to origin (0, 0, 0).
    2. Scale by Euclidean distance between wrist (0) and middle finger MCP (9).
       If scale is near zero (e.g., degenerate detection), use 1.0 to avoid division by zero.
    """
    if landmarks.shape != (NUM_LANDMARKS_PER_HAND, COORDS_PER_LANDMARK):
        raise ValueError(f"Expected shape ({NUM_LANDMARKS_PER_HAND}, {COORDS_PER_LANDMARK}), got {landmarks.shape}")

    wrist = landmarks[WRIST_INDEX]
    centered = landmarks - wrist

    # Distance from wrist to middle finger knuckle (MCP)
    mcp = centered[MIDDLE_MCP_INDEX]
    scale = float(np.linalg.norm(mcp))

    if scale < 1e-6:
        # Avoid division by zero
        scale = 1.0

    return (centered / scale).astype(np.float32)


def assemble_frame_features(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray]
) -> np.ndarray:
    """Assemble a 126-dimensional feature vector for a single frame.

    Order:
      - Indices 0..62: Left hand (normalized, or zeros if missing)
      - Indices 63..125: Right hand (normalized, or zeros if missing)
    """
    left_vec = (
        normalize_hand_landmarks(left_hand).flatten()
        if left_hand is not None
        else np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    )

    right_vec = (
        normalize_hand_landmarks(right_hand).flatten()
        if right_hand is not None
        else np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    )

    return np.concatenate([left_vec, right_vec], axis=0)


def compute_sequence_features(
    raw_sequence: np.ndarray,
    include_velocity: bool = False
) -> np.ndarray:
    """Take a sequence of shape (T, 126) and optionally append velocities.

    If include_velocity is True:
      - delta_t = sequence[t] - sequence[t - 1] (delta_0 = 0)
      - Returns shape (T, 252)
    Otherwise returns shape (T, 126).
    """
    if raw_sequence.ndim != 2 or raw_sequence.shape[1] != TOTAL_RAW_DIM:
        raise ValueError(f"Expected shape (T, {TOTAL_RAW_DIM}), got {raw_sequence.shape}")

    if not include_velocity:
        return raw_sequence.astype(np.float32)

    t, dim = raw_sequence.shape
    deltas = np.zeros_like(raw_sequence)
    if t > 1:
        deltas[1:] = raw_sequence[1:] - raw_sequence[:-1]

    return np.concatenate([raw_sequence, deltas], axis=-1).astype(np.float32)


class LandmarkExtractor:
    """Extracts and normalizes hand landmarks from RGB/BGR frames using MediaPipe Hands."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if mp is None:
            raise ImportError("MediaPipe is not installed. Please install mediapipe to use LandmarkExtractor.")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        is_mirrored: bool = True
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Process a BGR camera frame and extract the 126-dim normalized feature vector.

        Args:
            frame_bgr: BGR image from OpenCV.
            is_mirrored: True if the frame was flipped horizontally (cv2.flip(frame, 1)).
                Note on MediaPipe Handedness:
                MediaPipe assigns "Left" / "Right" from the camera's perspective.
                When an image is mirrored for user preview, a physical right hand appears
                on the right side of the screen and MediaPipe's raw label reflects the
                mirrored image. We map handedness so that canonical Left is always the
                signer's actual left hand, and Right is the signer's actual right hand.

        Returns:
            features: 126-dim np.ndarray (float32).
            has_hands: bool indicating if at least one hand was detected.
            debug_info: dict with raw landmark coordinates and hand labels for drawing.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_hand: Optional[np.ndarray] = None
        right_hand: Optional[np.ndarray] = None
        detected_hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                # Adjust for mirror flip:
                # If mirrored, MediaPipe's "Left" is actually the user's Right hand, and vice versa.
                actual_label = label
                if is_mirrored:
                    actual_label = "Right" if label == "Left" else "Left"

                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                    dtype=np.float32
                )
                if is_mirrored:
                    # The frame was horizontally flipped with cv2.flip(frame, 1)
                    # Flip x back so landmark geometry matches unmirrored training orientation
                    coords[:, 0] = 1.0 - coords[:, 0]

                if actual_label == "Left" and left_hand is None:
                    left_hand = coords
                elif actual_label == "Right" and right_hand is None:
                    right_hand = coords

                detected_hands.append({
                    "raw_label": label,
                    "actual_label": actual_label,
                    "score": score,
                    "landmarks": hand_lms
                })

        features = assemble_frame_features(left_hand, right_hand)
        has_hands = (left_hand is not None) or (right_hand is not None)

        return features, has_hands, {
            "results": results,
            "detected_hands": detected_hands,
            "has_left": left_hand is not None,
            "has_right": right_hand is not None
        }

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()

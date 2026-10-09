"""Demo script to verify feature extraction from webcam or synthetic frames.

Usage:
    python -m src.features_demo [--mock]
"""

import argparse
import time
import cv2
import numpy as np
from src.features import LandmarkExtractor, TOTAL_RAW_DIM


def run_demo(use_mock: bool = False, frames: int = 60):
    print("=" * 60)
    print("Starting Feature Extraction Demo")
    print(f"Mode: {'Synthetic Mock Generator' if use_mock else 'Live Webcam'}")
    print(f"Output Feature Dimension: {TOTAL_RAW_DIM}")
    print("=" * 60)

    if use_mock:
        extractor = LandmarkExtractor()
        for i in range(frames):
            # Generate a blank image
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a white rectangle as a dummy visual
            cv2.rectangle(mock_frame, (100, 100), (300, 300), (255, 255, 255), -1)
            features, has_hands, info = extractor.process_frame(mock_frame)
            print(f"Frame {i+1:02d}: hands_detected={has_hands}, feature_shape={features.shape}, min={features.min():.2f}, max={features.max():.2f}")
            time.sleep(0.02)
        extractor.close()
        print("\nMock demo finished successfully!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Could not open webcam at index 0. Re-running with synthetic frames...")
        run_demo(use_mock=True, frames=30)
        return

    extractor = LandmarkExtractor()
    print("Webcam opened. Press 'q' to exit. Showing live normalized features...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Mirror frame
            frame = cv2.flip(frame, 1)
            features, has_hands, info = extractor.process_frame(frame, is_mirrored=True)

            status_str = f"Hands: {'YES' if has_hands else 'NO'} | Left: {info['has_left']} | Right: {info['has_right']}"
            cv2.putText(frame, status_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if has_hands:
                # Show first 4 normalized coords
                sample_feat = ", ".join([f"{v:.2f}" for v in features[:4]])
                cv2.putText(frame, f"Feat[0:4]: [{sample_feat}...]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("Sign2Text - Landmark Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Feature Extractor Demo")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic frames instead of webcam")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to test in mock mode")
    args = parser.parse_args()
    run_demo(use_mock=args.mock, frames=args.frames)

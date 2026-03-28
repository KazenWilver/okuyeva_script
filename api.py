"""
Zero Barreiras — API de Streaming MJPEG + Classificação Dinâmica (LSTM)

Arquitectura:
  - MediaPipe Holistic (1662 features/frame) para tracking completo
  - Buffer circular de 30 frames para classificação temporal
  - LSTM bidireccional (PyTorch GPU) para gestos dinâmicos
  - MJPEG streaming directo para o browser
"""
import os
import time
import cv2 as cv
import threading
import numpy as np
from collections import deque

import mediapipe as mp
try:
    if not hasattr(mp, 'solutions'):
        import importlib
        mp.solutions = importlib.import_module('mediapipe.python.solutions')
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from features_v2 import (
    create_holistic_detector, process_frame, extract_keypoints,
    has_hands, draw_minimal_landmarks, NUM_FEATURES_V2,
)

# ── Load LSTM Classifier ──
try:
    from model.sequence_classifier.sequence_classifier import SequenceClassifier
    seq_classifier = SequenceClassifier(
        model_path='model/sequence_classifier/sequence_classifier.pt',
        labels_path='model/sequence_classifier/sequence_classifier_label.csv',
    )
    LSTM_AVAILABLE = seq_classifier.is_loaded
    if LSTM_AVAILABLE:
        print(f"[LSTM] Loaded successfully")
    else:
        print(f"[LSTM] Not trained yet")
except Exception as e:
    seq_classifier = None
    LSTM_AVAILABLE = False
    print(f"[LSTM] Not available: {e}")

# Sequence classifier labels
seq_labels = []
SEQ_LABEL_CSV = 'model/sequence_classifier/sequence_classifier_label.csv'
if os.path.exists(SEQ_LABEL_CSV):
    with open(SEQ_LABEL_CSV, encoding='utf-8-sig') as f:
        seq_labels = [row.strip() for row in f if row.strip()]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIANCA_MINIMA = 0.40  # Lowered from 0.70 — model with few classes needs less
SEQUENCE_LENGTH = 30
CLASSIFY_EVERY_N = 10  # Run LSTM every N new frames (sliding window)

# ── Shared State ──
global_gesture = ""
global_confidence = 0.0
global_gesture_type = "none"  # "dynamic" or "none"
video_initialized = False


class CameraState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()


cam_state = CameraState()


def capture_thread():
    global global_gesture, global_confidence, global_gesture_type, video_initialized

    # Probe cameras, skip virtual ones
    cap = None
    for i in range(3):
        temp_cap = cv.VideoCapture(i, cv.CAP_DSHOW)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                if cap is not None:
                    cap.release()
                cap = temp_cap
                print(f"[CAM] Using Camera Index: {i}")
                if i == 1:
                    break

    if cap is None or not cap.isOpened():
        print("[CAM] Camera Initialization Failed")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)

    # MediaPipe Holistic
    holistic = create_holistic_detector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Sequence buffer for LSTM
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frames_since_classify = 0
    last_debug_time = 0

    video_initialized = True
    print("[API] Capture thread started")

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = cv.flip(image, 1)

        # ── Holistic Processing ──
        results = process_frame(image, holistic)
        hands_visible = has_hands(results)

        # Draw landmarks on image
        draw_minimal_landmarks(image, results)

        # Extract holistic keypoints (1662 features)
        keypoints = extract_keypoints(results)

        # ── LSTM Dynamic Classification ──
        if LSTM_AVAILABLE and hands_visible:
            sequence_buffer.append(keypoints)
            frames_since_classify += 1

            if len(sequence_buffer) >= SEQUENCE_LENGTH and frames_since_classify >= CLASSIFY_EVERY_N:
                frames_since_classify = 0
                seq_array = np.array(list(sequence_buffer), dtype=np.float32)
                class_id, confidence = seq_classifier(seq_array)
                label = seq_classifier.get_label(class_id)

                # Debug logging (every 2 seconds max)
                now = time.time()
                if now - last_debug_time > 2.0:
                    print(f"[LSTM] Predicted: {label} ({confidence*100:.1f}%) | Threshold: {CONFIANCA_MINIMA*100:.0f}%")
                    last_debug_time = now

                if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
                    global_gesture = label
                    global_confidence = confidence
                    global_gesture_type = "dynamic"
                else:
                    global_gesture = ""
                    global_confidence = 0.0
                    global_gesture_type = "none"

        elif not hands_visible:
            sequence_buffer.clear()
            frames_since_classify = 0
            global_gesture = ""
            global_confidence = 0.0
            global_gesture_type = "none"

        # ── Encode JPEG ──
        _, buffer = cv.imencode('.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 80])

        with cam_state.lock:
            cam_state.frame = buffer.tobytes()


# Start background thread
thread = threading.Thread(target=capture_thread, daemon=True)
thread.start()


def frame_generator():
    while True:
        with cam_state.lock:
            frame = cam_state.frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/tracking_state")
def tracking_state():
    return {
        "gesture": global_gesture,
        "confidence": global_confidence,
        "gesture_type": global_gesture_type,
        "camera_active": video_initialized,
        "lstm_available": LSTM_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

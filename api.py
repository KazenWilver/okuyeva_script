"""
Zero Barreiras — API de Streaming MJPEG + Classificação Dinâmica (LSTM)

Arquitectura:
  - MediaPipe Holistic (1662 features/frame) para tracking completo
  - Buffer circular de 30 frames para classificação temporal
  - LSTM bidireccional (PyTorch GPU) para gestos dinâmicos
  - Fallback para MLP estático se LSTM não estiver treinado
  - MJPEG streaming directo para o browser
"""
import os
import time
import cv2 as cv
import threading
import numpy as np
from collections import deque, Counter

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

# Try to load sequence classifier (LSTM)
try:
    from model.sequence_classifier.sequence_classifier import SequenceClassifier
    seq_classifier = SequenceClassifier(
        model_path='model/sequence_classifier/sequence_classifier.pt',
        labels_path='model/sequence_classifier/sequence_classifier_label.csv',
    )
    LSTM_AVAILABLE = seq_classifier.is_loaded
    print(f"[LSTM] {'Loaded' if LSTM_AVAILABLE else 'Not trained yet — using static fallback'}")
except Exception as e:
    seq_classifier = None
    LSTM_AVAILABLE = False
    print(f"[LSTM] Not available: {e}")

# Try to load static classifier (MLP fallback)
try:
    from model import KeyPointClassifier
    from features import (
        NUM_FEATURES, MotionTracker, FeatureSmoother,
        extract_all_features, calc_bounding_rect, draw_body_refs,
    )
    static_classifier = KeyPointClassifier(model_path='model/keypoint_classifier/keypoint_classifier.pkl')
    STATIC_AVAILABLE = True

    static_labels = []
    LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            static_labels = [row.strip() for row in f if row.strip()]
    print(f"[STATIC] Loaded ({len(static_labels)} classes)")
except Exception as e:
    static_classifier = None
    STATIC_AVAILABLE = False
    static_labels = []
    print(f"[STATIC] Not available: {e}")

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

CONFIANCA_MINIMA = 0.70
SEQUENCE_LENGTH = 30
CLASSIFY_EVERY_N = 10  # Run LSTM every N new frames (sliding window)

# ── Shared State ──
global_gesture = ""
global_confidence = 0.0
global_gesture_type = "none"  # "static", "dynamic", or "none"
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

    # MediaPipe Holistic (replaces separate Hands + Pose)
    holistic = create_holistic_detector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Static classifier state (fallback)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    static_historico = deque(maxlen=10)
    static_smoother = FeatureSmoother(alpha=0.65) if STATIC_AVAILABLE else None
    static_motion = MotionTracker() if STATIC_AVAILABLE else None
    static_prev_wrists = None

    # Also create separate hands/pose for static features if needed
    if STATIC_AVAILABLE:
        hands_detector = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.65, min_tracking_confidence=0.4
        )
        pose_detector = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5
        )
    else:
        hands_detector = None
        pose_detector = None

    # Sequence buffer for LSTM
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frames_since_classify = 0

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

                if confidence > CONFIANCA_MINIMA:
                    label = seq_classifier.get_label(class_id)
                    if label.lower() != "neutro":
                        global_gesture = label
                        global_confidence = confidence
                        global_gesture_type = "dynamic"
                    else:
                        global_gesture = ""
                        global_confidence = 0.0
                        global_gesture_type = "none"
                else:
                    global_gesture = ""
                    global_confidence = 0.0
                    global_gesture_type = "none"

        # ── Static MLP Fallback ──
        elif STATIC_AVAILABLE and not LSTM_AVAILABLE and hands_visible:
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands_detector.process(rgb)
            pose_results = pose_detector.process(rgb)

            if hand_results.multi_hand_landmarks is not None:
                features, wrists, ordered = extract_all_features(
                    image, hand_results, pose_results, static_prev_wrists, static_motion)
                static_prev_wrists = wrists

                if features and len(features) == NUM_FEATURES:
                    smoothed = static_smoother.smooth(features)
                    sign_id, confidence = static_classifier(smoothed)
                    label = static_labels[sign_id] if sign_id < len(static_labels) else f"Classe {sign_id}"

                    global_confidence = float(confidence)
                    if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
                        static_historico.append(label)
                    else:
                        static_historico.append("_neutro")
                else:
                    static_historico.append("_neutro")
                    global_confidence = 0.0
            else:
                static_historico.append("_neutro")
                global_confidence = 0.0

            if len(static_historico) >= 3:
                contagem = Counter(static_historico)
                mais_comum, freq = contagem.most_common(1)[0]
                if mais_comum != "_neutro" and freq >= 6:
                    global_gesture = mais_comum
                    global_gesture_type = "static"
                else:
                    global_gesture = ""
                    global_gesture_type = "none"
            else:
                global_gesture = ""
                global_gesture_type = "none"

        elif not hands_visible:
            if not LSTM_AVAILABLE and STATIC_AVAILABLE:
                static_historico.clear()
                if static_motion:
                    static_motion.reset()
                if static_smoother:
                    static_smoother.reset()
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
        "static_available": STATIC_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

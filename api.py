"""
Okuyeva — API de Streaming MJPEG + Classificação Dinâmica (LSTM)

Arquitectura:
  - MediaPipe Holistic (1662 features/frame) para tracking completo
  - Buffer circular de 30 frames para classificação temporal
  - LSTM bidireccional (PyTorch GPU) para gestos dinâmicos
  - Suavização temporal: voto maioritário sobre últimas N predições
  - Gesto "hold": mantém o gesto visível por pelo menos 1.5s
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIANCA_MINIMA = 0.45
SEQUENCE_LENGTH = 30
CLASSIFY_EVERY_N = 3      # Run LSTM more frequently for better responsiveness
SMOOTHING_WINDOW = 5      # Vote over last N predictions
GESTURE_HOLD_TIME = 1.5   # Keep gesture visible for at least this many seconds

# MUDE AQUI O NÚMERO DA CÂMARA (0 = Webcam do PC, 1 ou 2 = Iriun Webcam/Externa)
CAMERA_INDEX = 2

# ── Shared State ──
global_gesture = ""
global_confidence = 0.0
global_gesture_type = "none"
video_initialized = False


class CameraState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()


cam_state = CameraState()


def capture_thread():
    global global_gesture, global_confidence, global_gesture_type, video_initialized

    print(f"[CAM] Inicializando Camara {CAMERA_INDEX}...")
    cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
    
    # Tenta inicializar sem DSHOW se falhar
    if not cap.isOpened():
        cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERRO] Falha ao abrir a câmara {CAMERA_INDEX}. Edite a variavel CAMERA_INDEX no codigo.")
        return

    # Mesma resolução que a coleta para consistência de detecção
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
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

    # ── Smoothing state ──
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    last_gesture_time = 0     # When the last gesture was first detected
    active_gesture = ""       # Currently displayed gesture
    active_confidence = 0.0

    video_initialized = True
    print("[API] Capture thread started")
    print(f"[API] Smoothing: {SMOOTHING_WINDOW} votes | Hold: {GESTURE_HOLD_TIME}s | Min confidence: {CONFIANCA_MINIMA}")

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = cv.flip(image, 1)

        # ── Holistic Processing ──
        results = process_frame(image, holistic)
        hands_visible = has_hands(results)

        # Draw landmarks
        draw_minimal_landmarks(image, results)

        # Extract holistic keypoints (1662 features)
        keypoints = extract_keypoints(results)

        # ── LSTM Dynamic Classification ──
        now = time.time()

        if LSTM_AVAILABLE:
            # Always accumulate keypoints (matches collection behavior)
            sequence_buffer.append(keypoints)
            frames_since_classify += 1

            # Only classify when hands are visible and buffer is full
            if hands_visible and len(sequence_buffer) >= SEQUENCE_LENGTH and frames_since_classify >= CLASSIFY_EVERY_N:
                frames_since_classify = 0
                seq_array = np.array(list(sequence_buffer), dtype=np.float32)
                class_id, confidence = seq_classifier(seq_array)
                label = seq_classifier.get_label(class_id)

                # Add to prediction history (only meaningful predictions)
                if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
                    prediction_history.append((label, confidence))
                else:
                    prediction_history.append(("", 0.0))

                # ── Majority Vote Smoothing ──
                if prediction_history:
                    valid_preds = [(lbl, conf) for lbl, conf in prediction_history if lbl]
                    
                    if len(valid_preds) >= 2:
                        label_counts = Counter(lbl for lbl, _ in valid_preds)
                        best_label, count = label_counts.most_common(1)[0]
                        
                        if count >= len(prediction_history) / 2:
                            avg_conf = np.mean([conf for lbl, conf in valid_preds if lbl == best_label])
                            active_gesture = best_label
                            active_confidence = avg_conf
                            last_gesture_time = now
                        
                    elif not valid_preds:
                        if now - last_gesture_time > GESTURE_HOLD_TIME:
                            active_gesture = ""
                            active_confidence = 0.0

                # Debug logging
                if now - last_debug_time > 2.0:
                    raw_label = label if confidence > CONFIANCA_MINIMA else "---"
                    print(f"[LSTM] Raw: {raw_label} ({confidence*100:.1f}%) | "
                          f"Smoothed: {active_gesture or '---'} ({active_confidence*100:.0f}%) | "
                          f"History: {len(prediction_history)}")
                    last_debug_time = now

            elif not hands_visible:
                # Don't clear buffer! Just respect hold time for display
                if now - last_gesture_time > GESTURE_HOLD_TIME:
                    active_gesture = ""
                    active_confidence = 0.0
                    prediction_history.clear()

        # Update global state
        if active_gesture:
            global_gesture = active_gesture
            global_confidence = active_confidence
            global_gesture_type = "dynamic"
        else:
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

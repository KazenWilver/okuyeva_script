"""
Zero Barreiras — API de Streaming MJPEG (single-thread, tudo sincronizado)
Câmara + MediaPipe + Classificador correm no mesmo ciclo = traços 100% colados às mãos.
"""
import os
import time
import cv2 as cv
import threading
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

from model import KeyPointClassifier
from features import (
    NUM_FEATURES, MotionTracker, FeatureSmoother,
    extract_all_features, calc_bounding_rect, draw_body_refs,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

model_path = 'model/keypoint_classifier/keypoint_classifier.pkl'
LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'

keypoint_classifier = KeyPointClassifier(model_path=model_path)
labels = []
if os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, encoding='utf-8-sig') as f:
        labels = [row.strip() for row in f if row.strip()]

CONFIANCA_MINIMA = 0.70
JANELA_SUAVIZACAO = 10
MAIORIA_MINIMA = 6

# Shared state
global_gesture = ""
global_confidence = 0.0
video_initialized = False

class CameraState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

cam_state = CameraState()

def capture_thread():
    global global_gesture, global_confidence, video_initialized

    # Probe cameras, skip virtual ones (Iriun, OBS)
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

    hands_detector = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=0.65, min_tracking_confidence=0.4
    )
    pose_detector = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5
    )

    historico = deque(maxlen=JANELA_SUAVIZACAO)
    smoother = FeatureSmoother(alpha=0.65)
    motion = MotionTracker()
    prev_wrists = None

    video_initialized = True

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = cv.flip(image, 1)

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = hands_detector.process(rgb)
        pose_results = pose_detector.process(rgb)
        rgb.flags.writeable = True

        draw_body_refs(image, pose_results)

        if hand_results.multi_hand_landmarks is not None:
            features, wrists, ordered = extract_all_features(
                image, hand_results, pose_results, prev_wrists, motion)
            prev_wrists = wrists

            for hand_data in ordered:
                brect = calc_bounding_rect(image, hand_data['landmarks'])
                mp_draw.draw_landmarks(image, hand_data['landmarks'], mp_hands.HAND_CONNECTIONS)
                cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                hand_side = hand_data['label']
                cv.rectangle(image, (brect[0], brect[1] - 22), (brect[2], brect[1]), (0, 0, 0), -1)
                cv.putText(image, hand_side, (brect[0] + 5, brect[1] - 4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            if features and len(features) == NUM_FEATURES:
                smoothed = smoother.smooth(features)
                sign_id, confidence = keypoint_classifier(smoothed)
                label = labels[sign_id] if sign_id < len(labels) else f"Classe {sign_id}"

                global_confidence = float(confidence)
                if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
                    historico.append(label)
                else:
                    historico.append("_neutro")
            else:
                historico.append("_neutro")
                global_confidence = 0.0
        else:
            historico.append("_neutro")
            global_confidence = 0.0
            motion.reset()
            smoother.reset()

        if len(historico) >= 3:
            contagem = Counter(historico)
            mais_comum, freq = contagem.most_common(1)[0]
            if mais_comum != "_neutro" and freq >= MAIORIA_MINIMA:
                global_gesture = mais_comum
            else:
                global_gesture = ""
        else:
            global_gesture = ""

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
        "camera_active": video_initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

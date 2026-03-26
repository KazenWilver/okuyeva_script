"""
Zero Barreiras - API REST + WebSocket
Serve o modelo de reconhecimento de LGA para o frontend React.

Endpoints:
  GET  /api/health          - Status da API
  GET  /api/labels          - Lista de gestos suportados
  POST /api/predict         - Classificar landmarks (HTTP)
  POST /api/sentence/clear  - Limpar frase da sessao
  POST /api/transcript/save - Guardar transcricao
  WS   /ws/predict          - Classificar em tempo real (WebSocket)

Iniciar:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import pickle
import time
import csv
from datetime import datetime
from collections import deque, Counter
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import (
    NUM_FEATURES, NUM_HAND_FEATURES, NUM_BODY_FEATURES, NUM_MOTION_FEATURES,
    MotionTracker, FeatureSmoother,
)

# ---------------------------------------------------------------------------
#  App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Zero Barreiras API",
    description="Tradução de Língua Gestual Angolana para texto",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Load model & labels
# ---------------------------------------------------------------------------

MODEL_PATH = 'model/keypoint_classifier/keypoint_classifier.pkl'
LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'
PASTA_CONSULTAS = 'consultas'

CONFIANCA_MINIMA = 0.70
JANELA_SUAVIZACAO = 10
MAIORIA_MINIMA = 6

model = None
labels = []


def load_model():
    global model, labels
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f if row.strip()]


load_model()

# ---------------------------------------------------------------------------
#  Session management (per-client state)
# ---------------------------------------------------------------------------

sessions = {}


class Session:
    def __init__(self):
        self.motion = MotionTracker()
        self.smoother = FeatureSmoother(alpha=0.65)
        self.historico = deque(maxlen=JANELA_SUAVIZACAO)
        self.frase = []
        self.ultimo_gesto = ""
        self.transcricao = []
        self.prev_wrists = None
        self.last_active = time.time()

    def reset_tracking(self):
        self.motion.reset()
        self.smoother.reset()
        self.prev_wrists = None


def get_session(session_id: str) -> Session:
    if session_id not in sessions:
        sessions[session_id] = Session()
    sessions[session_id].last_active = time.time()
    return sessions[session_id]


# ---------------------------------------------------------------------------
#  Landmark processing (convert frontend JSON to 144 features)
# ---------------------------------------------------------------------------

def process_hand_landmarks(hands_data, image_w, image_h):
    """Convert hand landmarks from frontend format to pixel coordinates.

    hands_data: list of {landmarks: [[x,y,z],...], handedness: "Right"|"Left"}
    Returns: (hand_features_84, wrist_positions) or (None, [])
    """
    if not hands_data:
        return None, []

    hands = []
    for hand in hands_data:
        lms = hand.get('landmarks', [])
        label = hand.get('handedness', 'Right')
        pts = []
        for lm in lms:
            x = min(int(lm[0] * image_w), image_w - 1)
            y = min(int(lm[1] * image_h), image_h - 1)
            pts.append([x, y])
        if len(pts) == 21:
            hands.append({'pts': pts, 'label': label, 'wrist': (pts[0][0], pts[0][1])})

    if not hands:
        return None, []

    hands.sort(key=lambda h: h['wrist'][0])

    wrist_positions = [h['wrist'] for h in hands]
    base_x = hands[0]['pts'][0][0]
    base_y = hands[0]['pts'][0][1]

    flat = []
    for point in hands[0]['pts']:
        flat.append(point[0] - base_x)
        flat.append(point[1] - base_y)

    if len(hands) >= 2:
        for point in hands[1]['pts']:
            flat.append(point[0] - base_x)
            flat.append(point[1] - base_y)
    else:
        flat.extend([0.0] * 42)

    max_val = max(map(abs, flat)) if flat else 1
    if max_val > 0:
        flat = [v / max_val for v in flat]

    return flat, wrist_positions


def process_pose_landmarks(pose_data, wrist_positions, image_w, image_h):
    """Convert pose landmarks from frontend format to body+face features (44).

    pose_data: {landmarks: [[x,y,z],...]} with 33 points
    Returns: body_features_44
    """
    if not pose_data or 'landmarks' not in pose_data:
        return [0.0] * NUM_BODY_FEATURES

    lms = pose_data['landmarks']
    if len(lms) < 17:
        return [0.0] * NUM_BODY_FEATURES

    def px(idx):
        return (lms[idx][0] * image_w, lms[idx][1] * image_h)

    l_shoulder = px(11)
    r_shoulder = px(12)
    shoulder_w = max(abs(l_shoulder[0] - r_shoulder[0]), 1.0)
    sc = ((l_shoulder[0] + r_shoulder[0]) / 2,
          (l_shoulder[1] + r_shoulder[1]) / 2)

    nose = px(0)
    mouth = ((px(9)[0] + px(10)[0]) / 2, (px(9)[1] + px(10)[1]) / 2)
    forehead = ((px(1)[0] + px(4)[0]) / 2, (px(1)[1] + px(4)[1]) / 2)
    chin = (mouth[0], mouth[1] + (mouth[1] - nose[1]) * 0.6)

    face_points = [nose, mouth, sc, forehead, px(2), px(5), px(7), px(8), chin]

    features = []

    for i in range(2):
        if i < len(wrist_positions):
            wx, wy = wrist_positions[i]
        else:
            wx, wy = sc

        for ref in face_points:
            features.append((wx - ref[0]) / shoulder_w)
            features.append((wy - ref[1]) / shoulder_w)

    le, re = px(13), px(14)
    ls, rs = l_shoulder, r_shoulder
    lw, rw = px(15), px(16)

    features.append((le[0] - ls[0]) / shoulder_w)
    features.append((le[1] - ls[1]) / shoulder_w)
    features.append((re[0] - rs[0]) / shoulder_w)
    features.append((re[1] - rs[1]) / shoulder_w)
    features.append((lw[0] - le[0]) / shoulder_w)
    features.append((lw[1] - le[1]) / shoulder_w)
    features.append((rw[0] - re[0]) / shoulder_w)
    features.append((rw[1] - re[1]) / shoulder_w)

    return features


def build_body_refs_from_pose(pose_data, image_w, image_h):
    """Build body_refs dict for MotionTracker from pose JSON."""
    if not pose_data or 'landmarks' not in pose_data:
        return None

    lms = pose_data['landmarks']
    if len(lms) < 17:
        return None

    def px(idx):
        return (lms[idx][0] * image_w, lms[idx][1] * image_h)

    l_shoulder = px(11)
    r_shoulder = px(12)
    shoulder_w = max(abs(l_shoulder[0] - r_shoulder[0]), 1.0)

    return {
        'nose': px(0),
        'mouth': ((px(9)[0] + px(10)[0]) / 2, (px(9)[1] + px(10)[1]) / 2),
        'shoulder_center': ((l_shoulder[0] + r_shoulder[0]) / 2,
                            (l_shoulder[1] + r_shoulder[1]) / 2),
        'shoulder_w': shoulder_w,
    }


def classify(features, session: Session):
    """Run classification with smoothing and majority voting.
    Returns (gesture_name, gesture_id, confidence, sentence, is_new_gesture).
    """
    if model is None:
        return "", -1, 0.0, session.frase, False

    smoothed = session.smoother.smooth(features)

    probs = model.predict_proba([smoothed])[0]
    idx = np.argmax(probs)
    sign_id = int(model.classes_[idx])
    confidence = float(probs[idx])
    label = labels[sign_id] if sign_id < len(labels) else f"Classe {sign_id}"

    if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
        session.historico.append(label)
    else:
        session.historico.append("_neutro")

    gesto_atual = ""
    if len(session.historico) >= 3:
        contagem = Counter(session.historico)
        mais_comum, freq = contagem.most_common(1)[0]
        if mais_comum != "_neutro" and freq >= MAIORIA_MINIMA:
            gesto_atual = mais_comum

    is_new = False
    if gesto_atual and gesto_atual != session.ultimo_gesto:
        session.frase.append(gesto_atual)
        session.transcricao.append((time.time(), 'paciente', gesto_atual))
        is_new = True
    if not gesto_atual:
        session.ultimo_gesto = ""
    else:
        session.ultimo_gesto = gesto_atual

    return gesto_atual, sign_id, confidence, list(session.frase), is_new


# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------

class HandData(BaseModel):
    landmarks: list[list[float]]
    handedness: str = "Right"

class PoseData(BaseModel):
    landmarks: list[list[float]]

class PredictRequest(BaseModel):
    hands: list[HandData]
    pose: Optional[PoseData] = None
    image_width: int = 960
    image_height: int = 540
    session_id: str = "default"

class DoctorMessage(BaseModel):
    text: str
    session_id: str = "default"

class SessionId(BaseModel):
    session_id: str = "default"


# ---------------------------------------------------------------------------
#  REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_labels": len(labels),
        "num_features": NUM_FEATURES,
    }


@app.get("/api/labels")
async def get_labels():
    return {
        "labels": labels,
        "mapping": {i: label for i, label in enumerate(labels)},
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")

    session = get_session(req.session_id)

    hands_raw = [{"landmarks": h.landmarks, "handedness": h.handedness}
                 for h in req.hands]

    hand_feats, wrists = process_hand_landmarks(
        hands_raw, req.image_width, req.image_height)

    if hand_feats is None:
        session.historico.append("_neutro")
        session.reset_tracking()
        return {
            "gesture": "",
            "gesture_id": -1,
            "confidence": 0.0,
            "sentence": list(session.frase),
            "num_hands": 0,
            "is_new_gesture": False,
        }

    pose_raw = {"landmarks": req.pose.landmarks} if req.pose else None
    body_feats = process_pose_landmarks(
        pose_raw, wrists, req.image_width, req.image_height)

    body_refs = build_body_refs_from_pose(
        pose_raw, req.image_width, req.image_height)
    motion_feats = session.motion.compute(wrists, body_refs)

    all_features = hand_feats + body_feats + motion_feats

    if len(all_features) != NUM_FEATURES:
        raise HTTPException(status_code=400,
                            detail=f"Feature count {len(all_features)} != {NUM_FEATURES}")

    gesto, sign_id, conf, sentence, is_new = classify(all_features, session)

    return {
        "gesture": gesto,
        "gesture_id": sign_id,
        "confidence": round(conf, 4),
        "sentence": sentence,
        "num_hands": len(req.hands),
        "is_new_gesture": is_new,
    }


@app.post("/api/doctor/message")
async def doctor_message(msg: DoctorMessage):
    session = get_session(msg.session_id)
    session.transcricao.append((time.time(), 'medico', msg.text))
    return {"status": "ok", "transcript_length": len(session.transcricao)}


@app.post("/api/sentence/clear")
async def clear_sentence(req: SessionId):
    session = get_session(req.session_id)
    session.frase.clear()
    session.ultimo_gesto = ""
    return {"sentence": []}


@app.post("/api/transcript/save")
async def save_transcript(req: SessionId):
    session = get_session(req.session_id)
    if not session.transcricao:
        return {"path": None, "message": "Transcricao vazia"}

    os.makedirs(PASTA_CONSULTAS, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(PASTA_CONSULTAS, f"consulta_{ts}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  ZERO BARREIRAS - Transcricao de Consulta\n")
        f.write(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for t, origem, texto in session.transcricao:
            hora = datetime.fromtimestamp(t).strftime('%H:%M:%S')
            if origem == 'paciente':
                f.write(f"  [{hora}] PACIENTE (LGA): {texto}\n")
            else:
                f.write(f"  [{hora}] MEDICO (voz):   {texto}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"  Total: {len(session.transcricao)} interaccoes\n")
        f.write("=" * 60 + "\n")

    return {"path": filepath, "total_events": len(session.transcricao)}


@app.post("/api/model/reload")
async def reload_model():
    load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
#  WebSocket (real-time prediction)
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    session = Session()

    try:
        while True:
            data = await websocket.receive_json()

            hands_raw = data.get('hands', [])
            pose_raw = data.get('pose', None)
            img_w = data.get('image_width', 960)
            img_h = data.get('image_height', 540)

            hand_feats, wrists = process_hand_landmarks(hands_raw, img_w, img_h)

            if hand_feats is None:
                session.historico.append("_neutro")
                session.reset_tracking()
                await websocket.send_json({
                    "gesture": "",
                    "gesture_id": -1,
                    "confidence": 0.0,
                    "sentence": list(session.frase),
                    "num_hands": 0,
                    "is_new_gesture": False,
                })
                continue

            body_feats = process_pose_landmarks(pose_raw, wrists, img_w, img_h)
            body_refs = build_body_refs_from_pose(pose_raw, img_w, img_h)
            motion_feats = session.motion.compute(wrists, body_refs)

            all_features = hand_feats + body_feats + motion_feats

            if len(all_features) != NUM_FEATURES:
                await websocket.send_json({"error": "feature count mismatch"})
                continue

            gesto, sign_id, conf, sentence, is_new = classify(all_features, session)

            await websocket.send_json({
                "gesture": gesto,
                "gesture_id": sign_id,
                "confidence": round(conf, 4),
                "sentence": sentence,
                "num_hands": len(hands_raw),
                "is_new_gesture": is_new,
            })

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    print("\n" + "=" * 62)
    print("  ZERO BARREIRAS - API")
    print(f"  Modelo: {'OK' if model else 'NAO ENCONTRADO'}")
    print(f"  Labels: {labels}")
    print(f"  Features: {NUM_FEATURES}")
    print("=" * 62)
    print("  Docs: http://localhost:8000/docs")
    print("=" * 62 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

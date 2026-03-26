"""
Zero Barreiras - Feature Extraction (partilhado entre coleta.py e app.py)

144 features:
  84 = forma das maos (42 por mao, relativa ao pulso primario, normalizada)
  44 = contexto corporal + rosto detalhado:
       Por mao (x2 = 36): nariz, boca, ombros, testa, olho E, olho D,
                           orelha E, orelha D, queixo
       Bracos (8): cotovelos e pulsos relativos aos ombros
  16 = movimento (velocidade, aproximacao ao rosto, inter-mao)

Robusto para:
  - Sinais que tocam no nariz, testa, orelha, boca, queixo
  - Sobreposicao de dedos e maos cruzadas
  - Uma mao ou duas maos
  - Sinais dinamicos com movimento
"""
import cv2 as cv
import numpy as np

NUM_HAND_FEATURES = 84
NUM_BODY_FEATURES = 44
NUM_MOTION_FEATURES = 16
NUM_FEATURES = NUM_HAND_FEATURES + NUM_BODY_FEATURES + NUM_MOTION_FEATURES


def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    pts = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * iw), iw - 1)
        y = min(int(lm.y * ih), ih - 1)
        pts.append([x, y])
    return pts


def calc_bounding_rect(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    xs, ys = [], []
    for lm in landmarks.landmark:
        xs.append(min(int(lm.x * iw), iw - 1))
        ys.append(min(int(lm.y * ih), ih - 1))
    return [min(xs), min(ys), max(xs), max(ys)]


def _dist_sq(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


# ---------------------------------------------------------------------------
#  Hand ordering (position-based for consistency across frames)
# ---------------------------------------------------------------------------

def order_hands(hand_results, image, prev_wrists=None):
    """Order hands consistently using nearest-neighbor to previous frame."""
    if not hand_results.multi_hand_landmarks:
        return []

    hands = []
    for hand_lm, handedness in zip(
            hand_results.multi_hand_landmarks, hand_results.multi_handedness):
        pts = calc_landmark_list(image, hand_lm)
        hands.append({
            'pts': pts,
            'label': handedness.classification[0].label,
            'landmarks': hand_lm,
            'wrist': (pts[0][0], pts[0][1]),
        })

    if len(hands) == 2 and prev_wrists and len(prev_wrists) >= 2:
        d_keep = _dist_sq(hands[0]['wrist'], prev_wrists[0]) + \
                 _dist_sq(hands[1]['wrist'], prev_wrists[1])
        d_swap = _dist_sq(hands[0]['wrist'], prev_wrists[1]) + \
                 _dist_sq(hands[1]['wrist'], prev_wrists[0])
        if d_swap < d_keep:
            hands = [hands[1], hands[0]]
    else:
        hands.sort(key=lambda h: h['wrist'][0])

    return hands


# ---------------------------------------------------------------------------
#  Hand shape features (84)
# ---------------------------------------------------------------------------

def extract_hand_features(ordered_hands):
    """84 hand shape features from ordered hands.
    Returns (features_84, wrist_pixel_positions) or (None, []).
    """
    if not ordered_hands:
        return None, []

    wrist_positions = [h['wrist'] for h in ordered_hands]

    base_x = ordered_hands[0]['pts'][0][0]
    base_y = ordered_hands[0]['pts'][0][1]

    flat = []
    for point in ordered_hands[0]['pts']:
        flat.append(point[0] - base_x)
        flat.append(point[1] - base_y)

    if len(ordered_hands) >= 2:
        for point in ordered_hands[1]['pts']:
            flat.append(point[0] - base_x)
            flat.append(point[1] - base_y)
    else:
        flat.extend([0.0] * 42)

    max_val = max(map(abs, flat)) if flat else 1
    if max_val > 0:
        flat = [v / max_val for v in flat]

    return flat, wrist_positions


# ---------------------------------------------------------------------------
#  Body + face context features (44)
# ---------------------------------------------------------------------------

# Pose landmark indices:
#  0 = nose
#  1 = left eye inner    4 = right eye inner
#  2 = left eye          5 = right eye
#  3 = left eye outer    6 = right eye outer
#  7 = left ear          8 = right ear
#  9 = mouth left       10 = mouth right
# 11 = left shoulder    12 = right shoulder
# 13 = left elbow       14 = right elbow
# 15 = left wrist       16 = right wrist

def _get_body_refs(image, pose_results):
    """Extract all key body + face reference points."""
    if not pose_results.pose_landmarks:
        return None

    lm = pose_results.pose_landmarks.landmark
    ih, iw = image.shape[:2]

    def px(landmark):
        return (landmark.x * iw, landmark.y * ih)

    l_shoulder = px(lm[11])
    r_shoulder = px(lm[12])
    shoulder_w = max(abs(l_shoulder[0] - r_shoulder[0]), 1.0)

    mouth = ((px(lm[9])[0] + px(lm[10])[0]) / 2,
             (px(lm[9])[1] + px(lm[10])[1]) / 2)
    nose = px(lm[0])

    # Forehead: midpoint between inner eyes (above nose)
    forehead = ((px(lm[1])[0] + px(lm[4])[0]) / 2,
                (px(lm[1])[1] + px(lm[4])[1]) / 2)

    # Chin: estimated below mouth (mouth + 60% of nose-to-mouth distance)
    chin = (mouth[0],
            mouth[1] + (mouth[1] - nose[1]) * 0.6)

    return {
        'nose': nose,
        'mouth': mouth,
        'forehead': forehead,
        'chin': chin,
        'l_eye': px(lm[2]),
        'r_eye': px(lm[5]),
        'l_ear': px(lm[7]),
        'r_ear': px(lm[8]),
        'shoulder_center': ((l_shoulder[0] + r_shoulder[0]) / 2,
                            (l_shoulder[1] + r_shoulder[1]) / 2),
        'shoulder_w': shoulder_w,
        'l_shoulder': l_shoulder,
        'r_shoulder': r_shoulder,
        'l_elbow': px(lm[13]),
        'r_elbow': px(lm[14]),
        'l_wrist_pose': px(lm[15]),
        'r_wrist_pose': px(lm[16]),
    }


def extract_body_features(wrist_positions, body_refs):
    """44 body + face context features normalized by shoulder width.

    Per hand (x2 = 36):
      hand to nose (2), hand to mouth (2), hand to shoulder_center (2),
      hand to forehead (2), hand to left eye (2), hand to right eye (2),
      hand to left ear (2), hand to right ear (2), hand to chin (2)
      = 18 per hand

    Arms (8):
      left elbow-shoulder (2), right elbow-shoulder (2),
      left wrist-elbow (2), right wrist-elbow (2)
    """
    if body_refs is None:
        return [0.0] * NUM_BODY_FEATURES

    sw = body_refs['shoulder_w']
    sc = body_refs['shoulder_center']

    face_points = [
        body_refs['nose'],
        body_refs['mouth'],
        sc,
        body_refs['forehead'],
        body_refs['l_eye'],
        body_refs['r_eye'],
        body_refs['l_ear'],
        body_refs['r_ear'],
        body_refs['chin'],
    ]

    features = []

    for i in range(2):
        if i < len(wrist_positions):
            wx, wy = wrist_positions[i]
        else:
            wx, wy = sc

        for ref in face_points:
            features.append((wx - ref[0]) / sw)
            features.append((wy - ref[1]) / sw)

    le, re = body_refs['l_elbow'], body_refs['r_elbow']
    ls, rs = body_refs['l_shoulder'], body_refs['r_shoulder']
    lw, rw = body_refs['l_wrist_pose'], body_refs['r_wrist_pose']

    features.append((le[0] - ls[0]) / sw)
    features.append((le[1] - ls[1]) / sw)
    features.append((re[0] - rs[0]) / sw)
    features.append((re[1] - rs[1]) / sw)
    features.append((lw[0] - le[0]) / sw)
    features.append((lw[1] - le[1]) / sw)
    features.append((rw[0] - re[0]) / sw)
    features.append((rw[1] - re[1]) / sw)

    return features


# ---------------------------------------------------------------------------
#  Motion features (16) - velocity/movement tracking
# ---------------------------------------------------------------------------

class MotionTracker:
    """Tracks hand movement across frames.

    16 features per frame:
      Per hand (x2 = 14):
        - wrist velocity (vx, vy)           : direction of movement
        - speed (magnitude)                  : how fast
        - wrist-to-nose approach (dvx, dvy)  : approaching/leaving face?
        - wrist-to-mouth approach (dvx, dvy) : approaching/leaving mouth?
      Inter-hand (2):
        - inter-hand distance change (dx, dy): hands together/apart?

    All normalized by shoulder width for scale invariance.
    """

    def __init__(self):
        self.prev_wrists = None
        self.prev_hand_nose = [None, None]
        self.prev_hand_mouth = [None, None]
        self.prev_inter_hand = None

    def compute(self, wrist_positions, body_refs):
        """Compute 16 motion features from current and previous frame."""
        if body_refs is None:
            self._store(wrist_positions, None, None, None)
            return [0.0] * NUM_MOTION_FEATURES

        sw = body_refs['shoulder_w']
        nose = body_refs['nose']
        mouth = body_refs['mouth']
        sc = body_refs['shoulder_center']

        features = []
        curr_nose = []
        curr_mouth = []

        for i in range(2):
            if i < len(wrist_positions):
                wx, wy = wrist_positions[i]
            else:
                wx, wy = sc

            if self.prev_wrists is not None and i < len(self.prev_wrists):
                pvx, pvy = self.prev_wrists[i]
                vx = (wx - pvx) / sw
                vy = (wy - pvy) / sw
            else:
                vx, vy = 0.0, 0.0

            features.append(vx)
            features.append(vy)
            features.append((vx ** 2 + vy ** 2) ** 0.5)

            hn_x = (wx - nose[0]) / sw
            hn_y = (wy - nose[1]) / sw
            curr_nose.append((hn_x, hn_y))

            if self.prev_hand_nose[i] is not None:
                features.append(hn_x - self.prev_hand_nose[i][0])
                features.append(hn_y - self.prev_hand_nose[i][1])
            else:
                features.extend([0.0, 0.0])

            hm_x = (wx - mouth[0]) / sw
            hm_y = (wy - mouth[1]) / sw
            curr_mouth.append((hm_x, hm_y))

            if self.prev_hand_mouth[i] is not None:
                features.append(hm_x - self.prev_hand_mouth[i][0])
                features.append(hm_y - self.prev_hand_mouth[i][1])
            else:
                features.extend([0.0, 0.0])

        if len(wrist_positions) >= 2:
            ih_x = (wrist_positions[0][0] - wrist_positions[1][0]) / sw
            ih_y = (wrist_positions[0][1] - wrist_positions[1][1]) / sw
        else:
            ih_x, ih_y = 0.0, 0.0

        curr_inter = (ih_x, ih_y)

        if self.prev_inter_hand is not None:
            features.append(ih_x - self.prev_inter_hand[0])
            features.append(ih_y - self.prev_inter_hand[1])
        else:
            features.extend([0.0, 0.0])

        self._store(wrist_positions, curr_nose, curr_mouth, curr_inter)
        return features

    def _store(self, wrists, nose_dists, mouth_dists, inter):
        self.prev_wrists = list(wrists) if wrists else None
        if nose_dists:
            self.prev_hand_nose = nose_dists
        if mouth_dists:
            self.prev_hand_mouth = mouth_dists
        self.prev_inter_hand = inter

    def reset(self):
        self.__init__()


# ---------------------------------------------------------------------------
#  Full pipeline
# ---------------------------------------------------------------------------

def extract_all_features(image, hand_results, pose_results,
                         prev_wrists=None, motion_tracker=None):
    """Full 144-feature extraction: hand shape + body/face + motion.

    Returns (features_144, wrist_positions, ordered_hands) or (None, prev, []).
    """
    ordered = order_hands(hand_results, image, prev_wrists)
    hand_feats, wrists = extract_hand_features(ordered)

    if hand_feats is None:
        if motion_tracker:
            motion_tracker.reset()
        return None, prev_wrists or [], []

    body_refs = _get_body_refs(image, pose_results)
    body_feats = extract_body_features(wrists, body_refs)

    if motion_tracker:
        motion_feats = motion_tracker.compute(wrists, body_refs)
    else:
        motion_feats = [0.0] * NUM_MOTION_FEATURES

    return hand_feats + body_feats + motion_feats, wrists, ordered


# ---------------------------------------------------------------------------
#  Drawing helpers
# ---------------------------------------------------------------------------

def draw_body_refs(image, pose_results):
    """Draw all tracked body + face reference points."""
    if not pose_results.pose_landmarks:
        return
    lm = pose_results.pose_landmarks.landmark
    ih, iw = image.shape[:2]

    refs = {
        0: (0, 255, 255),    # nose - cyan
        1: (255, 200, 0),    # left eye inner - orange
        2: (255, 150, 0),    # left eye - orange
        4: (255, 200, 0),    # right eye inner - orange
        5: (255, 150, 0),    # right eye - orange
        7: (0, 150, 255),    # left ear - blue
        8: (0, 150, 255),    # right ear - blue
        9: (255, 0, 255),    # mouth left - magenta
        10: (255, 0, 255),   # mouth right - magenta
        11: (255, 255, 0),   # left shoulder - yellow
        12: (255, 255, 0),   # right shoulder - yellow
        13: (200, 200, 0),   # left elbow
        14: (200, 200, 0),   # right elbow
        15: (0, 200, 200),   # left wrist
        16: (0, 200, 200),   # right wrist
    }
    for idx, color in refs.items():
        x = int(lm[idx].x * iw)
        y = int(lm[idx].y * ih)
        cv.circle(image, (x, y), 4, color, cv.FILLED)

    # Forehead (computed midpoint)
    fx = int((lm[1].x + lm[4].x) / 2 * iw)
    fy = int((lm[1].y + lm[4].y) / 2 * ih)
    cv.circle(image, (fx, fy), 5, (0, 255, 0), cv.FILLED)

    # Chin (estimated)
    mx = int((lm[9].x + lm[10].x) / 2 * iw)
    my = int((lm[9].y + lm[10].y) / 2 * ih)
    ny = int(lm[0].y * ih)
    cx, cy = mx, my + int((my - ny) * 0.6)
    cv.circle(image, (cx, cy), 5, (150, 0, 255), cv.FILLED)

    # Skeleton lines
    for a, b in [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]:
        ax, ay = int(lm[a].x * iw), int(lm[a].y * ih)
        bx, by = int(lm[b].x * iw), int(lm[b].y * ih)
        cv.line(image, (ax, ay), (bx, by), (80, 80, 80), 1)

    # Face contour lines
    for a, b in [(7, 2), (2, 0), (0, 5), (5, 8)]:
        ax, ay = int(lm[a].x * iw), int(lm[a].y * ih)
        bx, by = int(lm[b].x * iw), int(lm[b].y * ih)
        cv.line(image, (ax, ay), (bx, by), (60, 60, 60), 1)


# ---------------------------------------------------------------------------
#  Feature smoothing (use in app.py ONLY, not in coleta.py)
# ---------------------------------------------------------------------------

class FeatureSmoother:
    """Exponential Moving Average on feature vectors."""

    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.prev = None

    def smooth(self, features):
        if features is None:
            return None

        arr = np.array(features, dtype=np.float64)

        if self.prev is None:
            self.prev = arr
            return features

        smoothed = self.alpha * arr + (1 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed.tolist()

    def reset(self):
        self.prev = None

"""
Zero Barreiras — Feature Extraction v2 (MediaPipe Holistic)

Uses mp.solutions.holistic for simultaneous tracking of:
  - Pose:  33 landmarks × 4 (x,y,z,visibility)  = 132 features
  - Face: 468 landmarks × 3 (x,y,z)              = 1404 features
  - Left Hand:  21 landmarks × 3 (x,y,z)         = 63 features
  - Right Hand: 21 landmarks × 3 (x,y,z)         = 63 features
  Total: 1662 features per frame

This provides far richer data than v1 (144 features) and captures:
  - Full 3D face mesh (expressions, mouth shape, eye position)
  - Full 3D body pose with depth
  - Complete hand articulation with depth
"""
import numpy as np
import cv2 as cv

import mediapipe as mp
try:
    if not hasattr(mp, 'solutions'):
        import importlib
        mp.solutions = importlib.import_module('mediapipe.python.solutions')
except Exception:
    pass

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

NUM_FEATURES_V2 = 1662  # 132 + 1404 + 63 + 63


def create_holistic_detector(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create a Holistic detector instance."""
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=1,
    )


def process_frame(image, holistic):
    """Run holistic detection on a BGR image.
    Returns the mediapipe results object.
    """
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    return results


def extract_keypoints(results):
    """Extract 1662 keypoints from holistic results.

    Returns a flat numpy array of shape (1662,).
    If a body part is not detected, its features are filled with zeros.
    """
    # Pose: 33 landmarks × (x, y, z, visibility) = 132
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Face: 468 landmarks × (x, y, z) = 1404
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    # Left Hand: 21 landmarks × (x, y, z) = 63
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right Hand: 21 landmarks × (x, y, z) = 63
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


def has_hands(results):
    """Check if at least one hand is detected."""
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


def draw_landmarks(image, results):
    """Draw all holistic landmarks on the image."""
    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )


def draw_minimal_landmarks(image, results):
    """Draw only hands and upper body landmarks (lighter for streaming)."""
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )

    # Pose skeleton (upper body only)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ih, iw = image.shape[:2]

        upper_body_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # torso + arms
            (0, 11), (0, 12),  # head to shoulders (simplified)
        ]
        for a, b in upper_body_connections:
            ax, ay = int(lm[a].x * iw), int(lm[a].y * ih)
            bx, by = int(lm[b].x * iw), int(lm[b].y * ih)
            cv.line(image, (ax, ay), (bx, by), (80, 110, 80), 2)

        # Key points
        key_indices = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists
        for idx in key_indices:
            x = int(lm[idx].x * iw)
            y = int(lm[idx].y * ih)
            cv.circle(image, (x, y), 4, (0, 255, 128), cv.FILLED)

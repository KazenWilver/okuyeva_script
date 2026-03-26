"""
Zero Barreiras - Recolha de Dados LGA (maos + corpo + movimento, 120 features)

Uso:
  1. Clica na tecla da classe (0-9, A-E) -> gravacao inicia
     - Grava enquanto mao(s) estiverem visiveis (pausa auto sem maos)
  2. [N] Parar gravacao
  3. [V] Gravar video .mp4 do sinal
  4. [D] Mostrar/esconder dashboard de classes
  5. [Q/ESC] Sair
"""
import csv
import os
import time

import cv2 as cv

import mediapipe as mp
if not hasattr(mp, 'solutions'):
    import importlib
    mp.solutions = importlib.import_module('mediapipe.python.solutions')

from features import (
    NUM_FEATURES, MotionTracker,
    extract_all_features, calc_bounding_rect, draw_body_refs,
)

KEYPOINT_CSV = 'model/keypoint_classifier/keypoint.csv'
LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'
VIDEOS_DIR = 'gifs'

os.makedirs(os.path.dirname(KEYPOINT_CSV), exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


def load_labels():
    labels = []
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f if row.strip()]
    return labels


def count_samples():
    counts = {}
    if os.path.exists(KEYPOINT_CSV):
        with open(KEYPOINT_CSV, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    cls_id = int(row[0])
                    counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def key_to_class(key):
    if 48 <= key <= 57:
        return key - 48
    if 97 <= key <= 101:
        return key - 97 + 10
    return -1


def draw_dashboard(image, labels, counts, selected_class, x_start, y_start):
    """Draw a compact class dashboard overlay."""
    h_line = 18
    max_cols = 2
    items_per_col = (len(labels) + max_cols - 1) // max_cols
    col_w = 220

    bg_h = items_per_col * h_line + 30
    bg_w = col_w * max_cols + 10
    overlay = image.copy()
    cv.rectangle(overlay, (x_start, y_start),
                 (x_start + bg_w, y_start + bg_h),
                 (30, 30, 30), cv.FILLED)
    cv.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    cv.putText(image, "CLASSES (amostras):", (x_start + 5, y_start + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    for i, label in enumerate(labels):
        col = i // items_per_col
        row = i % items_per_col
        x = x_start + 5 + col * col_w
        y = y_start + 30 + row * h_line

        n = counts.get(i, 0)
        k = str(i) if i < 10 else chr(97 + i - 10)

        if i == selected_class:
            color = (0, 255, 0)
            txt = f">[{k}] {label}: {n}"
        elif n > 0:
            color = (200, 200, 100)
            txt = f" [{k}] {label}: {n}"
        else:
            color = (120, 120, 120)
            txt = f" [{k}] {label}: --"

        cv.putText(image, txt, (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def main():
    labels = load_labels()
    counts = count_samples()

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.4,
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    mp_draw = mp.solutions.drawing_utils

    recording = False
    selected_class = -1
    prev_wrists = None
    motion = MotionTracker()
    show_dashboard = True

    rec_video = False
    video_writer = None
    video_frames = 0
    MAX_VIDEO_FRAMES = 120

    samples_this_session = 0
    session_start = time.time()

    print("=" * 62)
    print("  COLETA DE DADOS - Zero Barreiras (LGA)")
    print("  Maos + Corpo + Movimento (120 features)")
    print("=" * 62)
    print("  [0-9] Classe 0-9   |  [A-E] Classe 10-14")
    print("  [N] Parar gravacao  |  [V] Gravar video")
    print("  [D] Dashboard       |  [Q/ESC] Sair")
    print()
    if labels:
        for i, label in enumerate(labels):
            n = counts.get(i, 0)
            k = str(i) if i < 10 else chr(97 + i - 10)
            print(f"    [{k}] {i:2d}: {label} ({n} amostras)")
    total = sum(counts.values())
    print(f"\n    Total: {total} amostras")
    print("=" * 62)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = image.copy()
        clean_image = image.copy()
        h, w = debug_image.shape[:2]

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = hands.process(rgb)
        pose_results = pose.process(rgb)
        rgb.flags.writeable = True

        hand_detected = hand_results.multi_hand_landmarks is not None
        num_hands = len(hand_results.multi_hand_landmarks) if hand_detected else 0
        pose_detected = pose_results.pose_landmarks is not None

        key = cv.waitKey(1) & 0xFF

        if key == ord('n') or key == ord('N'):
            if recording:
                cls_name = labels[selected_class] if selected_class < len(labels) else str(selected_class)
                print(f"[STOP] '{cls_name}' parada. Total: {counts.get(selected_class, 0)}")
            recording = False
            selected_class = -1
            motion.reset()

        cls_from_key = key_to_class(key)
        if cls_from_key >= 0:
            selected_class = cls_from_key
            recording = True
            prev_wrists = None
            motion.reset()
            samples_this_session = 0
            session_start = time.time()
            cls_name = labels[selected_class] if selected_class < len(labels) else str(selected_class)
            print(f"[REC] Classe [{selected_class}] '{cls_name}' - [N] para parar")

        if key == ord('v') or key == ord('V'):
            if selected_class >= 0 and not rec_video:
                cls_name = labels[selected_class] if selected_class < len(labels) else f"classe_{selected_class}"
                vpath = os.path.join(VIDEOS_DIR, f"{cls_name.lower()}.mp4")
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                video_writer = cv.VideoWriter(vpath, fourcc, 20.0, (w, h))
                rec_video = True
                video_frames = 0
                print(f"[VIDEO] A gravar -> {vpath}")

        if key == ord('d') or key == ord('D'):
            show_dashboard = not show_dashboard

        draw_body_refs(debug_image, pose_results)

        if hand_detected:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(debug_image, hand_lm, mp_hands.HAND_CONNECTIONS)

            if recording and selected_class >= 0:
                features, wrists, _ = extract_all_features(
                    image, hand_results, pose_results, prev_wrists, motion)
                prev_wrists = wrists

                if features and len(features) == NUM_FEATURES:
                    with open(KEYPOINT_CSV, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([selected_class, *features])
                    counts[selected_class] = counts.get(selected_class, 0) + 1
                    samples_this_session += 1
        else:
            if recording:
                motion.reset()
                prev_wrists = None

        if rec_video and video_writer is not None:
            video_writer.write(clean_image)
            video_frames += 1
            if video_frames >= MAX_VIDEO_FRAMES:
                video_writer.release()
                video_writer = None
                rec_video = False
                print(f"[VIDEO] Concluido! ({video_frames} frames)")
                video_frames = 0

        # === UI ===
        cv.rectangle(debug_image, (0, 0), (w, 40), (40, 40, 40), cv.FILLED)
        cv.putText(debug_image, "Coleta - Zero Barreiras (LGA)",
                   (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        status_parts = []
        if pose_detected:
            status_parts.append("Corpo:OK")
        else:
            status_parts.append("Corpo:--")
        status_parts.append(f"Total:{sum(counts.values())}")
        cv.putText(debug_image, " | ".join(status_parts), (w - 220, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if recording and selected_class >= 0:
            cls_name = labels[selected_class] if selected_class < len(labels) else f"Classe {selected_class}"
            n = counts.get(selected_class, 0)

            elapsed = time.time() - session_start
            rate = samples_this_session / elapsed if elapsed > 0.5 else 0

            if hand_detected:
                bg = (0, 100, 0)
                estado = f"GRAVANDO [{selected_class}] {cls_name}: {n}"
                estado += f" | Maos:{num_hands} | {rate:.0f} amostras/s"
            else:
                bg = (0, 0, 140)
                estado = f"[{selected_class}] {cls_name}: {n} | PAUSADO (sem maos)"

            cv.rectangle(debug_image, (0, 42), (w, 78), bg, cv.FILLED)
            cv.putText(debug_image, estado, (10, 66),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Progress bar (target: 300 samples)
            target = 300
            prog = min(n / target, 1.0)
            bar_w = 200
            bar_fill = int(bar_w * prog)
            bar_color = (0, 255, 0) if n >= target else (0, 200, 255)
            cv.rectangle(debug_image, (w - bar_w - 20, 50), (w - 20, 64),
                         (80, 80, 80), cv.FILLED)
            cv.rectangle(debug_image, (w - bar_w - 20, 50), (w - bar_w - 20 + bar_fill, 64),
                         bar_color, cv.FILLED)
            cv.putText(debug_image, f"{n}/{target}",
                       (w - bar_w - 20, 76), cv.FONT_HERSHEY_SIMPLEX, 0.35,
                       bar_color, 1)
        else:
            cv.putText(debug_image, "Clique [0-9,A-E] para gravar  |  [D] Dashboard",
                       (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if rec_video:
            prog = int((video_frames / MAX_VIDEO_FRAMES) * 200)
            cv.rectangle(debug_image, (w - 220, 80), (w - 220 + prog, 94),
                         (0, 0, 255), cv.FILLED)
            cv.rectangle(debug_image, (w - 220, 80), (w - 20, 94),
                         (255, 255, 255), 1)
            cv.putText(debug_image, f"Video: {video_frames}/{MAX_VIDEO_FRAMES}",
                       (w - 220, 108), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Dashboard overlay
        if show_dashboard and labels:
            draw_dashboard(debug_image, labels, counts, selected_class, 5, h - 200)

        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        hand_text = f"Maos: {num_hands}" if hand_detected else "Maos: 0"
        cv.putText(debug_image, hand_text, (w - 100, h - 45),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)

        cv.rectangle(debug_image, (0, h - 28), (w, h), (40, 40, 40), cv.FILLED)
        cv.putText(debug_image, "[V]Video  [N]Parar  [D]Dashboard  [0-9,A-E]Classe  [Q]Sair",
                   (10, h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv.imshow("Coleta de Dados", debug_image)

        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        try:
            if cv.getWindowProperty("Coleta de Dados", cv.WND_PROP_VISIBLE) < 1:
                break
        except cv.error:
            break

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv.destroyAllWindows()

    print("\n" + "=" * 62)
    print("  RESUMO FINAL")
    print("=" * 62)
    counts = count_samples()
    total = 0
    classes_ok = 0
    for cls_id in sorted(counts.keys()):
        label = labels[cls_id] if cls_id < len(labels) else f"Classe {cls_id}"
        n = counts[cls_id]
        status = "OK" if n >= 200 else "POUCO" if n >= 50 else "FALTA"
        print(f"  {cls_id:2d}: {label:20s} - {n:5d} amostras  [{status}]")
        total += n
        if n >= 200:
            classes_ok += 1
    print(f"\n  Total: {total} amostras | {classes_ok} classes prontas (>=200)")
    if classes_ok >= 2:
        print("\n  -> Pode treinar: python treinar.py")
    else:
        print("\n  -> Precisa de pelo menos 2 classes com >=200 amostras")
    print("=" * 62)


if __name__ == '__main__':
    main()

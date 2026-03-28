"""
Zero Barreiras — Recolha de Dados Dinâmica (Sequências de Movimento)

Usa MediaPipe Holistic para capturar 1662 features por frame.
Dois modos de recolha:
  [E] Modo ESTÁTICO: Grava frame-a-frame (como coleta.py original)
  [D] Modo DINÂMICO: Grava blocos de 30 frames (sequências de movimento)

Teclas:
  [0-9, A-E] Seleccionar classe
  [SPACE]    Iniciar/parar gravação (estático) ou gravar 1 sequência (dinâmico)
  [E]        Modo Estático
  [D]        Modo Dinâmico  
  [T]        Dashboard de classes
  [Q/ESC]    Sair
"""
import csv
import os
import time
import glob

import cv2 as cv
import numpy as np

from features_v2 import (
    create_holistic_detector, process_frame, extract_keypoints,
    has_hands, draw_landmarks, NUM_FEATURES_V2,
)

# ── Paths ──
SEQUENCE_DIR = 'model/sequence_classifier/sequences'
STATIC_CSV = 'model/sequence_classifier/static_keypoints.csv'
LABEL_CSV = 'model/sequence_classifier/sequence_classifier_label.csv'

SEQUENCE_LENGTH = 30   # frames per dynamic sequence

os.makedirs(SEQUENCE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATIC_CSV), exist_ok=True)


def load_labels():
    labels = []
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f if row.strip()]
    return labels


def count_sequences(labels):
    """Count existing dynamic sequences per class."""
    counts = {}
    for i, label in enumerate(labels):
        class_dir = os.path.join(SEQUENCE_DIR, label.lower())
        if os.path.exists(class_dir):
            counts[i] = len(glob.glob(os.path.join(class_dir, '*.npy')))
        else:
            counts[i] = 0
    return counts


def count_static_samples():
    """Count static samples per class from CSV."""
    counts = {}
    if os.path.exists(STATIC_CSV):
        with open(STATIC_CSV, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    cls_id = int(row[0])
                    counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def save_sequence(label_name, sequence_frames):
    """Save a sequence of keypoints as a .npy file.
    
    Args:
        label_name: gesture name (used as directory name)
        sequence_frames: list of numpy arrays, each (1662,)
    
    Returns:
        path to saved file
    """
    class_dir = os.path.join(SEQUENCE_DIR, label_name.lower())
    os.makedirs(class_dir, exist_ok=True)

    # Find next sample number
    existing = glob.glob(os.path.join(class_dir, 'seq_*.npy'))
    next_num = len(existing)
    
    # Stack frames into (seq_len, 1662) array
    sequence = np.array(sequence_frames)
    
    # If sequence is shorter than SEQUENCE_LENGTH, pad with last frame
    if len(sequence) < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - len(sequence)
        padding = np.tile(sequence[-1:], (pad_count, 1))
        sequence = np.concatenate([sequence, padding], axis=0)
    
    # If sequence is longer, trim to SEQUENCE_LENGTH
    if len(sequence) > SEQUENCE_LENGTH:
        # Uniformly sample SEQUENCE_LENGTH frames
        indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = sequence[indices]
    
    filepath = os.path.join(class_dir, f'seq_{next_num:04d}.npy')
    np.save(filepath, sequence)
    return filepath


def key_to_class(key):
    if 48 <= key <= 57:     # 0-9
        return key - 48
    if 97 <= key <= 101:    # a-e
        return key - 97 + 10
    return -1


def draw_dashboard(image, labels, dyn_counts, sta_counts, selected_class, mode, x_start, y_start):
    """Draw class dashboard with both dynamic and static counts."""
    h_line = 18
    max_cols = 2
    items_per_col = (len(labels) + max_cols - 1) // max_cols
    col_w = 260

    bg_h = items_per_col * h_line + 40
    bg_w = col_w * max_cols + 10
    overlay = image.copy()
    cv.rectangle(overlay, (x_start, y_start),
                 (x_start + bg_w, y_start + bg_h),
                 (30, 30, 30), cv.FILLED)
    cv.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    mode_str = "DINAMICO" if mode == 'dynamic' else "ESTATICO"
    cv.putText(image, f"CLASSES ({mode_str}):", (x_start + 5, y_start + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    for i, label in enumerate(labels):
        col = i // items_per_col
        row = i % items_per_col
        x = x_start + 5 + col * col_w
        y = y_start + 35 + row * h_line

        dn = dyn_counts.get(i, 0)
        sn = sta_counts.get(i, 0)
        k = str(i) if i < 10 else chr(97 + i - 10)

        if i == selected_class:
            color = (0, 255, 0)
            txt = f">[{k}] {label}: D={dn} S={sn}"
        elif dn > 0 or sn > 0:
            color = (200, 200, 100)
            txt = f" [{k}] {label}: D={dn} S={sn}"
        else:
            color = (120, 120, 120)
            txt = f" [{k}] {label}: --"

        cv.putText(image, txt, (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def main():
    labels = load_labels()
    if not labels:
        print("ERRO: Crie o ficheiro de labels primeiro:")
        print(f"  {LABEL_CSV}")
        print("Uma label por linha (ex: Neutro, Dor, Febre, ...)")
        exit(1)

    dyn_counts = count_sequences(labels)
    sta_counts = count_static_samples()

    # Camera setup — probe for physical camera
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

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    holistic = create_holistic_detector()

    # State
    mode = 'dynamic'  # 'dynamic' or 'static'
    selected_class = -1
    recording_static = False
    recording_dynamic = False
    dynamic_buffer = []
    show_dashboard = True
    session_start = time.time()
    samples_this_session = 0

    print("=" * 62)
    print("  COLETA DE DADOS v2 — Zero Barreiras")
    print("  MediaPipe Holistic (1662 features)")
    print("  Modos: [D] Dinâmico  [E] Estático")
    print("=" * 62)
    print("  [0-9] Classe 0-9   |  [A-E] Classe 10-14")
    print("  [SPACE] Gravar     |  [T] Dashboard")
    print("  [D] Modo Dinâmico  |  [E] Modo Estático")
    print("  [Q/ESC] Sair")
    print()
    if labels:
        for i, label in enumerate(labels):
            dn = dyn_counts.get(i, 0)
            sn = sta_counts.get(i, 0)
            k = str(i) if i < 10 else chr(97 + i - 10)
            print(f"    [{k}] {i:2d}: {label} (D={dn} S={sn})")
    print(f"\n    Modo actual: DINÂMICO")
    print("=" * 62)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        h, w = image.shape[:2]

        # Process with Holistic
        results = process_frame(image, holistic)
        hands_visible = has_hands(results)

        # Draw landmarks
        draw_landmarks(image, results)

        key = cv.waitKey(1) & 0xFF

        # ── Mode switching ──
        if key == ord('d') or key == ord('D'):
            mode = 'dynamic'
            recording_static = False
            dynamic_buffer = []
            recording_dynamic = False
            print(f"[MODE] Modo DINÂMICO activado")

        if key == ord('e') or key == ord('E'):
            mode = 'static'
            recording_static = False
            dynamic_buffer = []
            recording_dynamic = False
            print(f"[MODE] Modo ESTÁTICO activado")

        if key == ord('t') or key == ord('T'):
            show_dashboard = not show_dashboard

        # ── Class selection ──
        cls_from_key = key_to_class(key)
        if cls_from_key >= 0 and cls_from_key < len(labels):
            selected_class = cls_from_key
            samples_this_session = 0
            session_start = time.time()
            cls_name = labels[selected_class]
            print(f"[CLASS] Classe [{selected_class}] '{cls_name}' seleccionada")

        # ── Recording logic ──
        if mode == 'dynamic':
            # SPACE starts recording a 30-frame sequence
            if key == 32 and selected_class >= 0 and not recording_dynamic:
                recording_dynamic = True
                dynamic_buffer = []
                print(f"[REC] A gravar sequência dinâmica... ({SEQUENCE_LENGTH} frames)")

            if recording_dynamic:
                keypoints = extract_keypoints(results)
                dynamic_buffer.append(keypoints)

                # Check if we have enough frames
                if len(dynamic_buffer) >= SEQUENCE_LENGTH:
                    cls_name = labels[selected_class]
                    filepath = save_sequence(cls_name, dynamic_buffer)
                    dyn_counts[selected_class] = dyn_counts.get(selected_class, 0) + 1
                    samples_this_session += 1
                    recording_dynamic = False
                    dynamic_buffer = []
                    print(f"[OK] Sequência guardada: {filepath} (Total: {dyn_counts[selected_class]})")

        elif mode == 'static':
            # SPACE toggles continuous recording
            if key == 32 and selected_class >= 0:
                recording_static = not recording_static
                if recording_static:
                    print(f"[REC] Gravação estática ON — [SPACE] para parar")
                else:
                    print(f"[STOP] Gravação estática OFF")

            if recording_static and selected_class >= 0 and hands_visible:
                keypoints = extract_keypoints(results)
                with open(STATIC_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([selected_class, *keypoints.tolist()])
                sta_counts[selected_class] = sta_counts.get(selected_class, 0) + 1
                samples_this_session += 1

        # ══════════════════════════════════════════
        #  UI Rendering
        # ══════════════════════════════════════════

        # Top bar
        mode_color = (0, 180, 255) if mode == 'dynamic' else (255, 180, 0)
        mode_text = "DINAMICO" if mode == 'dynamic' else "ESTATICO"
        cv.rectangle(image, (0, 0), (w, 40), (40, 40, 40), cv.FILLED)
        cv.putText(image, f"Coleta v2 — Zero Barreiras | Modo: {mode_text}",
                   (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

        # Recording status bar
        if selected_class >= 0:
            cls_name = labels[selected_class]

            if mode == 'dynamic' and recording_dynamic:
                progress = len(dynamic_buffer) / SEQUENCE_LENGTH
                bar_w = w - 40
                bar_fill = int(bar_w * progress)

                cv.rectangle(image, (0, 42), (w, 85), (0, 100, 0), cv.FILLED)
                cv.putText(image, f"GRAVANDO [{selected_class}] {cls_name}: {len(dynamic_buffer)}/{SEQUENCE_LENGTH}",
                           (10, 62), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Progress bar
                cv.rectangle(image, (20, 68), (20 + bar_w, 80), (80, 80, 80), cv.FILLED)
                cv.rectangle(image, (20, 68), (20 + bar_fill, 80), (0, 255, 0), cv.FILLED)

            elif mode == 'dynamic' and not recording_dynamic:
                n = dyn_counts.get(selected_class, 0)
                cv.rectangle(image, (0, 42), (w, 78), (60, 60, 60), cv.FILLED)
                cv.putText(image, f"[{selected_class}] {cls_name}: {n} seq. | [SPACE] Gravar sequencia",
                           (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            elif mode == 'static' and recording_static:
                n = sta_counts.get(selected_class, 0)
                elapsed = time.time() - session_start
                rate = samples_this_session / elapsed if elapsed > 0.5 else 0

                if hands_visible:
                    bg = (0, 100, 0)
                    estado = f"GRAVANDO [{selected_class}] {cls_name}: {n} | {rate:.0f}/s"
                else:
                    bg = (0, 0, 140)
                    estado = f"[{selected_class}] {cls_name}: {n} | PAUSADO (sem maos)"

                cv.rectangle(image, (0, 42), (w, 78), bg, cv.FILLED)
                cv.putText(image, estado, (10, 65),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif mode == 'static' and not recording_static:
                n = sta_counts.get(selected_class, 0)
                cv.rectangle(image, (0, 42), (w, 78), (60, 60, 60), cv.FILLED)
                cv.putText(image, f"[{selected_class}] {cls_name}: {n} | [SPACE] Iniciar gravacao",
                           (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        else:
            cv.putText(image, "Seleccione classe [0-9, A-E] | [D]=Dinamico [E]=Estatico",
                       (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Hand indicator
        hand_color = (0, 255, 0) if hands_visible else (0, 0, 255)
        hand_text = "Maos: OK" if hands_visible else "Maos: --"
        cv.putText(image, hand_text, (w - 120, h - 45),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        # Dashboard
        if show_dashboard and labels:
            draw_dashboard(image, labels, dyn_counts, sta_counts,
                          selected_class, mode, 5, h - 220)

        # Bottom bar
        cv.rectangle(image, (0, h - 28), (w, h), (40, 40, 40), cv.FILLED)
        cv.putText(image, "[SPACE]Gravar  [D]Dinamico  [E]Estatico  [T]Dashboard  [0-9,A-E]Classe  [Q]Sair",
                   (10, h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

        cv.imshow("Coleta Dinamica — Zero Barreiras", image)

        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        try:
            if cv.getWindowProperty("Coleta Dinamica — Zero Barreiras", cv.WND_PROP_VISIBLE) < 1:
                break
        except cv.error:
            break

    holistic.close()
    cap.release()
    cv.destroyAllWindows()

    # Final summary
    dyn_counts = count_sequences(labels)
    sta_counts = count_static_samples()

    print("\n" + "=" * 62)
    print("  RESUMO FINAL")
    print("=" * 62)
    total_dyn = 0
    total_sta = 0
    for i, label in enumerate(labels):
        dn = dyn_counts.get(i, 0)
        sn = sta_counts.get(i, 0)
        total_dyn += dn
        total_sta += sn
        status = "OK" if dn >= 30 else "POUCO" if dn >= 10 else "FALTA"
        print(f"  {i:2d}: {label:20s} - D={dn:4d} S={sn:5d}  [{status}]")
    print(f"\n  Total Dinâmico: {total_dyn} sequências")
    print(f"  Total Estático: {total_sta} amostras")
    if total_dyn >= 30:
        print("\n  -> Pode treinar: python treinar_dinamico.py")
    else:
        print("\n  -> Precisa de pelo menos 30 sequências dinâmicas no total")
    print("=" * 62)


if __name__ == '__main__':
    main()

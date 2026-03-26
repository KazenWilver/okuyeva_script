"""
Zero Barreiras - Consulta Inclusiva (maos + corpo + movimento, 120 features)

Fluxo 1 (Paciente -> Medico): Webcam -> MediaPipe Hands+Pose -> MLP -> Texto
Fluxo 2 (Medico -> Paciente): Microfone -> Speech -> Keyword -> Video LGA

Teclas:
  [M] Microfone medico  |  [R] Repetir video
  [C] Limpar frase      |  [S] Guardar transcricao
  [Q/ESC] Sair (guarda transcricao automaticamente)
"""
import os
import threading
import time
import unicodedata
from collections import deque, Counter
from datetime import datetime

import cv2 as cv

import mediapipe as mp
if not hasattr(mp, 'solutions'):
    import importlib
    mp.solutions = importlib.import_module('mediapipe.python.solutions')

from utils import CvFpsCalc
from model import KeyPointClassifier
from features import (
    NUM_FEATURES, MotionTracker, FeatureSmoother,
    extract_all_features, calc_bounding_rect, draw_body_refs,
)

CONFIANCA_MINIMA = 0.70
JANELA_SUAVIZACAO = 10
MAIORIA_MINIMA = 6
PASTA_VIDEOS = 'gifs'
PASTA_CONSULTAS = 'consultas'
LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'


def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def guardar_transcricao(eventos, filepath):
    """Save consultation transcript to text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  ZERO BARREIRAS - Transcricao de Consulta\n")
        f.write(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for ts, origem, texto in eventos:
            hora = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            if origem == 'paciente':
                f.write(f"  [{hora}] PACIENTE (LGA): {texto}\n")
            else:
                f.write(f"  [{hora}] MEDICO (voz):   {texto}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"  Total: {len(eventos)} interaccoes\n")
        f.write("=" * 60 + "\n")
    return filepath


def main():
    model_path = 'model/keypoint_classifier/keypoint_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"ERRO: '{model_path}' nao encontrado!")
        print("Execute treinar.py primeiro.")
        exit(1)

    keypoint_classifier = KeyPointClassifier(model_path=model_path)

    labels = []
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f if row.strip()]
    print(f"Classes: {labels}")

    videos = {}
    if os.path.exists(PASTA_VIDEOS):
        for f_name in os.listdir(PASTA_VIDEOS):
            if f_name.lower().endswith('.mp4'):
                keyword = os.path.splitext(f_name)[0].lower()
                videos[keyword] = os.path.join(PASTA_VIDEOS, f_name)
        if videos:
            print(f"Videos LGA: {list(videos.keys())}")

    try:
        import speech_recognition as sr
        reconhecedor = sr.Recognizer()
        speech_available = True
        print("Voz: Disponivel")
    except ImportError:
        speech_available = False
        sr = None
        reconhecedor = None
        print("Voz: Indisponivel (pip install SpeechRecognition pyaudio)")

    lock = threading.Lock()
    state = {
        'mensagem': '',
        'alerta': False,
        'alerta_tempo': 0,
        'video_a_tocar': None,
        'a_escutar': False,
    }

    # Sentence builder & transcript
    frase = []
    ultimo_gesto_na_frase = ""
    transcricao = []
    ts_inicio = datetime.now().strftime('%Y%m%d_%H%M%S')
    transcript_path = os.path.join(PASTA_CONSULTAS, f"consulta_{ts_inicio}.txt")

    def escutar_medico():
        with lock:
            if state['a_escutar'] or not speech_available:
                return
            state['a_escutar'] = True

        try:
            with sr.Microphone() as source:
                reconhecedor.adjust_for_ambient_noise(source, duration=0.3)
                print("[MIC] A escutar...")
                try:
                    audio = reconhecedor.listen(source, timeout=7,
                                                phrase_time_limit=10)
                    texto = reconhecedor.recognize_google(audio,
                                                         language="pt-PT")
                    with lock:
                        state['mensagem'] = f"Medico: {texto}"
                        texto_norm = normalizar_texto(texto)
                        for palavra, caminho in videos.items():
                            p_norm = normalizar_texto(palavra)
                            if p_norm in texto_norm or p_norm.replace("_", " ") in texto_norm:
                                state['video_a_tocar'] = caminho
                                print(f"[MIC] Keyword: '{palavra}'")
                                break
                        state['alerta'] = True
                        state['alerta_tempo'] = time.time()
                    transcricao.append((time.time(), 'medico', texto))
                    print(f"[MIC] '{texto}'")

                except sr.WaitTimeoutError:
                    with lock:
                        state['mensagem'] = "Nenhuma fala detectada."
                        state['alerta'] = True
                        state['alerta_tempo'] = time.time()
                except sr.UnknownValueError:
                    with lock:
                        state['mensagem'] = "Nao entendi. Fale mais perto."
                        state['alerta'] = True
                        state['alerta_tempo'] = time.time()
                except sr.RequestError:
                    with lock:
                        state['mensagem'] = "Erro de rede."
                        state['alerta'] = True
                        state['alerta_tempo'] = time.time()
        except Exception as e:
            with lock:
                state['mensagem'] = f"Erro mic: {str(e)[:50]}"
                state['alerta'] = True
                state['alerta_tempo'] = time.time()
        finally:
            with lock:
                state['a_escutar'] = False

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("ERRO: Camera indisponivel.")
        exit(1)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.4,
    )

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    mp_draw = mp.solutions.drawing_utils

    fps_calc = CvFpsCalc(buffer_len=10)
    historico = deque(maxlen=JANELA_SUAVIZACAO)
    smoother = FeatureSmoother(alpha=0.65)
    motion = MotionTracker()
    prev_wrists = None

    cap_video = None
    video_ativo = False
    video_fps = 20
    ultimo_frame_video = 0
    ultimo_video_path = None

    gesto_atual = ""
    confianca_atual = 0.0

    print("\n" + "=" * 62)
    print("  CONSULTA INCLUSIVA - Zero Barreiras")
    print("  Lingua Gestual Angolana -> Texto (Maos+Corpo+Movimento)")
    print("=" * 62)
    print("  [M] Microfone    [R] Repetir video")
    print("  [C] Limpar frase [S] Guardar transcricao")
    print("  [Q/ESC] Sair")
    print("=" * 62)

    while True:
        fps = fps_calc.get()

        with lock:
            if state['video_a_tocar'] and not video_ativo:
                caminho = state['video_a_tocar']
                state['video_a_tocar'] = None
                if os.path.exists(caminho):
                    cap_video = cv.VideoCapture(caminho)
                    video_ativo = True
                    ultimo_frame_video = time.time()
                    ultimo_video_path = caminho

        if video_ativo and cap_video is not None:
            agora = time.time()
            if agora - ultimo_frame_video >= 1.0 / video_fps:
                ret_vid, frame_vid = cap_video.read()
                if ret_vid:
                    cv.putText(frame_vid, "Traducao LGA", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv.imshow("Traducao para Paciente", frame_vid)
                    ultimo_frame_video = agora
                else:
                    cap_video.release()
                    cap_video = None
                    video_ativo = False
                    try:
                        cv.destroyWindow("Traducao para Paciente")
                    except cv.error:
                        pass

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = image.copy()
        frame_h, frame_w = debug_image.shape[:2]

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = hands_detector.process(rgb)
        pose_results = pose_detector.process(rgb)
        rgb.flags.writeable = True

        num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        pose_detected = pose_results.pose_landmarks is not None

        draw_body_refs(debug_image, pose_results)

        if hand_results.multi_hand_landmarks is not None:
            features, wrists, ordered = extract_all_features(
                image, hand_results, pose_results, prev_wrists, motion)
            prev_wrists = wrists

            for hand_data in ordered:
                brect = calc_bounding_rect(debug_image, hand_data['landmarks'])
                mp_draw.draw_landmarks(
                    debug_image, hand_data['landmarks'], mp_hands.HAND_CONNECTIONS)
                cv.rectangle(debug_image,
                             (brect[0], brect[1]), (brect[2], brect[3]),
                             (0, 0, 0), 1)
                hand_side = hand_data['label']
                cv.rectangle(debug_image,
                             (brect[0], brect[1] - 22), (brect[2], brect[1]),
                             (0, 0, 0), -1)
                cv.putText(debug_image, hand_side,
                           (brect[0] + 5, brect[1] - 4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 1, cv.LINE_AA)

            if features and len(features) == NUM_FEATURES:
                smoothed = smoother.smooth(features)
                sign_id, confidence = keypoint_classifier(smoothed)
                label = labels[sign_id] if sign_id < len(labels) else f"Classe {sign_id}"
                confianca_atual = confidence

                if confidence > CONFIANCA_MINIMA and label.lower() != "neutro":
                    historico.append(label)
                else:
                    historico.append("_neutro")
            else:
                historico.append("_neutro")
                confianca_atual = 0.0
        else:
            historico.append("_neutro")
            confianca_atual = 0.0
            motion.reset()
            smoother.reset()

        prev_gesto = gesto_atual
        if len(historico) >= 3:
            contagem = Counter(historico)
            mais_comum, freq = contagem.most_common(1)[0]
            if mais_comum != "_neutro" and freq >= MAIORIA_MINIMA:
                gesto_atual = mais_comum
            else:
                gesto_atual = ""
        else:
            gesto_atual = ""

        # Sentence builder: add new gesture when it changes
        if gesto_atual and gesto_atual != ultimo_gesto_na_frase:
            frase.append(gesto_atual)
            ultimo_gesto_na_frase = gesto_atual
            transcricao.append((time.time(), 'paciente', gesto_atual))
            print(f"[GESTO] {gesto_atual} (frase: {' > '.join(frase)})")
        if not gesto_atual:
            ultimo_gesto_na_frase = ""

        # === UI ===
        # Header
        cv.rectangle(debug_image, (0, 0), (frame_w, 40), (40, 40, 40), cv.FILLED)
        cv.putText(debug_image, "Zero Barreiras - Consulta Inclusiva",
                   (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_txt = f"FPS:{fps} | Maos:{num_hands}"
        if pose_detected:
            info_txt += " | Corpo"
        cv.putText(debug_image, info_txt, (frame_w - 240, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Current gesture + confidence bar
        if gesto_atual:
            texto_gesto = f"Paciente diz: {gesto_atual.upper()}"
            (tw, _), _ = cv.getTextSize(texto_gesto,
                                        cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv.rectangle(debug_image, (5, 48), (tw + 25, 88),
                         (0, 80, 0), cv.FILLED)
            cv.putText(debug_image, texto_gesto, (10, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Confidence bar
            bar_x = tw + 35
            bar_w = 100
            bar_fill = int(bar_w * min(confianca_atual, 1.0))
            bar_color = (0, 255, 0) if confianca_atual > 0.85 else (0, 200, 255)
            cv.rectangle(debug_image, (bar_x, 60), (bar_x + bar_w, 75),
                         (80, 80, 80), cv.FILLED)
            cv.rectangle(debug_image, (bar_x, 60), (bar_x + bar_fill, 75),
                         bar_color, cv.FILLED)
            cv.putText(debug_image, f"{confianca_atual*100:.0f}%",
                       (bar_x + bar_w + 5, 73),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
        else:
            cv.putText(debug_image, "Paciente: (a aguardar gesto...)",
                       (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (120, 120, 120), 2)

        # Sentence display
        if frase:
            frase_txt = " > ".join(frase[-8:])
            if len(frase) > 8:
                frase_txt = "... > " + frase_txt
            y_frase = 110
            cv.rectangle(debug_image, (5, y_frase - 18), (frame_w - 5, y_frase + 8),
                         (60, 40, 0), cv.FILLED)
            cv.putText(debug_image, f"Frase: {frase_txt}", (10, y_frase),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        # Doctor message
        with lock:
            escutando = state['a_escutar']
            alerta = state['alerta']
            msg = state['mensagem']
            t_alerta = state['alerta_tempo']

        if alerta and time.time() - t_alerta > 5.0:
            with lock:
                state['alerta'] = False
                alerta = False

        if escutando:
            cv.circle(debug_image, (frame_w - 25, 65), 10, (0, 0, 255), cv.FILLED)
            cv.putText(debug_image, "A OUVIR...", (frame_w - 130, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if alerta and msg:
            y_msg = frame_h - 100
            cv.rectangle(debug_image, (5, y_msg - 5), (frame_w - 5, y_msg + 35),
                         (40, 40, 40), cv.FILLED)
            cv.rectangle(debug_image, (5, y_msg - 5), (frame_w - 5, y_msg + 35),
                         (0, 255, 255), 2)
            cv.putText(debug_image, msg[:80], (15, y_msg + 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Transcript counter
        if transcricao:
            cv.putText(debug_image, f"Transcricao: {len(transcricao)} msgs",
                       (10, frame_h - 45), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                       (150, 150, 150), 1)

        # Footer
        cv.rectangle(debug_image, (0, frame_h - 28), (frame_w, frame_h),
                     (40, 40, 40), cv.FILLED)
        cv.putText(debug_image,
                   "[M]Mic  [R]Repetir  [C]Limpar  [S]Guardar  [Q]Sair",
                   (10, frame_h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                   (180, 180, 180), 1)

        cv.imshow("Consulta Inclusiva", debug_image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('m') or key == ord('M'):
            threading.Thread(target=escutar_medico, daemon=True).start()
        elif key == ord('r') or key == ord('R'):
            if ultimo_video_path and not video_ativo:
                with lock:
                    state['video_a_tocar'] = ultimo_video_path
        elif key == ord('c') or key == ord('C'):
            frase.clear()
            ultimo_gesto_na_frase = ""
            print("[FRASE] Limpa")
        elif key == ord('s') or key == ord('S'):
            if transcricao:
                p = guardar_transcricao(transcricao, transcript_path)
                print(f"[SAVE] Transcricao guardada: {p}")
        elif key == 27 or key == ord('q') or key == ord('Q'):
            break

        try:
            if cv.getWindowProperty("Consulta Inclusiva", cv.WND_PROP_VISIBLE) < 1:
                break
        except cv.error:
            break

    # Auto-save transcript on exit
    if transcricao:
        p = guardar_transcricao(transcricao, transcript_path)
        print(f"\n[AUTO-SAVE] Transcricao: {p}")

    if cap_video is not None:
        cap_video.release()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

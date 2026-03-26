import cv2
import mediapipe as mp
import pickle
import numpy as np
import speech_recognition as sr
import threading
import time
import os
import unicodedata
from collections import deque, Counter

MODELO_PATH = "modelo_lga.pkl"
PASTA_VIDEOS = "gifs"
CONFIANCA_MINIMA = 0.75
JANELA_SUAVIZACAO = 8
MAIORIA_MINIMA = 4

def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()

if not os.path.exists(MODELO_PATH):
    print(f"ERRO: Modelo '{MODELO_PATH}' nao encontrado! Execute treinar.py primeiro.")
    exit(1)

with open(MODELO_PATH, "rb") as f:
    modelo = pickle.load(f)
print(f"Modelo carregado. Classes: {list(modelo.classes_)}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

palavras_chave_videos = {}
if os.path.exists(PASTA_VIDEOS):
    for f in os.listdir(PASTA_VIDEOS):
        if f.lower().endswith(".mp4"):
            palavra = os.path.splitext(f)[0].lower()
            palavras_chave_videos[palavra] = os.path.join(PASTA_VIDEOS, f)
    if palavras_chave_videos:
        print(f"Videos LGA disponiveis: {list(palavras_chave_videos.keys())}")
    else:
        print(f"AVISO: Pasta '{PASTA_VIDEOS}' sem ficheiros .mp4")
else:
    print(f"AVISO: Pasta '{PASTA_VIDEOS}' nao encontrada.")

lock = threading.Lock()
mensagem_medico = ""
mostrar_alerta = False
alerta_tempo = 0
video_a_tocar = None
a_escutar = False

reconhecedor = sr.Recognizer()

def escutar_medico():
    global mensagem_medico, mostrar_alerta, alerta_tempo, video_a_tocar, a_escutar

    with lock:
        if a_escutar:
            return
        a_escutar = True

    try:
        with sr.Microphone() as source:
            reconhecedor.adjust_for_ambient_noise(source, duration=0.3)
            print("[MIC] A escutar o medico...")

            try:
                audio = reconhecedor.listen(source, timeout=7, phrase_time_limit=10)
                texto = reconhecedor.recognize_google(audio, language="pt-PT")

                with lock:
                    mensagem_medico = f"Medico: {texto}"

                    texto_norm = normalizar_texto(texto)
                    for palavra, caminho in palavras_chave_videos.items():
                        palavra_norm = normalizar_texto(palavra)
                        if palavra_norm in texto_norm or palavra_norm.replace("_", " ") in texto_norm:
                            video_a_tocar = caminho
                            print(f"[MIC] Palavra-chave: '{palavra}' -> {caminho}")
                            break

                    mostrar_alerta = True
                    alerta_tempo = time.time()

                print(f"[MIC] Transcricao: {texto}")

            except sr.WaitTimeoutError:
                with lock:
                    mensagem_medico = "Nenhuma fala detectada. Tente novamente."
                    mostrar_alerta = True
                    alerta_tempo = time.time()
            except sr.UnknownValueError:
                with lock:
                    mensagem_medico = "Nao foi possivel entender. Fale mais perto."
                    mostrar_alerta = True
                    alerta_tempo = time.time()
            except sr.RequestError:
                with lock:
                    mensagem_medico = "Erro de rede. Verifique a conexao a Internet."
                    mostrar_alerta = True
                    alerta_tempo = time.time()
    except Exception as e:
        with lock:
            mensagem_medico = f"Erro no microfone: {str(e)[:50]}"
            mostrar_alerta = True
            alerta_tempo = time.time()
    finally:
        with lock:
            a_escutar = False

cap_video = None
video_ativo = False
video_fps = 20
ultimo_frame_video = 0

historico_gestos = deque(maxlen=JANELA_SUAVIZACAO)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Nao foi possivel abrir a camera.")
    exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

gesto_detectado = ""
gesto_confianca = 0.0
fps_cont = 0
fps_val = 0
fps_t = time.time()

print("\n" + "=" * 50)
print("  CONSULTA INCLUSIVA - Zero Barreiras")
print("=" * 50)
print("  [M] Activar microfone do medico")
print("  [R] Repetir ultimo video LGA")
print("  [Q] Sair")
print("=" * 50)

ultimo_video_path = None

while True:
    with lock:
        if video_a_tocar and not video_ativo:
            caminho = video_a_tocar
            video_a_tocar = None
            if os.path.exists(caminho):
                cap_video = cv2.VideoCapture(caminho)
                video_ativo = True
                ultimo_frame_video = time.time()
                ultimo_video_path = caminho

    if video_ativo and cap_video is not None:
        agora = time.time()
        if agora - ultimo_frame_video >= 1.0 / video_fps:
            ret_vid, frame_vid = cap_video.read()
            if ret_vid:
                cv2.putText(frame_vid, "Traducao para LGA", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Traducao para Paciente", frame_vid)
                ultimo_frame_video = agora
            else:
                cap_video.release()
                cap_video = None
                video_ativo = False
                try:
                    cv2.destroyWindow("Traducao para Paciente")
                except cv2.error:
                    pass

    sucesso, img = cap.read()
    if not sucesso:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    if resultados.multi_hand_landmarks:
        coordenadas = []
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                coordenadas.extend([lm.x, lm.y, lm.z])

        if len(coordenadas) < 126:
            coordenadas.extend([0.0] * (126 - len(coordenadas)))
        elif len(coordenadas) > 126:
            coordenadas = coordenadas[:126]

        probabilidades = modelo.predict_proba([coordenadas])[0]
        indice_maior = np.argmax(probabilidades)
        confianca = probabilidades[indice_maior]
        previsao = modelo.classes_[indice_maior]

        if confianca > CONFIANCA_MINIMA and previsao != "neutro":
            historico_gestos.append(previsao)
        else:
            historico_gestos.append("neutro")

        if len(historico_gestos) >= 3:
            contagem = Counter(historico_gestos)
            mais_comum, freq = contagem.most_common(1)[0]

            if mais_comum != "neutro" and freq >= MAIORIA_MINIMA:
                gesto_detectado = mais_comum.upper()
                gesto_confianca = confianca
            else:
                gesto_detectado = ""
                gesto_confianca = 0.0
    else:
        historico_gestos.append("neutro")
        gesto_detectado = ""
        gesto_confianca = 0.0

    fps_cont += 1
    if time.time() - fps_t >= 1.0:
        fps_val = fps_cont
        fps_cont = 0
        fps_t = time.time()

    cv2.rectangle(img, (0, 0), (frame_w, 45), (40, 40, 40), cv2.FILLED)
    cv2.putText(img, "Zero Barreiras - Consulta Inclusiva", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"{fps_val} FPS", (frame_w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if gesto_detectado:
        texto_gesto = f"Paciente: {gesto_detectado} ({gesto_confianca*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(texto_gesto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (5, 52), (tw + 20, 92), (0, 80, 0), cv2.FILLED)
        cv2.putText(img, texto_gesto, (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Paciente: (a aguardar gesto...)", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    with lock:
        escutando = a_escutar
        alerta = mostrar_alerta
        msg = mensagem_medico
        t_alerta = alerta_tempo

    if alerta and time.time() - t_alerta > 5.0:
        with lock:
            mostrar_alerta = False
            alerta = False

    if escutando:
        cv2.circle(img, (frame_w - 25, 70), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "A OUVIR...", (frame_w - 130, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if alerta and msg:
        y_msg = frame_h - 80
        cv2.rectangle(img, (5, y_msg - 5), (frame_w - 5, y_msg + 35), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(img, (5, y_msg - 5), (frame_w - 5, y_msg + 35), (0, 255, 255), 2)
        cv2.putText(img, msg[:70], (15, y_msg + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.rectangle(img, (0, frame_h - 30), (frame_w, frame_h), (40, 40, 40), cv2.FILLED)
    cv2.putText(img, "[M] Medico falar  |  [R] Repetir video  |  [Q] Sair",
                (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Consulta Inclusiva", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m') or key == ord('M'):
        threading.Thread(target=escutar_medico, daemon=True).start()
    elif key == ord('r') or key == ord('R'):
        if ultimo_video_path and not video_ativo:
            with lock:
                video_a_tocar = ultimo_video_path
    elif key == ord('q') or key == ord('Q'):
        break

    try:
        if cv2.getWindowProperty("Consulta Inclusiva", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

if cap_video is not None:
    cap_video.release()
cap.release()
cv2.destroyAllWindows()

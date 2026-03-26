import cv2

# Vamos testar as primeiras 3 portas
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Sucesso! A câmara {i} está a funcionar.")
        # Mostra um frame rápido para veres qual é qual
        ret, frame = cap.read()
        cv2.imshow(f"Camara {i}", frame)
        cv2.waitKey(2000) # Mostra a imagem por 2 segundos
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Câmara {i} não encontrada.")
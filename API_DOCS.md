# Zero Barreiras - API para o Frontend

## O que faz?

O paciente faz um gesto com as mãos em frente à câmera. O frontend (React) detecta as mãos, envia as coordenadas para o Python, e o Python responde com o nome do gesto (ex: "Dor", "Febre", "Sim").

```
CÂMERA (browser) → MediaPipe detecta mãos → envia para API Python → responde "Dor"
```

---

## Como ligar a API

```bash
cd okuyeva_script
pip install -r requirements.txt
python api.py
```

A API fica em: **http://localhost:8000**

Para testar se está a funcionar, abre no browser: **http://localhost:8000/docs**

---

## Resumo dos Endpoints

| O quê                       | Método | URL                        |
|-----------------------------|--------|----------------------------|
| Ver se API está OK          | GET    | `/api/health`              |
| Lista de gestos             | GET    | `/api/labels`              |
| **Detectar gesto**          | POST   | `/api/predict`             |
| Mensagem do médico          | POST   | `/api/doctor/message`      |
| Limpar frase do paciente    | POST   | `/api/sentence/clear`      |
| Guardar transcrição         | POST   | `/api/transcript/save`     |
| **Detectar em tempo real**  | WS     | `ws://localhost:8000/ws/predict` |

---

## Passo a passo para o Frontend

### 1. Instalar MediaPipe no projecto React

```bash
npm install @mediapipe/hands @mediapipe/camera_utils
```

### 2. Abrir a câmera e detectar mãos no browser

O MediaPipe corre no browser. Ele dá-te as coordenadas das mãos (21 pontos por mão, cada ponto com x, y, z entre 0 e 1).

```javascript
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

// Referência ao elemento <video>
const videoElement = document.getElementById('webcam');

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

hands.onResults((results) => {
  // results.multiHandLandmarks = array de mãos detectadas
  // Cada mão = array de 21 pontos {x, y, z}
  // results.multiHandedness = "Right" ou "Left" para cada mão

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    enviarParaAPI(results);
  }
});

const camera = new Camera(videoElement, {
  onFrame: async () => await hands.send({ image: videoElement }),
  width: 960,
  height: 540
});
camera.start();
```

### 3. Enviar para a API e receber o gesto

**Opção A - HTTP (mais simples, bom para começar):**

```javascript
async function enviarParaAPI(results) {
  // Montar os dados das mãos
  const hands = results.multiHandLandmarks.map((landmarks, i) => ({
    landmarks: landmarks.map(p => [p.x, p.y, p.z]),
    handedness: results.multiHandedness[i].label  // "Right" ou "Left"
  }));

  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      hands: hands,
      image_width: 960,
      image_height: 540,
      session_id: "consulta-1"
    })
  });

  const resultado = await response.json();

  // resultado.gesture     → "Dor", "Febre", "Sim", etc. (ou "" se nada)
  // resultado.confidence  → 0.0 a 1.0 (ex: 0.95 = 95%)
  // resultado.sentence    → ["Dor", "Cabeca"] (frase completa do paciente)
  // resultado.is_new_gesture → true se é um gesto novo na frase

  console.log('Gesto:', resultado.gesture);
  console.log('Frase:', resultado.sentence.join(' > '));
}
```

**Opção B - WebSocket (recomendado, mais rápido):**

```javascript
// Ligar uma vez quando o componente monta
const ws = new WebSocket('ws://localhost:8000/ws/predict');

// Receber resultados
ws.onmessage = (event) => {
  const resultado = JSON.parse(event.data);
  // Mesmos campos: gesture, confidence, sentence, is_new_gesture
  console.log('Gesto:', resultado.gesture);
};

// Enviar a cada frame (dentro do hands.onResults)
function enviarParaAPI(results) {
  const hands = results.multiHandLandmarks.map((landmarks, i) => ({
    landmarks: landmarks.map(p => [p.x, p.y, p.z]),
    handedness: results.multiHandedness[i].label
  }));

  ws.send(JSON.stringify({
    hands: hands,
    image_width: 960,
    image_height: 540
  }));
}
```

---

## O que a API responde (POST /api/predict)

### Tu envias:

```json
{
  "hands": [
    {
      "landmarks": [[0.45, 0.62, 0.0], [0.47, 0.58, -0.02], "... 21 pontos"],
      "handedness": "Right"
    }
  ],
  "image_width": 960,
  "image_height": 540,
  "session_id": "consulta-1"
}
```

### A API responde:

**Quando detecta um gesto:**
```json
{
  "gesture": "Dor",
  "gesture_id": 1,
  "confidence": 0.95,
  "sentence": ["Dor", "Cabeca"],
  "num_hands": 1,
  "is_new_gesture": true
}
```

**Quando não detecta nada (mãos paradas, sem gesto claro):**
```json
{
  "gesture": "",
  "gesture_id": -1,
  "confidence": 0.0,
  "sentence": ["Dor"],
  "num_hands": 1,
  "is_new_gesture": false
}
```

### Explicação dos campos da resposta:

| Campo          | O que é                                                       | Exemplo              |
|----------------|---------------------------------------------------------------|----------------------|
| gesture        | Nome do gesto detectado. Vazio `""` se nenhum                 | `"Dor"`              |
| gesture_id     | ID do gesto (número). `-1` se nenhum                          | `1`                  |
| confidence     | Certeza do modelo, de 0 a 1                                   | `0.95` (= 95%)      |
| sentence       | Todos os gestos que o paciente fez até agora                  | `["Dor", "Cabeca"]`  |
| num_hands      | Quantas mãos foram detectadas                                 | `1` ou `2`           |
| is_new_gesture | É um gesto novo? (útil para saber quando mostrar algo novo)   | `true` ou `false`    |

---

## Outros endpoints úteis

### Ver lista de gestos suportados

```javascript
// GET http://localhost:8000/api/labels
const res = await fetch('http://localhost:8000/api/labels');
const data = await res.json();
// data.labels = ["Neutro", "Dor", "Febre", "Cabeca", "Barriga", "Sim", "Nao",
//                "Ajuda", "Menstruacao", "Gravidez", "Enjoo", "Sangramento",
//                "Medicamento", "Obrigado", "Agua"]
```

### Registar o que o médico disse

```javascript
await fetch('http://localhost:8000/api/doctor/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "Onde sente a dor?",
    session_id: "consulta-1"
  })
});
```

### Limpar a frase do paciente (botão "Limpar")

```javascript
await fetch('http://localhost:8000/api/sentence/clear', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: "consulta-1" })
});
```

### Guardar a consulta toda num ficheiro

```javascript
const res = await fetch('http://localhost:8000/api/transcript/save', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: "consulta-1" })
});
const data = await res.json();
// data.path = "consultas/consulta_20260325_143052.txt"
```

### Verificar se API está online

```javascript
const res = await fetch('http://localhost:8000/api/health');
const data = await res.json();
// data.model_loaded = true/false (se false, modelo ainda não foi treinado)
```

---

## Hook React pronto a usar (copiar e colar)

Ficheiro `useGestureWebSocket.js`:

```javascript
import { useEffect, useRef, useState, useCallback } from 'react';

export default function useGestureWebSocket(url = 'ws://localhost:8000/ws/predict') {
  const wsRef = useRef(null);
  const [gesture, setGesture] = useState('');
  const [sentence, setSentence] = useState([]);
  const [confidence, setConfidence] = useState(0);
  const [connected, setConnected] = useState(false);
  const [isNewGesture, setIsNewGesture] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setGesture(data.gesture || '');
      setSentence(data.sentence || []);
      setConfidence(data.confidence || 0);
      setIsNewGesture(data.is_new_gesture || false);
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [url]);

  const sendLandmarks = useCallback((handsResults) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!handsResults.multiHandLandmarks?.length) return;

    const hands = handsResults.multiHandLandmarks.map((lm, i) => ({
      landmarks: lm.map(p => [p.x, p.y, p.z]),
      handedness: handsResults.multiHandedness?.[i]?.label || 'Right'
    }));

    wsRef.current.send(JSON.stringify({
      hands,
      image_width: 960,
      image_height: 540
    }));
  }, []);

  return { gesture, sentence, confidence, connected, isNewGesture, sendLandmarks };
}
```

Uso no componente:

```jsx
import useGestureWebSocket from './useGestureWebSocket';

function TelaConsulta() {
  const { gesture, sentence, confidence, connected, isNewGesture, sendLandmarks } =
    useGestureWebSocket();

  // No callback do MediaPipe Hands:
  // hands.onResults((results) => sendLandmarks(results));

  return (
    <div>
      <video id="webcam" autoPlay playsInline />

      <div className="status">
        {connected ? '🟢 Conectado' : '🔴 Desconectado'}
      </div>

      <div className="gesto">
        {gesture ? (
          <>
            <h2>{gesture}</h2>
            <p>{(confidence * 100).toFixed(0)}% certeza</p>
          </>
        ) : (
          <p>Faça um gesto...</p>
        )}
      </div>

      <div className="frase">
        <p>Paciente disse: {sentence.join(' → ') || '(nada ainda)'}</p>
      </div>
    </div>
  );
}
```

---

## Gestos que o sistema conhece

| ID | Gesto         |
|----|---------------|
| 0  | Neutro        |
| 1  | Dor           |
| 2  | Febre         |
| 3  | Cabeca        |
| 4  | Barriga       |
| 5  | Sim           |
| 6  | Nao           |
| 7  | Ajuda         |
| 8  | Menstruacao   |
| 9  | Gravidez      |
| 10 | Enjoo         |
| 11 | Sangramento   |
| 12 | Medicamento   |
| 13 | Obrigado      |
| 14 | Agua          |

---

## Erros que podem acontecer

| Problema                          | Resposta da API      | O que fazer                          |
|-----------------------------------|----------------------|--------------------------------------|
| API desligada                     | Sem resposta         | Correr `python api.py`               |
| Modelo não treinado               | 503 + mensagem       | Treinar primeiro com `python treinar.py` |
| Campos em falta no JSON           | 422 + detalhes       | Verificar que `hands` tem `landmarks` |
| Landmarks com formato errado      | 400 + mensagem       | Cada mão deve ter exactamente 21 pontos `[x, y, z]` |

---

## Resumo rápido

1. **Ligar a API**: `python api.py`
2. **No React**: abrir câmera → MediaPipe Hands detecta mãos → enviar `landmarks` para a API
3. **A API responde**: nome do gesto + frase acumulada
4. **Usar WebSocket** (`ws://localhost:8000/ws/predict`) para tempo real
5. **Cada mão** = 21 pontos, cada ponto = `[x, y, z]` (valores entre 0 e 1)
6. **session_id**: usar o mesmo em todos os pedidos da mesma consulta

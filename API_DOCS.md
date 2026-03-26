# Zero Barreiras - Documentação da API

> Tradução de Língua Gestual Angolana (LGA) para texto em contexto de consultas médicas.

## Índice

- [Início Rápido](#início-rápido)
- [Base URL](#base-url)
- [Endpoints REST](#endpoints-rest)
  - [GET /api/health](#get-apihealth)
  - [GET /api/labels](#get-apilabels)
  - [POST /api/predict](#post-apipredict)
  - [POST /api/doctor/message](#post-apidoctormessage)
  - [POST /api/sentence/clear](#post-apisentenceclear)
  - [POST /api/transcript/save](#post-apitranscriptsave)
  - [POST /api/model/reload](#post-apimodelreload)
- [WebSocket (Tempo Real)](#websocket-tempo-real)
  - [WS /ws/predict](#ws-wspredict)
- [Modelos de Dados](#modelos-de-dados)
- [Gestos Suportados](#gestos-suportados)
- [Guia de Integração Frontend](#guia-de-integração-frontend)
- [Códigos de Erro](#códigos-de-erro)

---

## Início Rápido

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Iniciar o servidor

```bash
python api.py
```

Ou com reload automático (para desenvolvimento):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

O servidor fica disponível em `http://localhost:8000`.

Documentação interactiva Swagger: `http://localhost:8000/docs`

---

## Base URL

```
http://localhost:8000
```

Todas as respostas são em JSON. CORS está habilitado para todas as origens (`*`).

---

## Endpoints REST

---

### GET /api/health

Verifica se a API está a funcionar e se o modelo está carregado.

**Request:**

```
GET /api/health
```

**Response 200:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "num_labels": 15,
  "num_features": 144
}
```

| Campo          | Tipo    | Descrição                              |
|----------------|---------|----------------------------------------|
| status         | string  | Sempre `"ok"` se o servidor responder  |
| model_loaded   | boolean | `true` se o modelo .pkl foi carregado  |
| num_labels     | integer | Número de gestos suportados            |
| num_features   | integer | Número de features que o modelo espera |

**Exemplo React:**

```javascript
const checkHealth = async () => {
  const res = await fetch('http://localhost:8000/api/health');
  const data = await res.json();
  console.log('API pronta:', data.model_loaded);
};
```

---

### GET /api/labels

Retorna a lista de todos os gestos LGA suportados.

**Request:**

```
GET /api/labels
```

**Response 200:**

```json
{
  "labels": [
    "Neutro", "Dor", "Febre", "Cabeca", "Barriga",
    "Sim", "Nao", "Ajuda", "Menstruacao", "Gravidez",
    "Enjoo", "Sangramento", "Medicamento", "Obrigado", "Agua"
  ],
  "mapping": {
    "0": "Neutro",
    "1": "Dor",
    "2": "Febre",
    "3": "Cabeca",
    "4": "Barriga",
    "5": "Sim",
    "6": "Nao",
    "7": "Ajuda",
    "8": "Menstruacao",
    "9": "Gravidez",
    "10": "Enjoo",
    "11": "Sangramento",
    "12": "Medicamento",
    "13": "Obrigado",
    "14": "Agua"
  }
}
```

| Campo   | Tipo            | Descrição                                    |
|---------|-----------------|----------------------------------------------|
| labels  | string[]        | Lista ordenada dos nomes dos gestos          |
| mapping | object          | Mapeamento `id -> nome` para cada gesto      |

---

### POST /api/predict

**Endpoint principal.** Envia landmarks das mãos (e opcionalmente do corpo) detectadas pelo MediaPipe no frontend, e recebe a predição do gesto.

**Request:**

```
POST /api/predict
Content-Type: application/json
```

**Body:**

```json
{
  "hands": [
    {
      "landmarks": [[0.45, 0.62, 0.0], [0.47, 0.58, -0.02], ...],
      "handedness": "Right"
    },
    {
      "landmarks": [[0.55, 0.60, 0.0], [0.53, 0.56, -0.01], ...],
      "handedness": "Left"
    }
  ],
  "pose": {
    "landmarks": [[0.50, 0.15, 0.0], [0.48, 0.14, 0.0], ...]
  },
  "image_width": 960,
  "image_height": 540,
  "session_id": "consulta-1"
}
```

| Campo        | Tipo        | Obrigatório | Default     | Descrição                                                  |
|--------------|-------------|-------------|-------------|------------------------------------------------------------|
| hands        | HandData[]  | **Sim**     | -           | Array com 1 ou 2 mãos detectadas                          |
| pose         | PoseData    | Não         | `null`      | 33 landmarks do corpo (MediaPipe Pose)                     |
| image_width  | integer     | Não         | `960`       | Largura do vídeo da webcam em pixels                       |
| image_height | integer     | Não         | `540`       | Altura do vídeo da webcam em pixels                        |
| session_id   | string      | Não         | `"default"` | ID da sessão (mantém estado entre pedidos)                 |

**HandData:**

| Campo      | Tipo         | Obrigatório | Default   | Descrição                                              |
|------------|--------------|-------------|-----------|--------------------------------------------------------|
| landmarks  | float[21][3] | **Sim**     | -         | 21 pontos [x, y, z] normalizados 0-1 (MediaPipe Hand) |
| handedness | string       | Não         | `"Right"` | `"Right"` ou `"Left"`                                  |

**PoseData:**

| Campo     | Tipo         | Obrigatório | Descrição                                          |
|-----------|--------------|-------------|----------------------------------------------------|
| landmarks | float[33][3] | **Sim**     | 33 pontos [x, y, z] normalizados 0-1 (MediaPipe Pose) |

**Response 200 (mãos detectadas):**

```json
{
  "gesture": "Dor",
  "gesture_id": 1,
  "confidence": 0.9523,
  "sentence": ["Dor", "Cabeca"],
  "num_hands": 2,
  "is_new_gesture": true
}
```

**Response 200 (sem mãos / gesto neutro):**

```json
{
  "gesture": "",
  "gesture_id": -1,
  "confidence": 0.0,
  "sentence": ["Dor"],
  "num_hands": 0,
  "is_new_gesture": false
}
```

| Campo          | Tipo     | Descrição                                                                |
|----------------|----------|--------------------------------------------------------------------------|
| gesture        | string   | Nome do gesto detectado. `""` se nenhum gesto estável                    |
| gesture_id     | integer  | ID numérico do gesto. `-1` se nenhum                                     |
| confidence     | float    | Confiança do modelo (0.0 a 1.0)                                         |
| sentence       | string[] | Frase acumulada do paciente (todos os gestos detectados na sessão)       |
| num_hands      | integer  | Número de mãos enviadas neste pedido                                     |
| is_new_gesture | boolean  | `true` se este gesto é novo e foi adicionado à frase                     |

**Exemplo React:**

```javascript
const predict = async (handLandmarks, poseLandmarks) => {
  const body = {
    hands: handLandmarks.map(hand => ({
      landmarks: hand.landmarks,
      handedness: hand.handedness
    })),
    image_width: videoWidth,
    image_height: videoHeight,
    session_id: sessionId
  };

  // Pose é opcional, mas melhora a detecção de sinais que tocam o rosto
  if (poseLandmarks) {
    body.pose = { landmarks: poseLandmarks };
  }

  const res = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  return await res.json();
};
```

---

### POST /api/doctor/message

Regista uma mensagem do médico (transcrita por voz) na transcrição da consulta.

**Request:**

```
POST /api/doctor/message
Content-Type: application/json
```

**Body:**

```json
{
  "text": "Onde sente a dor?",
  "session_id": "consulta-1"
}
```

| Campo      | Tipo   | Obrigatório | Default     | Descrição                       |
|------------|--------|-------------|-------------|---------------------------------|
| text       | string | **Sim**     | -           | Texto transcrito do médico      |
| session_id | string | Não         | `"default"` | ID da sessão                    |

**Response 200:**

```json
{
  "status": "ok",
  "transcript_length": 5
}
```

---

### POST /api/sentence/clear

Limpa a frase acumulada do paciente na sessão.

**Request:**

```
POST /api/sentence/clear
Content-Type: application/json
```

**Body:**

```json
{
  "session_id": "consulta-1"
}
```

**Response 200:**

```json
{
  "sentence": []
}
```

---

### POST /api/transcript/save

Guarda a transcrição completa da consulta num ficheiro `.txt` no servidor.

**Request:**

```
POST /api/transcript/save
Content-Type: application/json
```

**Body:**

```json
{
  "session_id": "consulta-1"
}
```

**Response 200:**

```json
{
  "path": "consultas/consulta_20260325_143052.txt",
  "total_events": 12
}
```

| Campo        | Tipo    | Descrição                                        |
|--------------|---------|--------------------------------------------------|
| path         | string  | Caminho do ficheiro guardado (`null` se vazio)    |
| total_events | integer | Número total de interacções na transcrição        |

---

### POST /api/model/reload

Recarrega o modelo ML sem reiniciar o servidor. Útil após re-treinar com novos dados.

**Request:**

```
POST /api/model/reload
```

**Response 200:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "labels": ["Neutro", "Dor", "Febre", "..."]
}
```

---

## WebSocket (Tempo Real)

### WS /ws/predict

Conexão WebSocket para predições frame-a-frame em tempo real. **Recomendado para a webcam** pois evita o overhead de HTTP em cada frame.

**URL:**

```
ws://localhost:8000/ws/predict
```

Cada conexão WebSocket tem a sua própria sessão independente (motion tracker, sentence builder, etc).

**Mensagem enviada (cliente -> servidor):**

```json
{
  "hands": [
    {
      "landmarks": [[0.45, 0.62, 0.0], ...],
      "handedness": "Right"
    }
  ],
  "pose": {
    "landmarks": [[0.50, 0.15, 0.0], ...]
  },
  "image_width": 960,
  "image_height": 540
}
```

O formato é idêntico ao body do `POST /api/predict` (sem `session_id`).

**Mensagem recebida (servidor -> cliente):**

```json
{
  "gesture": "Dor",
  "gesture_id": 1,
  "confidence": 0.9523,
  "sentence": ["Dor"],
  "num_hands": 1,
  "is_new_gesture": true
}
```

O formato é idêntico à response do `POST /api/predict`.

**Exemplo React completo:**

```javascript
import { useEffect, useRef, useState } from 'react';

function useGestureWebSocket(url = 'ws://localhost:8000/ws/predict') {
  const wsRef = useRef(null);
  const [gesture, setGesture] = useState('');
  const [sentence, setSentence] = useState([]);
  const [confidence, setConfidence] = useState(0);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket conectado');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setGesture(data.gesture);
      setSentence(data.sentence);
      setConfidence(data.confidence);
    };

    ws.onclose = () => {
      console.log('WebSocket desconectado');
      setConnected(false);
    };

    ws.onerror = (err) => {
      console.error('WebSocket erro:', err);
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [url]);

  const sendLandmarks = (hands, pose, imageWidth, imageHeight) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        hands,
        pose,
        image_width: imageWidth,
        image_height: imageHeight
      }));
    }
  };

  return { gesture, sentence, confidence, connected, sendLandmarks };
}

export default useGestureWebSocket;
```

**Uso no componente:**

```javascript
function GestureDetector() {
  const { gesture, sentence, confidence, connected, sendLandmarks } =
    useGestureWebSocket();

  // Chamado a cada frame do MediaPipe
  const onResults = (handResults, poseResults) => {
    const hands = handResults.multiHandLandmarks?.map((lm, i) => ({
      landmarks: lm.map(p => [p.x, p.y, p.z]),
      handedness: handResults.multiHandedness[i].label
    })) || [];

    const pose = poseResults?.poseLandmarks
      ? { landmarks: poseResults.poseLandmarks.map(p => [p.x, p.y, p.z]) }
      : null;

    sendLandmarks(hands, pose, 960, 540);
  };

  return (
    <div>
      <p>Status: {connected ? 'Conectado' : 'Desconectado'}</p>
      <p>Gesto: {gesture || '(aguardando...)'}</p>
      <p>Confiança: {(confidence * 100).toFixed(0)}%</p>
      <p>Frase: {sentence.join(' > ')}</p>
    </div>
  );
}
```

---

## Modelos de Dados

### Landmarks das Mãos (MediaPipe Hands)

Cada mão tem **21 pontos**, cada ponto com `[x, y, z]` normalizados entre 0 e 1:

```
 0: WRIST (pulso)
 1: THUMB_CMC          5: INDEX_FINGER_MCP     9: MIDDLE_FINGER_MCP
 2: THUMB_MCP          6: INDEX_FINGER_PIP    10: MIDDLE_FINGER_PIP
 3: THUMB_IP           7: INDEX_FINGER_DIP    11: MIDDLE_FINGER_DIP
 4: THUMB_TIP          8: INDEX_FINGER_TIP    12: MIDDLE_FINGER_TIP
13: RING_FINGER_MCP   17: PINKY_MCP
14: RING_FINGER_PIP   18: PINKY_PIP
15: RING_FINGER_DIP   19: PINKY_DIP
16: RING_FINGER_TIP   20: PINKY_TIP
```

### Landmarks do Corpo (MediaPipe Pose)

33 pontos, mas a API usa principalmente estes:

```
 0: Nariz               9: Boca esquerda      11: Ombro esquerdo
 1: Olho interno esq.  10: Boca direita       12: Ombro direito
 2: Olho esquerdo       7: Orelha esquerda    13: Cotovelo esquerdo
 4: Olho interno dir.   8: Orelha direita     14: Cotovelo direito
 5: Olho direito       15: Pulso esquerdo     16: Pulso direito
```

### Sessões

A API mantém estado por `session_id`:
- **Sentence builder** - acumula gestos detectados numa frase
- **Motion tracker** - calcula velocidade e direcção do movimento
- **Feature smoother** - suaviza os landmarks para menos ruído
- **Transcrição** - regista toda a interacção paciente/médico

Usa o mesmo `session_id` para todos os pedidos da mesma consulta.

---

## Gestos Suportados

| ID | Gesto         | Descrição                                |
|----|---------------|------------------------------------------|
| 0  | Neutro        | Posição neutra (sem sinal activo)        |
| 1  | Dor           | Indicar dor                              |
| 2  | Febre         | Indicar febre                            |
| 3  | Cabeca        | Referir à cabeça                         |
| 4  | Barriga       | Referir à barriga                        |
| 5  | Sim           | Afirmação                                |
| 6  | Nao           | Negação                                  |
| 7  | Ajuda         | Pedir ajuda                              |
| 8  | Menstruacao   | Referir menstruação                      |
| 9  | Gravidez      | Indicar gravidez                         |
| 10 | Enjoo         | Indicar enjoo/náusea                     |
| 11 | Sangramento   | Indicar sangramento                      |
| 12 | Medicamento   | Referir medicamento                      |
| 13 | Obrigado      | Agradecimento                            |
| 14 | Agua          | Pedir água                               |

---

## Guia de Integração Frontend

### 1. Verificar se a API está pronta

```javascript
// Ao iniciar a app, verificar conexão
const res = await fetch('http://localhost:8000/api/health');
const { model_loaded } = await res.json();
if (!model_loaded) {
  alert('Modelo ainda não foi treinado!');
}
```

### 2. Carregar labels para o UI

```javascript
const res = await fetch('http://localhost:8000/api/labels');
const { labels, mapping } = await res.json();
// labels = ["Neutro", "Dor", "Febre", ...]
// mapping = { 0: "Neutro", 1: "Dor", ... }
```

### 3. Escolher: HTTP ou WebSocket?

| Método    | Quando usar                                     | Latência |
|-----------|--------------------------------------------------|----------|
| HTTP POST | Enviar landmarks pontualmente ou a cada 200-500ms | Maior    |
| WebSocket | Stream contínuo frame-a-frame (recomendado)       | Menor    |

### 4. Fluxo completo de uma consulta

```
1. Frontend liga ao /ws/predict
2. A cada frame da webcam:
   - MediaPipe Hands detecta mãos → envia landmarks via WebSocket
   - (opcional) MediaPipe Pose detecta corpo → envia junto
3. Backend responde com gesto + frase acumulada
4. Quando médico fala → POST /api/doctor/message com texto
5. Para limpar frase → POST /api/sentence/clear
6. No fim → POST /api/transcript/save
```

### 5. MediaPipe no Frontend (JavaScript)

Para detectar landmarks no browser, usar:

```bash
npm install @mediapipe/hands @mediapipe/pose @mediapipe/camera_utils
```

Ou via CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
```

### 6. Notas importantes

- **Pose é opcional mas recomendado**: Sem pose, sinais que tocam o rosto (nariz, boca, testa) não são tão bem detectados. Os 44 features do corpo vão como zeros.
- **session_id**: Usa o mesmo ID durante toda a consulta para acumular a frase correctamente.
- **Confiança mínima**: O backend usa 70% como limiar. Gestos abaixo disso são tratados como "neutro".
- **Suavização temporal**: O backend usa uma janela de 10 frames com votação por maioria (mínimo 6 votos) para estabilizar a predição. Não é necessário suavizar no frontend.

---

## Códigos de Erro

| Status | Detalhe                            | Quando                                |
|--------|------------------------------------|---------------------------------------|
| 200    | -                                  | Sucesso                               |
| 400    | `"Feature count X != 144"`         | Landmarks mal formatados              |
| 503    | `"Modelo nao carregado"`           | Ficheiro .pkl não existe (treinar primeiro) |
| 422    | Validation Error (automático)      | Campos obrigatórios em falta / tipo errado |

---

*Zero Barreiras - Hackathon 2026*

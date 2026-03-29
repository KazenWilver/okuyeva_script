# 🎬 Guia de Coleta de Gestos Dinâmicos — Okuyeva

## O que é uma "sequência dinâmica"?

Uma sequência é um **vídeo curto** do seu gesto (1-3 segundos), convertido em **30 frames de dados corporais**.
O modelo LSTM aprende a reconhecer o **movimento ao longo do tempo**, não apenas uma pose estática.

---

## 🚀 Passo a Passo Completo

### 1. Abrir a ferramenta de coleta

```bash
python coleta_dinamica.py
```

Vai abrir uma janela com a sua câmara e um painel de controlo.

### 2. Seleccionar o gesto

Prima a tecla correspondente:

| Tecla | Gesto        |
|-------|-------------|
| `0`   | Neutro       |
| `1`   | Dor          |
| `2`   | Febre        |
| `3`   | Cabeça       |
| ...   | (ver dashboard com T) |

### 3. Confirmar modo DINÂMICO

Prima `D` para activar (painel mostra "DINAMICO" em laranja).

### 4. Gravar — O PASSO MAIS IMPORTANTE

O processo é:
```
[SPACE] → Contagem 3-2-1 → BIP! → GRAVAÇÃO → [SPACE] → Guardado!
```

#### Fase 1: Contagem (3-2-1)
- **3...** — Posicione-se em frente à câmara
- **2...** — Mãos visíveis, prepare o gesto
- **1...** — Pronto!

⚠️ **NÃO comece o gesto durante a contagem!** Espere o BIP.

#### Fase 2: Gravação (frames a subir)

Quando ouvir o BIP e o ecrã ficar vermelho "REC":

1. **Faça o gesto UMA VEZ de forma natural**
2. No ecrã vai ver:
   - **Frames:** 10, 15, 20, 25... (sobe automaticamente)
   - **Tempo:** 0.5s, 1.0s, 1.5s...
3. **Quando terminar o gesto, prima SPACE**

### ❓ O que fazer enquanto os frames sobem?

**NADA DE ESPECIAL!** Simplesmente faça o gesto ao seu ritmo.
Os frames capturam-se sozinhos (~30 por segundo).
Cada frame é uma "foto" dos seus 1662 pontos corporais.

**Exemplo — Gesto "Sim" (acenar cabeça):**
```
Frame  0-10  → Cabeça na posição normal
Frame 10-20  → Cabeça desce (acenar)
Frame 20-30  → Cabeça sobe
Frame 30-40  → Segundo aceno
[SPACE] → Parar!  (40 frames, ~1.3 segundos)
```

**Exemplo — Gesto "Dor" (apontar zona):**
```
Frame  0-10  → Mãos em repouso
Frame 10-25  → Mão move para zona de dor
Frame 25-40  → Mão pressiona/indica
Frame 40-50  → Mão regressa
[SPACE] → Parar!  (50 frames, ~1.7 segundos)
```

💡 O sistema aceita 10 a 200+ frames e normaliza tudo para 30. Não se preocupe com a duração exacta!

---

## 📊 Quantas amostras gravar?

| Quantidade       | Qualidade        |
|-----------------|-----------------|
| 5-10 por gesto  | ⚠️ Mínimo       |
| **20-30**       | ✅ Recomendado   |
| 50+             | 🎯 Óptimo       |

---

## 🎯 Dicas para QUALIDADE

### ✅ FAZER
1. **Variar velocidade** — lento, médio, rápido
2. **Variar posição** — esquerda, centro, direita
3. **Variar distância** — perto e longe da câmara
4. **Usar ambas as mãos** — alternar mão dominante
5. **Sessões diferentes** — não grave tudo seguido

### ❌ EVITAR
1. Repetir sempre igual (decora em vez de aprender)
2. Ficar estático num gesto dinâmico
3. Esconder as mãos
4. Gravar mais de 4 segundos

---

## O gesto "Neutro" é especial

É a posição "nada acontece". Grave com as mãos paradas em posições diferentes:
ao lado do corpo, no colo, cruzadas. Ajuda o modelo a distinguir actividade de repouso.

---

## 🔧 Depois de gravar

```bash
python treinar_dinamico.py              # treina o modelo
python -m uvicorn api:app --port 8000   # inicia a API
cd frontend && npm run dev              # inicia o frontend
```

Abrir http://localhost:5173/consulta para testar em tempo real!

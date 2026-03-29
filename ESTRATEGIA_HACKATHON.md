# 🎯 Análise Hackathon — Estratégia de Gravação

## Contexto do Pitch

O guião requer que a **paciente surda** faça gestos que são traduzidos em texto para o médico.
As falas da paciente no guião são:

1. "Bom dia. Sim. Tenho muitas cólicas. A minha menstruação está atrasada há um mês."
2. "Tenho medo. E também quero mudar a minha pílula. Sinto-me mal."
3. "Muito obrigada. Ter privacidade e respeito ajuda muito."

---

## ⚡ Estratégia: Palavras-Chave (não frases inteiras)

O sistema traduz **palavras/gestos individuais**, não frases completas.
A paciente fará uma sequência de gestos e o sistema mostrará cada palavra.

### Gestos PRIORITÁRIOS para o pitch (gravar PRIMEIRO):

| # | Gesto | Usado na fala | Qtd mínima |
|---|-------|--------------|------------|
| 0 | **Neutro** | (entre gestos) | 25 |
| 1 | **Ola** | "Bom dia" | 20 |
| 2 | **Dor** | "cólicas" | 20 |
| 3 | **Menstruacao** | "menstruação atrasada" | 20 |
| 4 | **Colica** | "muitas cólicas" | 20 |
| 5 | **Medo** | "Tenho medo" | 20 |
| 6 | **Pilula** | "mudar a minha pílula" | 20 |
| 7 | **Obrigada** | "Muito obrigada" | 20 |
| 8 | **Privacidade** | "Ter privacidade" | 20 |
| 14 | **Sim** | "Sim" | 20 |

### Gestos SECUNDÁRIOS (gravar se houver tempo):

| # | Gesto | Qtd mínima |
|---|-------|------------|
| 5 | Sangramento | 15 |
| 6 | Gravidez | 15 |
| 8 | Exame | 15 |
| 10 | Confortavel | 15 |
| 11 | Ajuda | 15 |
| 13 | Adeus | 15 |
| 15 | Nao | 15 |

### Gestos EXTRAS (gravar se sobrar tempo):

Corpo, Duvida, Tratamento

---

## ⏱️ Estimativa de Tempo

- Cada gesto: ~3 segundos (countdown + gravação)
- 20 repetições: ~1 minuto por gesto
- **10 gestos prioritários × 1 min = ~10-15 min**
- **7 gestos secundários × 1 min = ~7-10 min**
- **Treino do modelo: ~2-5 min**
- **Total realista: 30-40 minutos**

---

## 📋 Checklist antes de ir gravar

- [ ] `python coleta_dinamica.py` funciona (câmara abre)
- [ ] Labels actualizadas (20 gestos)
- [ ] Intérprete sabe os 10 gestos prioritários em LGA
- [ ] Boa iluminação no local
- [ ] Câmara capta rosto + mãos

## 🔄 Depois de gravar

```bash
python treinar_dinamico.py
python -m uvicorn api:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

## 💡 Dica para o pitch

Na demo ao vivo, a paciente pode fazer gestos numa sequência:
**OLA → DOR → COLICA → MENSTRUACAO** (pausa entre cada)

O médico vê no ecrã cada palavra aparecer individualmente.
O público vê a tradução a acontecer em tempo real.

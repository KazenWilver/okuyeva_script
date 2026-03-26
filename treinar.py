"""
Zero Barreiras - Treino do Modelo (144 features: maos + corpo/rosto + movimento)
Usa scikit-learn MLPClassifier com data augmentation.
"""
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

KEYPOINT_CSV = 'model/keypoint_classifier/keypoint.csv'
LABEL_CSV = 'model/keypoint_classifier/keypoint_classifier_label.csv'
MODEL_SAVE_PATH = 'model/keypoint_classifier/keypoint_classifier.pkl'

NUM_FEATURES = 144
RANDOM_SEED = 42
AUGMENT_FACTOR = 3
AUGMENT_NOISE = 0.015

print("=" * 62)
print("  TREINO - Zero Barreiras (144 feat: maos+rosto+corpo+movimento)")
print("=" * 62)

labels = []
if os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, encoding='utf-8-sig') as f:
        labels = [row.strip() for row in f if row.strip()]
    print(f"Labels: {labels}")

if not os.path.exists(KEYPOINT_CSV):
    print(f"\nERRO: '{KEYPOINT_CSV}' nao encontrado!")
    print("Execute coleta.py primeiro.")
    exit(1)

raw = np.loadtxt(KEYPOINT_CSV, delimiter=',', dtype='float32')
n_cols = raw.shape[1] - 1

if n_cols != NUM_FEATURES:
    print(f"\nAVISO: CSV tem {n_cols} features, esperado {NUM_FEATURES}.")
    if n_cols < NUM_FEATURES:
        print("A preencher features em falta com zeros (dados antigos?)...")
        pad = np.zeros((raw.shape[0], NUM_FEATURES - n_cols), dtype='float32')
        raw = np.hstack([raw[:, :1], raw[:, 1:], pad])
    else:
        print(f"A usar apenas as primeiras {NUM_FEATURES} features...")
        raw = raw[:, :NUM_FEATURES + 1]

X = raw[:, 1:]
y = raw[:, 0].astype('int32')

NUM_CLASSES = len(np.unique(y))
print(f"\nDataset original: {len(X)} amostras, {NUM_CLASSES} classes, {X.shape[1]} features")

unique_classes, class_counts = np.unique(y, return_counts=True)
print("\nDistribuicao:")
for cls_id, cnt in zip(unique_classes, class_counts):
    label = labels[int(cls_id)] if int(cls_id) < len(labels) else f"Classe {int(cls_id)}"
    pct = cnt / len(y) * 100
    bar = "#" * (cnt // 20)
    print(f"  {int(cls_id):2d}: {label:20s} {cnt:5d} ({pct:5.1f}%)  {bar}")

if NUM_CLASSES < 2:
    print("\nERRO: Precisa de pelo menos 2 classes.")
    exit(1)

# --- Data Augmentation ---
print(f"\nData Augmentation: {AUGMENT_FACTOR}x com ruido {AUGMENT_NOISE}...")
X_aug = [X]
y_aug = [y]
np.random.seed(RANDOM_SEED)
for i in range(AUGMENT_FACTOR):
    noise = np.random.normal(0, AUGMENT_NOISE, X.shape).astype(np.float32)
    X_aug.append(X + noise)
    y_aug.append(y)

X_full = np.vstack(X_aug)
y_full = np.hstack(y_aug)
print(f"Dataset aumentado: {len(X_full)} amostras ({AUGMENT_FACTOR + 1}x)")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
)
print(f"Split: {len(X_train)} treino / {len(X_test)} teste")

# --- Train ---
print("\nA treinar MLP (72-36 neuronios, early stopping)...")
modelo = MLPClassifier(
    hidden_layer_sizes=(72, 36),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
    verbose=False,
)
modelo.fit(X_train, y_train)
print(f"Convergiu em {modelo.n_iter_} iteracoes")

# --- Evaluate ---
previsoes = modelo.predict(X_test)
precisao = accuracy_score(y_test, previsoes)
print(f"\nPrecisao teste: {precisao * 100:.2f}%")

target_names = [labels[int(i)] if int(i) < len(labels) else f"Classe {int(i)}"
                for i in sorted(np.unique(y))]
print("\nRelatorio por classe:")
print(classification_report(y_test, previsoes, target_names=target_names,
                            zero_division=0))

# --- Cross-validation on original data ---
print("Validacao cruzada (5-fold, dados originais)...")
scores = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(72, 36), activation='relu', solver='adam',
                  max_iter=1000, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=RANDOM_SEED, verbose=False),
    X, y, cv=min(5, NUM_CLASSES), scoring='accuracy'
)
print(f"Precisao media CV: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

# --- Save ---
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(modelo, f)

size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
print(f"\nModelo guardado: '{MODEL_SAVE_PATH}' ({size_kb:.1f} KB)")
print(f"Classes: {list(modelo.classes_)}")

# --- Quick test ---
prob_test = modelo.predict_proba([X_test[0]])[0]
idx = np.argmax(prob_test)
pred = int(modelo.classes_[idx])
pred_label = labels[pred] if pred < len(labels) else f"Classe {pred}"
print(f"Teste rapido: classe={pred} ({pred_label}) confianca={prob_test[idx]*100:.1f}%")

print("\n" + "=" * 62)
if precisao >= 0.95:
    print("  MODELO PRONTO PARA DEMO!")
elif precisao >= 0.85:
    print("  BOM! Considere mais dados para melhorar.")
else:
    print("  ATENCAO: Precisao baixa. Recolha mais dados.")
print("=" * 62)

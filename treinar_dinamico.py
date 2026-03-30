"""
Okuyeva — Treino do Modelo Dinâmico (LSTM)

Treina um LSTM bidireccional sobre sequências de 30 frames × 1662 features.
Suporta GPU (CUDA) automaticamente.

Optimizações para datasets pequenos:
  - Modelo menor (64 hidden, 2 layers) para evitar overfitting
  - Class weights para balancear classes desiguais
  - Feature normalization (mean/std guardados no checkpoint)
  - Augmentação: ruído, time-warp, mirror, feature dropout
  - Gradient clipping para estabilidade
  - Verificação de colapso do modelo

Uso:
  python treinar_dinamico.py
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model.sequence_classifier.sequence_classifier import GestureLSTM

# ── Config ──
SEQUENCE_DIR = 'model/sequence_classifier/sequences'
LABEL_CSV = 'model/sequence_classifier/sequence_classifier_label.csv'
MODEL_SAVE_PATH = 'model/sequence_classifier/sequence_classifier.pt'
SEQUENCE_LENGTH = 30

# ── Feature Selection ──
# Layout original (1662 features):
#   Pose:       0:132   (33 landmarks × 4)  = 132
#   Face:     132:1536  (468 landmarks × 3)  = 1404  ← REMOVIDO (causa drift)
#   Left Hand: 1536:1599 (21 landmarks × 3)  = 63
#   Right Hand:1599:1662 (21 landmarks × 3)  = 63
#
# A face é 84.5% do sinal mas NÃO distingue gestos!
# Mudanças mínimas na posição da cara em tempo real destroem as previsões.
# Solução: usar apenas Pose + Mãos = 258 features (mãos = 49% do sinal)
FEATURE_INDICES = list(range(0, 132)) + list(range(1536, 1662))
NUM_FEATURES = len(FEATURE_INDICES)  # 258
NUM_FEATURES_RAW = 1662  # raw features from MediaPipe

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0005
AUGMENT_FACTOR = 5
AUGMENT_NOISE = 0.015  # slightly higher noise since fewer features
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ──
class GestureSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ── Augmentation Functions ──

def time_warp(sequence, sigma=0.2):
    """Apply random temporal warping. Simulates different gesture speeds."""
    seq_len = len(sequence)
    orig_steps = np.arange(seq_len)
    warp = np.random.normal(loc=1.0, scale=sigma, size=seq_len)
    warp = np.cumsum(warp)
    warp = warp / warp[-1] * (seq_len - 1)

    warped = np.zeros_like(sequence)
    for feat_idx in range(sequence.shape[1]):
        warped[:, feat_idx] = np.interp(orig_steps, warp, sequence[:, feat_idx])
    return warped


def mirror_sequence(sequence):
    """Mirror left/right by flipping x-coords and swapping hands.
    
    Feature layout after face removal (258 total):
      Pose:  0:132   (33 landmarks × 4: x,y,z,vis)  — x every 4 values
      LH:    132:195  (21 landmarks × 3: x,y,z)      — x every 3 values
      RH:    195:258  (21 landmarks × 3: x,y,z)      — x every 3 values
    """
    mirrored = sequence.copy()
    
    # Flip x coordinates in pose (every 4th value starting at 0)
    for i in range(0, 132, 4):
        mirrored[:, i] = 1.0 - mirrored[:, i]
    
    # Flip x in hands (every 3rd value starting at 132)
    for i in range(132, 258, 3):
        mirrored[:, i] = 1.0 - mirrored[:, i]
    
    # Swap left hand (132:195) and right hand (195:258)
    lh = mirrored[:, 132:195].copy()
    rh = mirrored[:, 195:258].copy()
    mirrored[:, 132:195] = rh
    mirrored[:, 195:258] = lh
    
    return mirrored


def feature_dropout(sequence, drop_rate=0.05):
    """Randomly zero out some features (simulates occlusion)."""
    mask = np.random.random(sequence.shape) > drop_rate
    return sequence * mask


def speed_variation(sequence, factor_range=(0.7, 1.3)):
    """Simulate different gesture speeds by resampling."""
    seq_len = len(sequence)
    factor = np.random.uniform(*factor_range)
    new_len = max(5, int(seq_len * factor))
    
    indices_orig = np.linspace(0, seq_len - 1, new_len)
    indices_target = np.linspace(0, seq_len - 1, seq_len)
    
    resampled = np.zeros_like(sequence)
    for feat_idx in range(sequence.shape[1]):
        values = np.interp(indices_orig, np.arange(seq_len), sequence[:, feat_idx])
        resampled[:, feat_idx] = np.interp(indices_target, indices_orig, values)
    
    return resampled


def shift_sequence(sequence, max_shift=3):
    """Simulate sliding window misalignment by shifting left or right."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return sequence.copy()
    
    res = np.empty_like(sequence)
    if shift > 0:
        res[shift:] = sequence[:-shift]
        res[:shift] = sequence[0]
    else:
        shift = abs(shift)
        res[:-shift] = sequence[shift:]
        res[-shift:] = sequence[-1]
    return res


def augment_sequence(sequence, noise_std=AUGMENT_NOISE):
    """Apply random combination of augmentations."""
    aug = sequence.copy()
    
    if np.random.random() < 0.4:
        aug = shift_sequence(aug)
        
    aug = aug + np.random.normal(0, noise_std, aug.shape)
    
    r = np.random.random()
    if r < 0.25:
        aug = time_warp(aug.astype(np.float32), sigma=0.15)
    elif r < 0.4:
        aug = speed_variation(aug.astype(np.float32), factor_range=(0.8, 1.2))
    
    if np.random.random() < 0.15:
        aug = feature_dropout(aug.astype(np.float32), drop_rate=0.03)
    
    return aug.astype(np.float32)


# ── Main ──
def main():
    print("=" * 62)
    print("  TREINO DINÂMICO — Okuyeva (LSTM v2)")
    print(f"  Device: {device}")
    print("=" * 62)

    # Load labels
    labels = []
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f if row.strip()]
    print(f"Labels: {labels}")

    # Load sequences
    all_sequences = []
    all_labels = []
    label_map = {name.lower(): idx for idx, name in enumerate(labels)}

    for class_name in os.listdir(SEQUENCE_DIR):
        class_dir = os.path.join(SEQUENCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name.lower() not in label_map:
            print(f"  AVISO: Pasta '{class_name}' não tem label correspondente, ignorando.")
            continue

        class_id = label_map[class_name.lower()]
        npy_files = glob.glob(os.path.join(class_dir, '*.npy'))

        for npy_path in npy_files:
            seq = np.load(npy_path)
            # Accept raw 1662-feature sequences from disk
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES_RAW):
                # Select only pose + hands features (remove face)
                seq = seq[:, FEATURE_INDICES]
                all_sequences.append(seq)
                all_labels.append(class_id)
            elif seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                # Already filtered (shouldn't happen, but handle gracefully)
                all_sequences.append(seq)
                all_labels.append(class_id)
            else:
                print(f"  AVISO: {npy_path} shape {seq.shape} ignorado")

    if len(all_sequences) < 2:
        print(f"\nERRO: Apenas {len(all_sequences)} sequências encontradas.")
        print("Execute coleta_dinamica.py primeiro para gravar pelo menos 30 sequências.")
        exit(1)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\n  Features: {NUM_FEATURES_RAW} (raw) → {NUM_FEATURES} (pose+mãos, sem face)")
    print(f"  Face removida: 1404 features de ruído eliminadas!")

    unique_classes = sorted(np.unique(y))
    NUM_CLASSES = len(unique_classes)

    # Remap class IDs to 0..N-1
    class_remap = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}
    class_id_to_label = {new_id: labels[old_id] if old_id < len(labels) else f"Classe {old_id}" 
                         for old_id, new_id in class_remap.items()}
    y = np.array([class_remap[c] for c in y], dtype=np.int64)
    print(f"\nClass remap: {class_remap}")
    print(f"Labels activas: {class_id_to_label}")

    print(f"\nDataset: {len(X)} sequências, {NUM_CLASSES} classes")
    print(f"Shape: {X.shape}")

    print("\nDistribuição:")
    for new_id in sorted(class_id_to_label.keys()):
        count = int(np.sum(y == new_id))
        label = class_id_to_label[new_id]
        bar = "#" * (count // 2)
        print(f"  {new_id:2d}: {label:20s} {count:5d}  {bar}")

    if NUM_CLASSES < 2:
        print("\nERRO: Precisa de pelo menos 2 classes com dados.")
        exit(1)

    # ── Feature Normalization ──
    X_flat = X.reshape(-1, NUM_FEATURES)
    feat_mean = X_flat.mean(axis=0)
    feat_std = X_flat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    X_raw = X.copy()
    X = (X - feat_mean[np.newaxis, np.newaxis, :]) / feat_std[np.newaxis, np.newaxis, :]
    print(f"\nFeature normalization: mean range [{feat_mean.min():.4f}, {feat_mean.max():.4f}]")
    print(f"                       std range  [{feat_std.min():.4f}, {feat_std.max():.4f}]")

    # ── Class Weights ──
    class_counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: {dict(zip([class_id_to_label[i] for i in range(NUM_CLASSES)], [f'{w:.2f}' for w in class_weights]))}")

    # ── Data Augmentation ──
    print(f"\nAugmentation: {AUGMENT_FACTOR}x (ruído + time-warp + dropout + speed)...")
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(AUGMENT_FACTOR):
        augmented = np.array([augment_sequence(seq) for seq in X])
        X_aug.append(augmented)
        y_aug.append(y)
    
    # Mirror: applied on RAW data (before normalization) to keep x-coords in [0,1]
    X_mirror_raw = np.array([mirror_sequence(seq) for seq in X_raw])
    X_mirror_norm = (X_mirror_raw - feat_mean[np.newaxis, np.newaxis, :]) / feat_std[np.newaxis, np.newaxis, :]
    X_aug.append(X_mirror_norm)
    y_aug.append(y)

    X_full = np.vstack(X_aug)
    y_full = np.hstack(y_aug)
    print(f"Dataset aumentado: {len(X_full)} sequências ({len(X_aug)}x fontes)")

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
    )
    print(f"Split: {len(X_train)} treino / {len(X_test)} teste")

    # ── Datasets ──
    train_dataset = GestureSequenceDataset(X_train, y_train)
    test_dataset = GestureSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model (smaller, optimized for small datasets) ──
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    
    model = GestureLSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=0.4,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {total_params:,} parâmetros (vs 2.6M antes)")
    print(f"Arch: LSTM({HIDDEN_SIZE}x{NUM_LAYERS} BiDir) + FC(64→{NUM_CLASSES})")
    print(f"A treinar por {EPOCHS} epochs...\n")

    # ── Training Loop ──
    best_acc = 0.0
    patience_counter = 0
    PATIENCE = 35

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = train_correct / train_total
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                val_preds.extend(predicted.cpu().tolist())

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)

        if epoch % 10 == 0 or epoch == 1:
            pred_dist = Counter(val_preds)
            dist_str = " ".join(f"{class_id_to_label.get(k, '?')[:3]}={v}" 
                                for k, v in sorted(pred_dist.items()))
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {val_acc*100:.1f}% | "
                  f"Loss: {avg_val_loss:.4f} | "
                  f"Pred: [{dist_str}]")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'input_size': NUM_FEATURES,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'sequence_length': SEQUENCE_LENGTH,
                'labels': labels,
                'class_id_to_label': class_id_to_label,
                'class_remap': class_remap,
                'feat_mean': feat_mean.tolist(),
                'feat_std': feat_std.tolist(),
                'feature_indices': FEATURE_INDICES,
                'best_accuracy': best_acc,
            }, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping na epoch {epoch} (paciência: {PATIENCE})")
                break

    # ── Final Evaluation ──
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n{'=' * 62}")
    print(f"  MELHOR PRECISÃO: {best_acc * 100:.2f}%")

    class_correct = {}
    class_total = {}
    all_preds = []
    all_actuals = []

    # Confusion matrix storage: confusion[actual][predicted] = count
    confusion = {i: {j: 0 for j in range(NUM_CLASSES)} for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)

            for pred, actual in zip(predicted, batch_y):
                p = int(pred.item())
                a = int(actual.item())
                all_preds.append(p)
                all_actuals.append(a)
                confusion[a][p] = confusion[a].get(p, 0) + 1
                class_total[a] = class_total.get(a, 0) + 1
                if p == a:
                    class_correct[a] = class_correct.get(a, 0) + 1

    print("\nRelatório por classe:")
    for cls_id in sorted(class_total.keys()):
        correct = class_correct.get(cls_id, 0)
        total = class_total[cls_id]
        acc = correct / total if total > 0 else 0
        label = class_id_to_label.get(cls_id, f"Classe {cls_id}")
        print(f"  {cls_id:2d}: {label:20s} {acc*100:6.1f}%  ({correct}/{total})")

    # ── Confusion Matrix ──
    print(f"\n{'─' * 62}")
    print("  MATRIZ DE CONFUSÃO (linhas=real, colunas=previsto)")
    print(f"{'─' * 62}")
    header = "         "
    for j in range(NUM_CLASSES):
        short = class_id_to_label.get(j, f"C{j}")[:6]
        header += f"{short:>7s}"
    print(header)
    for i in range(NUM_CLASSES):
        row_label = class_id_to_label.get(i, f"C{i}")[:8]
        row = f"  {row_label:8s}"
        for j in range(NUM_CLASSES):
            val = confusion[i][j]
            if i == j and val > 0:
                row += f"  [{val:3d}]"
            elif val > 0:
                row += f"   {val:3d} "
            else:
                row += f"     . "
        print(row)

    # ── Collapse Detection ──
    pred_counts = Counter(all_preds)
    total_preds = len(all_preds)
    dominant_class = pred_counts.most_common(1)[0] if pred_counts else (0, 0)
    dominant_ratio = dominant_class[1] / total_preds if total_preds > 0 else 0

    if dominant_ratio > 0.6:
        dominant_label = class_id_to_label.get(dominant_class[0], f"Classe {dominant_class[0]}")
        print(f"\n  ⚠️  ALERTA DE COLAPSO: O modelo prevê '{dominant_label}' em "
              f"{dominant_ratio*100:.0f}% dos casos!")
        print(f"  Distribuição de previsões: ", end="")
        for cls_id, count in sorted(pred_counts.items()):
            lbl = class_id_to_label.get(cls_id, f"C{cls_id}")
            print(f"{lbl}={count} ", end="")
        print()
        print(f"\n  POSSÍVEIS SOLUÇÕES:")
        print(f"    1. Recolher mais dados variados (posições, distâncias diferentes)")
        print(f"    2. Garantir que gestos são bem distintos entre classes")
        print(f"    3. Verificar se os dados foram recolhidos correctamente")
    else:
        print(f"\n  Distribuição de previsões: ", end="")
        for cls_id, count in sorted(pred_counts.items()):
            lbl = class_id_to_label.get(cls_id, f"C{cls_id}")
            print(f"{lbl}={count} ", end="")
        print()

    size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    print(f"\nModelo guardado: '{MODEL_SAVE_PATH}' ({size_kb:.1f} KB)")
    print(f"Device: {device}")

    print("\n" + "=" * 62)
    if dominant_ratio > 0.6:
        print("  ❌ MODELO COM COLAPSO — sempre prevê a mesma classe!")
        print("     Re-execute coleta_dinamica.py e recolha dados mais variados.")
    elif best_acc >= 0.90:
        print("  OK MODELO PRONTO PARA DEMO!")
    elif best_acc >= 0.75:
        print("  RAZOÁVEL. Recolha mais dados para melhorar.")
    else:
        print("  PRECISA DE MAIS DADOS. Use coleta_dinamica.py")
    print("=" * 62)


if __name__ == '__main__':
    main()

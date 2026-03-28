"""
Zero Barreiras — Treino do Modelo Dinâmico (LSTM)

Treina um LSTM bidireccional sobre sequências de 30 frames × 1662 features.
Suporta GPU (CUDA) automaticamente.

Uso:
  python treinar_dinamico.py
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model.sequence_classifier.sequence_classifier import GestureLSTM

# ── Config ──
SEQUENCE_DIR = 'model/sequence_classifier/sequences'
LABEL_CSV = 'model/sequence_classifier/sequence_classifier_label.csv'
MODEL_SAVE_PATH = 'model/sequence_classifier/sequence_classifier.pt'
SEQUENCE_LENGTH = 30
NUM_FEATURES = 1662

BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001
AUGMENT_FACTOR = 3
AUGMENT_NOISE = 0.01
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


def time_warp(sequence, sigma=0.2):
    """Apply random temporal warping to a sequence.
    Simulates different gesture speeds.
    """
    seq_len = len(sequence)
    # Generate random time steps
    orig_steps = np.arange(seq_len)
    warp = np.random.normal(loc=1.0, scale=sigma, size=seq_len)
    warp = np.cumsum(warp)
    warp = warp / warp[-1] * (seq_len - 1)  # normalize to original range

    # Interpolate each feature
    warped = np.zeros_like(sequence)
    for feat_idx in range(sequence.shape[1]):
        warped[:, feat_idx] = np.interp(orig_steps, warp, sequence[:, feat_idx])

    return warped


def augment_sequence(sequence, noise_std=AUGMENT_NOISE):
    """Add Gaussian noise + temporal warping."""
    noisy = sequence + np.random.normal(0, noise_std, sequence.shape)
    return time_warp(noisy.astype(np.float32))


# ── Main ──
def main():
    print("=" * 62)
    print("  TREINO DINÂMICO — Zero Barreiras (LSTM)")
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
            # Ensure correct shape
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                all_sequences.append(seq)
                all_labels.append(class_id)
            else:
                print(f"  AVISO: {npy_path} shape {seq.shape} != ({SEQUENCE_LENGTH}, {NUM_FEATURES})")

    if len(all_sequences) < 2:
        print(f"\nERRO: Apenas {len(all_sequences)} sequências encontradas.")
        print("Execute coleta_dinamica.py primeiro para gravar pelo menos 30 sequências.")
        exit(1)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    unique_classes = sorted(np.unique(y))
    NUM_CLASSES = len(unique_classes)

    # Remap class IDs to 0..N-1 (critical for softmax with few classes)
    class_remap = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}
    class_id_to_label = {new_id: labels[old_id] if old_id < len(labels) else f"Classe {old_id}" 
                         for old_id, new_id in class_remap.items()}
    y = np.array([class_remap[c] for c in y], dtype=np.int64)
    print(f"\nClass remap: {class_remap}")
    print(f"Labels activas: {class_id_to_label}")

    print(f"\nDataset: {len(X)} sequências, {NUM_CLASSES} classes")
    print(f"Shape: {X.shape}")

    print("\nDistribuição:")
    for cls_id in sorted(unique_classes):
        count = np.sum(y == cls_id)
        label = labels[cls_id] if cls_id < len(labels) else f"Classe {cls_id}"
        bar = "#" * (count // 2)
        print(f"  {cls_id:2d}: {label:20s} {count:5d}  {bar}")

    if NUM_CLASSES < 2:
        print("\nERRO: Precisa de pelo menos 2 classes com dados.")
        exit(1)

    # ── Data Augmentation ──
    print(f"\nAugmentation: {AUGMENT_FACTOR}x (ruído + time-warping)...")
    X_aug = [X]
    y_aug = [y]
    for _ in range(AUGMENT_FACTOR):
        augmented = np.array([augment_sequence(seq) for seq in X])
        X_aug.append(augmented)
        y_aug.append(y)

    X_full = np.vstack(X_aug)
    y_full = np.hstack(y_aug)
    print(f"Dataset aumentado: {len(X_full)} sequências ({AUGMENT_FACTOR + 1}x)")

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

    # ── Model ──
    model = GestureLSTM(
        input_size=NUM_FEATURES,
        hidden_size=128,
        num_layers=3,
        num_classes=NUM_CLASSES,  # Only classes with actual training data
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {total_params:,} parâmetros")
    print(f"A treinar por {EPOCHS} epochs...\n")

    # ── Training Loop ──
    best_acc = 0.0
    patience_counter = 0
    PATIENCE = 25

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
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)

        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {val_acc*100:.1f}% | "
                  f"Loss: {avg_val_loss:.4f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'input_size': NUM_FEATURES,
                'sequence_length': SEQUENCE_LENGTH,
                'labels': labels,
                'class_id_to_label': class_id_to_label,
                'class_remap': class_remap,
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

    # Per-class accuracy
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)

            for pred, actual in zip(predicted, batch_y):
                cls = int(actual.item())
                class_total[cls] = class_total.get(cls, 0) + 1
                if pred == actual:
                    class_correct[cls] = class_correct.get(cls, 0) + 1

    print("\nRelatório por classe:")
    for cls_id in sorted(class_total.keys()):
        correct = class_correct.get(cls_id, 0)
        total = class_total[cls_id]
        acc = correct / total if total > 0 else 0
        label = labels[cls_id] if cls_id < len(labels) else f"Classe {cls_id}"
        print(f"  {cls_id:2d}: {label:20s} {acc*100:6.1f}%  ({correct}/{total})")

    size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    print(f"\nModelo guardado: '{MODEL_SAVE_PATH}' ({size_kb:.1f} KB)")
    print(f"Device: {device}")

    print("\n" + "=" * 62)
    if best_acc >= 0.90:
        print("  MODELO PRONTO PARA DEMO!")
    elif best_acc >= 0.75:
        print("  BOM! Recolha mais dados para melhorar.")
    else:
        print("  PRECISA DE MAIS DADOS. Use coleta_dinamica.py")
    print("=" * 62)


if __name__ == '__main__':
    main()

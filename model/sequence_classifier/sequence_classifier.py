"""
Okuyeva — Sequence Classifier (LSTM for Dynamic Gestures)

Architecture:
  Input:  (batch, seq_len, 1662)  — 30 frames of holistic features
  LSTM:   2 layers bidirectional (64 hidden) with dropout
  Output: (batch, num_classes)    — gesture probabilities

Optimized for small datasets (<100 samples) with:
  - Smaller architecture to prevent overfitting
  - Feature normalization (mean/std from training data)
  - Class mapping for partial label training
"""
import os
import numpy as np
import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    """Bidirectional LSTM for temporal gesture classification.
    
    Optimized for small datasets:
    - 2 layers instead of 3 (less overfitting)
    - 64 hidden units instead of 128
    - Total ~660K params vs 2.6M before
    """

    def __init__(self, input_size=1662, hidden_size=64, num_layers=2,
                 num_classes=15, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Bidirectional → output is 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        # Use both first and last hidden states for richer representation
        last_fwd = lstm_out[:, -1, :self.hidden_size]
        last_bwd = lstm_out[:, 0, self.hidden_size:]
        combined = torch.cat([last_fwd, last_bwd], dim=1)
        return self.classifier(combined)


class SequenceClassifier:
    """Wrapper for inference with pre-trained GestureLSTM.

    Usage:
        classifier = SequenceClassifier('model/sequence_classifier/sequence_classifier.pt')
        gesture_id, confidence = classifier(sequence_of_30_frames)
    """

    def __init__(self, model_path=None, labels_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.labels = []
        self.class_id_to_label = {}
        self.feat_mean = None
        self.feat_std = None
        self.feature_indices = None  # indices to select from raw 1662 features

        if labels_path and os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8-sig') as f:
                self.labels = [row.strip() for row in f if row.strip()]

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get('num_classes', len(self.labels))
            input_size = checkpoint.get('input_size', 1662)
            hidden_size = checkpoint.get('hidden_size', 64)
            num_layers = checkpoint.get('num_layers', 2)

            # Load feature indices (for face removal)
            if 'feature_indices' in checkpoint:
                self.feature_indices = checkpoint['feature_indices']
                print(f"[SEQ] Feature filter: {input_size} features (from 1662 raw)")

            # Load class mapping
            if 'class_id_to_label' in checkpoint:
                self.class_id_to_label = {
                    int(k): v for k, v in checkpoint['class_id_to_label'].items()
                }
            else:
                self.class_id_to_label = {
                    i: self.labels[i] if i < len(self.labels) else f"Classe {i}"
                    for i in range(num_classes)
                }

            # Load normalization stats
            if 'feat_mean' in checkpoint:
                self.feat_mean = torch.FloatTensor(checkpoint['feat_mean']).to(self.device)
                self.feat_std = torch.FloatTensor(checkpoint['feat_std']).to(self.device)
                # Replace zeros to avoid division by zero
                self.feat_std[self.feat_std < 1e-8] = 1.0

            self.model = GestureLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[SEQ] Model: {model_path} on {self.device}")
            print(f"[SEQ] Classes: {num_classes} | Arch: LSTM({hidden_size}x{num_layers}) | Features: {input_size}")
            print(f"[SEQ] Labels: {self.class_id_to_label}")
            print(f"[SEQ] Normalization: {'Yes' if self.feat_mean is not None else 'No'}")

    @property
    def is_loaded(self):
        return self.model is not None

    @torch.no_grad()
    def __call__(self, sequence):
        """Classify a sequence of features.

        Args:
            sequence: numpy array of shape (seq_len, num_features)
                      Can be raw 1662 features or already filtered.

        Returns:
            (class_id, confidence) tuple
        """
        if not self.is_loaded:
            return -1, 0.0

        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Filter features if model was trained with feature selection
        if self.feature_indices is not None and x.shape[2] > len(self.feature_indices):
            idx = torch.LongTensor(self.feature_indices).to(self.device)
            x = torch.index_select(x, 2, idx)

        # Apply feature normalization if available
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / self.feat_std

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)

        return int(idx.item()), float(confidence.item())

    def get_label(self, class_id):
        if class_id in self.class_id_to_label:
            return self.class_id_to_label[class_id]
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"Classe {class_id}"

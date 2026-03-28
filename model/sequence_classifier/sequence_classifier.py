"""
Zero Barreiras — Sequence Classifier (LSTM for Dynamic Gestures)

Architecture:
  Input:  (batch, seq_len, 1662)  — 30 frames of holistic features
  LSTM:   3 layers (64→128→64) with dropout
  Output: (batch, num_classes)    — gesture probabilities

Handles variable-length sequences via padding.
Supports GPU acceleration (CUDA) when available.
"""
import os
import numpy as np
import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    """Bidirectional LSTM for temporal gesture classification."""

    def __init__(self, input_size=1662, hidden_size=128, num_layers=3,
                 num_classes=15, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


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
        self.class_id_to_label = {}  # Maps model output ID → gesture name

        if labels_path and os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8-sig') as f:
                self.labels = [row.strip() for row in f if row.strip()]

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get('num_classes', len(self.labels))
            input_size = checkpoint.get('input_size', 1662)

            # Load class mapping (new format with remapped IDs)
            if 'class_id_to_label' in checkpoint:
                # Convert string keys back to int
                self.class_id_to_label = {
                    int(k): v for k, v in checkpoint['class_id_to_label'].items()
                }
            else:
                # Legacy format: assume 1:1 mapping
                self.class_id_to_label = {
                    i: self.labels[i] if i < len(self.labels) else f"Classe {i}"
                    for i in range(num_classes)
                }

            self.model = GestureLSTM(
                input_size=input_size,
                num_classes=num_classes,
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[SEQ] Loaded model from {model_path} on {self.device}")
            print(f"[SEQ] Classes: {num_classes}, Input: {input_size}")
            print(f"[SEQ] Labels: {self.class_id_to_label}")

    @property
    def is_loaded(self):
        return self.model is not None

    @torch.no_grad()
    def __call__(self, sequence):
        """Classify a sequence of features.

        Args:
            sequence: numpy array of shape (seq_len, num_features)

        Returns:
            (class_id, confidence) tuple
        """
        if not self.is_loaded:
            return -1, 0.0

        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
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

"""PyTorch neural network architectures for sign language sequence recognition.

Models:
- SignGRU: 2-layer GRU with Dropout and Linear classification head.
- Sign1DCNN: Temporal 1D-CNN with Conv1d, BatchNorm, ReLU, AdaptiveAvgPool1d, and Linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignGRU(nn.Module):
    """Gated Recurrent Unit (GRU) for temporal gesture sequence modeling."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Use last timestep representation
        last_timestep = out[:, -1, :]
        logits = self.fc(last_timestep)
        return logits


class Sign1DCNN(nn.Module):
    """1D Convolutional Neural Network for rapid temporal feature extraction."""

    def __init__(
        self,
        input_dim: int = 126,
        num_classes: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        # PyTorch Conv1d expects (batch, channels, seq_len)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len / 2

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # pool temporal dimension to 1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        features = self.conv_net(x).squeeze(-1)  # (batch_size, 128)
        logits = self.classifier(features)
        return logits


def build_model(
    model_type: str,
    input_dim: int = 126,
    num_classes: int = 6,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "gru":
        return SignGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type in ("cnn", "1dcnn"):
        return Sign1DCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gru' or 'cnn'.")

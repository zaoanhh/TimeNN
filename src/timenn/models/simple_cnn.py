"""
Simple 1D CNN for Time Series Classification.

Architecture: Input -> Conv1D blocks -> GlobalPool -> MLP -> Output
Based on SimpleCNN1D from lms_cnn_net.py but as standalone model.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimpleCNNConfig:
    """Configuration for Simple 1D CNN model."""

    # CNN parameters
    cnn_layers: Tuple[int, ...] = (64, 128, 256)
    cnn_kernel_size: int = 5
    cnn_stride: int = 2
    cnn_dropout: float = 0.0

    # Head parameters
    mlp_hidden: int = 256
    dropout: float = 0.1


class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN backbone for feature extraction.

    Architecture: Conv1D -> BN -> GELU -> MaxPool (repeat) -> GlobalPool
    """

    def __init__(
        self,
        in_channels: int,
        layers: Tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.layers = layers
        self.num_layers = len(layers)

        # Build convolutional layers
        conv_layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(layers):
            conv = nn.Conv1d(
                prev_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride if i > 0 else 1,  # No stride on first layer
                padding=kernel_size // 2,
            )
            conv_layers.extend(
                [
                    conv,
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_layers)

        # Calculate output features
        self.out_features = prev_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN layers."""
        return self.conv_blocks(x)


class SimpleCNNClassifier(nn.Module):
    """
    Simple 1D CNN Classifier for time series classification.

    Architecture: Input -> InstanceNorm -> CNN -> GlobalPool -> MLP -> Output
    """

    def __init__(
        self,
        c_in: int,
        input_length: int,
        num_classes: int,
        config: SimpleCNNConfig = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.input_length = input_length
        self.num_classes = num_classes

        if config is None:
            config = SimpleCNNConfig()

        self.config = config

        # Input normalization
        self.norm = nn.InstanceNorm1d(c_in, affine=True)

        # CNN backbone
        self.cnn = SimpleCNN1D(
            in_channels=c_in,
            layers=config.cnn_layers,
            kernel_size=config.cnn_kernel_size,
            stride=config.cnn_stride,
            dropout=config.cnn_dropout,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.cnn.out_features, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, L]

        Returns:
            Logits [B, num_classes]
        """
        # Normalize input
        out = self.norm(x)

        # CNN processing
        out = self.cnn(out)

        # Global pooling
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # [B, cnn_out_features]

        return self.head(out)


def build_simple_cnn(
    c_in: int,
    input_length: int,
    num_classes: int,
    cnn_layers: Tuple[int, ...] = (64, 128, 256),
    cnn_kernel_size: int = 5,
    cnn_stride: int = 2,
    cnn_dropout: float = 0.0,
    mlp_hidden: int = 256,
    dropout: float = 0.1,
) -> SimpleCNNClassifier:
    """
    Convenience function to build Simple CNN classifier.

    Args:
        c_in: Number of input channels
        input_length: Input sequence length
        num_classes: Number of output classes
        cnn_layers: CNN layer configuration (output channels per layer)
        cnn_kernel_size: Kernel size for CNN layers
        cnn_stride: Stride for CNN layers (except first)
        cnn_dropout: Dropout rate in CNN layers
        mlp_hidden: Hidden dimension of MLP head
        dropout: Dropout rate in MLP head

    Returns:
        Configured SimpleCNNClassifier instance
    """
    config = SimpleCNNConfig(
        cnn_layers=cnn_layers,
        cnn_kernel_size=cnn_kernel_size,
        cnn_stride=cnn_stride,
        cnn_dropout=cnn_dropout,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    )

    return SimpleCNNClassifier(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        config=config,
    )

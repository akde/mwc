"""
1D Convolutional Neural Network classifier for thermal time-series classification.

Architecture:
    Input (batch, seq_len, 1) → Conv Blocks → Global Pooling → FC → Softmax

Key features:
- Simple 1D convolutions (no dilations or causal constraints)
- Batch normalization for stable training
- MaxPooling between conv blocks
- Multiple pooling strategies (last, avg, max)

Motivation: TCN success (F1=0.5396 with 1 layer) suggests simpler CNNs might work well.
This implementation tests whether removing TCN's causal/dilated constraints helps or hurts.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    Basic 1D convolution block: Conv1d → BatchNorm → ReLU → Dropout.

    Follows standard CNN patterns without causal constraints.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()

        # Same padding to preserve sequence length
        padding = kernel_size // 2

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        self.block = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Thermal1DCNNClassifier(nn.Module):
    """
    Simple 1D CNN for thermal time-series classification.

    Processes univariate thermal intensity sequences and outputs
    class probabilities for 4 material classes.
    """

    def __init__(
        self,
        input_size: int = 1,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = 'last',
        use_batch_norm: bool = True,
        pool_stride: int = 2
    ):
        """
        Args:
            input_size: Input feature dimension (1 for univariate thermal)
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Global pooling strategy ('last', 'avg', 'max')
            use_batch_norm: Whether to use batch normalization
            pool_stride: Stride for MaxPool1d between conv blocks
        """
        super().__init__()

        # Default architecture: 3 conv blocks with decreasing kernels
        if num_filters is None:
            num_filters = [64, 128, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        assert len(num_filters) == len(kernel_sizes), \
            "num_filters and kernel_sizes must have same length"

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.pooling = pooling
        self.pool_stride = pool_stride

        # Build conv blocks with max pooling between them
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = input_size
        for i, (filters, kernel) in enumerate(zip(num_filters, kernel_sizes)):
            self.conv_blocks.append(
                ConvBlock(in_channels, filters, kernel, dropout, use_batch_norm)
            )
            # Add max pooling between blocks (except after last block)
            if i < len(num_filters) - 1:
                self.pools.append(nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride))
            else:
                self.pools.append(nn.Identity())  # No pooling after last block
            in_channels = filters

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters[-1], num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            lengths: Actual sequence lengths [batch]
            mask: Attention mask [batch, seq_len] where 1=real data, 0=padding

        Returns:
            Class logits [batch, num_classes]
        """
        # Transpose for Conv1d: [batch, input_size, seq_len]
        x = x.transpose(1, 2)

        # Apply conv blocks with pooling
        for conv_block, pool in zip(self.conv_blocks, self.pools):
            x = pool(conv_block(x))

        # Global pooling over time dimension
        # x shape: [batch, num_filters[-1], reduced_seq_len]
        if self.pooling == 'last':
            # Use the last timestep (real data is at end with beginning padding)
            pooled = x[:, :, -1]
        elif self.pooling == 'avg':
            # Global average pooling
            pooled = x.mean(dim=2)
        elif self.pooling == 'max':
            # Global max pooling
            pooled, _ = x.max(dim=2)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> str:
        """Return a string describing the architecture."""
        layers_str = "→".join([f"Conv({k}@{f})" for k, f in zip(self.kernel_sizes, self.num_filters)])
        return f"Input → {layers_str} → {self.pooling.upper()} → FC({self.num_filters[-1]}→4)"


def create_cnn_1d_model(
    input_size: int = 1,
    num_filters: List[int] = None,
    kernel_sizes: List[int] = None,
    num_classes: int = 4,
    dropout: float = 0.3,
    pooling: str = 'last',
    use_batch_norm: bool = True,
    pool_stride: int = 2
) -> Thermal1DCNNClassifier:
    """Factory function to create 1D CNN model."""
    return Thermal1DCNNClassifier(
        input_size=input_size,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        use_batch_norm=use_batch_norm,
        pool_stride=pool_stride
    )


if __name__ == '__main__':
    print("Testing Thermal1DCNNClassifier...")

    # Create model with default architecture
    model = create_cnn_1d_model()

    print(f"Architecture: {model.get_architecture_summary()}")
    print(f"Parameters: {model.get_num_parameters():,}")

    # Test with synthetic data
    batch_size = 8
    seq_len = 1000
    x = torch.randn(batch_size, seq_len, 1)
    lengths = torch.tensor([1000, 900, 800, 700, 600, 500, 400, 300])

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output sample: {torch.softmax(logits[0], dim=0)}")

    # Test different pooling strategies
    print("\nPooling strategy comparison:")
    for pooling in ['last', 'avg', 'max']:
        model_p = create_cnn_1d_model(pooling=pooling)
        with torch.no_grad():
            logits_p = model_p(x, lengths)
        print(f"  {pooling.upper():4s}: output shape {logits_p.shape}, params={model_p.get_num_parameters():,}")

    # Test different architectures
    print("\nArchitecture variants:")
    configs = [
        ([64, 128, 128], [7, 5, 3]),       # Default
        ([64, 128], [7, 5]),               # 2 layers
        ([64, 128, 128, 128], [7, 5, 3, 3]), # 4 layers
        ([128, 128, 128], [7, 5, 3]),      # Wider
        ([64, 128, 128], [3, 3, 3]),       # Small kernels
    ]

    for filters, kernels in configs:
        model_v = create_cnn_1d_model(num_filters=filters, kernel_sizes=kernels)
        print(f"  {filters} kernels={kernels}: {model_v.get_num_parameters():,} params")

    print("\nTest passed!")

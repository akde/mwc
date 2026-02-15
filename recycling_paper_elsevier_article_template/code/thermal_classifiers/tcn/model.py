"""
Temporal Convolutional Network (TCN) classifier for thermal time-series classification.

Architecture:
    Input (batch, 1, seq_len) → TCN Blocks (dilated causal convolutions) →
    Global Average Pooling → FC(hidden, num_classes) → Softmax

Key features:
- Dilated causal convolutions with exponentially increasing dilation factors
- Residual connections for gradient flow
- Parallel computation (faster than LSTM/GRU)
- Effective receptive field grows exponentially with depth

Reference: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
           for Sequence Modeling" (Bai et al., 2018)
"""

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from typing import List, Optional


class Chomp1d(nn.Module):
    """Remove the extra padding at the end of the sequence for causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single temporal block with dilated causal convolutions.

    Structure:
        Conv1d → Chomp → ReLU → Dropout → Conv1d → Chomp → ReLU → Dropout + Residual
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential for the main path
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Initialize weights using normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN).

    Stacks temporal blocks with exponentially increasing dilation factors,
    creating a large receptive field with relatively few layers.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # Causal padding

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, seq_len]

        Returns:
            Output tensor [batch, num_channels[-1], seq_len]
        """
        return self.network(x)


class ThermalTCNClassifier(nn.Module):
    """
    TCN for thermal time-series classification.

    Processes univariate thermal intensity sequences and outputs
    class probabilities for 4 material classes.
    """

    def __init__(
        self,
        input_size: int = 1,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
        pooling: str = 'avg'
    ):
        """
        Args:
            input_size: Input feature dimension (1 for univariate thermal)
            num_channels: List of channel sizes for each TCN layer
            kernel_size: Convolution kernel size
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Pooling strategy ('avg', 'max', or 'last')
        """
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64, 64]  # 4 layers, 64 channels each

        self.num_channels = num_channels
        self.pooling = pooling

        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], num_classes)
        )

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, len(num_channels))

    def _calculate_receptive_field(self, kernel_size: int, num_layers: int) -> int:
        """Calculate the effective receptive field of the TCN."""
        # RF = 1 + sum(dilation * (kernel_size - 1) * 2) for each layer
        # With dilation = 2^i and 2 convs per block
        rf = 1
        for i in range(num_layers):
            dilation = 2 ** i
            rf += dilation * (kernel_size - 1) * 2
        return rf

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
                  Required when padding is at BEGINNING (aligned with RGB)

        Returns:
            Class logits [batch, num_classes]
        """
        # Transpose for Conv1d: [batch, input_size, seq_len]
        x = x.transpose(1, 2)

        # TCN forward
        x = self.tcn(x)  # [batch, num_channels[-1], seq_len]

        # Pool over time dimension
        if self.pooling == 'avg':
            # Masked average pooling
            pooled = self._masked_avg_pool(x, lengths, mask)
        elif self.pooling == 'max':
            # Masked max pooling
            pooled = self._masked_max_pool(x, lengths, mask)
        elif self.pooling == 'last':
            # Use the last valid position (always last position with beginning padding)
            pooled = self._get_last_valid(x, lengths)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits

    def _masked_avg_pool(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Average pooling over valid positions (handles beginning padding)."""
        batch_size, channels, seq_len = x.shape
        device = x.device

        if mask is not None:
            # Use explicit mask: [batch, seq_len] -> [batch, 1, seq_len]
            pool_mask = mask.unsqueeze(1).float()
        else:
            # Fallback: create mask assuming beginning padding
            # Valid positions are (seq_len - length) to (seq_len - 1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
            real_start = (seq_len - lengths).unsqueeze(1).unsqueeze(2)
            pool_mask = (positions >= real_start).float()

        # Masked average
        masked_x = x * pool_mask
        sum_x = masked_x.sum(dim=2)  # [batch, channels]
        lengths_expanded = lengths.float().unsqueeze(1)
        avg_x = sum_x / lengths_expanded.clamp(min=1)

        return avg_x

    def _masked_max_pool(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Max pooling over valid positions (handles beginning padding)."""
        batch_size, channels, seq_len = x.shape
        device = x.device

        if mask is not None:
            # Use explicit mask: [batch, seq_len] -> [batch, 1, seq_len]
            # Must convert to bool for ~ operator
            pool_mask = mask.unsqueeze(1).bool()
        else:
            # Fallback: create mask assuming beginning padding
            positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
            real_start = (seq_len - lengths).unsqueeze(1).unsqueeze(2)
            pool_mask = positions >= real_start

        # Set padded positions to very negative value
        x_masked = x.clone()
        x_masked[~pool_mask.expand_as(x)] = float('-inf')

        # Max pooling
        max_x, _ = x_masked.max(dim=2)

        return max_x

    def _get_last_valid(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the last valid position for each sample.

        With beginning padding, the last valid position is always (seq_len - 1)
        since real data is at the end of the sequence.
        """
        # With beginning padding, real data extends to the end
        # So last valid position is always seq_len - 1
        last_x = x[:, :, -1]  # [batch, channels]
        return last_x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tcn_model(
    input_size: int = 1,
    num_channels: List[int] = None,
    kernel_size: int = 3,
    num_classes: int = 4,
    dropout: float = 0.2,
    pooling: str = 'avg'
) -> ThermalTCNClassifier:
    """Factory function to create TCN model."""
    return ThermalTCNClassifier(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling
    )


if __name__ == '__main__':
    print("Testing ThermalTCNClassifier...")

    model = create_tcn_model(
        num_channels=[64, 64, 64, 64],
        kernel_size=3,
        dropout=0.2,
        pooling='avg'
    )

    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Receptive field: {model.receptive_field} frames")

    # Test with synthetic data
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 1)
    lengths = torch.tensor([100, 90, 80, 70, 60, 50, 40, 30])

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test different pooling strategies
    for pooling in ['avg', 'max', 'last']:
        model_p = create_tcn_model(pooling=pooling)
        with torch.no_grad():
            logits_p = model_p(x, lengths)
        print(f"{pooling.capitalize()} pooling output shape: {logits_p.shape}")

    # Calculate receptive fields for different configurations
    print("\nReceptive fields for different configurations:")
    for kernel_size in [3, 5, 7]:
        for num_layers in [4, 6, 8]:
            rf = 1
            for i in range(num_layers):
                rf += (2 ** i) * (kernel_size - 1) * 2
            print(f"  kernel={kernel_size}, layers={num_layers}: RF={rf}")

    print("\nTest passed!")

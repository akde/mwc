"""
Transformer encoder classifier for thermal time-series classification.

Architecture:
    Input (batch, seq_len, 1) → Linear(1, d_model) → PositionalEncoding →
    TransformerEncoder(layers, heads) → CLS token pooling → FC(d_model, num_classes)

Key features:
- Learnable positional embeddings (not sinusoidal)
- CLS token prepended for classification
- Padding mask for variable-length sequences
- Multi-head self-attention captures long-range dependencies
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.

    Unlike sinusoidal, these are learned during training and can adapt
    to the specific patterns in thermal time-series.
    """

    def __init__(self, d_model: int, max_len: int = 1001, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (+1 for CLS token)
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class ThermalTransformerClassifier(nn.Module):
    """
    Transformer encoder for thermal time-series classification.

    Uses a CLS token prepended to the sequence for classification,
    similar to BERT architecture.
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
        max_len: int = 1001,
        pooling: str = 'cls'
    ):
        """
        Args:
            input_size: Input feature dimension (1 for univariate thermal)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            max_len: Maximum sequence length (+1 for CLS token)
            pooling: Pooling strategy ('cls' or 'mean')
        """
        super().__init__()

        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
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
                  Required when padding is at BEGINNING (aligned with RGB)

        Returns:
            Class logits [batch, num_classes]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input to d_model dimension
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask (True = masked/ignored)
        # With beginning padding, we use the mask directly
        src_key_padding_mask = self._create_padding_mask(lengths, seq_len, device, mask)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool
        if self.pooling == 'cls':
            # Use CLS token representation
            pooled = x[:, 0, :]  # [batch, d_model]
        elif self.pooling == 'mean':
            # Mean pooling over non-padded positions (excluding CLS)
            pooled = self._mean_pool(x[:, 1:, :], lengths, mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits

    def _create_padding_mask(
        self,
        lengths: torch.Tensor,
        seq_len: int,
        device: torch.device,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create attention padding mask for transformer.

        With beginning padding (aligned with RGB):
        - Padding is at positions 1 to (seq_len - length) after CLS
        - Real data is at positions (seq_len - length + 1) to seq_len after CLS
        - CLS token at position 0 is never masked

        Args:
            lengths: Actual sequence lengths [batch] (NOT including CLS)
            seq_len: Sequence length (NOT including CLS)
            device: Torch device
            mask: Optional explicit mask [batch, seq_len] where 1=real, 0=padding

        Returns:
            Boolean mask [batch, seq_len+1] where True = padded position (ignored)
        """
        batch_size = lengths.size(0)
        max_len_with_cls = seq_len + 1

        if mask is not None:
            # Use explicit mask: prepend False for CLS, invert (1=real -> False, 0=pad -> True)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            # Invert: 0 in input mask (padding) -> True in output (masked)
            padding_mask = (mask < 0.5)  # [batch, seq_len]
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [batch, seq_len+1]
            return full_mask
        else:
            # Fallback: create mask from lengths assuming beginning padding
            # Position 0 = CLS (not masked)
            # Positions 1 to (seq_len - length) = padding (masked)
            # Positions (seq_len - length + 1) to seq_len = real data (not masked)
            positions = torch.arange(max_len_with_cls, device=device).unsqueeze(0)
            # Start of real data (after padding) in CLS-prepended sequence
            real_start = seq_len - lengths.unsqueeze(1) + 1  # +1 for CLS
            # Mask is True for padding: position > 0 AND position < real_start
            full_mask = (positions > 0) & (positions < real_start)
            return full_mask

    def _mean_pool(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean pooling over non-padded positions (handles beginning padding)."""
        batch_size, max_len, d_model = x.shape
        device = x.device

        if mask is not None:
            # Use explicit mask directly (1=real, 0=padding)
            pool_mask = mask.unsqueeze(2).float()  # [batch, max_len, 1]
        else:
            # Fallback: create mask from lengths assuming beginning padding
            # Real data is at positions (max_len - length) to (max_len - 1)
            positions = torch.arange(max_len, device=device).unsqueeze(0)
            real_start = max_len - lengths.unsqueeze(1)
            pool_mask = (positions >= real_start).unsqueeze(2).float()

        masked_x = x * pool_mask
        sum_x = masked_x.sum(dim=1)
        lengths_expanded = lengths.float().unsqueeze(1)
        mean_x = sum_x / lengths_expanded.clamp(min=1)

        return mean_x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(
    input_size: int = 1,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 128,
    num_classes: int = 4,
    dropout: float = 0.1,
    max_len: int = 1001,
    pooling: str = 'cls'
) -> ThermalTransformerClassifier:
    """Factory function to create Transformer model."""
    return ThermalTransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        dropout=dropout,
        max_len=max_len,
        pooling=pooling
    )


if __name__ == '__main__':
    print("Testing ThermalTransformerClassifier...")

    model = create_transformer_model(
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        pooling='cls'
    )

    print(f"Model parameters: {model.get_num_parameters():,}")

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

    # Test mean pooling
    model_mean = create_transformer_model(pooling='mean')
    with torch.no_grad():
        logits_mean = model_mean(x, lengths)
    print(f"Mean pooling output shape: {logits_mean.shape}")

    # Verify attention mask
    print("\nVerifying attention mask...")
    mask = model._create_padding_mask(lengths, seq_len + 1, x.device)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum per sample (should increase with shorter sequences): {mask.sum(dim=1).tolist()}")

    print("\nTest passed!")

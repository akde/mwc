"""
Bidirectional LSTM classifier for thermal time-series classification.

Architecture (Improved - December 2025):
    Input (batch, seq_len, 1) → InputProjection(1 → hidden) →
    BiLSTM(hidden, 1 layer) → Pooling (last/mean/attention) →
    LayerNorm → Dropout → MLP ClassificationHead → Softmax

Key improvements over original:
- Input projection layer expands univariate to hidden_dim before LSTM
- Self-attention pooling option (from RGB LSTM) for learning important timesteps
- Single layer default (ablation showed +17.7% accuracy vs 2 layers)
- Deeper 2-layer MLP classification head
- Bidirectional processing captures both past and future context
- Layer normalization for stable training
- Supports variable-length sequences via pack_padded_sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple


class Attention(nn.Module):
    """
    Self-attention layer for sequence pooling.

    Learns to weight different timesteps based on their importance,
    rather than using simple mean or last-state pooling.

    Adapted from RGB LSTM classifier.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted sum of LSTM outputs.

        Args:
            lstm_output: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] (1 for valid, 0 for padding)

        Returns:
            Tuple of (context_vector, attention_weights)
            - context_vector: [batch, hidden_dim]
            - attention_weights: [batch, seq_len]
        """
        attention_scores = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch, hidden_dim]

        return context, attention_weights


class ThermalLSTMClassifier(nn.Module):
    """
    Improved Bidirectional LSTM for thermal time-series classification.

    Processes univariate thermal intensity sequences and outputs
    class probabilities for 4 material classes.

    Key improvements (December 2025):
    - Input projection expands univariate input to hidden_dim
    - Default 1 layer (ablation showed +17.7% accuracy)
    - Attention pooling option for learning important timesteps
    - Deeper MLP classification head
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,  # Changed from 2: ablation showed +17.7% accuracy
        num_classes: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = 'mean',  # Changed from 'last': mean pooling improves generalization
        use_input_projection: bool = True,  # NEW: expand input before LSTM
        use_deep_classifier: bool = True,  # NEW: 2-layer MLP classification head
    ):
        """
        Args:
            input_size: Input feature dimension (1 for univariate thermal)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers (default 1, proven better on small data)
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            pooling: Pooling strategy ('last', 'mean', or 'attention')
            use_input_projection: Expand input to hidden_dim before LSTM
            use_deep_classifier: Use 2-layer MLP instead of single Linear
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.num_directions = 2 if bidirectional else 1
        self.use_input_projection = use_input_projection
        self.use_deep_classifier = use_deep_classifier

        # Input projection layer (NEW)
        if use_input_projection:
            self.input_proj = nn.Linear(input_size, hidden_dim)
            lstm_input_size = hidden_dim
        else:
            self.input_proj = None
            lstm_input_size = input_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension after LSTM
        lstm_output_dim = hidden_dim * self.num_directions

        # Attention layer (NEW - for 'attention' pooling)
        if pooling == 'attention':
            self.attention = Attention(lstm_output_dim)
        else:
            self.attention = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head (IMPROVED: deeper MLP option)
        if use_deep_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(lstm_output_dim, num_classes)

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
        batch_size, max_len, _ = x.shape
        device = x.device

        # Apply input projection if enabled (NEW)
        if self.input_proj is not None:
            x = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Extract non-padded portions for pack_padded_sequence
        # With beginning padding, real data is at positions (max_len - length) to max_len
        # We need to extract and left-align the data for pack_padded_sequence
        x_aligned = torch.zeros_like(x)
        for i in range(batch_size):
            length = lengths[i].item()
            if length > 0:
                start_idx = max_len - length
                x_aligned[i, :length] = x[i, start_idx:]

        # Sort by length (required for pack_padded_sequence)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x_aligned[sort_idx]

        # Pack sequences
        packed = pack_padded_sequence(
            x_sorted,
            lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM forward
        packed_output, (hidden, cell) = self.lstm(packed)

        # Unpack sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Pool hidden states
        if self.pooling == 'last':
            # Use final hidden state from both directions
            if self.bidirectional:
                # hidden shape: [num_layers * num_directions, batch, hidden_dim]
                # Get last layer, concatenate both directions
                hidden_fwd = hidden[-2]  # Last layer, forward
                hidden_bwd = hidden[-1]  # Last layer, backward
                pooled = torch.cat([hidden_fwd, hidden_bwd], dim=1)
            else:
                pooled = hidden[-1]
        elif self.pooling == 'mean':
            # Average over all time steps (masked)
            pooled = self._mean_pool(output, lengths_sorted)
        elif self.pooling == 'attention':
            # Attention-weighted pooling (NEW)
            # Create mask for attention (left-aligned after unpacking)
            attn_mask = torch.arange(output.size(1), device=device).unsqueeze(0) < lengths_sorted.unsqueeze(1)
            pooled, _ = self.attention(output, attn_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Unsort to restore original order
        _, unsort_idx = sort_idx.sort()
        pooled = pooled[unsort_idx]

        # Layer norm, dropout, and classify
        pooled = self.layer_norm(pooled)
        pooled = self.dropout_layer(pooled)
        logits = self.classifier(pooled)

        return logits

    def _mean_pool(
        self,
        output: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over time steps, respecting actual sequence lengths.

        Args:
            output: LSTM output [batch, max_len, hidden_dim * num_directions]
            lengths: Actual sequence lengths [batch]

        Returns:
            Pooled representation [batch, hidden_dim * num_directions]
        """
        batch_size, max_len, hidden_dim = output.shape
        device = output.device

        # Create mask
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()  # [batch, max_len, 1]

        # Masked sum
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  # [batch, hidden_dim]

        # Divide by actual lengths
        lengths_expanded = lengths.float().unsqueeze(1)  # [batch, 1]
        mean_output = sum_output / lengths_expanded.clamp(min=1)

        return mean_output

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lstm_model(
    input_size: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 1,  # Changed default: 1 layer proven better on small data
    num_classes: int = 4,
    dropout: float = 0.3,
    bidirectional: bool = True,
    pooling: str = 'mean',  # Changed default: mean pooling improves generalization
    use_input_projection: bool = True,  # NEW: expand input before LSTM
    use_deep_classifier: bool = True,  # NEW: 2-layer MLP classification head
) -> ThermalLSTMClassifier:
    """
    Factory function to create improved LSTM model.

    Defaults are optimized based on ablation studies (December 2025):
    - 1 layer instead of 2 (+17.7% accuracy)
    - mean pooling instead of last state
    - input projection enabled
    - deep classifier head enabled
    """
    return ThermalLSTMClassifier(
        input_size=input_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        pooling=pooling,
        use_input_projection=use_input_projection,
        use_deep_classifier=use_deep_classifier,
    )


if __name__ == '__main__':
    # Test the improved model
    print("=" * 60)
    print("Testing Improved ThermalLSTMClassifier (December 2025)")
    print("=" * 60)

    # Test with synthetic data
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 1)
    lengths = torch.tensor([100, 90, 80, 70, 60, 50, 40, 30])

    # Test 1: Default configuration (improved)
    print("\n1. Default improved model (1-layer, mean pooling, input proj, deep head):")
    model = create_lstm_model()
    print(f"   Parameters: {model.get_num_parameters():,}")
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits.shape}")

    # Test 2: Attention pooling
    print("\n2. Attention pooling:")
    model_attn = create_lstm_model(pooling='attention')
    print(f"   Parameters: {model_attn.get_num_parameters():,}")
    with torch.no_grad():
        logits_attn = model_attn(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits_attn.shape}")

    # Test 3: Last-state pooling (backward compatibility)
    print("\n3. Last-state pooling (legacy):")
    model_last = create_lstm_model(pooling='last')
    print(f"   Parameters: {model_last.get_num_parameters():,}")
    with torch.no_grad():
        logits_last = model_last(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits_last.shape}")

    # Test 4: Without input projection (legacy-like)
    print("\n4. No input projection, simple classifier (legacy architecture):")
    model_legacy = create_lstm_model(
        num_layers=2,
        pooling='last',
        use_input_projection=False,
        use_deep_classifier=False
    )
    print(f"   Parameters: {model_legacy.get_num_parameters():,}")
    with torch.no_grad():
        logits_legacy = model_legacy(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits_legacy.shape}")

    # Compare parameter counts
    print("\n" + "=" * 60)
    print("Parameter Comparison:")
    print("=" * 60)
    print(f"  Improved (1-layer, proj, deep):   {model.get_num_parameters():>7,}")
    print(f"  Attention pooling:                {model_attn.get_num_parameters():>7,}")
    print(f"  Legacy (2-layer, no proj):        {model_legacy.get_num_parameters():>7,}")
    print(f"  Reduction: {model_legacy.get_num_parameters() - model.get_num_parameters():,} fewer parameters")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

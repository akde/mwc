import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learnable attention pooling over temporal dimension."""

    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )
        self.temperature = temperature

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D) frame features
            mask: (B, T) boolean mask, True = valid frame

        Returns:
            (B, D) pooled feature vector
        """
        scores = self.attn_proj(x) / self.temperature  # (B, T, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = scores.softmax(dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)

    def get_attention_weights(self, x, mask=None):
        """Return attention weights for interpretability. Shape: (B, T)"""
        scores = self.attn_proj(x) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return scores.softmax(dim=1).squeeze(-1)  # (B, T)

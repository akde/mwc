import torch.nn as nn


class MLPHead(nn.Module):
    """MLP classification head with LayerNorm."""

    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, D) pooled features

        Returns:
            (B, num_classes) raw logits
        """
        return self.head(x)

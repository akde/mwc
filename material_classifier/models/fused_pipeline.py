import torch
import torch.nn as nn

from material_classifier.models.attention_pool import AttentionPool
from material_classifier.models.classifier import MLPHead


class FusedCachedMaterialClassifier(nn.Module):
    """
    Late fusion model for RGB + thermal classification.
    Each modality is pooled independently via its own AttentionPool,
    then concatenated and classified by a shared MLP head.
    """

    def __init__(self, feat_dim=1024, hidden_dim=256,
                 num_classes=4, dropout=0.3, temperature=1.0):
        super().__init__()
        self.rgb_pool = AttentionPool(feat_dim, temperature)
        self.thermal_pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim * 2, hidden_dim, num_classes, dropout)

    def forward(self, rgb_features, rgb_mask, thermal_features, thermal_mask):
        """
        Args:
            rgb_features: (B, T_rgb, D) precomputed RGB [CLS] features
            rgb_mask: (B, T_rgb) boolean mask
            thermal_features: (B, T_thermal, D) precomputed thermal [CLS] features
            thermal_mask: (B, T_thermal) boolean mask

        Returns:
            (B, num_classes) logits
        """
        rgb_pooled = self.rgb_pool(rgb_features, rgb_mask)           # (B, D)
        thermal_pooled = self.thermal_pool(thermal_features, thermal_mask)  # (B, D)
        fused = torch.cat([rgb_pooled, thermal_pooled], dim=1)       # (B, 2*D)
        return self.head(fused)                                       # (B, num_classes)

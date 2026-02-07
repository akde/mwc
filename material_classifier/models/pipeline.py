import torch
import torch.nn as nn

from material_classifier.models.feature_extractor import DINOv2Extractor
from material_classifier.models.attention_pool import AttentionPool
from material_classifier.models.classifier import MLPHead


class MaterialClassifier(nn.Module):
    """Full pipeline: DINOv2 -> Attention Pool -> MLP Head."""

    def __init__(self, backbone="dinov2_vitl14", hidden_dim=256,
                 num_classes=4, dropout=0.3, temperature=1.0):
        super().__init__()
        self.extractor = DINOv2Extractor(backbone)
        feat_dim = self.extractor.model.embed_dim  # 1024 for ViT-L
        self.pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim, hidden_dim, num_classes, dropout)

    def forward(self, frames, mask=None):
        """
        Args:
            frames: (B, T, 3, 518, 518)
            mask: (B, T) boolean

        Returns:
            (B, num_classes) logits
        """
        B, T = frames.shape[:2]
        flat = frames.view(B * T, *frames.shape[2:])
        with torch.amp.autocast("cuda"):
            features = self.extractor(flat)           # (B*T, D)
        features = features.view(B, T, -1)            # (B, T, D)
        pooled = self.pool(features, mask)             # (B, D)
        logits = self.head(pooled)                     # (B, num_classes)
        return logits


class CachedMaterialClassifier(nn.Module):
    """Lightweight model for training on precomputed features (no DINOv2)."""

    def __init__(self, feat_dim=1024, hidden_dim=256,
                 num_classes=4, dropout=0.3, temperature=1.0):
        super().__init__()
        self.pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim, hidden_dim, num_classes, dropout)

    def forward(self, features, mask=None):
        """
        Args:
            features: (B, T, D) precomputed [CLS] features
            mask: (B, T) boolean

        Returns:
            (B, num_classes) logits
        """
        pooled = self.pool(features, mask)
        return self.head(pooled)

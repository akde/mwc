import torch
import torch.nn as nn


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT backbone for [CLS] token extraction."""

    def __init__(self, model_name="dinov2_vitl14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B*T, 3, 518, 518) normalized images

        Returns:
            (B*T, 1024) [CLS] token features for ViT-L
        """
        return self.model(x)

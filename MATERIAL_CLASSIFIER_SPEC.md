# Material Classification Pipeline — Implementation Specification

## Overview

Multi-view material classification from conveyor belt RGB video. Objects move along a conveyor under a fixed monocular camera. The temporal axis provides viewpoint diversity, not action dynamics. The goal: classify each object into one of four material classes: **plastic, metal, glass, paper**.

---

## Architecture

```
Conveyor Video (.mp4/.avi)
     │
     ▼
Frame Sampling (every Nth frame or uniform temporal sampling)
     │
     ▼
Object Segmentation (background subtraction or Detectron2)
     │
     ▼
Masked Crop + Resize to 518×518
     │
     ▼
DINOv2 ViT-L/14 (frozen) → [CLS] token per frame → (B, T, 1024)
     │
     ▼
Attention Pooling over T frames → (B, 1024)
     │
     ▼
MLP Head → 4-class softmax → {plastic, metal, glass, paper}
```

---

## Project Structure

```
material_classifier/
├── config/
│   └── default.yaml              # all hyperparameters and paths
├── data/
│   ├── dataset.py                # VideoMaterialDataset (PyTorch Dataset)
│   └── preprocessing.py          # frame sampling, segmentation, masking, resizing
├── models/
│   ├── feature_extractor.py      # DINOv2 frozen backbone wrapper
│   ├── attention_pool.py         # learnable attention pooling module
│   ├── classifier.py             # MLP classification head
│   └── pipeline.py               # full model: extractor + pool + head
├── train.py                      # training loop
├── evaluate.py                   # evaluation + metrics
├── inference.py                  # single video inference
├── requirements.txt
└── README.md
```

---

## Data Format

### Expected directory structure for training data

```
dataset/
├── train/
│   ├── plastic/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── ...
│   ├── metal/
│   ├── glass/
│   └── paper/
├── val/
│   ├── plastic/
│   ├── metal/
│   ├── glass/
│   └── paper/
└── test/
    ├── plastic/
    ├── metal/
    ├── glass/
    └── paper/
```

Each video contains one object traversing the conveyor belt. Class is determined by parent folder name.

---

## Module Specifications

### 1. Preprocessing (`data/preprocessing.py`)

#### Frame Sampling
- Input: video path → Output: list of PIL Images or numpy arrays
- Strategy: uniform temporal sampling of `N` frames (default `N=8`)
- If video has fewer than N frames, use all frames (no padding)
- If video has more than N frames, sample uniformly spaced indices: `np.linspace(0, total_frames-1, N).astype(int)`

#### Object Segmentation — Two modes, selectable via config

**Mode A: Simple background subtraction (default, preferred for single-object controlled environments)**
- Compute median frame across the entire video as background model (or accept a precomputed background image)
- Absolute difference between each frame and background → grayscale → threshold (Otsu) → morphological close (kernel=15) → largest connected component → binary mask
- Use OpenCV only. No deep learning dependency.

**Mode B: Detectron2 instance segmentation (for multi-object or cluttered scenes)**
- Use Detectron2 Mask R-CNN with ResNet-50 FPN backbone
- Pretrained on COCO. Fine-tune on domain data if available.
- Take the highest-confidence detection mask per frame
- Fallback: if no detection, use the full frame (log a warning)

#### Masked Crop
- Apply binary mask: `masked = frame * mask[..., None] + (1 - mask[..., None]) * 128`
- The `128` neutral gray fill prevents background pixels from contributing features
- Crop to mask bounding box
- Resize to 518×518 (DINOv2 ViT-L/14 native resolution using `transforms.Resize((518, 518))`)
- Normalize with DINOv2 expected values: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

---

### 2. Feature Extractor (`models/feature_extractor.py`)

```python
import torch
import torch.nn as nn

class DINOv2Extractor(nn.Module):
    def __init__(self, model_name="dinov2_vitl14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # x: (B*T, 3, 518, 518)
        out = self.model(x)  # (B*T, 1024) for ViT-L
        return out
```

- **Model**: `dinov2_vitl14` (ViT-Large, patch size 14, output dim 1024)
- **Completely frozen**. No gradients. Use `torch.no_grad()` during forward.
- Extracts `[CLS]` token only (default behavior of `model.forward()`)
- Smaller alternatives if compute-constrained: `dinov2_vitb14` (768-dim), `dinov2_vits14` (384-dim)

---

### 3. Attention Pooling (`models/attention_pool.py`)

```python
import torch
import torch.nn as nn

class AttentionPool(nn.Module):
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
        self.temperature = temperature

    def forward(self, x, mask=None):
        # x: (B, T, D)
        # mask: (B, T) boolean, True = valid frame, False = padding
        scores = self.attn_proj(x) / self.temperature  # (B, T, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = scores.softmax(dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)
```

- Two-layer projection with Tanh activation (Ilse et al., 2018 attention MIL style) — slightly more expressive than single linear
- Supports variable-length sequences via boolean mask
- Temperature parameter for controlling attention sharpness (default 1.0)
- **This module IS trainable**

---

### 4. MLP Classifier Head (`models/classifier.py`)

```python
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)
```

- LayerNorm before projection (stabilizes frozen feature distribution)
- Single hidden layer, 256 units
- GELU activation, 0.3 dropout
- Output: raw logits (4 classes). Loss function applies softmax internally.
- **This module IS trainable**

---

### 5. Full Pipeline (`models/pipeline.py`)

```python
class MaterialClassifier(nn.Module):
    def __init__(self, backbone="dinov2_vitl14", hidden_dim=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.extractor = DINOv2Extractor(backbone)
        feat_dim = self.extractor.model.embed_dim  # 1024 for ViT-L
        self.pool = AttentionPool(feat_dim)
        self.head = MLPHead(feat_dim, hidden_dim, num_classes, dropout)

    def forward(self, frames, mask=None):
        # frames: (B, T, 3, 518, 518)
        B, T = frames.shape[:2]
        flat = frames.view(B * T, *frames.shape[2:])    # (B*T, 3, 518, 518)
        features = self.extractor(flat)                   # (B*T, D)
        features = features.view(B, T, -1)                # (B, T, D)
        pooled = self.pool(features, mask)                # (B, D)
        logits = self.head(pooled)                        # (B, 4)
        return logits
```

---

## Training Specification

### Hyperparameters (defaults)

```yaml
# config/default.yaml
data:
  dataset_root: "./dataset"
  num_frames: 8                    # frames sampled per video
  image_size: 518                  # DINOv2 native resolution
  segmentation_mode: "background_subtraction"  # or "detectron2"
  background_image: null           # path to precomputed background (optional)

model:
  backbone: "dinov2_vitl14"        # dinov2_vits14, dinov2_vitb14, dinov2_vitl14
  hidden_dim: 256
  num_classes: 4
  dropout: 0.3
  attention_temperature: 1.0

training:
  batch_size: 8
  epochs: 50
  lr: 1e-3                         # only attention pool + MLP are trained
  weight_decay: 1e-4
  scheduler: "cosine"              # cosine annealing to 0
  warmup_epochs: 5
  label_smoothing: 0.1

augmentation:
  random_horizontal_flip: true
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05
  random_erasing: 0.1              # simulate partial occlusion
```

### Training Details

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) — only over `pool` and `head` parameters
- **Loss**: CrossEntropyLoss with label_smoothing=0.1
- **Scheduler**: CosineAnnealingLR with linear warmup for first 5 epochs
- **Only ~200K trainable parameters** (attention pool + MLP). Backbone is completely frozen.
- **Gradient accumulation**: if GPU memory is tight, accumulate over 2-4 steps
- **Mixed precision**: use `torch.amp.autocast("cuda")` for the DINOv2 forward pass (reduces memory, backbone is frozen so no precision concerns for gradients)

### Data Augmentation

Applied **per-frame** during training, **after** segmentation and cropping:
- Random horizontal flip (applied consistently across all frames of a video)
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing (simulates partial occlusion/dirt on lens)
- **No** random crop — object is already cropped to mask bounding box

### DataLoader Collation

Variable-length sequences require a custom collate function:
- Pad all sequences in a batch to the length of the longest sequence
- Return a boolean mask tensor: `True` for real frames, `False` for padding
- Padding frames can be zeros — they'll be masked out in attention pooling

```python
def collate_fn(batch):
    # batch: list of (frames_tensor, label) where frames_tensor is (T_i, 3, 518, 518)
    max_T = max(f.shape[0] for f, _ in batch)
    B = len(batch)
    padded = torch.zeros(B, max_T, 3, 518, 518)
    masks = torch.zeros(B, max_T, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for i, (f, l) in enumerate(batch):
        T_i = f.shape[0]
        padded[i, :T_i] = f
        masks[i, :T_i] = True
        labels[i] = l
    return padded, masks, labels
```

---

## Evaluation

### Metrics
- **Primary**: Accuracy, Macro F1-Score (handles class imbalance)
- **Per-class**: Precision, Recall, F1 for each material
- **Confusion matrix**: save as image after each eval
- Use `torchmetrics` or `sklearn.metrics`

### Inference (`inference.py`)
- Input: single video file path
- Output: predicted material class + confidence score + attention weights per frame
- Optionally visualize which frames received highest attention (useful for interpretability in thesis)

---

## Dependencies

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
numpy
opencv-python
Pillow
pyyaml
tqdm
scikit-learn
matplotlib
```

Optional (only if using Detectron2 segmentation mode):
```
detectron2  # install via: python -m pip install detectron2 -f https://dl.fbaipublicmodels.com/detectron2/wheels/cu118/torch2.1/index.html
```

---

## Important Implementation Notes

1. **DINOv2 must remain completely frozen throughout training.** Never set `requires_grad=True` on backbone parameters. The optimizer should only receive parameters from `pool` and `head`.

2. **Image size must be 518×518 for ViT-L/14.** DINOv2 uses patch size 14. Input must be divisible by 14. The model was trained at 518×518. Using other sizes works but degrades feature quality.

3. **Neutral gray (128) fill for masked regions**, not black (0). Black pixels at DINOv2's normalization become non-zero values that contaminate features. Gray is closer to the normalized mean and is thus more neutral.

4. **Segmentation failures**: if the background subtraction or Detectron2 fails to find an object in a frame, skip that frame entirely rather than using the full frame. A clean subset of frames is better than a contaminated full set.

5. **Feature caching optimization**: since DINOv2 is frozen, you can precompute and cache all `[CLS]` features to disk during a one-time preprocessing pass, then train the attention pool + MLP on cached features without loading DINOv2 at all during training. This dramatically speeds up training iteration.

```python
# Precompute and save features
# features/train/plastic/video_001.pt → tensor of shape (T, 1024)
```

6. **Reproducibility**: set random seeds for `torch`, `numpy`, `random`, and use `torch.backends.cudnn.deterministic = True`.

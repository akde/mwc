# Material Classification Pipeline — Implementation Specification

## Overview

Multi-view material classification from conveyor belt RGB video. Multiple objects move simultaneously along a conveyor under a fixed monocular camera. The temporal axis provides viewpoint diversity, not action dynamics. Objects are detected per-frame with Detectron2, tracked across frames with OC-SORT to form per-object tracklets, then classified. The goal: classify each tracked object into one of four material classes: **plastic, metal, glass, paper**.

---

## Architecture

```
Conveyor Video (.mp4/.avi)
     │
     ▼
Frame Extraction (every frame)
     │
     ▼
Detectron2 Mask R-CNN (frozen, det2_model/) → per-frame detections + masks
     │
     ▼
OC-SORT Tracking → per-object tracklets (track_id persists across frames)
     │
     ▼
Per-tracklet: Masked Crop + Resize to 518×518 (gray fill 128)
     │
     ▼
Per-tracklet: Uniform sample N frames from tracklet
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
├── __init__.py
├── config/
│   ├── default.yaml              # RGB pipeline hyperparameters and paths
│   ├── thermal.yaml              # thermal pipeline hyperparameters and paths
│   └── fused.yaml                # late fusion pipeline hyperparameters and paths
├── data/
│   ├── dataset.py                # TrackletDataset, CachedFeatureDataset, collate_fn
│   ├── fused_dataset.py          # FusedCachedFeatureDataset, fused_collate_fn
│   └── preprocessing.py          # masking, cropping, frame sampling, augmentation
├── tracking/
│   ├── detector.py               # Detectron2 inference wrapper (frozen model)
│   └── ocsort.py                 # OC-SORT multi-object tracker (with OCR + merge + dedup)
├── models/
│   ├── feature_extractor.py      # DINOv2 frozen backbone wrapper
│   ├── attention_pool.py         # learnable attention pooling module
│   ├── classifier.py             # MLP classification head
│   ├── pipeline.py               # MaterialClassifier + CachedMaterialClassifier
│   └── fused_pipeline.py         # FusedCachedMaterialClassifier (late fusion)
├── thermal/
│   ├── __init__.py
│   ├── dataset.py                # ThermalTrackletDataset (thermal frames + warped masks)
│   ├── train.py                  # thermal feature caching + training entry point
│   ├── evaluate.py               # thermal model evaluation entry point
│   ├── visualize.py              # thermal overlay video (synced to RGB frame count)
│   └── utils.py                  # frame matching, homography warping, grayscale conversion
├── train.py                      # feature caching + training loop
├── evaluate.py                   # evaluation metrics + confusion matrix
├── inference.py                  # detect-and-track (single/batch) + full inference
├── label.py                      # interactive tracklet labeling tool (OpenCV GUI)
├── visualize.py                  # tracklet overlay visualization on video
└── analyze_gaps.py               # detection gap analysis + frame extraction

cross_validate.py                 # 5-fold stratified cross-validation (RGB)
thermal_cross_validate.py         # 5-fold stratified cross-validation (thermal)
fused_cross_validate.py           # 5-fold stratified cross-validation (late fusion)
fused_train.py                    # late fusion training entry point
labels.csv                        # 550 tracklet labels (19 videos, 4 classes)
videos/                           # symlinks to experiment videos in ~/Downloads/
```

---

## Data Format

### Input videos

Videos contain multiple objects of potentially different classes on the conveyor simultaneously. Videos are stored in a flat directory:

```
videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### Track labels (manual annotation)

After running detection + tracking, each tracklet is assigned a `track_id`. Labels are provided in a CSV file mapping each tracklet to its material class:

```csv
video,track_id,split,class
video_001.mp4,1,train,plastic
video_001.mp4,2,train,metal
video_001.mp4,3,val,glass
video_002.mp4,1,train,paper
video_002.mp4,2,test,plastic
```

- `video`: source video filename
- `track_id`: OC-SORT assigned track ID (persistent across frames for one object)
- `split`: one of `train`, `val`, `test`
- `class`: one of `plastic`, `metal`, `glass`, `paper`

---

## Module Specifications

### 1. Detection & Tracking (`tracking/`)

#### Detectron2 Detection (`tracking/detector.py`)
- Pre-trained Detectron2 Mask R-CNN model at `det2_model/` (config.yaml + model_best.pth)
- **Frozen — not trained further.** Load and run inference only.
- Produces per-frame: bounding boxes, instance masks, confidence scores
- Multiple detections per frame (multiple objects on conveyor simultaneously)

#### OC-SORT Tracking (`tracking/ocsort.py`)
- Associates detections across frames into persistent tracklets using OC-SORT
- Each tracklet has a unique `track_id` that persists across the object's lifetime
- Uses IoU + observation-centric momentum for association
- **Observation-Centric Recovery (OCR):** Second association pass re-matches remaining unmatched detections against unmatched trackers using their last observed position (not Kalman prediction), recovering tracks where prediction drifted
- **Duplicate detection suppression:** Before spawning new tracks, discards unmatched detections that overlap (IoU >= `merge_iou_threshold`) with already-matched tracks — prevents NMS failures from creating competing tracks
- **Track merging:** After each frame, checks all pairs of recently-updated tracks for IoU overlap; merges pairs that maintain overlap >= `merge_iou_threshold` for `merge_patience` consecutive frames (older track ID kept)
- Reference implementation: `/home/akde/phd/spatio_temporal_alignment/object_tracker_ocsort.py`

### 2. Preprocessing (`data/preprocessing.py`)

#### Per-Tracklet Frame Sampling
- Input: tracklet (all frames where a specific track_id was detected)
- Strategy: uniform temporal sampling of `N` frames (default `N=8`) from the tracklet
- If tracklet has fewer than N frames, use all frames (no padding)
- If tracklet has more than N frames, sample uniformly spaced indices: `np.linspace(0, total_frames-1, N).astype(int)`

#### Masked Crop
- Apply binary mask: `masked = frame * mask[..., None] + (1 - mask[..., None]) * 128`
- The `128` neutral gray fill prevents background pixels from contributing features
- Crop to mask bounding box
- Resize to 518×518 (DINOv2 ViT-L/14 native resolution using `transforms.Resize((518, 518))`)
- Normalize with DINOv2 expected values: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

---

### 3. Feature Extractor (`models/feature_extractor.py`)

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

### 4. Attention Pooling (`models/attention_pool.py`)

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

### 5. MLP Classifier Head (`models/classifier.py`)

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

### 6. Full Pipeline (`models/pipeline.py`)

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

### 7. Late Fusion Pipeline (`models/fused_pipeline.py`)

```python
class FusedCachedMaterialClassifier(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256,
                 num_classes=4, dropout=0.3, temperature=1.0):
        super().__init__()
        self.rgb_pool = AttentionPool(feat_dim, temperature)
        self.thermal_pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim * 2, hidden_dim, num_classes, dropout)

    def forward(self, rgb_features, rgb_mask, thermal_features, thermal_mask):
        rgb_pooled = self.rgb_pool(rgb_features, rgb_mask)           # (B, D)
        thermal_pooled = self.thermal_pool(thermal_features, thermal_mask)  # (B, D)
        fused = torch.cat([rgb_pooled, thermal_pooled], dim=1)       # (B, 2*D)
        return self.head(fused)                                       # (B, num_classes)
```

- Two independent attention pools (one per modality) with modality-specific temporal weighting
- Concatenation fusion: RGB (1024) + Thermal (1024) → 2048-dim input to MLP head
- MLP head input dimension doubles from 1024 to 2048
- **~1.05M trainable parameters** (2× attention pools + wider MLP head, 0.35% of backbone)
- Trains on pre-cached features from both modalities — no DINOv2 loaded during training

---

## Training Specification

### Hyperparameters (defaults)

```yaml
# config/default.yaml
data:
  videos_dir: "./videos"
  labels_csv: "./labels.csv"       # manual track labels (video, track_id, split, class)
  tracklets_dir: "./tracklets"     # detect-and-track output directory
  features_dir: "./features"       # cached DINOv2 features
  num_frames: 8                    # frames sampled per tracklet
  image_size: 518                  # DINOv2 native resolution

detection:
  model_dir: "./det2_model"          # config.yaml + model_best.pth
  confidence_threshold: 0.5

tracking:
  det_thresh: 0.3
  max_age: 60
  min_hits: 1
  iou_threshold: 0.3
  delta_t: 3
  inertia: 0.2
  merge_iou_threshold: 0.7          # IoU threshold for merging duplicate tracks (0=disable)
  merge_patience: 3                  # consecutive overlap frames before merging

model:
  backbone: "dinov2_vitl14"        # dinov2_vits14, dinov2_vitb14, dinov2_vitl14
  hidden_dim: 256
  num_classes: 4
  dropout: 0.3
  attention_temperature: 1.0

training:
  batch_size: 8
  epochs: 20
  lr: 1e-3                         # only attention pool + MLP are trained
  weight_decay: 1e-4
  scheduler: "cosine"              # cosine annealing to 0
  warmup_epochs: 5
  label_smoothing: 0.1
  grad_accumulation_steps: 1
  seed: 42

augmentation:
  enabled: true
  random_horizontal_flip: true
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05
  random_erasing: 0.1              # simulate partial occlusion

output:
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

### Training Details

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) — only over `pool` and `head` parameters
- **Loss**: CrossEntropyLoss with label_smoothing=0.1
- **Scheduler**: CosineAnnealingLR with linear warmup for first 5 epochs
- **~528K trainable parameters** (attention pool + MLP). Backbone is completely frozen.
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

### Current Results

**5-Fold Stratified Cross-Validation** on 550 labeled tracklets from 19 experiment videos:

| Pipeline | Params | Mean Accuracy | Mean Macro F1 | Errors |
|----------|--------|---------------|---------------|--------|
| **RGB** | 528K | 0.9509 +/- 0.0093 | **0.9507 +/- 0.0110** | 27 |
| **Thermal** | 528K | 0.9127 +/- 0.0384 | **0.9056 +/- 0.0405** | 48 |
| **Late Fusion** | 1.05M | 0.9636 +/- 0.0100 | **0.9602 +/- 0.0136** | 20 |

Per-class F1 (pooled across all folds):

| Class | RGB | Thermal | Late Fusion | Count |
|-------|-----|---------|-------------|-------|
| glass | 0.985 | 0.970 | 0.992 | 132 |
| metal | 0.948 | 0.931 | 0.966 | 131 |
| paper | 0.932 | 0.804 | 0.920 | 98 |
| plastic | 0.938 | 0.917 | 0.964 | 189 |

### Metrics
- **Primary**: Accuracy, Macro F1-Score (handles class imbalance)
- **Per-class**: Precision, Recall, F1 for each material
- **Confusion matrix**: save as image after each eval
- Uses `sklearn.metrics`

### Inference (`inference.py`)

**Detect-and-track mode (Phase 1):**
- Input: single video (`--video`) or batch directory (`--video-dir`)
- Output: per-video `tracklet_data.csv` + binary mask PNGs
- Batch mode discovers 4-digit subdirs containing `IMG_XXXX_synched_cropped.mp4`
- Supports `--overwrite` / `--no-overwrite` to skip already-processed videos

**Full inference mode (Phase 2):**
- Input: single video + trained checkpoint + config YAML
- Output: per-object predicted material class + confidence score + attention weights per frame
- Runs full pipeline: Detectron2 detection → OC-SORT tracking → per-tracklet classification

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
scipy
detectron2  # install via: python -m pip install detectron2 -f https://dl.fbaipublicmodels.com/detectron2/wheels/cu118/torch2.1/index.html
```

---

## Important Implementation Notes

1. **DINOv2 must remain completely frozen throughout training.** Never set `requires_grad=True` on backbone parameters. The optimizer should only receive parameters from `pool` and `head`.

2. **Image size must be 518×518 for ViT-L/14.** DINOv2 uses patch size 14. Input must be divisible by 14. The model was trained at 518×518. Using other sizes works but degrades feature quality.

3. **Neutral gray (128) fill for masked regions**, not black (0). Black pixels at DINOv2's normalization become non-zero values that contaminate features. Gray is closer to the normalized mean and is thus more neutral.

4. **Detection failures**: if Detectron2 finds no detection for a tracked object in a frame (track gap), that frame is naturally absent from the tracklet. OC-SORT handles this via its lost-track mechanism. Do not interpolate or hallucinate masks for missing frames.

5. **Feature caching optimization**: since DINOv2 is frozen, you can precompute and cache all `[CLS]` features to disk during a one-time preprocessing pass, then train the attention pool + MLP on cached features without loading DINOv2 at all during training. This dramatically speeds up training iteration.

```python
# Precompute and save features per tracklet
# features/video_001_track_1.pt → tensor of shape (T, 1024)
# features/video_001_track_2.pt → tensor of shape (T, 1024)
```

6. **Reproducibility**: set random seeds for `torch`, `numpy`, `random`, and use `torch.backends.cudnn.deterministic = True`.

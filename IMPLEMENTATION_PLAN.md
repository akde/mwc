# Implementation Plan: Material Classification Pipeline

## 1. Current Project State

- **Documentation**: `MATERIAL_CLASSIFIER_SPEC.md` (authoritative spec), `CLAUDE.md` (project guide)
- **Frozen Detectron2 model**: `det2_model/config.yaml` + `det2_model/model_best.pth` (Mask R-CNN, ResNet-50-FPN, 4 classes, MASK_ON=true)
- **OC-SORT reference**: `/home/akde/phd/spatio_temporal_alignment/object_tracker_ocsort.py` (self-contained, ~900 lines)
- **No implementation code exists yet** — everything below must be created from scratch

---

## 2. Full Pipeline

```
Video (.mp4/.avi)
  │
  ▼
Frame-by-frame extraction (cv2.VideoCapture)
  │
  ▼
Detectron2 Mask R-CNN (frozen, det2_model/)
  → per-frame: bboxes [x1,y1,x2,y2], binary masks, confidence scores
  → INPUT.FORMAT = BGR, NUM_CLASSES = 4, SCORE_THRESH_TEST = 0.5, MASK_ON = true
  │
  ▼
OC-SORT Tracking
  → input: [N, 5] array of [x1, y1, x2, y2, score] per frame
  → output: list of dicts with track_id (1-indexed, persistent across frames)
  → also carries per-detection masks through the tracker
  │
  ▼
[User manually labels each tracklet in labels.csv]
  │
  ▼
Per-tracklet preprocessing:
  1. Apply binary mask: masked = frame * mask + (1 - mask) * 128  (gray fill)
  2. Crop to mask bounding box
  3. Resize to 518×518
  4. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  │
  ▼
Per-tracklet: uniform sample N=8 frames
  → np.linspace(0, total_frames-1, N).astype(int)
  → if fewer than N frames, use all (no padding at this stage)
  │
  ▼
DINOv2 ViT-L/14 (frozen)
  → torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
  → [CLS] token per frame → (T, 1024)
  │
  ▼
Attention Pooling (trainable, ~200K params)
  → (T, 1024) → (1024,)
  │
  ▼
MLP Head (trainable)
  → (1024,) → 4-class logits {plastic, metal, glass, paper}
```

---

## 3. Implementation Phases

### Phase 1: Detection & Tracking (5 files)

**Goal**: Process raw videos into tracklets so the user can label them. This must work before any training can happen.

| # | File | Purpose |
|---|------|---------|
| 1 | `material_classifier/__init__.py` | Package init |
| 2 | `material_classifier/tracking/__init__.py` | Subpackage init |
| 3 | `material_classifier/tracking/detector.py` | Detectron2 inference wrapper |
| 4 | `material_classifier/tracking/ocsort.py` | OC-SORT tracker (adapted from reference) |
| 5 | `material_classifier/inference.py` | CLI entry point for `--detect-and-track` mode |

### Phase 2: Data Pipeline + Models (8 files)

**Goal**: Build the dataset, feature extractor, attention pool, classifier, and full pipeline model.

| # | File | Purpose |
|---|------|---------|
| 6 | `material_classifier/config/default.yaml` | All hyperparameters and paths |
| 7 | `material_classifier/data/__init__.py` | Subpackage init |
| 8 | `material_classifier/data/preprocessing.py` | Masking, cropping, frame sampling, augmentation |
| 9 | `material_classifier/data/dataset.py` | TrackletDataset + CachedFeatureDataset + collate_fn |
| 10 | `material_classifier/models/__init__.py` | Subpackage init |
| 11 | `material_classifier/models/feature_extractor.py` | DINOv2 frozen wrapper |
| 12 | `material_classifier/models/attention_pool.py` | Learnable attention pooling |
| 13 | `material_classifier/models/classifier.py` | MLP classification head |

### Phase 3: Training, Evaluation, Full Inference (3 files)

| # | File | Purpose |
|---|------|---------|
| 14 | `material_classifier/models/pipeline.py` | Full model assembly (MaterialClassifier) |
| 15 | `material_classifier/train.py` | Training loop + feature caching |
| 16 | `material_classifier/evaluate.py` | Evaluation metrics + confusion matrix |

Plus update `material_classifier/inference.py` to add `--video --checkpoint` full-inference mode.

**Also create**: `requirements.txt` at project root.

---

## 4. Per-File Specifications

### File 1: `material_classifier/__init__.py`

Empty or minimal version string.

---

### File 2: `material_classifier/tracking/__init__.py`

Empty.

---

### File 3: `material_classifier/tracking/detector.py`

```python
class Detectron2Detector:
    """Frozen Detectron2 Mask R-CNN inference wrapper."""

    def __init__(self, model_dir: str, confidence_threshold: float = 0.5, device: str = "cuda"):
        """
        Load Detectron2 model from model_dir containing config.yaml and model_best.pth.

        Critical implementation details:
        - Use get_cfg() from detectron2.config (NOT bare CfgNode() — it lacks VERSION field)
        - cfg.set_new_allowed(True)  # REQUIRED: config has custom keys like AUGMENTATION_ENABLED
        - cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
        - cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_best.pth")
        - cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        - cfg.MODEL.DEVICE = device
        - self.predictor = DefaultPredictor(cfg)
        """

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame (OpenCV format).

        Args:
            frame: BGR numpy array (H, W, 3), uint8

        Returns:
            List of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] (floats)
                - 'score': float
                - 'mask': binary mask (H, W), numpy bool array

        Implementation:
            outputs = self.predictor(frame)
            instances = outputs["instances"].to("cpu")
            # instances.pred_boxes, instances.scores, instances.pred_masks
        """
```

**Key gotchas**:
- `cfg.set_new_allowed(True)` is mandatory because the config contains custom keys (`AUGMENTATION_ENABLED`, `AUGMENTATION_STRATEGY`, `FLOAT32_PRECISION`). Without this, `merge_from_file` will raise `KeyError`.
- The config specifies `INPUT.FORMAT: BGR` — Detectron2's `DefaultPredictor` expects BGR input by default, which matches OpenCV's `cv2.imread` / `VideoCapture.read()` output. No conversion needed at this stage.
- `NUM_CLASSES = 4` is already set in the config (`ROI_HEADS.NUM_CLASSES: 4`).
- `MASK_ON = true` is already set.
- `SCORE_THRESH_TEST = 0.5` is already set but should be overridable via constructor parameter.

---

### File 4: `material_classifier/tracking/ocsort.py`

**Source**: Copy and adapt from `/home/akde/phd/spatio_temporal_alignment/object_tracker_ocsort.py`.

**What to keep** (core OC-SORT algorithm):
- `convert_bbox_to_z`, `convert_x_to_bbox`
- `speed_direction`, `speed_direction_batch`
- `iou_batch`, `giou_batch`
- `linear_assignment`, `associate`
- `KalmanFilter`, `KalmanBoxTracker`
- `OCSort` class

**What to modify**:
- Simplify `OCSort.update()` to carry masks alongside bboxes. The metadata_list mechanism already exists in the reference — keep it.
- Remove `process_experiment()` and `main()` functions (those are specific to the reference pipeline).
- Remove `pandas` dependency (not needed in our tracker).
- Remove `parse_box()` (not needed).
- The tracker output already returns dicts with `track_id`, `bbox`, and arbitrary metadata — this is exactly what we need.

**Interface**:
```python
class OCSort:
    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, inertia=0.2): ...

    def update(self, dets: np.ndarray, metadata_list: list[dict] = None) -> list[dict]:
        """
        Args:
            dets: [N, 5] array of [x1, y1, x2, y2, score]
            metadata_list: list of dicts with 'mask' key (binary mask array)

        Returns:
            List of dicts: {'bbox': [x1,y1,x2,y2], 'track_id': int, 'mask': np.ndarray, ...}
        """
```

---

### File 5: `material_classifier/inference.py`

Two modes of operation:

**Mode 1: `--detect-and-track`** (Phase 1)
```python
def detect_and_track(video_path: str, output_dir: str, detector: Detectron2Detector,
                     tracker_params: dict) -> None:
    """
    Process a video: detect objects per frame, track across frames, save tracklets.

    Output:
        tracklets/{video_stem}/
            tracklet_data.csv       # frame, track_id, x1, y1, x2, y2, score
            masks/frame_{:06d}_track_{}.png  # binary masks per detection per frame

    The CSV enables the user to review tracklets and create labels.csv.
    """
```

**Mode 2: `--video --checkpoint`** (Phase 3, added later)
```python
def full_inference(video_path: str, checkpoint_path: str, config_path: str) -> list[dict]:
    """
    Full pipeline: detect → track → classify each tracklet.

    Returns:
        List of dicts per tracklet:
            {'track_id': int, 'class': str, 'confidence': float,
             'attention_weights': np.ndarray, 'frame_indices': list[int]}
    """
```

**CLI**:
```bash
# Phase 1:
python material_classifier/inference.py --detect-and-track --video path/to/video.mp4 --output-dir tracklets/

# Phase 3:
python material_classifier/inference.py --video path/to/video.mp4 --checkpoint path/to/best.pt --config config/default.yaml
```

---

### File 6: `material_classifier/config/default.yaml`

```yaml
data:
  videos_dir: "./videos"
  labels_csv: "./labels.csv"
  tracklets_dir: "./tracklets"
  features_dir: "./features"
  num_frames: 8
  image_size: 518

detection:
  model_dir: "./det2_model"
  confidence_threshold: 0.5

tracking:
  det_thresh: 0.5
  max_age: 40          # max_frames_missing
  min_hits: 3
  iou_threshold: 0.3
  delta_t: 3
  inertia: 0.2

model:
  backbone: "dinov2_vitl14"
  hidden_dim: 256
  num_classes: 4
  dropout: 0.3
  attention_temperature: 1.0

training:
  batch_size: 8
  epochs: 50
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: "cosine"
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
  random_erasing: 0.1

output:
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

---

### File 7: `material_classifier/data/__init__.py`

Empty.

---

### File 8: `material_classifier/data/preprocessing.py`

```python
def apply_mask_and_crop(frame: np.ndarray, mask: np.ndarray, gray_fill: int = 128) -> np.ndarray:
    """
    Apply binary mask with gray fill and crop to bounding box.

    Args:
        frame: (H, W, 3) uint8 BGR or RGB image
        mask: (H, W) binary mask (bool or 0/1)
        gray_fill: fill value for non-object pixels (default 128)

    Returns:
        cropped: (h, w, 3) masked and cropped image

    Implementation:
        masked = frame * mask[..., None] + (1 - mask[..., None]) * gray_fill
        # Find bbox of mask
        rows, cols = np.where(mask > 0)
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1
        cropped = masked[y1:y2, x1:x2]
    """


def get_transform(image_size: int = 518, train: bool = False, aug_config: dict = None):
    """
    Build torchvision transform pipeline.

    Base transforms (always applied):
        transforms.ToPILImage()
        transforms.Resize((image_size, image_size))
        transforms.ToTensor()
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Training augmentations (inserted before ToTensor):
        transforms.RandomHorizontalFlip(p=0.5)
        transforms.ColorJitter(brightness, contrast, saturation, hue)
        transforms.RandomErasing(p=0.1)  # applied after ToTensor

    Note: RandomHorizontalFlip must be applied consistently per tracklet.
    Strategy: set a per-tracklet random seed so all frames in a tracklet
    get the same flip decision.
    """


def sample_frames(total_frames: int, num_frames: int = 8) -> list[int]:
    """
    Uniform temporal sampling of frame indices.

    If total_frames <= num_frames: return list(range(total_frames))
    Else: return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    """
```

**Critical detail**: BGR→RGB conversion must happen here. Detectron2 works in BGR, but DINOv2 expects RGB (ImageNet normalization). Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before applying the transform pipeline.

---

### File 9: `material_classifier/data/dataset.py`

Two dataset classes:

```python
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]  # alphabetical order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class TrackletDataset(torch.utils.data.Dataset):
    """
    Dataset that loads raw frames + masks for each tracklet.
    Used during feature extraction (with DINOv2) and during full-pipeline training.

    Args:
        labels_csv: path to labels.csv
        tracklets_dir: path to tracklets/ directory
        videos_dir: path to videos/ directory
        split: 'train', 'val', or 'test'
        num_frames: frames to sample per tracklet
        transform: torchvision transform
    """

    def __init__(self, labels_csv, tracklets_dir, videos_dir, split, num_frames=8, transform=None): ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """
        Returns:
            frames: (T, 3, 518, 518) tensor
            label: int class index
        """


class CachedFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset that loads precomputed DINOv2 [CLS] features from disk.
    Used during training (no need to load DINOv2 backbone).

    Features stored as: features/{video_stem}_track_{track_id}.pt → (T, 1024)

    Args:
        labels_csv: path to labels.csv
        features_dir: path to features/ directory
        split: 'train', 'val', or 'test'
    """

    def __init__(self, labels_csv, features_dir, split): ...

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """
        Returns:
            features: (T, 1024) tensor of precomputed [CLS] features
            label: int class index
        """


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate for variable-length sequences.

    Pads all sequences to max length in batch, creates boolean mask.

    Returns:
        padded: (B, max_T, ...) — either (B, max_T, 3, 518, 518) or (B, max_T, 1024)
        masks: (B, max_T) boolean, True = valid frame
        labels: (B,) long tensor
    """
    max_T = max(f.shape[0] for f, _ in batch)
    B = len(batch)
    first_shape = batch[0][0].shape[1:]  # e.g., (3, 518, 518) or (1024,)
    padded = torch.zeros(B, max_T, *first_shape)
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

### File 10: `material_classifier/models/__init__.py`

Empty.

---

### File 11: `material_classifier/models/feature_extractor.py`

```python
class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT backbone for [CLS] token extraction."""

    def __init__(self, model_name: str = "dinov2_vitl14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, 3, 518, 518) normalized images

        Returns:
            (B*T, 1024) [CLS] token features for ViT-L
        """
        return self.model(x)
```

**Key details**:
- `self.model.embed_dim` gives the feature dimension (1024 for ViT-L, 768 for ViT-B, 384 for ViT-S)
- `model.forward(x)` returns the [CLS] token by default
- Always keep `model.eval()` — never switch to train mode
- Use `torch.amp.autocast("cuda")` during forward for mixed precision (saves GPU memory)

---

### File 12: `material_classifier/models/attention_pool.py`

```python
class AttentionPool(nn.Module):
    """Learnable attention pooling over temporal dimension."""

    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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

    def get_attention_weights(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Return attention weights for interpretability. Shape: (B, T)"""
        scores = self.attn_proj(x) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return scores.softmax(dim=1).squeeze(-1)  # (B, T)
```

**Trainable parameters**: `dim * (dim // 4) + (dim // 4)` (first linear) + `(dim // 4) * 1 + 1` (second linear) = for dim=1024: 262,144 + 256 + 256 + 1 = ~263K params.

---

### File 13: `material_classifier/models/classifier.py`

```python
class MLPHead(nn.Module):
    """MLP classification head with LayerNorm."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) pooled features

        Returns:
            (B, num_classes) raw logits
        """
        return self.head(x)
```

**Trainable parameters**: 1024 (LN scale) + 1024 (LN bias) + 1024*256 + 256 (first linear) + 256*4 + 4 (second linear) = ~265K params.

**Total trainable**: ~263K (pool) + ~265K (head) ≈ ~528K (the spec says ~200K — this is approximate, the exact count depends on dim//4 rounding and whether we count LayerNorm).

---

### File 14: `material_classifier/models/pipeline.py`

```python
class MaterialClassifier(nn.Module):
    """Full pipeline: DINOv2 → Attention Pool → MLP Head."""

    def __init__(self, backbone: str = "dinov2_vitl14", hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.3, temperature: float = 1.0):
        super().__init__()
        self.extractor = DINOv2Extractor(backbone)
        feat_dim = self.extractor.model.embed_dim  # 1024 for ViT-L
        self.pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim, hidden_dim, num_classes, dropout)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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
            features = self.extractor(flat)          # (B*T, D)
        features = features.view(B, T, -1)           # (B, T, D)
        pooled = self.pool(features, mask)            # (B, D)
        logits = self.head(pooled)                    # (B, num_classes)
        return logits


class CachedMaterialClassifier(nn.Module):
    """Lightweight model for training on precomputed features (no DINOv2)."""

    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.3, temperature: float = 1.0):
        super().__init__()
        self.pool = AttentionPool(feat_dim, temperature)
        self.head = MLPHead(feat_dim, hidden_dim, num_classes, dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) precomputed [CLS] features
            mask: (B, T) boolean

        Returns:
            (B, num_classes) logits
        """
        pooled = self.pool(features, mask)
        return self.head(pooled)
```

---

### File 15: `material_classifier/train.py`

```python
def cache_features(config: dict) -> None:
    """
    One-time preprocessing: extract DINOv2 [CLS] features for all tracklets.

    For each tracklet in labels.csv:
        1. Load video frames + masks from tracklets_dir
        2. Apply mask with gray fill, crop, resize to 518×518, normalize
        3. Sample N frames uniformly
        4. Forward through frozen DINOv2
        5. Save features to features/{video_stem}_track_{track_id}.pt → (T, 1024)

    Uses mixed precision (torch.amp.autocast) for DINOv2 forward.
    Skips tracklets whose feature files already exist.
    """


def train(config: dict) -> None:
    """
    Training loop for attention pool + MLP head on cached features.

    Setup:
        - model = CachedMaterialClassifier(...)
        - optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        - criterion = CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        - scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs - config.training.warmup_epochs)
        - Linear warmup: for first warmup_epochs, scale lr linearly from 0 to config.training.lr

    Training loop:
        for epoch in range(epochs):
            model.train()
            for features, masks, labels in train_loader:
                logits = model(features, masks)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader)

            # Save best checkpoint by val_f1
            if val_f1 > best_f1:
                save_checkpoint(model, optimizer, epoch, val_f1)

            scheduler.step()

    Reproducibility:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    """


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="material_classifier/config/default.yaml")
    parser.add_argument("--cache-features", action="store_true",
                        help="Run feature caching only (no training)")
    args = parser.parse_args()
    # Load config with PyYAML
    # If --cache-features: call cache_features(config)
    # Else: call cache_features(config) then train(config)
```

**Warmup implementation**:
```python
# Linear warmup scheduler wrapper
def get_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr  # cosine scheduler handles the rest

# Use torch.optim.lr_scheduler.LambdaLR for warmup,
# then switch to CosineAnnealingLR, or use SequentialLR to chain them.
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
```

**Checkpoint format**:
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_f1': best_f1,
    'config': config,
}, path)
```

---

### File 16: `material_classifier/evaluate.py`

```python
def evaluate(model, dataloader, device, class_names) -> dict:
    """
    Full evaluation on a dataset split.

    Returns:
        {
            'accuracy': float,
            'macro_f1': float,
            'per_class': {class_name: {'precision': float, 'recall': float, 'f1': float}},
            'confusion_matrix': np.ndarray (4x4),
            'all_preds': list[int],
            'all_labels': list[int],
        }

    Uses sklearn.metrics:
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    """


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    """Save confusion matrix as PNG using matplotlib."""


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
```

---

### File 17: `requirements.txt`

```
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
```

Note: `detectron2` must be installed separately (wheel URL depends on CUDA/PyTorch version):
```bash
python -m pip install detectron2 -f https://dl.fbaipublicmodels.com/detectron2/wheels/cu118/torch2.1/index.html
```

---

## 5. Critical Technical Details

### Detectron2

- **Config loading**: Must call `cfg.set_new_allowed(True)` before `cfg.merge_from_file()` because the saved config contains non-standard keys (`AUGMENTATION_ENABLED`, `AUGMENTATION_STRATEGY`, `FLOAT32_PRECISION`).
- **Input format**: BGR (matches OpenCV). `DefaultPredictor` handles preprocessing internally.
- **Model architecture**: ResNet-50-FPN backbone, Mask R-CNN with `StandardROIHeads`.
- **Key config values**: `ROI_HEADS.NUM_CLASSES: 4`, `ROI_HEADS.SCORE_THRESH_TEST: 0.5`, `MASK_ON: true`, `INPUT.FORMAT: BGR`.
- **Output**: `outputs["instances"]` has `.pred_boxes` (Boxes object), `.scores` (tensor), `.pred_masks` (bool tensor, one per instance).

### OC-SORT

- **Self-contained**: No external dependencies beyond numpy, scipy (for `linear_sum_assignment`).
- **Input**: `[N, 5]` numpy array of `[x1, y1, x2, y2, score]` per frame.
- **Output**: List of dicts with `track_id` (1-indexed), `bbox`, and any metadata passed through.
- **Key parameters**: `det_thresh=0.5`, `max_age=40`, `min_hits=3`, `iou_threshold=0.3`, `delta_t=3`, `inertia=0.2`.
- **Stateful**: Call `tracker.update(dets)` once per frame, sequentially. Create a fresh `OCSort()` instance per video.
- **ID counter**: `KalmanBoxTracker.count` must be reset per video (the constructor does `KalmanBoxTracker.count = 0`).

### DINOv2

- **Loading**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")` — downloads model on first use.
- **Frozen**: All parameters have `requires_grad = False`. Always in `eval()` mode.
- **Input**: `(B, 3, 518, 518)` tensor, normalized with ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.
- **Output**: `(B, 1024)` [CLS] token (default `forward()` behavior). For ViT-B it's 768, for ViT-S it's 384.
- **Image size**: Must be 518×518 (= 37 patches × 14 pixels). Other sizes technically work but degrade quality.
- **Mixed precision**: Safe to use `torch.amp.autocast("cuda")` since no gradients flow through.

### Color Space Conversion

- **Detectron2** works in BGR (OpenCV native).
- **DINOv2** expects RGB (ImageNet convention).
- Conversion point: In `preprocessing.py`, after extracting frame and mask from the tracklet data, convert BGR→RGB with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before applying the torchvision transform pipeline.

### Gray Fill

- Background pixels (outside object mask) must be filled with `128`, not `0` (black).
- Black becomes non-zero after ImageNet normalization (`(0 - mean) / std` ≠ 0) and contaminates features.
- Gray (128/255 ≈ 0.502) is close to the normalized mean, making it approximately zero after normalization.
- Formula: `masked = frame * mask[..., None] + (1 - mask[..., None]) * 128` (ensure mask is float for this).

### Feature Caching

- Cache path pattern: `features/{video_stem}_track_{track_id}.pt`
- Each file contains a `(T, 1024)` float32 tensor (T = number of sampled frames for that tracklet).
- Caching is done once, then training loads only the cached `.pt` files.
- This eliminates the need to load DINOv2 during training, reducing GPU memory from ~5GB to near zero for the backbone.

---

## 6. Data Formats

### Input Videos

```
videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

Flat directory. Each video contains multiple objects on a conveyor belt.

### Detection + Tracking Output (Phase 1)

```
tracklets/
├── video_001/
│   ├── tracklet_data.csv
│   └── masks/
│       ├── frame_000001_track_1.png
│       ├── frame_000001_track_2.png
│       ├── frame_000005_track_1.png
│       └── ...
└── video_002/
    ├── tracklet_data.csv
    └── masks/
        └── ...
```

**tracklet_data.csv** columns:
```csv
frame,track_id,x1,y1,x2,y2,score
1,1,120.5,80.3,340.2,290.1,0.95
1,2,450.0,100.0,600.0,310.0,0.87
5,1,125.0,82.0,345.0,292.0,0.93
```

**Masks**: binary PNG images (0 or 255), same resolution as original video frame. One mask per detection per frame.

### Labels (Manual Annotation)

```csv
video,track_id,split,class
video_001.mp4,1,train,plastic
video_001.mp4,2,train,metal
video_001.mp4,3,val,glass
video_002.mp4,1,train,paper
video_002.mp4,2,test,plastic
```

Created by the user after reviewing tracklet output.

### Cached Features

```
features/
├── video_001_track_1.pt    # tensor shape (T, 1024)
├── video_001_track_2.pt
├── video_002_track_1.pt
└── ...
```

### Checkpoints

```
checkpoints/
├── best_model.pt           # best by val macro-F1
└── last_model.pt           # latest epoch
```

---

## 7. Training Details

### Optimizer & Scheduler

- **Optimizer**: `AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)`
  - Only pool + head parameters (all of `CachedMaterialClassifier`)
- **Loss**: `CrossEntropyLoss(label_smoothing=0.1)`
- **Scheduler**: `SequentialLR` chaining:
  1. `LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=5)` — 5-epoch warmup
  2. `CosineAnnealingLR(optimizer, T_max=45)` — cosine decay to 0

### Mixed Precision

- Use `torch.amp.autocast("cuda")` only around the DINOv2 forward pass (in `MaterialClassifier.forward` and `cache_features`).
- Not needed for `CachedMaterialClassifier` training (it's just linear layers).

### Augmentation

Applied **per-frame, after masking and cropping**, during training only:

1. **Random horizontal flip** (p=0.5) — must be consistent across all frames of a tracklet. Implementation: seed the random state per tracklet so all frames get the same flip.
2. **Color jitter**: brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05.
3. **Random erasing** (p=0.1) — applied after `ToTensor`, simulates partial occlusion.
4. **No random crop** — object is already cropped to its mask bounding box.

### DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False,
)
```

### Custom Collation

Variable-length tracklets require padding:
- Pad all sequences in batch to `max_T` (the longest in the batch)
- Create boolean mask: `True` for real frames, `False` for padding
- Padding frames filled with zeros — masked out by attention pooling

---

## 8. Verification Steps

### Phase 1 Verification

1. **detector.py**: Load Detectron2 model, run on a single frame, verify output has bboxes + masks + scores. Check that `cfg.set_new_allowed(True)` prevents config errors.
2. **ocsort.py**: Process a short video (50 frames), verify track_ids are consistent and 1-indexed.
3. **inference.py `--detect-and-track`**: Process one video end-to-end, verify output directory structure, CSV content, and mask PNG files are valid binary masks.
4. **Mask quality**: Visually inspect a few masked crops to confirm gray fill and proper cropping.

```bash
python material_classifier/inference.py --detect-and-track --video videos/test_video.mp4 --output-dir tracklets/
# Check: tracklets/test_video/tracklet_data.csv exists, masks/ dir has PNG files
```

### Phase 2 Verification

5. **preprocessing.py**: Load a frame + mask, apply `apply_mask_and_crop`, verify output is correctly masked with gray fill and cropped.
6. **feature_extractor.py**: Load DINOv2, pass a `(1, 3, 518, 518)` tensor, verify output shape is `(1, 1024)`. Confirm `requires_grad=False` on all backbone params.
7. **dataset.py**: Instantiate `TrackletDataset`, iterate one sample, verify shapes. Test `collate_fn` with variable-length samples.
8. **attention_pool.py**: Pass `(2, 8, 1024)` tensor through, verify output `(2, 1024)`. Test with mask.
9. **classifier.py**: Pass `(2, 1024)` tensor, verify output `(2, 4)`.

```python
# Quick shape test
extractor = DINOv2Extractor()
x = torch.randn(1, 3, 518, 518).cuda()
out = extractor(x)
assert out.shape == (1, 1024)
assert not any(p.requires_grad for p in extractor.parameters())
```

### Phase 3 Verification

10. **Feature caching**: Run `cache_features`, verify `.pt` files exist with correct shapes.
11. **Training**: Run 5 epochs, verify loss decreases, checkpoint saved.
12. **Evaluation**: Load checkpoint, evaluate on val split, verify metrics are computed correctly.
13. **Full inference**: Run `--video --checkpoint` mode, verify per-tracklet predictions are returned.
14. **Optimizer params**: Verify only pool + head parameters are in the optimizer (no DINOv2 params).

```bash
# Feature caching
python material_classifier/train.py --config material_classifier/config/default.yaml --cache-features
# Check: features/ dir has .pt files

# Training
python material_classifier/train.py --config material_classifier/config/default.yaml
# Check: checkpoints/ dir has best_model.pt, loss decreases in logs

# Evaluation
python material_classifier/evaluate.py --config material_classifier/config/default.yaml --checkpoint checkpoints/best_model.pt
# Check: prints accuracy, macro-F1, per-class metrics, saves confusion_matrix.png

# Full inference
python material_classifier/inference.py --video videos/test_video.mp4 --checkpoint checkpoints/best_model.pt --config material_classifier/config/default.yaml
# Check: prints per-tracklet predictions with class and confidence
```

---

## 9. Implementation Order Summary

```
Phase 1 (Detection & Tracking):
  1. tracking/detector.py       ← needs det2_model/
  2. tracking/ocsort.py         ← adapt from reference
  3. inference.py (--detect-and-track mode)
  4. [USER: run on videos, label tracklets → labels.csv]

Phase 2 (Data + Models):
  5. config/default.yaml
  6. data/preprocessing.py
  7. data/dataset.py
  8. models/feature_extractor.py
  9. models/attention_pool.py
  10. models/classifier.py

Phase 3 (Train + Evaluate + Full Inference):
  11. models/pipeline.py
  12. train.py (with cache_features)
  13. evaluate.py
  14. inference.py (add --video --checkpoint mode)
  15. requirements.txt
```

Each phase is independently testable. Phase 1 produces tracklets the user can label. Phase 2 builds all components. Phase 3 ties everything together for training and inference.

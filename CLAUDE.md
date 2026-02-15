# CLAUDE.md

## Project Overview

Multi-view material classification from conveyor belt video. Multiple objects move simultaneously on a conveyor under a fixed monocular camera. Objects are detected per-frame (Detectron2 Mask R-CNN), tracked across frames (OC-SORT) into per-object tracklets, then classified. Goal: classify each tracked object as **glass, metal, paper, or plastic**.

## Architecture

### Single-Modal (RGB or Thermal)

```
Video (.mp4)
  → Detectron2 Mask R-CNN (frozen, det2_model/) → per-frame detections + masks
  → OC-SORT Tracking (with OCR, duplicate suppression, track merging)
      → per-object tracklets (persistent track_id)
  → Per-tracklet: Masked Crop + Resize (518x518, gray fill 128)
  → Per-tracklet: Uniform sample 8 frames
  → DINOv2 ViT-L/14 (frozen) → [CLS] per frame → 8 × 1024
  → Attention Pooling (trainable) → 1 × 1024
  → MLP Head (trainable) → 4-class logits → {glass, metal, paper, plastic}
```

~528K trainable parameters (attention pool + MLP head). Detectron2 and DINOv2 are completely frozen.

### Late Fusion (RGB + Thermal)

```
RGB cached features (T × 1024)  →  AttentionPool_RGB   → 1024 ─┐
                                                                 ├─ cat → 2048 → MLP Head → 4-class logits
Thermal cached features (T × 1024) → AttentionPool_Thermal → 1024 ─┘
```

~1.05M trainable parameters (two independent attention pools + wider MLP head). DINOv2 frozen; trains on pre-cached features from both modalities.

## Current Results

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

Late fusion reduces errors by 26% vs RGB alone. Glass is near-perfect across all modalities. Paper is the hardest class (paper↔plastic confusion persists in all three).

## Project Structure

```
material_classifier/
├── __init__.py
├── config/
│   ├── default.yaml              # RGB pipeline hyperparameters and paths
│   ├── thermal.yaml              # thermal pipeline hyperparameters and paths
│   └── fused.yaml                # late fusion pipeline hyperparameters and paths
├── tracking/
│   ├── detector.py               # Detectron2 Mask R-CNN inference wrapper
│   └── ocsort.py                 # OC-SORT tracker (with OCR + merge + dedup)
├── data/
│   ├── dataset.py                # TrackletDataset, CachedFeatureDataset, collate_fn
│   ├── fused_dataset.py          # FusedCachedFeatureDataset, fused_collate_fn
│   └── preprocessing.py          # masking, cropping, frame sampling, augmentation
├── models/
│   ├── feature_extractor.py      # DINOv2 frozen backbone wrapper
│   ├── attention_pool.py         # learnable attention pooling
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
├── inference.py                  # detect-and-track (single/batch) + full inference
├── train.py                      # feature caching + training loop
├── evaluate.py                   # evaluation metrics + confusion matrix
├── label.py                      # interactive tracklet labeling tool (OpenCV GUI)
├── visualize.py                  # tracklet overlay visualization on video
└── analyze_gaps.py               # detection gap analysis + frame extraction
```

Other project files:
```
cross_validate.py                 # 5-fold stratified cross-validation (RGB)
thermal_cross_validate.py         # 5-fold stratified cross-validation (thermal)
fused_cross_validate.py           # 5-fold stratified cross-validation (late fusion)
fused_train.py                    # late fusion training entry point
labels.csv                        # 550 tracklet labels (19 videos, 4 classes)
videos/                           # symlinks to experiment videos in ~/Downloads/
det2_model/                       # frozen Detectron2 model (config.yaml + model_best.pth)
tracklets/                        # detect-and-track output (generated, not committed)
features/                         # cached DINOv2 [CLS] features from RGB (generated)
thermal_features/                 # cached DINOv2 [CLS] features from thermal (generated)
checkpoints/                      # trained RGB model weights (generated)
thermal_checkpoints/              # trained thermal model weights (generated)
fused_checkpoints/                # trained late fusion model weights (generated)
thermal_viz/                      # thermal annotated overlay videos (generated)
thermal_logs/                     # thermal training logs (generated)
fused_logs/                       # late fusion training logs (generated)
classification_report.tex         # LaTeX report (RGB, thermal, fusion comparison)
requirements.txt                  # Python dependencies
MATERIAL_CLASSIFIER_SPEC.md       # authoritative spec (module code, hyperparameters, data formats)
```

## Commands

### Detection + Tracking

Processes video(s) through Detectron2 + OC-SORT, outputs tracklet CSVs and binary masks.

```bash
# Single video
python material_classifier/inference.py --detect-and-track \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4

# Batch: all experiment videos in a directory
# Discovers 4-digit subdirs (0797/, 0798/, ...) containing IMG_XXXX_synched_cropped.mp4
python material_classifier/inference.py --detect-and-track \
  --video-dir ~/Downloads/

# Batch, skip already processed videos
python material_classifier/inference.py --detect-and-track \
  --video-dir ~/Downloads/ --no-overwrite
```

**Tracker parameters** (all have sensible defaults):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence-threshold` | 0.3 | Detectron2 detection confidence |
| `--max-age` | 60 | Frames to keep lost track alive |
| `--min-hits` | 1 | Min detections before track is reported |
| `--iou-threshold` | 0.3 | IoU threshold for association |
| `--delta-t` | 3 | Velocity estimation lookback |
| `--inertia` | 0.2 | VDC weight for direction consistency |
| `--merge-iou-threshold` | 0.7 | IoU threshold for merging duplicate tracks (0=disable) |
| `--merge-patience` | 3 | Consecutive overlap frames before merging |

Output structure per video:
```
tracklets/{video_stem}/
  tracklet_data.csv                    # frame, track_id, x1, y1, x2, y2, score
  masks/frame_{:06d}_track_{}.png      # binary masks (0/255)
```

### Gap Analysis

Analyze detection gaps in tracklets and extract missed frames for Detectron2 retraining.

```bash
# Analyze all experiments
python material_classifier/analyze_gaps.py \
  --video-dir ~/Downloads/ --tracklets-dir tracklets/

# Single video, report only (no frame extraction)
python material_classifier/analyze_gaps.py \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4 \
  --tracklets-dir tracklets/ --no-extract

# Analyze specific tracks
python material_classifier/analyze_gaps.py \
  --video-dir ~/Downloads/ --tracklets-dir tracklets/ \
  --tracks 28 45 102
```

**What it does:**
- Finds frame gaps within each track's lifetime (frames where detection failed)
- Extracts midpoint frames from long gaps (most informative for retraining)
- Filters near-duplicate frames (minimum 5s apart by default)
- Saves gap analysis CSV and report per video

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-track-length` | 50 | Ignore tracks shorter than this |
| `--min-gap-length` | 10 | Only extract from gaps this long |
| `--min-distance` | 5.0 | Minimum seconds between extracted frames |
| `--tracks` | all | Specific track IDs to analyze |
| `--no-extract` | false | Report only, don't extract frames |

### Labeling

Interactive OpenCV GUI for assigning material classes to tracklets.

```bash
# Label tracklets interactively (1=glass 2=metal 3=paper 4=plastic, u=undo, s=skip, q=quit, c=capture)
python material_classifier/label.py \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4 \
  --tracklets-dir tracklets/ \
  --min-track-length 1000

# Assign stratified train/val splits after labeling
python material_classifier/label.py --assign-splits --labels-csv labels.csv
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | (required) | Path to source video |
| `--tracklets-dir` | `tracklets/` | Base tracklets directory |
| `--labels-csv` | `labels.csv` | Output labels CSV path |
| `--min-track-length` | 0 | Only show tracks with at least this many frames (use 1000 to filter to high-quality tracklets) |
| `--assign-splits` | false | Assign stratified train/val splits instead of labeling |
| `--train-ratio` | 0.8 | Train split ratio for `--assign-splits` |

### Visualization (RGB)

Produces an annotated video with colored mask overlays and track_id labels for review.

```bash
python material_classifier/visualize.py \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4 \
  --tracklets-dir tracklets/
```

### Visualization (Thermal)

Produces thermal overlay videos synced to RGB frame count (same frame count and FPS), enabling side-by-side comparison. Iterates over RGB frames, maps each to its thermal counterpart via the frame matching CSV, and warps RGB tracklet masks into thermal space via homography. Holds the last thermal frame for RGB frames without a thermal match.

```bash
# Single experiment
python material_classifier/thermal/visualize.py \
  --experiment 0798 --tracklets-dir tracklets/

# All experiments
python material_classifier/thermal/visualize.py \
  --all --tracklets-dir tracklets/
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--experiment` | | Experiment ID (e.g. 0798), mutually exclusive with `--all` |
| `--all` | | Process all experiments with tracklets |
| `--tracklets-dir` | `tracklets/` | Base tracklets directory |
| `--video-dir` | `videos/` | Directory with RGB videos (for frame count/FPS) |
| `--downloads-dir` | `~/Downloads` | Directory containing experiment folders (thermal data) |
| `--output-dir` | `thermal_viz/` | Output directory |
| `--alpha` | 0.4 | Mask overlay transparency |
| `--fps` | 30.0 | Fallback FPS if RGB video unavailable |
| `--scale` | 3.0 | Upscale factor (thermal is 270x164) |
| `--colormap` | `inferno` | Colormap: inferno, jet, hot, magma, plasma, viridis, gray |

### Training (RGB)

```bash
# Cache DINOv2 features first (one-time), then train
python material_classifier/train.py --config material_classifier/config/default.yaml

# Cache features only (no training)
python material_classifier/train.py --config material_classifier/config/default.yaml --cache-features
```

### Training (Thermal)

```bash
# Cache DINOv2 features from thermal frames, then train
python material_classifier/thermal/train.py --config material_classifier/config/thermal.yaml

# Cache features only
python material_classifier/thermal/train.py --config material_classifier/config/thermal.yaml --cache-features
```

### Training (Late Fusion)

```bash
# Requires pre-cached RGB and thermal features
python fused_train.py --config material_classifier/config/fused.yaml
```

### Evaluation (RGB)

```bash
python material_classifier/evaluate.py \
  --config material_classifier/config/default.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split val
```

### Evaluation (Thermal)

```bash
python material_classifier/thermal/evaluate.py \
  --config material_classifier/config/thermal.yaml \
  --checkpoint thermal_checkpoints/best_model.pt \
  --split val
```

### Cross-Validation (RGB)

```bash
# 5-fold stratified CV (uses cached features, restores original splits when done)
python cross_validate.py
```

### Cross-Validation (Thermal)

```bash
python thermal_cross_validate.py
```

### Cross-Validation (Late Fusion)

```bash
python fused_cross_validate.py
```

### Full Inference (detect + track + classify)

```bash
python material_classifier/inference.py \
  --video path/to/video.mp4 \
  --checkpoint checkpoints/best_model.pt \
  --config material_classifier/config/default.yaml
```

## Workflow

### RGB Pipeline
1. **Detect & track** — run `inference.py --detect-and-track` on all videos
2. **Review** — run `visualize.py` to inspect tracklets visually
3. **Analyze gaps** — run `analyze_gaps.py` to find detection failures and extract frames for retraining
4. **Label** — run `label.py --min-track-length 1000` to assign material classes interactively
5. **Split** — run `label.py --assign-splits` for stratified 80/20 train/val
6. **Cache features** — run `train.py --cache-features` to precompute DINOv2 [CLS] tokens
7. **Train** — run `train.py` to train attention pool + MLP head on cached features
8. **Evaluate** — run `evaluate.py --split val` on validation split
9. **Cross-validate** — run `cross_validate.py` for 5-fold stratified CV
10. **Infer** — run `inference.py --video --checkpoint` for end-to-end prediction

### Thermal Pipeline
Uses the same tracklets (detection + tracking from RGB) but extracts features from thermal frames instead. Requires per-experiment thermal data: thermal frame PNGs, homography matrix, and RGB-to-thermal frame matching CSV in `~/Downloads/{exp_id}/`.

1. **Cache thermal features** — run `thermal/train.py --cache-features` (warps RGB masks to thermal space via homography)
2. **Train** — run `thermal/train.py` to train on thermal features
3. **Evaluate** — run `thermal/evaluate.py --split val`
4. **Cross-validate** — run `thermal_cross_validate.py`
5. **Visualize** — run `thermal/visualize.py --all` for overlay videos (synced to RGB frame count)

### Late Fusion Pipeline
Combines pre-cached RGB and thermal features via modality-specific attention pools + shared MLP head. Requires both RGB and thermal features to be cached first.

1. **Cache RGB features** — run `train.py --cache-features` (if not already done)
2. **Cache thermal features** — run `thermal/train.py --cache-features` (if not already done)
3. **Train** — run `fused_train.py` to train fusion model on both feature sets
4. **Cross-validate** — run `fused_cross_validate.py`

## Dataset

19 experiment videos, 550 labeled tracklets (filtered to tracks with >= 1000 frames):

| Class | Count | Percentage |
|-------|-------|------------|
| glass | 132 | 24.0% |
| metal | 131 | 23.8% |
| paper | 98 | 17.8% |
| plastic | 189 | 34.4% |

Split: 438 train (80%) / 112 val (20%), stratified by class.

## Data Formats

### Labels CSV (`labels.csv`)

```csv
video,track_id,split,class
IMG_0798_synched_cropped.mp4,1,train,plastic
IMG_0798_synched_cropped.mp4,2,val,metal
```

### Cached Features

```
features/{video_stem}_track_{track_id}.pt  → tensor (T, 1024)
```

### Cached Thermal Features

```
thermal_features/{video_stem}_track_{track_id}.pt  → tensor (T, 1024)
```

### Thermal Experiment Data (per experiment in `~/Downloads/{exp_id}/`)

```
thermal_frames/FLIR_frame_{:06d}.png    # grayscale thermal PNGs (sparse indices)
homography_*.joblib                      # 3x3 RGB-to-thermal homography matrix
frame_matches_*.csv                      # RGB frame index → thermal frame index mapping
```

### Classes (alphabetical, 0-indexed)

```python
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
```

## Video Naming Convention

Experiment videos live in `~/Downloads/` in 4-digit folders:
```
~/Downloads/0797/IMG_0797_synched_cropped.mp4
~/Downloads/0798/IMG_0798_synched_cropped.mp4
...
~/Downloads/0854/IMG_0854_synched_cropped.mp4
```

The `--video-dir` flag in `inference.py` auto-discovers this pattern. The `videos/` directory contains symlinks to these files for use by the training pipeline.

## Pre-trained Models (frozen)

- **Detectron2**: `det2_model/config.yaml` + `det2_model/model_best.pth` (Mask R-CNN, ResNet-50-FPN, 4 classes, MASK_ON=true, BGR input)
- **DINOv2**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")` (ViT-L/14, 1024-dim [CLS] output)

## Critical Implementation Constraints

1. **DINOv2 must stay frozen.** Never set `requires_grad=True` on backbone parameters. The optimizer must only receive `pool` and `head` parameters.

2. **Detectron2 model is frozen.** Use the pre-trained model at `det2_model/` for inference only.

3. **Detectron2 config loading:** Must use `get_cfg()` (not bare `CfgNode()`) then `cfg.set_new_allowed(True)` before `merge_from_file`. Bare `CfgNode()` lacks the `VERSION` field -> `AttributeError: VERSION`.

4. **Image size must be 518x518.** DINOv2 ViT-L/14 patch size is 14; input must be divisible by 14. The model was trained at 518x518.

5. **Masked regions filled with neutral gray (128), not black.** Black pixels become non-zero after DINOv2 normalization and contaminate features. Gray is closer to the normalized mean.

6. **Detection gaps are natural.** OC-SORT handles frames where an object is not detected. Do not interpolate or hallucinate masks for missing frames. Use `analyze_gaps.py` to find and extract gap frames for Detectron2 retraining.

7. **OC-SORT has three deduplication mechanisms:** (a) Observation-Centric Recovery (OCR) — a second association pass using last-observed position instead of Kalman prediction, recovering tracks where prediction drifted; (b) Duplicate detection suppression — discards unmatched detections that overlap with already-matched tracks before spawning new tracks; (c) Track merging — merges tracks that consistently co-exist with high spatial overlap for `merge_patience` consecutive frames.

8. **Augmentation is per-frame, applied after masking and cropping** — not before.

9. **Feature caching:** Since DINOv2 is frozen, precompute [CLS] features per tracklet to disk and train pool+head on cached features without loading DINOv2. Training uses `CachedMaterialClassifier` (pool+head only, no backbone) or `FusedCachedMaterialClassifier` (two pools + wider MLP head) for late fusion. Checkpoint saved from `CachedMaterialClassifier` is loaded into `MaterialClassifier` pool+head for inference.

10. **Color space:** Detectron2 works in BGR (OpenCV native). DINOv2 expects RGB. Conversion happens in preprocessing after extracting frames.

## Tech Stack

- Python 3.11, PyTorch >= 2.1, torchvision >= 0.16
- DINOv2 via torch.hub
- Detectron2 (Mask R-CNN)
- OC-SORT for multi-object tracking
- OpenCV, Pillow, NumPy, SciPy, scikit-learn, matplotlib, PyYAML, tqdm
- Conda environment: `mwc`

## Reference

`MATERIAL_CLASSIFIER_SPEC.md` is the authoritative specification — consult it for exact module code, hyperparameters, collation logic, and data format details.

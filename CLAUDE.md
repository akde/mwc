# CLAUDE.md

## Project Overview

Multi-view material classification from conveyor belt video. Multiple objects move simultaneously on a conveyor under a fixed monocular camera. Objects are detected per-frame (Detectron2 Mask R-CNN), tracked across frames (OC-SORT) into per-object tracklets, then classified. Goal: classify each tracked object as **glass, metal, paper, or plastic**.

## Architecture

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

## Current Results

**5-Fold Stratified Cross-Validation** on 550 labeled tracklets from 19 experiment videos:

| Metric | Value |
|--------|-------|
| Mean Accuracy | **0.9509 +/- 0.0093** |
| Mean Macro F1 | **0.9507 +/- 0.0110** |

Per-class (pooled across all folds):

| Class | Precision | Recall | F1 | Count |
|-------|-----------|--------|----|-------|
| glass | 0.978 | 0.992 | 0.985 | 132 |
| metal | 0.921 | 0.977 | 0.948 | 131 |
| paper | 0.957 | 0.908 | 0.932 | 98 |
| plastic | 0.951 | 0.926 | 0.938 | 189 |

Confusion matrix (pooled):
```
          glass  metal  paper  plastic
glass    [131      1      0      0]
metal    [  0    128      0      3]
paper    [  0      3     89      6]
plastic  [  3      7      4    175]
```

27 misclassifications out of 550. Glass is near-perfect. Most confusion is between plastic/metal/paper.

## Project Structure

```
material_classifier/
├── __init__.py
├── config/
│   └── default.yaml              # all hyperparameters and paths
├── tracking/
│   ├── detector.py               # Detectron2 Mask R-CNN inference wrapper
│   └── ocsort.py                 # OC-SORT tracker (with OCR + merge + dedup)
├── data/
│   ├── dataset.py                # TrackletDataset, CachedFeatureDataset, collate_fn
│   └── preprocessing.py          # masking, cropping, frame sampling, augmentation
├── models/
│   ├── feature_extractor.py      # DINOv2 frozen backbone wrapper
│   ├── attention_pool.py         # learnable attention pooling
│   ├── classifier.py             # MLP classification head
│   └── pipeline.py               # MaterialClassifier + CachedMaterialClassifier
├── inference.py                  # detect-and-track (single/batch) + full inference
├── train.py                      # feature caching + training loop
├── evaluate.py                   # evaluation metrics + confusion matrix
├── label.py                      # interactive tracklet labeling tool (OpenCV GUI)
├── visualize.py                  # tracklet overlay visualization on video
└── analyze_gaps.py               # detection gap analysis + frame extraction
```

Other project files:
```
cross_validate.py                 # 5-fold stratified cross-validation script
labels.csv                        # 550 tracklet labels (19 videos, 4 classes)
videos/                           # symlinks to experiment videos in ~/Downloads/
det2_model/                       # frozen Detectron2 model (config.yaml + model_best.pth)
tracklets/                        # detect-and-track output (generated, not committed)
features/                         # cached DINOv2 [CLS] features (generated, not committed)
checkpoints/                      # trained model weights (generated, not committed)
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

### Visualization

Produces an annotated video with colored mask overlays and track_id labels for review.

```bash
python material_classifier/visualize.py \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4 \
  --tracklets-dir tracklets/
```

### Training

```bash
# Cache DINOv2 features first (one-time), then train
python material_classifier/train.py --config material_classifier/config/default.yaml

# Cache features only (no training)
python material_classifier/train.py --config material_classifier/config/default.yaml --cache-features
```

### Evaluation

```bash
python material_classifier/evaluate.py \
  --config material_classifier/config/default.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split val
```

### Cross-Validation

```bash
# 5-fold stratified CV (uses cached features, restores original splits when done)
python cross_validate.py
```

### Full Inference (detect + track + classify)

```bash
python material_classifier/inference.py \
  --video path/to/video.mp4 \
  --checkpoint checkpoints/best_model.pt \
  --config material_classifier/config/default.yaml
```

## Workflow

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

9. **Feature caching:** Since DINOv2 is frozen, precompute [CLS] features per tracklet to disk and train pool+head on cached features without loading DINOv2. Training uses `CachedMaterialClassifier` (pool+head only, no backbone). Checkpoint saved from `CachedMaterialClassifier` is loaded into `MaterialClassifier` pool+head for inference.

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

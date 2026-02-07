# CLAUDE.md

## Project Overview

Multi-view material classification from conveyor belt video. Multiple objects move simultaneously on a conveyor under a fixed monocular camera. Objects are detected per-frame (Detectron2 Mask R-CNN), tracked across frames (OC-SORT) into per-object tracklets, then classified. Goal: classify each tracked object as **glass, metal, paper, or plastic**.

## Architecture

```
Video (.mp4)
  → Detectron2 Mask R-CNN (frozen, det2_model/) → per-frame detections + masks
  → OC-SORT Tracking → per-object tracklets (persistent track_id)
  → Per-tracklet: Masked Crop + Resize (518x518, gray fill 128)
  → Per-tracklet: Uniform sample N frames
  → DINOv2 ViT-L/14 (frozen) → [CLS] per frame → (B, T, 1024)
  → Attention Pooling (trainable) → (B, 1024)
  → MLP Head (trainable) → 4-class logits → {glass, metal, paper, plastic}
```

~528K trainable parameters (attention pool + MLP head). Detectron2 and DINOv2 are completely frozen.

## Project Structure

```
material_classifier/
├── __init__.py
├── config/
│   └── default.yaml              # all hyperparameters and paths
├── tracking/
│   ├── detector.py               # Detectron2 Mask R-CNN inference wrapper
│   └── ocsort.py                 # OC-SORT multi-object tracker
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
└── visualize.py                  # tracklet overlay visualization on video
```

Other project files:
```
det2_model/                       # frozen Detectron2 model (config.yaml + model_best.pth)
requirements.txt                  # Python dependencies
MATERIAL_CLASSIFIER_SPEC.md       # authoritative spec (module code, hyperparameters, data formats)
tracklets/                        # detect-and-track output (generated, not committed)
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

Output structure per video:
```
tracklets/{video_stem}/
  tracklet_data.csv                    # frame, track_id, x1, y1, x2, y2, score
  masks/frame_{:06d}_track_{}.png      # binary masks (0/255)
```

### Labeling

Interactive OpenCV GUI for assigning material classes to tracklets.

```bash
# Label tracklets interactively (1=glass 2=metal 3=paper 4=plastic, u=undo, s=skip, q=quit)
python material_classifier/label.py \
  --video ~/Downloads/0798/IMG_0798_synched_cropped.mp4 \
  --tracklets-dir tracklets/

# Assign stratified train/val splits after labeling
python material_classifier/label.py --assign-splits --labels-csv labels.csv
```

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
  --checkpoint checkpoints/best_model.pt
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
3. **Label** — run `label.py` to assign material classes interactively
4. **Split** — run `label.py --assign-splits` for stratified train/val
5. **Cache features** — run `train.py --cache-features` to precompute DINOv2 [CLS] tokens
6. **Train** — run `train.py` to train attention pool + MLP head on cached features
7. **Evaluate** — run `evaluate.py` on test split
8. **Infer** — run `inference.py --video --checkpoint` for end-to-end prediction

## Data Formats

### Labels CSV (manual annotation after step 3)

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
```

The `--video-dir` flag in `inference.py` auto-discovers this pattern.

## Pre-trained Models (frozen)

- **Detectron2**: `det2_model/config.yaml` + `det2_model/model_best.pth` (Mask R-CNN, ResNet-50-FPN, 4 classes, MASK_ON=true, BGR input)
- **DINOv2**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")` (ViT-L/14, 1024-dim [CLS] output)

## Critical Implementation Constraints

1. **DINOv2 must stay frozen.** Never set `requires_grad=True` on backbone parameters. The optimizer must only receive `pool` and `head` parameters.

2. **Detectron2 model is frozen.** Use the pre-trained model at `det2_model/` for inference only.

3. **Detectron2 config loading:** Must use `get_cfg()` (not bare `CfgNode()`) then `cfg.set_new_allowed(True)` before `merge_from_file`. Bare `CfgNode()` lacks the `VERSION` field → `AttributeError: VERSION`.

4. **Image size must be 518x518.** DINOv2 ViT-L/14 patch size is 14; input must be divisible by 14. The model was trained at 518x518.

5. **Masked regions filled with neutral gray (128), not black.** Black pixels become non-zero after DINOv2 normalization and contaminate features. Gray is closer to the normalized mean.

6. **Detection gaps are natural.** OC-SORT handles frames where an object is not detected. Do not interpolate or hallucinate masks for missing frames.

7. **Augmentation is per-frame, applied after masking and cropping** — not before.

8. **Feature caching:** Since DINOv2 is frozen, precompute [CLS] features per tracklet to disk and train pool+head on cached features without loading DINOv2. Training uses `CachedMaterialClassifier` (pool+head only, no backbone). Checkpoint saved from `CachedMaterialClassifier` is loaded into `MaterialClassifier` pool+head for inference.

9. **Color space:** Detectron2 works in BGR (OpenCV native). DINOv2 expects RGB. Conversion happens in preprocessing after extracting frames.

## Tech Stack

- Python 3.11, PyTorch >= 2.1, torchvision >= 0.16
- DINOv2 via torch.hub
- Detectron2 (Mask R-CNN)
- OC-SORT for multi-object tracking
- OpenCV, Pillow, NumPy, SciPy, scikit-learn, matplotlib, PyYAML, tqdm
- Conda environment: `mwc`

## Reference

`MATERIAL_CLASSIFIER_SPEC.md` is the authoritative specification — consult it for exact module code, hyperparameters, collation logic, and data format details.

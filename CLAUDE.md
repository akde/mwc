# CLAUDE.md

## Project Overview

Multi-view material classification from conveyor belt video. Multiple objects move simultaneously on the conveyor under a fixed monocular camera. Objects are detected per-frame (Detectron2), tracked across frames (OC-SORT) into per-object tracklets, then classified. Goal: classify each tracked object as **plastic, metal, glass, or paper**.

## Architecture

```
Video → Frame Extraction
      → Detectron2 Mask R-CNN (frozen, det2_model/) → per-frame detections + masks
      → OC-SORT Tracking → per-object tracklets (persistent track_id)
      → Per-tracklet: Masked Crop + Resize (518x518, gray fill 128)
      → Per-tracklet: Uniform sample N frames
      → DINOv2 ViT-L/14 (frozen) → [CLS] per frame → (B, T, 1024)
      → Attention Pooling (trainable) → (B, 1024)
      → MLP Head (trainable) → 4-class logits
```

Only ~200K parameters are trainable (attention pool + MLP head). Both Detectron2 and DINOv2 are completely frozen.

## Project Structure (planned)

```
material_classifier/
├── config/default.yaml
├── tracking/
│   ├── detector.py             # Detectron2 inference wrapper
│   └── ocsort.py               # OC-SORT tracker
├── data/
│   ├── dataset.py              # TrackletDataset
│   └── preprocessing.py        # masking, cropping, frame sampling
├── models/
│   ├── feature_extractor.py    # DINOv2 frozen wrapper
│   ├── attention_pool.py       # learnable attention pooling
│   ├── classifier.py           # MLP head
│   └── pipeline.py             # full model assembly
├── train.py
├── evaluate.py
├── inference.py
└── requirements.txt
```

## Data Format

Videos contain multiple objects of different classes simultaneously. After detection + tracking, tracklets are manually labeled:

```csv
video,track_id,split,class
video_001.mp4,1,train,plastic
video_001.mp4,2,train,metal
video_002.mp4,1,val,glass
```

## Pre-trained Models (frozen, not trained further)

- **Detectron2**: `det2_model/config.yaml` + `det2_model/model_best.pth`
- **DINOv2**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")`
- **OC-SORT reference**: `/home/akde/phd/spatio_temporal_alignment/object_tracker_ocsort.py`

## Critical Implementation Constraints

1. **DINOv2 must stay frozen.** Never set `requires_grad=True` on backbone parameters. The optimizer must only receive `pool` and `head` parameters.

2. **Detectron2 model is frozen.** Use the pre-trained model at `det2_model/` for inference only. Do not train it further.

3. **Detectron2 config loading:** Must use `get_cfg()` (not bare `CfgNode()`) then `cfg.set_new_allowed(True)` before `merge_from_file`. A bare `CfgNode()` lacks the `VERSION` field that `merge_from_file` expects, causing `AttributeError: VERSION`.

3. **Image size must be 518x518.** DINOv2 ViT-L/14 patch size is 14; input must be divisible by 14. The model was trained at 518x518 — other sizes degrade feature quality.

4. **Masked regions filled with neutral gray (128), not black.** Black pixels become non-zero after DINOv2 normalization and contaminate features. Gray is closer to the normalized mean.

5. **Detection gaps are natural.** OC-SORT handles frames where an object is not detected. Do not interpolate or hallucinate masks for missing frames.

6. **Augmentation is per-frame, applied after masking and cropping** — not before.

7. **Feature caching:** Since DINOv2 is frozen, precompute [CLS] features per tracklet to disk and train pool+head on cached features without loading DINOv2.

## Tech Stack

- Python 3, PyTorch >= 2.1, torchvision >= 0.16
- DINOv2 via torch.hub
- Detectron2 (Mask R-CNN)
- OC-SORT for multi-object tracking
- OpenCV, Pillow, NumPy, scikit-learn, matplotlib, PyYAML, tqdm

## Commands (once implemented)

```bash
# Detection + tracking (produces tracklets for labeling)
python material_classifier/inference.py --detect-and-track --video path/to/video.mp4

# Training (on labeled tracklets)
python material_classifier/train.py --config config/default.yaml

# Evaluation
python material_classifier/evaluate.py --config config/default.yaml --checkpoint path/to/best.pt

# Full inference (detect + track + classify)
python material_classifier/inference.py --video path/to/video.mp4 --checkpoint path/to/best.pt
```

## Reference

`MATERIAL_CLASSIFIER_SPEC.md` is the authoritative specification — consult it for exact module code, hyperparameters, collation logic, and data format details.

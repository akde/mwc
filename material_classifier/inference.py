import argparse
import csv
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def discover_videos(video_dir):
    """
    Discover experiment videos in a directory.

    Looks for subdirectories with 4-digit names (e.g., 0797, 0798) containing
    a video named IMG_XXXX_synched_cropped.mp4.

    Returns sorted list of video paths found.
    """
    video_dir = os.path.expanduser(video_dir)
    pattern = os.path.join(video_dir, "[0-9][0-9][0-9][0-9]")
    subdirs = sorted(glob.glob(pattern))

    videos = []
    for subdir in subdirs:
        dirname = os.path.basename(subdir)
        if not os.path.isdir(subdir):
            continue
        # Look for IMG_XXXX_synched_cropped.mp4
        video_pattern = os.path.join(subdir, f"IMG_{dirname}_synched_cropped.mp4")
        matches = glob.glob(video_pattern)
        if matches:
            videos.append(matches[0])

    print(f"Discovered {len(videos)} videos in {video_dir}")
    for v in videos:
        print(f"  {os.path.relpath(v, video_dir)}")

    return videos


def detect_and_track(video_path, output_dir, detector, tracker_params):
    """
    Process a video: detect objects per frame, track across frames, save tracklets.

    Output:
        {output_dir}/{video_stem}/
            tracklet_data.csv       # frame, track_id, x1, y1, x2, y2, score
            masks/frame_{:06d}_track_{}.png  # binary masks per detection per frame
    """
    from material_classifier.tracking.ocsort import OCSort

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_stem)
    masks_dir = os.path.join(video_output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    tracker = OCSort(
        det_thresh=tracker_params.get("det_thresh", 0.5),
        max_age=tracker_params.get("max_age", 40),
        min_hits=tracker_params.get("min_hits", 3),
        iou_threshold=tracker_params.get("iou_threshold", 0.3),
        delta_t=tracker_params.get("delta_t", 3),
        inertia=tracker_params.get("inertia", 0.2),
        merge_iou_threshold=tracker_params.get("merge_iou_threshold", 0.7),
        merge_patience=tracker_params.get("merge_patience", 3),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    csv_rows = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_stem}", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        if len(detections) > 0:
            dets_array = np.array(
                [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["score"]]
                 for d in detections]
            )
            metadata_list = [{"mask": d["mask"]} for d in detections]
        else:
            dets_array = np.empty((0, 5))
            metadata_list = []

        tracks = tracker.update(dets_array, metadata_list)

        for t in tracks:
            track_id = t["track_id"]
            bbox = t["bbox"]
            mask = t["mask"]

            # Save binary mask as PNG
            if mask is not None:
                mask_filename = f"frame_{frame_idx:06d}_track_{track_id}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                mask_uint8 = (mask.astype(np.uint8)) * 255
                cv2.imwrite(mask_path, mask_uint8)

            # Find the detection score for this track (use bbox IoU to match back)
            score = 0.0
            for d in detections:
                db = d["bbox"]
                # Simple check: if the track bbox is close to a detection bbox
                if (abs(db[0] - bbox[0]) < 50 and abs(db[1] - bbox[1]) < 50):
                    score = d["score"]
                    break

            csv_rows.append([
                frame_idx, track_id,
                f"{bbox[0]:.1f}", f"{bbox[1]:.1f}",
                f"{bbox[2]:.1f}", f"{bbox[3]:.1f}",
                f"{score:.4f}",
            ])

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Write tracklet CSV
    csv_path = os.path.join(video_output_dir, "tracklet_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score"])
        writer.writerows(csv_rows)

    print(f"Saved {len(csv_rows)} track entries to {csv_path}")
    print(f"Masks saved to {masks_dir}")


def full_inference(video_path, checkpoint_path, config):
    """
    Full pipeline: detect -> track -> classify each tracklet.

    Returns:
        List of dicts per tracklet:
            {'track_id': int, 'class': str, 'confidence': float,
             'attention_weights': np.ndarray, 'frame_indices': list[int]}
    """
    import torch
    import yaml

    from material_classifier.tracking.detector import Detectron2Detector
    from material_classifier.tracking.ocsort import OCSort
    from material_classifier.data.preprocessing import apply_mask_and_crop, get_transform, sample_frames
    from material_classifier.data.dataset import CLASS_NAMES
    from material_classifier.models.pipeline import MaterialClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    # Load detector
    detector = Detectron2Detector(
        config["detection"]["model_dir"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        device=device,
    )

    # Load model
    model_cfg = config["model"]
    model = MaterialClassifier(
        backbone=model_cfg["backbone"],
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg.get("attention_temperature", 1.0),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load pool + head weights (checkpoint was saved from CachedMaterialClassifier)
    state_dict = checkpoint["model_state_dict"]
    model.pool.load_state_dict(
        {k.replace("pool.", ""): v for k, v in state_dict.items() if k.startswith("pool.")}
    )
    model.head.load_state_dict(
        {k.replace("head.", ""): v for k, v in state_dict.items() if k.startswith("head.")}
    )
    model = model.to(device)
    model.eval()

    transform = get_transform(image_size=config["data"]["image_size"], train=False)

    # Tracker params
    tracking_cfg = config["tracking"]
    tracker = OCSort(
        det_thresh=tracking_cfg["det_thresh"],
        max_age=tracking_cfg["max_age"],
        min_hits=tracking_cfg["min_hits"],
        iou_threshold=tracking_cfg["iou_threshold"],
        delta_t=tracking_cfg["delta_t"],
        inertia=tracking_cfg["inertia"],
        merge_iou_threshold=tracking_cfg.get("merge_iou_threshold", 0.7),
        merge_patience=tracking_cfg.get("merge_patience", 3),
    )

    # Process video: collect per-track frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # track_id -> list of (frame_idx, masked_cropped_rgb)
    tracklet_frames = {}

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Detecting & tracking", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        if len(detections) > 0:
            dets_array = np.array(
                [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["score"]]
                 for d in detections]
            )
            metadata_list = [{"mask": d["mask"]} for d in detections]
        else:
            dets_array = np.empty((0, 5))
            metadata_list = []

        tracks = tracker.update(dets_array, metadata_list)

        for t in tracks:
            track_id = t["track_id"]
            mask = t["mask"]
            if mask is not None:
                cropped = apply_mask_and_crop(frame, mask, gray_fill=128)
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                if track_id not in tracklet_frames:
                    tracklet_frames[track_id] = []
                tracklet_frames[track_id].append((frame_idx, rgb))

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Classify each tracklet
    num_frames = config["data"]["num_frames"]
    results = []

    with torch.no_grad():
        for track_id, frame_list in tracklet_frames.items():
            indices = sample_frames(len(frame_list), num_frames)
            sampled = [frame_list[i] for i in indices]
            frame_indices = [s[0] for s in sampled]

            frames_tensor = torch.stack([transform(s[1]) for s in sampled])  # (T, 3, 518, 518)
            frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, T, 3, 518, 518)
            mask_tensor = torch.ones(1, len(indices), dtype=torch.bool, device=device)

            logits = model(frames_tensor, mask_tensor)  # (1, num_classes)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)

            # Get attention weights
            B, T = frames_tensor.shape[:2]
            flat = frames_tensor.view(B * T, *frames_tensor.shape[2:])
            with torch.amp.autocast("cuda"):
                features = model.extractor(flat)
            features = features.view(B, T, -1)
            attn_weights = model.pool.get_attention_weights(features, mask_tensor)

            results.append({
                "track_id": track_id,
                "class": CLASS_NAMES[pred_idx.item()],
                "confidence": confidence.item(),
                "attention_weights": attn_weights.cpu().numpy().squeeze(),
                "frame_indices": frame_indices,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Material classifier inference")
    parser.add_argument("--detect-and-track", action="store_true",
                        help="Run detection + tracking only (Phase 1)")
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing experiment folders (4-digit names) with videos")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Overwrite already processed videos (default: True, use --no-overwrite to skip)")
    parser.add_argument("--output-dir", default="tracklets/",
                        help="Output directory for tracklets (detect-and-track mode)")
    parser.add_argument("--model-dir", default="det2_model/",
                        help="Detectron2 model directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=60)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--delta-t", type=int, default=3)
    parser.add_argument("--inertia", type=float, default=0.2)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.7,
                        help="IoU threshold for merging duplicate tracks (0=disable)")
    parser.add_argument("--merge-patience", type=int, default=3,
                        help="Consecutive frames of overlap before merging")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint (full inference mode)")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (full inference mode)")
    args = parser.parse_args()

    if args.detect_and_track:
        if not args.video and not args.video_dir:
            parser.error("--detect-and-track requires either --video or --video-dir")

        from material_classifier.tracking.detector import Detectron2Detector

        detector = Detectron2Detector(
            args.model_dir,
            confidence_threshold=args.confidence_threshold,
        )
        tracker_params = {
            "det_thresh": args.confidence_threshold,
            "max_age": args.max_age,
            "min_hits": args.min_hits,
            "iou_threshold": args.iou_threshold,
            "delta_t": args.delta_t,
            "inertia": args.inertia,
            "merge_iou_threshold": args.merge_iou_threshold,
            "merge_patience": args.merge_patience,
        }

        if args.video_dir:
            videos = discover_videos(args.video_dir)
            if not videos:
                print("No videos found. Exiting.")
                return

            for i, video_path in enumerate(videos, 1):
                video_stem = os.path.splitext(os.path.basename(video_path))[0]
                output_csv = os.path.join(args.output_dir, video_stem, "tracklet_data.csv")
                if os.path.exists(output_csv) and not args.overwrite:
                    print(f"[{i}/{len(videos)}] Skipping {video_stem} (output already exists)")
                    continue
                print(f"\n[{i}/{len(videos)}] Processing {video_stem}...")
                detect_and_track(video_path, args.output_dir, detector, tracker_params)
        else:
            detect_and_track(args.video, args.output_dir, detector, tracker_params)
    elif args.checkpoint and args.config:
        if not args.video:
            parser.error("Full inference mode requires --video")
        results = full_inference(args.video, args.checkpoint, args.config)
        print(f"\n{'='*60}")
        print(f"Classification Results ({len(results)} tracklets)")
        print(f"{'='*60}")
        for r in results:
            print(f"  Track {r['track_id']:3d}: {r['class']:8s} "
                  f"(confidence: {r['confidence']:.3f})")
            if r["attention_weights"] is not None:
                top_idx = np.argsort(r["attention_weights"])[-3:][::-1]
                print(f"             Top attention frames: "
                      f"{[r['frame_indices'][i] for i in top_idx]}")
    else:
        parser.error("Use --detect-and-track for Phase 1, "
                      "or --checkpoint + --config for full inference")


if __name__ == "__main__":
    main()

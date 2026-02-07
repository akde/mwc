"""Overlay tracked masks on original video for labeling review.

Produces an annotated video with colored mask overlays and track_id labels,
useful for visually inspecting tracklets before creating labels.csv.

Usage:
    python material_classifier/visualize.py \
        --video /path/to/video.mp4 \
        --tracklets-dir tracklets/
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def track_id_to_color(track_id: int) -> tuple:
    """Map track_id to a distinct BGR color via HSV."""
    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def parse_tracklet_data(csv_path: str) -> dict:
    """Parse tracklet_data.csv into {frame_idx: [(track_id, x1, y1, x2, y2), ...]}."""
    frame_tracks = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            track_id = int(row["track_id"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            frame_tracks[frame_idx].append((track_id, x1, y1, x2, y2))
    return dict(frame_tracks)


def main():
    parser = argparse.ArgumentParser(description="Visualize tracklets on video")
    parser.add_argument("--video", required=True, help="Path to original video")
    parser.add_argument("--tracklets-dir", required=True, help="Base tracklets directory")
    parser.add_argument("--output", default=None, help="Output video path (default: auto)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    args = parser.parse_args()

    video_stem = Path(args.video).stem
    tracklet_dir = Path(args.tracklets_dir) / video_stem
    csv_path = tracklet_dir / "tracklet_data.csv"
    masks_dir = tracklet_dir / "masks"

    if not csv_path.exists():
        raise FileNotFoundError(f"tracklet_data.csv not found at {csv_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found at {masks_dir}")

    output_path = args.output or str(tracklet_dir / "annotated_video.mp4")

    # Parse tracklet CSV
    print(f"Parsing {csv_path} ...")
    frame_tracks = parse_tracklet_data(str(csv_path))
    total_annotations = sum(len(v) for v in frame_tracks.values())
    print(f"  {len(frame_tracks)} frames with annotations, {total_annotations} total masks")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps:.1f} fps, {total_frames} frames")

    # Open writer
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for: {output_path}")

    print(f"Writing annotated video to {output_path} ...")

    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        tracks = frame_tracks.get(frame_idx, [])
        for track_id, x1, y1, x2, y2 in tracks:
            mask_path = masks_dir / f"frame_{frame_idx:06d}_track_{track_id}.png"
            if not mask_path.exists():
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            color = track_id_to_color(track_id)
            mask_bool = mask > 127

            # Semi-transparent colored overlay
            overlay = frame.copy()
            overlay[mask_bool] = color
            cv2.addWeighted(overlay, args.alpha, frame, 1.0 - args.alpha, 0, frame)

            # Find centroid for label placement
            ys, xs = np.where(mask_bool)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                cx, cy = int((x1 + x2) / 2), int(y1)

            # Draw track_id label with dark background
            label = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            tx, ty = cx - tw // 2, cy - 4
            cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done. Output: {output_path} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()

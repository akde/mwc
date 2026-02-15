"""Overlay warped tracklet masks on thermal frames for visual review.

Produces an annotated video synced to RGB frame count with colored mask overlays,
using the RGB-to-thermal homography to warp masks into thermal space. Iterates
over RGB frames (not thermal) so the output has the same frame count and FPS as
the RGB overlay video, enabling side-by-side comparison.

Usage:
    python material_classifier/thermal/visualize.py \
        --experiment 0798 \
        --tracklets-dir tracklets/

    # All experiments
    python material_classifier/thermal/visualize.py \
        --all --tracklets-dir tracklets/
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from material_classifier.thermal.utils import (
    get_experiment_id,
    load_experiment_resources,
    warp_mask_to_thermal,
)


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



def _find_rgb_video(exp_id: str, video_dir: str, downloads_dir: str) -> str | None:
    """Locate the RGB source video, checking video_dir first, then downloads_dir."""
    video_name = f"IMG_{exp_id}_synched_cropped.mp4"
    # Check video_dir (may contain symlinks)
    candidate = os.path.join(video_dir, video_name)
    if os.path.exists(candidate):
        return candidate
    # Check downloads_dir/{exp_id}/
    candidate = os.path.join(downloads_dir, exp_id, video_name)
    if os.path.exists(candidate):
        return candidate
    return None


def visualize_thermal_experiment(
    exp_id: str,
    downloads_dir: str,
    tracklets_dir: str,
    output_dir: str,
    video_dir: str = "videos/",
    alpha: float = 0.4,
    fps: float = 30.0,
    colormap: int = cv2.COLORMAP_INFERNO,
    scale: float = 3.0,
):
    """Create annotated thermal video synced to RGB frame count."""
    exp_dir = os.path.join(downloads_dir, exp_id)
    res = load_experiment_resources(exp_dir, exp_id)
    if res is None:
        print(f"  Skipping {exp_id}: thermal resources not found")
        return None

    H = res["H"]
    frame_map = res["frame_map"]  # rgb_frame -> thermal_frame
    thermal_dir = res["thermal_frames_dir"]

    # Open RGB video to get total frame count and FPS
    rgb_video_path = _find_rgb_video(exp_id, video_dir, downloads_dir)
    if rgb_video_path is None:
        print(f"  Skipping {exp_id}: RGB video not found")
        return None

    rgb_cap = cv2.VideoCapture(rgb_video_path)
    if not rgb_cap.isOpened():
        print(f"  Skipping {exp_id}: cannot open RGB video")
        return None

    total_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_cap.release()

    if video_fps > 0:
        fps = video_fps  # Use RGB video's FPS; --fps is fallback

    # Find the video name for this experiment
    video_name = f"IMG_{exp_id}_synched_cropped"
    tracklet_csv = os.path.join(tracklets_dir, video_name, "tracklet_data.csv")
    masks_dir = os.path.join(tracklets_dir, video_name, "masks")

    if not os.path.exists(tracklet_csv):
        print(f"  Skipping {exp_id}: no tracklet_data.csv")
        return None

    # Parse tracklets (indexed by RGB frame)
    frame_tracks = parse_tracklet_data(tracklet_csv)

    # Read one thermal frame to get dimensions
    sample_frame = None
    pattern = re.compile(r"FLIR_frame_(\d+)\.png")
    for f in os.listdir(thermal_dir):
        if pattern.match(f):
            sample_frame = cv2.imread(
                os.path.join(thermal_dir, f), cv2.IMREAD_GRAYSCALE
            )
            if sample_frame is not None:
                break
    if sample_frame is None:
        print(f"  Skipping {exp_id}: no readable thermal frames")
        return None

    th, tw = sample_frame.shape[:2]
    out_w, out_h = int(tw * scale), int(th * scale)

    # Set up output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"thermal_{exp_id}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"  Cannot open writer for {output_path}")
        return None

    n_annotated = 0
    last_thermal_bgr = None  # Hold last thermal frame for gaps
    gray_frame = np.full((th, tw), 128, dtype=np.uint8)  # Fallback before first thermal

    print(f"  {total_frames} RGB frames, {len(frame_map)} mapped, FPS={fps:.1f}")

    for rgb_idx in tqdm(range(total_frames), desc=f"  {exp_id}"):
        # Look up corresponding thermal frame
        thermal_idx = frame_map.get(rgb_idx)

        if thermal_idx is not None:
            thermal_path = os.path.join(
                thermal_dir, f"FLIR_frame_{thermal_idx:06d}.png"
            )
            thermal_gray = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_gray is not None:
                last_thermal_bgr = cv2.applyColorMap(thermal_gray, colormap)

        # Use last seen thermal frame, or gray fallback
        if last_thermal_bgr is not None:
            frame_bgr = last_thermal_bgr.copy()
        else:
            frame_bgr = cv2.applyColorMap(gray_frame, colormap)

        # Overlay tracklet masks for this RGB frame
        tracks = frame_tracks.get(rgb_idx, [])
        for track_id, x1, y1, x2, y2 in tracks:
            mask_path = os.path.join(
                masks_dir, f"frame_{rgb_idx:06d}_track_{track_id}.png"
            )
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = (mask > 127).astype(np.uint8) * 255

            # Warp mask to thermal space
            warped = warp_mask_to_thermal(mask, H, (th, tw))
            warped_bool = warped > 127

            if not warped_bool.any():
                continue

            n_annotated += 1
            color = track_id_to_color(track_id)

            # Semi-transparent overlay
            overlay = frame_bgr.copy()
            overlay[warped_bool] = color
            cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

            # Track label at centroid
            ys, xs = np.where(warped_bool)
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (ltw, lth), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx, ty = cx - ltw // 2, cy - 2
            cv2.rectangle(
                frame_bgr,
                (tx - 1, ty - lth - 1),
                (tx + ltw + 1, ty + 1),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame_bgr,
                label,
                (tx, ty),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Scale up for visibility
        frame_out = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        writer.write(frame_out)

    writer.release()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output: {output_path} ({file_size:.1f} MB, {n_annotated} mask overlays)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize thermal tracklets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=str, help="Experiment ID (e.g. 0798)")
    group.add_argument("--all", action="store_true", help="Process all experiments")
    parser.add_argument(
        "--tracklets-dir", default="tracklets/", help="Base tracklets directory"
    )
    parser.add_argument(
        "--downloads-dir",
        default="~/Downloads",
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--output-dir",
        default="thermal_viz/",
        help="Output directory for annotated videos",
    )
    parser.add_argument(
        "--video-dir",
        default="videos/",
        help="Directory containing RGB source videos (for frame count/FPS)",
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Fallback FPS if RGB video unavailable"
    )
    parser.add_argument(
        "--scale", type=float, default=3.0, help="Upscale factor (thermal is 270x164)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        choices=["inferno", "jet", "hot", "magma", "plasma", "viridis", "gray"],
        help="Colormap for thermal visualization",
    )
    args = parser.parse_args()

    downloads_dir = os.path.expanduser(args.downloads_dir)

    cmap_lookup = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "gray": None,
    }
    colormap = cmap_lookup[args.colormap]

    if args.experiment:
        exp_ids = [args.experiment]
    else:
        # Discover all experiments that have tracklets
        exp_ids = []
        for entry in sorted(os.listdir(args.tracklets_dir)):
            m = re.match(r"IMG_(\d{4})_synched_cropped", entry)
            if m:
                exp_ids.append(m.group(1))

    print(f"Processing {len(exp_ids)} experiment(s)...")
    outputs = []
    for exp_id in exp_ids:
        print(f"\nExperiment {exp_id}:")
        result = visualize_thermal_experiment(
            exp_id=exp_id,
            downloads_dir=downloads_dir,
            tracklets_dir=args.tracklets_dir,
            output_dir=args.output_dir,
            video_dir=args.video_dir,
            alpha=args.alpha,
            fps=args.fps,
            colormap=colormap if colormap is not None else cv2.COLORMAP_BONE,
            scale=args.scale,
        )
        if result:
            outputs.append(result)

    print(f"\nDone. Created {len(outputs)} annotated video(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()

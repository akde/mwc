"""Interactive tracklet labeling tool for material classification.

Shows a visual montage per tracklet (context frame + masked crops) and captures
the material class via keypress. Supports resume, undo, skip, and quit.

Usage:
    # Label interactively
    python material_classifier/label.py \
        --video /path/to/video.mp4 \
        --tracklets-dir tracklets/

    # Assign stratified train/val splits after labeling
    python material_classifier/label.py --assign-splits --labels-csv labels.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# Class mapping: key code -> class name
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
KEY_TO_CLASS = {
    ord("1"): "glass",
    ord("2"): "metal",
    ord("3"): "paper",
    ord("4"): "plastic",
}
KEY_UNDO = ord("u")
KEY_SKIP = ord("s")
KEY_QUIT = ord("q")
KEY_CAPTURE = ord("c")


def track_id_to_color(track_id: int) -> tuple:
    """Map track_id to a distinct BGR color via HSV."""
    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def sample_frames(total_frames, num_frames=6):
    """Uniform temporal sampling of frame indices."""
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()


def parse_tracklets_by_track(csv_path: str) -> dict:
    """Parse tracklet_data.csv grouped by track_id.

    Returns:
        {track_id: [(frame_idx, x1, y1, x2, y2), ...]} sorted by frame_idx
    """
    tracks = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            frame_idx = int(row["frame"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            tracks[track_id].append((frame_idx, x1, y1, x2, y2))
    # Sort each track by frame index
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return dict(tracks)


def load_existing_labels(labels_csv: str) -> list:
    """Load existing labels.csv rows. Returns list of dicts."""
    if not os.path.exists(labels_csv):
        return []
    rows = []
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_labels(labels_csv: str, rows: list):
    """Write labels list to CSV."""
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "track_id", "split", "class"])
        writer.writeheader()
        writer.writerows(rows)


def append_label(labels_csv: str, row: dict):
    """Append a single label row to CSV. Creates file with header if needed."""
    exists = os.path.exists(labels_csv)
    with open(labels_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "track_id", "split", "class"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_montage(cap, track_id, track_data, sample_indices):
    """Build a montage image for a tracklet.

    Layout:
        - Header text row
        - Context frame (full frame with bbox, scaled to ~600px wide)
        - 2x3 grid of cropped objects (200x200 each)
        - Footer legend row
    """
    color = track_id_to_color(track_id)
    crop_size = 200
    context_width = 600
    n_cols = 3

    # Collect frames and crops
    crops = []
    context_frame = None

    for i, sample_idx in enumerate(sample_indices):
        frame_idx, x1, y1, x2, y2 = track_data[sample_idx]

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            crops.append((np.full((crop_size, crop_size, 3), 128, dtype=np.uint8), frame_idx))
            continue

        # Context frame: first sample, bbox only (no mask overlay)
        if i == 0:
            ctx = frame.copy()
            cv2.rectangle(ctx, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Scale to context_width
            h, w = ctx.shape[:2]
            scale = context_width / w
            context_frame = cv2.resize(ctx, (context_width, int(h * scale)))

        # Crop with padding
        h_frame, w_frame = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * 0.2
        pad_y = bh * 0.2
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w_frame, int(x2 + pad_x))
        cy2 = min(h_frame, int(y2 + pad_y))

        crop = frame[cy1:cy2, cx1:cx2].copy()

        # Resize to crop_size x crop_size
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = cv2.resize(crop, (crop_size, crop_size))
        else:
            crop = np.full((crop_size, crop_size, 3), 128, dtype=np.uint8)

        # Add frame number label
        label = f"F{frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(crop, label, (5, crop_size - 8), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(crop, label, (5, crop_size - 8), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        crops.append((crop, frame_idx))

    # Pad crops to fill 2x3 grid
    while len(crops) < 6:
        crops.append((np.full((crop_size, crop_size, 3), 128, dtype=np.uint8), -1))

    # Build grid: 2 rows x 3 cols
    grid_width = n_cols * crop_size
    n_rows = 2
    grid_rows = []
    for r in range(n_rows):
        row_crops = [crops[r * n_cols + c][0] for c in range(n_cols)]
        grid_rows.append(np.hstack(row_crops))
    crop_grid = np.vstack(grid_rows)

    # Scale context frame to match grid width
    if context_frame is not None:
        ctx_h, ctx_w = context_frame.shape[:2]
        scale = grid_width / ctx_w
        context_frame = cv2.resize(context_frame, (grid_width, int(ctx_h * scale)))
    else:
        context_frame = np.full((200, grid_width, 3), 128, dtype=np.uint8)

    # Assemble montage
    montage_parts = [context_frame, crop_grid]
    montage = np.vstack(montage_parts)

    return montage


def draw_header_footer(montage, track_id, total_frames, current_idx, total_tracks):
    """Add header and footer text bars to montage."""
    h, w = montage.shape[:2]
    header_h = 35
    footer_h = 35

    # Create padded canvas
    canvas = np.full((header_h + h + footer_h, w, 3), 40, dtype=np.uint8)
    canvas[header_h:header_h + h] = montage

    # Header
    header_text = f"Track {track_id} | {total_frames} frames | [{current_idx + 1}/{total_tracks}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, header_text, (10, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer
    footer_text = "1=glass  2=metal  3=paper  4=plastic  |  u=undo  s=skip  c=capture  q=quit"
    cv2.putText(canvas, footer_text, (10, header_h + h + 25), font, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def label_interactive(video_path: str, tracklets_dir: str, labels_csv: str,
                      min_track_length: int = 0):
    """Main interactive labeling loop."""
    video_stem = Path(video_path).stem
    video_name = Path(video_path).name
    tracklet_dir = Path(tracklets_dir) / video_stem
    csv_path = tracklet_dir / "tracklet_data.csv"
    masks_dir = tracklet_dir / "masks"

    if not csv_path.exists():
        raise FileNotFoundError(f"tracklet_data.csv not found at {csv_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found at {masks_dir}")

    # Parse tracklets
    print(f"Parsing {csv_path} ...")
    tracks = parse_tracklets_by_track(str(csv_path))
    total_before_filter = len(tracks)

    # Filter by minimum track length
    if min_track_length > 0:
        tracks = {tid: data for tid, data in tracks.items()
                  if len(data) >= min_track_length}
        print(f"  Found {total_before_filter} tracks, {len(tracks)} passed "
              f"min-track-length filter ({min_track_length})")
    else:
        print(f"  Found {total_before_filter} tracks")

    all_track_ids = sorted(tracks.keys())

    # Load existing labels for resume
    existing_labels = load_existing_labels(labels_csv)
    labeled_set = {(r["video"], int(r["track_id"])) for r in existing_labels}
    label_stack = list(existing_labels)  # for undo support
    print(f"  Already labeled: {len(labeled_set)} tracks")

    # Filter to unlabeled tracks
    unlabeled_ids = [tid for tid in all_track_ids if (video_name, tid) not in labeled_set]
    if not unlabeled_ids:
        print("All tracks are already labeled!")
        return

    print(f"  Remaining: {len(unlabeled_ids)} tracks to label")
    print()
    print("Keys: 1=glass  2=metal  3=paper  4=plastic  u=undo  s=skip  c=capture  q=quit")
    print()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cv2.namedWindow("Label Tracklet", cv2.WINDOW_NORMAL)

    i = 0
    while i < len(unlabeled_ids):
        track_id = unlabeled_ids[i]
        track_data = tracks[track_id]
        total_frames = len(track_data)

        # Sample 6 frames
        sample_indices = sample_frames(total_frames, num_frames=6)

        # Build montage
        montage = make_montage(cap, track_id, track_data, sample_indices)

        # Overall index (including already labeled)
        overall_idx = len(labeled_set) + i
        canvas = draw_header_footer(montage, track_id, total_frames,
                                    overall_idx, len(all_track_ids))

        cv2.imshow("Label Tracklet", canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in KEY_TO_CLASS:
            cls = KEY_TO_CLASS[key]
            row = {"video": video_name, "track_id": str(track_id), "split": "", "class": cls}
            label_stack.append(row)
            labeled_set.add((video_name, track_id))
            append_label(labels_csv, row)
            print(f"  Track {track_id} -> {cls}")
            i += 1

        elif key == KEY_SKIP:
            print(f"  Track {track_id} -> skipped")
            i += 1

        elif key == KEY_UNDO:
            if label_stack:
                undone = label_stack.pop()
                undone_tid = int(undone["track_id"])
                labeled_set.discard((undone["video"], undone_tid))
                # Rewrite labels.csv without the undone row
                write_labels(labels_csv, label_stack)
                print(f"  Undone track {undone_tid} ({undone['class']})")
                # Go back: if we can find the undone track in unlabeled_ids, go to it
                if undone_tid in unlabeled_ids:
                    i = unlabeled_ids.index(undone_tid)
                else:
                    # The undone track was already in the original labeled set;
                    # insert it at current position
                    unlabeled_ids.insert(i, undone_tid)
            else:
                print("  Nothing to undo")

        elif key == KEY_CAPTURE:
            # Save the first sampled frame (original, no annotations)
            first_frame_idx = track_data[sample_indices[0]][0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
            ret, raw_frame = cap.read()
            if ret:
                save_name = f"{video_stem}_frame_{first_frame_idx:06d}.png"
                cv2.imwrite(save_name, raw_frame)
                print(f"  Saved {save_name}")
            else:
                print(f"  Failed to read frame {first_frame_idx}")

        elif key == KEY_QUIT:
            print(f"\nQuitting. {len(label_stack)} labels saved to {labels_csv}")
            break

    cap.release()
    cv2.destroyAllWindows()

    labeled_count = len(label_stack)
    total_count = len(all_track_ids)
    print(f"\nLabeling session done. {labeled_count}/{total_count} tracks labeled.")
    if labeled_count < total_count:
        print(f"Re-run to continue labeling the remaining {total_count - labeled_count} tracks.")


def assign_splits(labels_csv: str, train_ratio: float = 0.8, seed: int = 42):
    """Assign stratified train/val splits to labeled tracklets."""
    rows = load_existing_labels(labels_csv)
    if not rows:
        print(f"No labels found in {labels_csv}")
        return

    # Filter to rows that have a class label
    labeled = [r for r in rows if r.get("class")]
    if not labeled:
        print("No labeled rows found")
        return

    rng = np.random.RandomState(seed)

    # Group by class for stratified split
    by_class = defaultdict(list)
    for idx, r in enumerate(labeled):
        by_class[r["class"]].append(idx)

    for cls, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        for j, idx in enumerate(indices):
            labeled[idx]["split"] = "train" if j < n_train else "val"

    write_labels(labels_csv, labeled)

    # Print summary
    train_count = sum(1 for r in labeled if r["split"] == "train")
    val_count = sum(1 for r in labeled if r["split"] == "val")
    print(f"Assigned splits: {train_count} train, {val_count} val")
    for cls in sorted(by_class.keys()):
        cls_rows = [r for r in labeled if r["class"] == cls]
        t = sum(1 for r in cls_rows if r["split"] == "train")
        v = sum(1 for r in cls_rows if r["split"] == "val")
        print(f"  {cls}: {t} train, {v} val")


def main():
    parser = argparse.ArgumentParser(description="Interactive tracklet labeling tool")
    parser.add_argument("--video", help="Path to original video")
    parser.add_argument("--tracklets-dir", default="tracklets/",
                        help="Base tracklets directory (default: tracklets/)")
    parser.add_argument("--labels-csv", default="labels.csv",
                        help="Path to labels CSV (default: labels.csv)")
    parser.add_argument("--min-track-length", type=int, default=0,
                        help="Only show tracks with at least this many frames (default: 0, no filter)")
    parser.add_argument("--assign-splits", action="store_true",
                        help="Assign stratified train/val splits to existing labels")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train split ratio (default: 0.8)")
    args = parser.parse_args()

    if args.assign_splits:
        assign_splits(args.labels_csv, args.train_ratio)
    else:
        if not args.video:
            parser.error("--video is required for interactive labeling")
        label_interactive(args.video, args.tracklets_dir, args.labels_csv,
                          min_track_length=args.min_track_length)


if __name__ == "__main__":
    main()

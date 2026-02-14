"""Analyze detection gaps in OC-SORT tracklets.

Parses tracklet_data.csv files, identifies frames where Detectron2 failed to
detect a tracked object, prints a summary report, and extracts the gap frames
from the original video with bounding box annotations for visual inspection.
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
from tqdm import tqdm

from material_classifier.inference import discover_videos


# Minimum track length (in frames) to include in analysis.
# Short tracks are typically spurious and not worth analyzing.
MIN_TRACK_LENGTH = 50

# Minimum gap length (in frames) to extract a frame for.
# Short gaps are detection flicker — the midpoint looks identical to detected frames.
MIN_GAP_LENGTH = 10

# Minimum distance (in seconds) between extracted frames.
# At 30fps nearby frames are visually identical — no value in annotating both.
MIN_FRAME_DISTANCE_SEC = 5.0


def analyze_tracklet_gaps(csv_path, min_track_length=MIN_TRACK_LENGTH,
                          track_ids=None):
    """Parse tracklet_data.csv and find per-track frame gaps.

    Args:
        csv_path: Path to tracklet_data.csv.
        min_track_length: Minimum track length to include in analysis.
        track_ids: If provided, only analyze these track IDs.

    Returns:
        dict: {track_id: {
            'start_frame': int, 'end_frame': int, 'total_frames': int,
            'present_frames': int,
            'gaps': [(gap_start, gap_end), ...],
            'total_missing': int,
            'frame_bboxes': {frame_idx: (x1, y1, x2, y2)},
        }}
    """
    # Read CSV and group by track_id
    track_frames = defaultdict(set)
    track_bboxes = defaultdict(dict)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            frame = int(row["frame"])
            track_frames[track_id].add(frame)
            track_bboxes[track_id][frame] = (
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            )

    gaps_by_track = {}

    ids_to_analyze = sorted(track_frames.keys())
    if track_ids is not None:
        ids_to_analyze = [t for t in ids_to_analyze if t in set(track_ids)]

    for track_id in ids_to_analyze:
        frames = sorted(track_frames[track_id])
        start_frame = frames[0]
        end_frame = frames[-1]
        total_frames = end_frame - start_frame + 1

        if total_frames < min_track_length:
            continue

        # Find missing frames within the track's range
        present = set(frames)
        all_in_range = set(range(start_frame, end_frame + 1))
        missing = sorted(all_in_range - present)

        # Group consecutive missing frames into gaps
        gaps = []
        if missing:
            gap_start = missing[0]
            prev = missing[0]
            for m in missing[1:]:
                if m == prev + 1:
                    prev = m
                else:
                    gaps.append((gap_start, prev))
                    gap_start = m
                    prev = m
            gaps.append((gap_start, prev))

        gaps_by_track[track_id] = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "present_frames": len(frames),
            "gaps": gaps,
            "total_missing": len(missing),
            "frame_bboxes": track_bboxes[track_id],
        }

    return gaps_by_track


def extract_gap_frames(video_path, gaps_by_track, tracklet_dir,
                       min_gap_length=MIN_GAP_LENGTH,
                       min_distance_sec=MIN_FRAME_DISTANCE_SEC):
    """Extract midpoint frames from gaps where detection failed.

    For each gap longer than min_gap_length, extracts the single midpoint
    frame — the frame furthest from successful detections on either side,
    and therefore most valuable for Detectron2 retraining. Frames are
    spaced at least min_distance_sec apart to avoid near-duplicates.

    Args:
        video_path: Path to the original video file.
        gaps_by_track: Output from analyze_tracklet_gaps().
        tracklet_dir: Directory for this video's tracklet outputs.
        min_gap_length: Only extract from gaps with at least this many frames.
        min_distance_sec: Minimum seconds between extracted frames.
    """
    # Get video fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_distance_frames = int(min_distance_sec * fps)

    # Collect midpoint frame index from each qualifying gap, with gap length
    # for prioritization (longer gaps = more interesting detection failures)
    candidates = []
    for track_id, info in gaps_by_track.items():
        for gap_start, gap_end in info["gaps"]:
            gap_length = gap_end - gap_start + 1
            if gap_length >= min_gap_length:
                midpoint = (gap_start + gap_end) // 2
                candidates.append((midpoint, gap_length))

    if not candidates:
        print("  No gap frames to extract.")
        cap.release()
        return

    # Sort by gap length descending — keep the most valuable frames first,
    # then greedily filter out frames that are too close to an already-kept frame
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept = []
    for midpoint, _ in candidates:
        if all(abs(midpoint - k) >= min_distance_frames for k in kept):
            kept.append(midpoint)
    kept.sort()

    if not kept:
        print("  No gap frames to extract after distance filtering.")
        cap.release()
        return

    gap_frames_dir = os.path.join(tracklet_dir, "gap_frames")
    os.makedirs(gap_frames_dir, exist_ok=True)

    saved = 0

    for frame_idx in tqdm(kept, desc="  Extracting gap frames", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        out_path = os.path.join(gap_frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    print(f"  Saved {saved} gap frame images to {gap_frames_dir}")



def save_gap_csv(tracklet_dir, gaps_by_track):
    """Save gap analysis results to CSV.

    Args:
        tracklet_dir: Directory for this video's tracklet outputs.
        gaps_by_track: Output from analyze_tracklet_gaps().
    """
    csv_path = os.path.join(tracklet_dir, "gap_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "gap_start", "gap_end", "gap_length"])
        for track_id in sorted(gaps_by_track.keys()):
            for gap_start, gap_end in gaps_by_track[track_id]["gaps"]:
                writer.writerow([track_id, gap_start, gap_end, gap_end - gap_start + 1])
    print(f"  Saved gap analysis to {csv_path}")


def print_report(video_stem, gaps_by_track):
    """Print formatted gap analysis report to terminal."""
    print(f"\n=== {video_stem} ===")

    if not gaps_by_track:
        print("  No tracks with sufficient length found.")
        return

    total_gaps = 0
    total_missing = 0
    total_tracked = 0
    tracks_with_gaps = 0

    for track_id in sorted(gaps_by_track.keys()):
        info = gaps_by_track[track_id]
        n_gaps = len(info["gaps"])
        total_gaps += n_gaps
        total_missing += info["total_missing"]
        total_tracked += info["total_frames"]
        if n_gaps > 0:
            tracks_with_gaps += 1

        gap_str = f"{n_gaps} gap{'s' if n_gaps != 1 else ''}"
        missing_str = f"{info['total_missing']} missing frame{'s' if info['total_missing'] != 1 else ''}"
        print(f"Track {track_id:3d}: {info['total_frames']} frames, {gap_str} ({missing_str})")

        for gap_start, gap_end in info["gaps"]:
            gap_len = gap_end - gap_start + 1
            print(f"  Gap: frames {gap_start}-{gap_end} ({gap_len} frame{'s' if gap_len != 1 else ''})")

    n_tracks = len(gaps_by_track)
    pct = (total_missing / total_tracked * 100) if total_tracked > 0 else 0
    print(f"Summary: {n_tracks} track{'s' if n_tracks != 1 else ''}, "
          f"{tracks_with_gaps} with gaps, "
          f"{total_gaps} gap{'s' if total_gaps != 1 else ''} total, "
          f"{total_missing} missing frames ({pct:.1f}% of tracked frames)")


def process_video(video_path, tracklets_dir, min_track_length=MIN_TRACK_LENGTH,
                   min_gap_length=MIN_GAP_LENGTH, min_distance_sec=MIN_FRAME_DISTANCE_SEC,
                   track_ids=None, extract=True):
    """Run full gap analysis for a single video.

    Args:
        video_path: Path to the video file.
        tracklets_dir: Base tracklets directory (e.g., "tracklets/").
        min_track_length: Minimum track length to include in analysis.
        min_gap_length: Only extract frames from gaps with at least this many frames.
        extract: Whether to extract gap frames from the video.
    """
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    tracklet_dir = os.path.join(tracklets_dir, video_stem)
    csv_path = os.path.join(tracklet_dir, "tracklet_data.csv")

    if not os.path.exists(csv_path):
        print(f"\n=== {video_stem} ===")
        print(f"  No tracklet_data.csv found at {csv_path}. Skipping.")
        return

    gaps_by_track = analyze_tracklet_gaps(csv_path, min_track_length, track_ids)
    print_report(video_stem, gaps_by_track)
    save_gap_csv(tracklet_dir, gaps_by_track)
    if extract:
        extract_gap_frames(video_path, gaps_by_track, tracklet_dir,
                           min_gap_length, min_distance_sec)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze detection gaps in OC-SORT tracklets"
    )
    parser.add_argument("--video", default=None, help="Path to a single video")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing experiment folders (4-digit names)")
    parser.add_argument("--tracklets-dir", default="tracklets/",
                        help="Base directory for tracklet outputs")
    parser.add_argument("--min-track-length", type=int, default=MIN_TRACK_LENGTH,
                        help=f"Minimum track length to analyze (default: {MIN_TRACK_LENGTH})")
    parser.add_argument("--min-gap-length", type=int, default=MIN_GAP_LENGTH,
                        help=f"Only extract from gaps with at least this many frames (default: {MIN_GAP_LENGTH})")
    parser.add_argument("--min-distance", type=float, default=MIN_FRAME_DISTANCE_SEC,
                        help=f"Minimum seconds between extracted frames (default: {MIN_FRAME_DISTANCE_SEC})")
    parser.add_argument("--tracks", type=int, nargs="+", default=None,
                        help="Only analyze these track IDs (e.g., --tracks 28 45 102)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extracting gap frames (report only)")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.error("Provide either --video or --video-dir")

    extract = not args.no_extract

    if args.video_dir:
        videos = discover_videos(args.video_dir)
        if not videos:
            print("No videos found. Exiting.")
            return
        for i, video_path in enumerate(videos, 1):
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            print(f"\n[{i}/{len(videos)}] Analyzing {video_stem}...")
            process_video(video_path, args.tracklets_dir,
                          args.min_track_length, args.min_gap_length,
                          args.min_distance, args.tracks, extract)
    else:
        video_path = os.path.expanduser(args.video)
        process_video(video_path, args.tracklets_dir,
                      args.min_track_length, args.min_gap_length,
                      args.min_distance, args.tracks, extract)


if __name__ == "__main__":
    main()

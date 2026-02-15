#!/usr/bin/env python3
"""
Extract thermal intensity time-series for labeled tracklets.

For each labeled tracklet, this script:
1. Loads the binary segmentation mask (RGB frame space)
2. Warps the mask to thermal frame space via a pre-computed homography H
3. Looks up the corresponding thermal frame via the frame matching CSV
4. Applies the warped mask to the thermal frame
5. Computes the mean pixel intensity of the masked region
6. Assembles the per-frame intensities into a time-series

Inputs:
    - labels.csv: Labeled tracklets (video, track_id, split, class)
    - tracklets/: Per-video tracklet data and binary masks
    - ~/Downloads/{exp_id}/H_{exp_id}.joblib: Homography matrices (RGB → thermal)
    - ~/Downloads/{exp_id}/frame_matches_*.csv: RGB-thermal frame correspondences
    - ~/Downloads/{exp_id}/thermal_frames/: Thermal frame images

Outputs:
    - data/final_thermal_train_dataset.csv
    - data/final_thermal_validation_dataset.csv
    - data/final_thermal_test_dataset.csv

Usage:
    python extract_thermal_data.py
    python extract_thermal_data.py --labels-csv ../../labels.csv --downloads-dir ~/Downloads
    python extract_thermal_data.py --dry-run
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_frame_matching(csv_path: str) -> dict:
    """
    Load RGB-to-thermal frame mapping from a frame matching CSV.

    The CSV has comment lines starting with '#', then a header row with
    'query_frame' and 'matched_frame' columns.

    Returns:
        Dict mapping RGB frame index (int) → thermal frame index (int).
    """
    # Skip comment lines, find the header
    header_line = None
    data_start = 0
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                header_line = stripped
                data_start = i + 1
                break

    if header_line is None:
        raise ValueError(f"No header found in {csv_path}")

    cols = [c.strip() for c in header_line.split(',')]
    df = pd.read_csv(csv_path, skiprows=data_start, header=None, names=cols,
                     comment='#')
    df['query_frame'] = pd.to_numeric(df['query_frame'], errors='coerce').astype('Int64')
    df['matched_frame'] = pd.to_numeric(df['matched_frame'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['query_frame', 'matched_frame'])

    return dict(zip(df['query_frame'].astype(int), df['matched_frame'].astype(int)))


def warp_mask_to_thermal(mask: np.ndarray, H: np.ndarray,
                         thermal_shape: tuple) -> np.ndarray:
    """
    Warp a binary mask from RGB frame space to thermal frame space.

    Args:
        mask: Binary mask in RGB space (H_rgb, W_rgb), values 0 or 255.
        H: 3x3 homography matrix mapping RGB coordinates to thermal coordinates.
        thermal_shape: (height, width) of the thermal frame.

    Returns:
        Binary mask in thermal space (H_thermal, W_thermal).
    """
    warped = cv2.warpPerspective(
        mask, H,
        (thermal_shape[1], thermal_shape[0]),  # (width, height)
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    return warped


def extract_tracklet_thermal_series(
    tracklet_frames: pd.DataFrame,
    mask_dir: str,
    H: np.ndarray,
    frame_map: dict,
    thermal_frames_dir: str,
    thermal_shape: tuple = (164, 270),
) -> list:
    """
    Extract the mean thermal intensity time-series for a single tracklet.

    Args:
        tracklet_frames: DataFrame rows for this tracklet from tracklet_data.csv,
                         sorted by frame number.
        mask_dir: Directory containing binary mask PNGs.
        H: 3x3 homography matrix (RGB → thermal).
        frame_map: Dict mapping RGB frame index → thermal frame index.
        thermal_frames_dir: Directory containing thermal frame PNGs.
        thermal_shape: Expected (height, width) of thermal frames.

    Returns:
        List of mean thermal intensity values (one per frame with valid data).
    """
    intensities = []

    for _, row in tracklet_frames.iterrows():
        rgb_frame = int(row['frame'])
        track_id = int(row['track_id'])

        # Look up corresponding thermal frame
        thermal_frame_idx = frame_map.get(rgb_frame)
        if thermal_frame_idx is None:
            continue

        # Load thermal frame
        thermal_path = os.path.join(
            thermal_frames_dir, f"FLIR_frame_{thermal_frame_idx:06d}.png"
        )
        if not os.path.exists(thermal_path):
            continue

        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_img is None:
            continue

        # Load binary mask (RGB space)
        mask_path = os.path.join(
            mask_dir, f"frame_{rgb_frame:06d}_track_{track_id}.png"
        )
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Warp mask to thermal space
        warped_mask = warp_mask_to_thermal(mask, H, thermal_img.shape[:2])

        # Compute mean intensity of masked region
        pixels = thermal_img[warped_mask > 0]
        if pixels.size == 0:
            continue

        intensities.append(float(np.mean(pixels)))

    return intensities


def main():
    parser = argparse.ArgumentParser(
        description='Extract thermal intensity time-series for labeled tracklets'
    )
    parser.add_argument(
        '--labels-csv', type=str,
        default=str(Path(__file__).resolve().parent.parent / 'labels.csv'),
        help='Path to labels.csv'
    )
    parser.add_argument(
        '--tracklets-dir', type=str,
        default=str(Path(__file__).resolve().parent.parent / 'tracklets'),
        help='Base tracklets directory'
    )
    parser.add_argument(
        '--downloads-dir', type=str,
        default=os.path.expanduser('~/Downloads'),
        help='Base directory containing experiment folders'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(Path(__file__).resolve().parent / 'data'),
        help='Output directory for thermal dataset CSVs'
    )
    parser.add_argument(
        '--min-frames', type=int, default=10,
        help='Minimum number of valid thermal frames per tracklet'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Check data availability without extracting'
    )
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    tracklets_dir = Path(args.tracklets_dir)
    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)

    # Load labels
    df_labels = pd.read_csv(labels_csv)
    print(f"Loaded {len(df_labels)} labeled tracklets from {labels_csv}")
    print(f"Splits: {dict(df_labels['split'].value_counts())}")
    print(f"Classes: {dict(df_labels['class'].value_counts())}")

    # Group by video (experiment)
    videos = df_labels['video'].unique()
    print(f"\nProcessing {len(videos)} experiments...")

    # Pre-load per-experiment resources
    experiment_resources = {}
    missing_resources = []

    for video in sorted(videos):
        # Extract experiment ID from video filename: IMG_0797_synched_cropped.mp4 → 0797
        exp_id = video.split('_')[1]
        exp_dir = downloads_dir / exp_id

        # Homography
        h_path = exp_dir / f"H_{exp_id}.joblib"
        if not h_path.exists():
            missing_resources.append(f"{exp_id}: H matrix not found at {h_path}")
            continue

        # Frame matching CSV
        fm_files = sorted(glob.glob(str(exp_dir / "frame_matches_*.csv")))
        if not fm_files:
            missing_resources.append(f"{exp_id}: No frame matching CSV found")
            continue

        # Thermal frames directory
        thermal_dir = exp_dir / "thermal_frames"
        if not thermal_dir.is_dir():
            missing_resources.append(f"{exp_id}: thermal_frames/ not found")
            continue

        # Tracklet data
        video_stem = video.replace('.mp4', '')
        tracklet_csv = tracklets_dir / video_stem / "tracklet_data.csv"
        mask_dir = tracklets_dir / video_stem / "masks"
        if not tracklet_csv.exists():
            missing_resources.append(f"{exp_id}: tracklet_data.csv not found")
            continue
        if not mask_dir.is_dir():
            missing_resources.append(f"{exp_id}: masks/ not found")
            continue

        experiment_resources[video] = {
            'exp_id': exp_id,
            'h_path': str(h_path),
            'fm_path': fm_files[0],  # Use the first (most recent) frame matching
            'thermal_dir': str(thermal_dir),
            'tracklet_csv': str(tracklet_csv),
            'mask_dir': str(mask_dir),
        }

    if missing_resources:
        print(f"\nWarning: {len(missing_resources)} missing resources:")
        for msg in missing_resources:
            print(f"  - {msg}")

    print(f"\nReady to process {len(experiment_resources)}/{len(videos)} experiments")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without extraction.")
        return

    # Extract thermal time-series for each tracklet
    results = []  # (class, uniqueID, track_length, intensities_str)
    skipped = 0

    for video in tqdm(sorted(experiment_resources.keys()), desc="Experiments"):
        res = experiment_resources[video]
        exp_id = res['exp_id']

        # Load experiment resources
        H = joblib.load(res['h_path'])
        frame_map = load_frame_matching(res['fm_path'])
        tracklet_data = pd.read_csv(res['tracklet_csv'])
        tracklet_data['frame'] = tracklet_data['frame'].astype(int)
        tracklet_data['track_id'] = tracklet_data['track_id'].astype(int)

        # Get labeled tracklets for this video
        video_labels = df_labels[df_labels['video'] == video]

        for _, label_row in video_labels.iterrows():
            track_id = int(label_row['track_id'])
            material_class = label_row['class']
            split = label_row['split']

            # Get frames for this tracklet
            track_frames = tracklet_data[tracklet_data['track_id'] == track_id]
            track_frames = track_frames.sort_values('frame')

            if len(track_frames) == 0:
                skipped += 1
                continue

            # Extract thermal time-series
            intensities = extract_tracklet_thermal_series(
                tracklet_frames=track_frames,
                mask_dir=res['mask_dir'],
                H=H,
                frame_map=frame_map,
                thermal_frames_dir=res['thermal_dir'],
            )

            if len(intensities) < args.min_frames:
                print(f"  Warning: {exp_id}_{track_id} has only "
                      f"{len(intensities)} thermal frames (min={args.min_frames}), skipping")
                skipped += 1
                continue

            unique_id = f"{exp_id}_{track_id}"
            intensities_str = ' '.join(f'{v:.4f}' for v in intensities)
            results.append({
                'class': material_class,
                'uniqueID': unique_id,
                'track_length': len(intensities),
                'mean_thermal_intensity_array': intensities_str,
                'split': split,
            })

    print(f"\nExtracted {len(results)} tracklets, skipped {skipped}")

    if not results:
        print("Error: No results extracted!")
        sys.exit(1)

    df_results = pd.DataFrame(results)

    # Summary statistics
    print(f"\nResults summary:")
    print(f"  Total tracklets: {len(df_results)}")
    print(f"  Track lengths: min={df_results['track_length'].min()}, "
          f"median={df_results['track_length'].median():.0f}, "
          f"max={df_results['track_length'].max()}")
    print(f"\n  By split:")
    for split_name in ['train', 'val', 'test']:
        split_df = df_results[df_results['split'] == split_name]
        if len(split_df) > 0:
            print(f"    {split_name}: {len(split_df)} tracklets")
            print(f"      Classes: {dict(split_df['class'].value_counts())}")

    # Save per-split CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cols = ['class', 'uniqueID', 'track_length', 'mean_thermal_intensity_array']

    split_files = {
        'train': 'final_thermal_train_dataset.csv',
        'val': 'final_thermal_validation_dataset.csv',
        'test': 'final_thermal_test_dataset.csv',
    }

    for split_name, filename in split_files.items():
        split_df = df_results[df_results['split'] == split_name][output_cols]
        out_path = output_dir / filename
        split_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(split_df)} {split_name} tracklets to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

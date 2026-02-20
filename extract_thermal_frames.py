#!/usr/bin/env python3
"""
Extracts specific frames from a thermal video corresponding to RGB frames,
based on a mapping CSV file. Deduces all necessary input/output paths.
(v10: More robust CSV parsing)
"""
import os
import cv2
import pandas as pd
import argparse
import glob
from tqdm import tqdm

def get_required_frames_from_csv(csv_path):
    """
    Robustly reads the mapping CSV by dynamically finding the header and data
    start, then using the correct delimiter.
    Returns a dictionary of {rgb_frame: thermal_frame}.
    """
    try:
        header_line = None
        data_start_line = 0
        
        # Find the header line and the line number where the actual data starts
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                # A valid line is not empty and does not start with a comment hash
                if line.strip() and not line.strip().startswith('#'):
                    header_line = line.strip()
                    data_start_line = i + 1
                    break
        
        if not header_line:
            raise ValueError("Could not find a header line (a line not starting with '#') in the CSV.")

        # Automatically detect the delimiter based on the header
        if ',' in header_line:
            delimiter = ','
            col_names = [col.strip() for col in header_line.split(',')]
        else:
            delimiter = r'\s+' # Fallback to one-or-more-spaces
            col_names = header_line.split()

        # Read the data using the dynamically found info
        df = pd.read_csv(
            csv_path,
            header=None,
            names=col_names,
            skiprows=data_start_line,
            sep=delimiter,
            engine='python',
            # Ignore comment lines that might be interspersed in the data
            comment='#'
        )

        rgb_col = 'query_frame'
        thermal_col = 'matched_frame'

        if rgb_col not in df.columns or thermal_col not in df.columns:
            raise ValueError(f"CSV must contain '{rgb_col}' and '{thermal_col}' columns after parsing. Found columns: {df.columns.tolist()}")

        # Ensure data is numeric and clean
        df[rgb_col] = pd.to_numeric(df[rgb_col], errors='coerce')
        df[thermal_col] = pd.to_numeric(df[thermal_col], errors='coerce')
        df.dropna(subset=[rgb_col, thermal_col], inplace=True)
        df = df.astype({rgb_col: int, thermal_col: int})

        mapping = pd.Series(df[thermal_col].values, index=df[rgb_col]).to_dict()
        
        if not mapping:
             print("Warning: Parsed the CSV, but found 0 valid frame mapping rows. The file might be empty or incorrectly formatted.")

        return mapping

    except (FileNotFoundError, IndexError, ValueError, Exception) as e:
        print(f"FATAL ERROR: Could not read or parse the mapping CSV '{csv_path}'. Reason: {e}")
        return None


def extract_frames(video_path, frame_indices, output_dir):
    """Extracts a specific set of frame indices from a video and saves them."""
    if not os.path.exists(video_path):
        print(f"Error: Thermal video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    sorted_indices = sorted(list(set(frame_indices)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=len(sorted_indices), desc="Extracting Thermal Frames")

    for target_idx in sorted_indices:
        if target_idx >= frame_count:
            tqdm.write(f"Warning: Frame index {target_idx} is out of bounds for video with {frame_count} frames. Skipping.")
            pbar.update(1)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()

        if ret:
            output_filename = f"FLIR_frame_{target_idx:06d}.png"
            cv2.imwrite(os.path.join(output_dir, output_filename), frame)
        else:
            tqdm.write(f"Warning: Could not read frame at index {target_idx}. It might be corrupt.")
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"✅ Finished extraction. Frames saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract thermal frames corresponding to RGB frames from a mapping CSV."
    )

    parser.add_argument('--base-path', type=str, required=True,
                        help='Base directory containing experiment folders.')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Output parent directory. Defaults to --base-path.')
    parser.add_argument('--experiments', type=str, required=True, nargs='+',
                        help='One or more experiment IDs (e.g., 0797 0798).')

    args = parser.parse_args()
    output_base = args.output_base if args.output_base else args.base_path

    for experiment_id in args.experiments:
        print(f"\n--- Processing Experiment: {experiment_id} ---")

        base_exp_path = os.path.join(os.path.expanduser(args.base_path), experiment_id)
        output_exp_path = os.path.join(os.path.expanduser(output_base), experiment_id)

        thermal_video_path = os.path.join(base_exp_path, f'FLIR{experiment_id}_synched_cropped.mp4')
        output_dir = os.path.join(output_exp_path, 'thermal_frames')

        # Find frame matching CSV (frame_matches_*.csv)
        csv_files = sorted(glob.glob(os.path.join(base_exp_path, 'frame_matches_*.csv')))
        if not csv_files:
            print(f"Error: No frame_matches_*.csv found in '{base_exp_path}'. Skipping.")
            continue
        mapping_csv_path = csv_files[0]

        print(f"  Mapping CSV:       {mapping_csv_path}")
        print(f"  Thermal Video:     {thermal_video_path}")
        print(f"  Output Directory:  {output_dir}")

        frame_mapping = get_required_frames_from_csv(mapping_csv_path)
        if frame_mapping is None:
            print("Skipping due to mapping CSV error.")
            continue

        unique_thermal_indices = list(set(frame_mapping.values()))
        print(f"  Mappings: {len(frame_mapping)} RGB frames -> {len(unique_thermal_indices)} unique thermal frames")

        extract_frames(thermal_video_path, unique_thermal_indices, output_dir)

"""Utilities for the thermal classification pipeline.

Handles RGB-to-thermal frame matching, homography warping, and grayscale conversion.
Ported from recycling_paper_elsevier_article_template/code/extract_thermal_data.py.
"""

import glob
import os

import cv2
import joblib
import numpy as np


def load_frame_matching(csv_path):
    """
    Load RGB-to-thermal frame mapping from a frame matching CSV.

    The CSV has comment lines starting with '#', then a header row with
    'query_frame' and 'matched_frame' columns.

    Returns:
        Dict mapping RGB frame index (int) -> thermal frame index (int).
    """
    frame_map = {}
    header = None
    qi, mi = None, None

    with open(csv_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in stripped.split(",")]
                qi = header.index("query_frame")
                mi = header.index("matched_frame")
                continue
            cols = stripped.split(",")
            try:
                frame_map[int(float(cols[qi]))] = int(float(cols[mi]))
            except (ValueError, IndexError):
                continue

    if header is None:
        raise ValueError(f"No header found in {csv_path}")

    return frame_map


def warp_mask_to_thermal(mask, H, thermal_shape):
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
        mask,
        H,
        (thermal_shape[1], thermal_shape[0]),  # (width, height)
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped


def get_experiment_id(video_name):
    """Extract experiment ID from video filename.

    "IMG_0798_synched_cropped.mp4" -> "0798"
    """
    return video_name.split("_")[1]


def find_frame_matching_csv(exp_dir):
    """Find the frame matching CSV in an experiment directory.

    Returns the first match or None.
    """
    matches = sorted(glob.glob(os.path.join(exp_dir, "frame_matches_*.csv")))
    return matches[0] if matches else None


def load_experiment_resources(exp_dir, exp_id):
    """Load all thermal resources for an experiment.

    Args:
        exp_dir: Path to the experiment directory (e.g., ~/Downloads/0798/).
        exp_id: Experiment ID string (e.g., "0798").

    Returns:
        Dict with 'H', 'frame_map', 'thermal_frames_dir' or None if missing.
    """
    h_path = os.path.join(exp_dir, f"H_{exp_id}.joblib")
    if not os.path.exists(h_path):
        return None

    fm_path = find_frame_matching_csv(exp_dir)
    if fm_path is None:
        return None

    thermal_dir = os.path.join(exp_dir, "thermal_frames")
    if not os.path.isdir(thermal_dir):
        return None

    H = joblib.load(h_path)
    frame_map = load_frame_matching(fm_path)

    return {
        "H": H,
        "frame_map": frame_map,
        "thermal_frames_dir": thermal_dir,
    }


def grayscale_to_3channel(img, colormap=None):
    """Convert a grayscale image to 3-channel.

    Args:
        img: (H, W) uint8 grayscale image.
        colormap: None for simple replication, or a string like "inferno"
                  to apply a matplotlib-style colormap via cv2.applyColorMap.

    Returns:
        (H, W, 3) uint8 image in RGB order.
    """
    if colormap is None:
        return np.stack([img, img, img], axis=-1)

    cmap_id = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }.get(colormap)

    if cmap_id is None:
        raise ValueError(f"Unknown colormap: {colormap}. Use one of: inferno, jet, hot, magma, plasma, viridis")

    bgr = cv2.applyColorMap(img, cmap_id)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

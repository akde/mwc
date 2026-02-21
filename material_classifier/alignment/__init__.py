"""Spatial alignment pipeline for RGB-thermal registration.

Provides homography computation and frame-to-frame temporal matching
between RGB and thermal video streams using SuperPoint-SuperGlue features.
"""

from material_classifier.alignment.matcher import SuperGlueMatcher
from material_classifier.alignment.homography import (
    compute_homography,
    calculate_homography,
    extract_frame,
    extract_eo_frame,
    extract_thermal_frame,
    crop_and_rotate_eo_frame,
    warp_image,
)
from material_classifier.alignment.frame_matching import match_video_frames
from material_classifier.alignment.experiment_metadata import (
    load_experiment_metadata,
    get_experiment_params,
)
from material_classifier.alignment.io import read_matching_csv, write_matching_csv

__all__ = [
    "SuperGlueMatcher",
    "compute_homography",
    "calculate_homography",
    "extract_frame",
    "extract_eo_frame",
    "extract_thermal_frame",
    "crop_and_rotate_eo_frame",
    "warp_image",
    "match_video_frames",
    "load_experiment_metadata",
    "get_experiment_params",
    "read_matching_csv",
    "write_matching_csv",
]

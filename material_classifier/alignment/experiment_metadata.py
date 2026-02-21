"""Experiment calibration metadata.

Loads per-experiment ROI coordinates, registration frame numbers, and HSV
thresholds from the calibration_metadata.yaml shipped with this package.
"""

import os
from pathlib import Path

import yaml


def load_experiment_metadata(config_path=None):
    """Load calibration metadata for all experiments.

    Args:
        config_path: Path to calibration_metadata.yaml.  If None, uses the
            default location at material_classifier/config/calibration_metadata.yaml.

    Returns:
        Dict mapping experiment ID (str) to a dict with keys:
            roi: dict with x, y, w, h
            registration: dict with eo_frame, th_frame
            experiment_range: dict with eo_start, eo_end, th_start
            lower_threshold: list of 3 ints (HSV lower bound)
    """
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent / "config" / "calibration_metadata.yaml"
        )
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Calibration metadata not found: {config_path}\n"
            f"Expected at: material_classifier/config/calibration_metadata.yaml"
        )

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    metadata = {}
    for exp in data["experiments"]:
        metadata[str(exp["id"])] = {
            "roi": exp["roi"],
            "registration": exp["registration"],
            "experiment_range": exp["experiment_range"],
            "lower_threshold": exp["lower_threshold"],
        }

    return metadata


def get_experiment_params(experiment_id, config_path=None):
    """Get calibration parameters for a single experiment.

    Args:
        experiment_id: Experiment ID string (e.g. "0798").
        config_path: Optional path to calibration_metadata.yaml.

    Returns:
        Dict with roi, registration, experiment_range, lower_threshold.

    Raises:
        KeyError: If experiment_id is not found in metadata.
    """
    metadata = load_experiment_metadata(config_path)
    exp_id = str(experiment_id)
    if exp_id not in metadata:
        available = sorted(metadata.keys())
        raise KeyError(
            f"Experiment '{exp_id}' not found. "
            f"Available: {', '.join(available)}"
        )
    return metadata[exp_id]

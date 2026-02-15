"""
Feature extraction for SVM baseline classifier.

Extracts 5 statistical features from thermal time-series:
1. Mean intensity (μ_I): Average thermal intensity
2. Median intensity (Ĩ): Median thermal intensity (robust to outliers)
3. Standard deviation (σ_I): Variability in thermal response
4. Rise time (t_r): Frames to reach 70% of peak intensity (thermal diffusivity)
5. Fall time (t_f): Frames from peak until below 70% (heat retention capacity)

Physical interpretation:
- Mean/median: Overall thermal emission level (material-dependent emissivity)
- Std: Temporal variability (heating/cooling dynamics)
- Rise time: How quickly material heats up (thermal diffusivity)
- Fall time: How long material retains heat (specific heat capacity, mass)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from .config import (
    CLASS_NAMES, CLASS_TO_IDX, SVM_FEATURES, RISE_FALL_THRESHOLD,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH
)


def calculate_mean_intensity(thermal_array: np.ndarray) -> float:
    """
    Calculate mean thermal intensity (μ_I).

    Physical meaning: Average thermal emission level, related to material emissivity
    and temperature during observation.
    """
    return float(np.mean(thermal_array))


def calculate_median_intensity(thermal_array: np.ndarray) -> float:
    """
    Calculate median thermal intensity (Ĩ).

    Physical meaning: Central tendency of thermal emission, robust to outliers
    (e.g., temporary occlusions or sensor noise).
    """
    return float(np.median(thermal_array))


def calculate_std_intensity(thermal_array: np.ndarray) -> float:
    """
    Calculate standard deviation of thermal intensity (σ_I).

    Physical meaning: Variability in thermal response over time.
    High std indicates significant heating/cooling dynamics.
    """
    return float(np.std(thermal_array))


def calculate_rise_time(
    thermal_array: np.ndarray,
    threshold: float = RISE_FALL_THRESHOLD
) -> float:
    """
    Calculate rise time (t_r): frames to reach threshold of peak intensity.

    Physical meaning: Thermal diffusivity - how quickly the material reaches
    near-peak temperature when exposed to heat.

    Args:
        thermal_array: 1D array of thermal intensities
        threshold: Fraction of peak intensity (default 0.7 = 70%)

    Returns:
        Rise time in frames (normalized by sequence length for comparability)
    """
    if len(thermal_array) < 2:
        return 0.0

    # Find peak and baseline
    peak_value = np.max(thermal_array)
    baseline_value = np.min(thermal_array)
    range_value = peak_value - baseline_value

    if range_value < 1e-8:  # No significant variation
        return 0.0

    # Threshold level
    target = baseline_value + threshold * range_value

    # Find first frame where we reach the threshold
    peak_idx = np.argmax(thermal_array)
    for i in range(peak_idx + 1):
        if thermal_array[i] >= target:
            # Normalize by sequence length for scale-invariance
            return float(i) / len(thermal_array)

    return float(peak_idx) / len(thermal_array)


def calculate_fall_time(
    thermal_array: np.ndarray,
    threshold: float = RISE_FALL_THRESHOLD
) -> float:
    """
    Calculate fall time (t_f): frames from peak until below threshold.

    Physical meaning: Heat retention capacity - how long the material maintains
    elevated temperature after reaching peak (related to specific heat and mass).

    Args:
        thermal_array: 1D array of thermal intensities
        threshold: Fraction of peak intensity (default 0.7 = 70%)

    Returns:
        Fall time in frames (normalized by sequence length for comparability)
    """
    if len(thermal_array) < 2:
        return 0.0

    # Find peak and baseline
    peak_value = np.max(thermal_array)
    baseline_value = np.min(thermal_array)
    range_value = peak_value - baseline_value

    if range_value < 1e-8:  # No significant variation
        return 0.0

    # Threshold level
    target = baseline_value + threshold * range_value

    # Find peak position
    peak_idx = np.argmax(thermal_array)

    # Find first frame after peak where we fall below threshold
    for i in range(peak_idx, len(thermal_array)):
        if thermal_array[i] < target:
            fall_time = i - peak_idx
            return float(fall_time) / len(thermal_array)

    # If never falls below threshold, return remaining sequence length
    return float(len(thermal_array) - peak_idx) / len(thermal_array)


def extract_features(thermal_array: np.ndarray) -> Dict[str, float]:
    """
    Extract all 5 statistical features from a thermal time-series.

    Args:
        thermal_array: 1D array of thermal intensities

    Returns:
        Dict with keys: 'mean', 'median', 'std', 'rise_time', 'fall_time'
    """
    return {
        'mean': calculate_mean_intensity(thermal_array),
        'median': calculate_median_intensity(thermal_array),
        'std': calculate_std_intensity(thermal_array),
        'rise_time': calculate_rise_time(thermal_array),
        'fall_time': calculate_fall_time(thermal_array)
    }


def load_and_extract_features(
    csv_path: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load thermal data from CSV and extract features for all samples.

    Args:
        csv_path: Path to CSV file
        verbose: Print progress

    Returns:
        Tuple of (X, y, unique_ids)
        - X: Feature matrix [n_samples, 5]
        - y: Labels [n_samples]
        - unique_ids: List of track identifiers
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"Loading and extracting features from: {csv_path}")
        print(f"  Samples: {len(df)}")

    features_list = []
    labels = []
    unique_ids = []

    for idx, row in df.iterrows():
        # Parse thermal intensity array
        thermal_array = np.array([float(v) for v in str(row['mean_thermal_intensity_array']).split()])

        # Extract features
        features = extract_features(thermal_array)
        features_list.append([features[f] for f in SVM_FEATURES])

        labels.append(CLASS_TO_IDX[row['class']])
        unique_ids.append(str(row['uniqueID']))

    X = np.array(features_list)
    y = np.array(labels)

    if verbose:
        print(f"  Feature shape: {X.shape}")
        print(f"  Features: {SVM_FEATURES}")
        print(f"  Classes: {dict(pd.Series([CLASS_NAMES[l] for l in labels]).value_counts())}")

    return X, y, unique_ids


def get_feature_dataframe(
    csv_path: str
) -> pd.DataFrame:
    """
    Load data and return features as a DataFrame with class labels.

    Useful for exploratory data analysis and visualization.
    """
    X, y, unique_ids = load_and_extract_features(csv_path, verbose=False)

    df = pd.DataFrame(X, columns=SVM_FEATURES)
    df['class'] = [CLASS_NAMES[label] for label in y]
    df['unique_id'] = unique_ids

    return df


if __name__ == '__main__':
    print("Testing feature extraction...")

    # Load training data
    X_train, y_train, ids_train = load_and_extract_features(str(TRAIN_DATA_PATH))
    X_val, y_val, ids_val = load_and_extract_features(str(VAL_DATA_PATH))
    X_test, y_test, ids_test = load_and_extract_features(str(TEST_DATA_PATH))

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Feature statistics
    print("\nFeature statistics (training set):")
    df = pd.DataFrame(X_train, columns=SVM_FEATURES)
    print(df.describe())

    # Per-class feature means
    print("\nPer-class feature means:")
    df['class'] = [CLASS_NAMES[l] for l in y_train]
    print(df.groupby('class').mean())

    print("\nFeature extraction test passed!")

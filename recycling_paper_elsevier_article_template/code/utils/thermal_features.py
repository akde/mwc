"""
Shared thermal feature extraction utilities.

This module contains reusable functions for extracting thermal features
from thermal timeseries data. These functions are used by:
- thermal_baseline_classifier.py (classification)
- thermal_eda_comparative.py (exploratory data analysis)
- thermal_eda.py (exploratory data analysis)

Features extracted:
1. Mean thermal intensity - Time-averaged thermal intensity
2. Median thermal intensity - Robust central tendency
3. Standard deviation - Thermal variability/stability
4. Rise time - Frames to reach 70% of max (heating dynamics)
5. Fall time - Frames from peak to drop below 70% (heat retention)

Created: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict


def load_and_parse_thermal_data(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load CSV and parse thermal arrays into numpy arrays.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: class, uniqueID, track_length, mean_thermal_intensity_array
    verbose : bool
        Whether to print loading information

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'thermal_values' column containing numpy arrays
    """
    if verbose:
        print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse the space-separated thermal values into numpy arrays
    df['thermal_values'] = df['mean_thermal_intensity_array'].apply(
        lambda x: np.array([float(v) for v in str(x).split()])
    )

    # Validate track_length matches array length
    df['actual_length'] = df['thermal_values'].apply(len)
    mismatches = df[df['track_length'] != df['actual_length']]
    if len(mismatches) > 0:
        print(f"  [WARNING] {len(mismatches)} tracks have length mismatches")

    if verbose:
        print(f"  Loaded {len(df)} tracks")
        if 'class' in df.columns:
            print(f"  Classes: {sorted(df['class'].unique().tolist())}")

    return df


def calculate_rise_time(thermal_values: np.ndarray) -> int:
    """
    Calculate rise time: frames until intensity reaches 70% of maximum.

    This measures the initial thermal response time - how quickly
    the object heats up when entering the heating zone.

    Parameters
    ----------
    thermal_values : np.ndarray
        Array of thermal intensity values over time

    Returns
    -------
    int
        Number of frames until 70% of maximum is reached
    """
    max_val = np.max(thermal_values)
    threshold = 0.7 * max_val

    for i, val in enumerate(thermal_values):
        if val >= threshold:
            return i
    return len(thermal_values)


def calculate_fall_time(thermal_values: np.ndarray) -> int:
    """
    Calculate fall time: frames from peak to decay below 70% of maximum.

    This measures heat retention ability - how long the object
    maintains its temperature after leaving the heating zone.

    Parameters
    ----------
    thermal_values : np.ndarray
        Array of thermal intensity values over time

    Returns
    -------
    int
        Number of frames from peak until dropping below 70% of maximum
    """
    max_val = np.max(thermal_values)
    peak_idx = np.argmax(thermal_values)
    threshold = 0.7 * max_val

    for i in range(peak_idx, len(thermal_values)):
        if thermal_values[i] < threshold:
            return i - peak_idx
    return len(thermal_values) - peak_idx


def extract_five_features(thermal_values: np.ndarray) -> Dict[str, float]:
    """
    Extract the 5 thermal features described in the paper (Section 6.2.4).

    Parameters
    ----------
    thermal_values : np.ndarray
        Array of thermal intensity values over time

    Returns
    -------
    dict
        Dictionary with keys: mean, median, std, rise_time, fall_time
    """
    return {
        'mean': np.mean(thermal_values),
        'median': np.median(thermal_values),
        'std': np.std(thermal_values),
        'rise_time': calculate_rise_time(thermal_values),
        'fall_time': calculate_fall_time(thermal_values)
    }


def extract_features_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract all 5 features for each track in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'thermal_values' column containing numpy arrays
    verbose : bool
        Whether to print extraction progress

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: class, uniqueID, track_length, mean, median, std, rise_time, fall_time
    """
    if verbose:
        print("\n" + "=" * 60)
        print("EXTRACTING 5 THERMAL FEATURES")
        print("=" * 60)

    features_list = []
    for idx, row in df.iterrows():
        thermal = row['thermal_values']
        features = extract_five_features(thermal)
        features['class'] = row['class']
        features['uniqueID'] = row['uniqueID']
        features['track_length'] = row['track_length']
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Reorder columns
    cols = ['class', 'uniqueID', 'track_length', 'mean', 'median', 'std', 'rise_time', 'fall_time']
    features_df = features_df[cols]

    if verbose:
        print(f"  Extracted features for {len(features_df)} tracks")
        print(f"  Features: mean, median, std, rise_time, fall_time")

    return features_df


def get_feature_names() -> list:
    """
    Return the list of feature names in order.

    Returns
    -------
    list
        List of feature names: ['mean', 'median', 'std', 'rise_time', 'fall_time']
    """
    return ['mean', 'median', 'std', 'rise_time', 'fall_time']

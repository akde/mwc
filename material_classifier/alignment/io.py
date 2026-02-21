"""CSV I/O for frame matching results.

Reads and writes the frame-matching CSV format consumed by the thermal
pipeline (material_classifier/thermal/utils.py:load_frame_matching).
"""

from datetime import datetime

import pandas as pd


def write_matching_csv(path, df, config=None, completed=False):
    """Write a frame-matching DataFrame with metadata comment headers.

    The output format is compatible with thermal/utils.py:load_frame_matching()
    which expects comment lines starting with '#' followed by a header row
    with 'query_frame' and 'matched_frame' columns.

    Args:
        path: Output CSV file path.
        df: DataFrame with at least 'query_frame' and 'matched_frame' columns.
        config: Optional dict of matching parameters to record in the header.
        completed: Whether the matching run finished (vs. was interrupted).
    """
    with open(path, "w") as f:
        status = "Completed" if completed else "Checkpoint"
        f.write(f"# Frame Matching Results - {status}\n")
        if config:
            for key, value in config.items():
                f.write(f"# {key}: {value}\n")
        f.write(f"# Frames Processed: {len(df)}\n")
        f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        df.to_csv(f, index=False)


def read_matching_csv(csv_path):
    """Read a frame-matching CSV, skipping comment headers.

    Returns a DataFrame with all available columns.  Core columns are
    'query_frame', 'matched_frame', 'search_center'.  Quality metrics
    (min/max/mean/std distance, gradient, snr) and window_length may
    also be present.

    Derived columns (frame_diff, search_diff, gradient, snr) are computed
    if missing but computable from other columns.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        pandas DataFrame with matching results.
    """
    df = pd.read_csv(csv_path, comment="#")

    # Compute derived metrics if missing
    if "frame_diff" not in df.columns and {"query_frame", "matched_frame"} <= set(df.columns):
        df["frame_diff"] = (df["query_frame"] - df["matched_frame"]).abs()

    if "search_diff" not in df.columns and {"matched_frame", "search_center"} <= set(df.columns):
        df["search_diff"] = (df["matched_frame"] - df["search_center"]).abs()

    if "gradient" not in df.columns and "mean_distance" in df.columns:
        df["gradient"] = df["mean_distance"].diff()

    if "snr" not in df.columns and {"mean_distance", "std_distance"} <= set(df.columns):
        df["snr"] = df.apply(
            lambda r: r["mean_distance"] / r["std_distance"]
            if r["std_distance"] > 0
            else float("nan"),
            axis=1,
        )

    return df

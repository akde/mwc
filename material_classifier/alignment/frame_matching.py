"""Frame-to-frame temporal matching between RGB and thermal videos.

Finds the best-matching thermal frame for each RGB frame by sliding a
search window through the thermal video and comparing SuperPoint-SuperGlue
feature distances.  Produces a CSV mapping RGB frame indices to thermal
frame indices, compatible with thermal/utils.py:load_frame_matching().
"""

import gc
import os
import signal
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from material_classifier.alignment.homography import (
    calculate_homography,
    extract_frame,
)
from material_classifier.alignment.io import read_matching_csv, write_matching_csv


# ---------------------------------------------------------------------------
# Low-level matching helpers
# ---------------------------------------------------------------------------

def calculate_match_distances(query_frame, target_frame, matcher, best_percent=50):
    """Calculate spatial distances between matched keypoints.

    Uses the homography pipeline to find matches, filters by ROI, keeps the
    best *best_percent*% of matches by descriptor distance, then computes
    the sum and mean of spatial keypoint distances.

    Args:
        query_frame: BGR query image.
        target_frame: BGR target image.
        matcher: A SuperGlueMatcher instance.
        best_percent: Percentage of best matches to use (e.g. 50).

    Returns:
        Tuple (sum_distance, avg_distance) or (None, None) if no matches.
    """
    try:
        H, valid_matches, kpts_src, kpts_dst = calculate_homography(
            query_frame, target_frame, matcher,
        )

        if valid_matches is None:
            return None, None

        # ROI filtering
        roi_matches = matcher.filter_matches_by_roi(kpts_src, kpts_dst, valid_matches)
        if not roi_matches:
            return None, None

        # Keep best N% by descriptor distance
        sorted_matches = sorted(roi_matches, key=lambda m: m.distance)
        n_keep = max(1, int(len(sorted_matches) * best_percent / 100))
        best = sorted_matches[:n_keep]

        # Compute spatial distances
        total = 0.0
        for m in best:
            src_pt = np.array(kpts_src[m.queryIdx].pt)
            dst_pt = np.array(kpts_dst[m.trainIdx].pt)
            total += np.linalg.norm(src_pt - dst_pt)

        avg = total / len(best)
        return total, avg

    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


def find_most_similar_frame(video_path, search_center, window_length,
                            query_frame, matcher, best_percent=50):
    """Find the most similar frame to *query_frame* within a search window.

    Args:
        video_path: Path to the target video.
        search_center: Center frame of the search window.
        window_length: Half-width of the search window.
        query_frame: Reference BGR frame to compare against.
        matcher: A SuperGlueMatcher instance.
        best_percent: Percentage of best matches for distance calculation.

    Returns:
        Tuple (best_frame_img, best_frame_num, frame_numbers, avg_distances).

    Raises:
        RuntimeError: If no valid matches are found in the window.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    start = max(0, search_center - window_length)
    end = search_center + window_length

    min_avg = float("inf")
    best_img = None
    best_num = None
    frame_nums = []
    avg_dists = []

    try:
        count = 0
        for fnum in tqdm(range(start, end + 1), leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, current = cap.read()

            if not ret:
                frame_nums.append(fnum)
                avg_dists.append(np.nan)
                continue

            try:
                _, avg = calculate_match_distances(
                    query_frame, current, matcher, best_percent,
                )
                frame_nums.append(fnum)

                if avg is None:
                    avg_dists.append(np.nan)
                else:
                    avg_dists.append(avg)
                    if avg < min_avg:
                        min_avg = avg
                        if best_img is not None:
                            del best_img
                        best_img = current.copy()
                        best_num = fnum
            except Exception:
                frame_nums.append(fnum)
                avg_dists.append(np.nan)
            finally:
                del current

            count += 1
            if count % 50 == 0:
                gc.collect()
    finally:
        cap.release()
        gc.collect()

    if best_img is None:
        raise RuntimeError("No valid matches found in the specified window")

    return (
        best_img,
        best_num,
        np.array(frame_nums, dtype=np.int32),
        np.array(avg_dists, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Matching config and helpers
# ---------------------------------------------------------------------------

@dataclass
class MatchingConfig:
    """Configuration for a frame-matching run."""
    query_video_path: str
    query_frame_number: int
    target_video_path: str
    search_center: int
    initial_window_length: int
    match_threshold: float
    stop_frame: Optional[int] = None
    best_percent: int = 50
    use_adaptive_window: bool = False
    save_interval: int = 50
    max_window_length: int = 200


class _SignalHandler:
    """Manages graceful termination via Ctrl-C."""

    def __init__(self):
        self.termination_requested = False
        self._original = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, sig, frame):
        if not self.termination_requested:
            print("\n\nGraceful termination requested. Saving progress...")
            self.termination_requested = True
        else:
            print("\nForced exit.")
            if self._original:
                self._original(sig, frame)
            else:
                sys.exit(1)

    def cleanup(self):
        signal.signal(signal.SIGINT, self._original)


class _MatchStatistics:
    """Calculate per-frame match quality metrics."""

    @staticmethod
    def calculate(distances, prev_mean=None):
        try:
            distances = np.asarray(distances, dtype=np.float32)
        except Exception:
            try:
                distances = np.array(distances.tolist(), dtype=np.float32)
            except Exception:
                return {k: np.nan for k in
                        ("min_distance", "max_distance", "mean_distance",
                         "std_distance", "gradient", "snr")}

        valid = distances[~np.isnan(distances)] if len(distances) > 0 else np.array([])

        stats = {
            "min_distance": np.min(valid) if len(valid) > 0 else np.nan,
            "max_distance": np.max(valid) if len(valid) > 0 else np.nan,
            "mean_distance": np.mean(valid) if len(valid) > 0 else np.nan,
            "std_distance": np.std(valid) if len(valid) > 0 else np.nan,
        }

        stats["gradient"] = np.nan
        if prev_mean is not None and not np.isnan(prev_mean):
            stats["gradient"] = stats["mean_distance"] - prev_mean

        stats["snr"] = np.nan
        if not np.isnan(stats["std_distance"]) and stats["std_distance"] > 0:
            stats["snr"] = stats["mean_distance"] / stats["std_distance"]

        return stats


class _AdaptiveWindow:
    """Adjust search window based on SNR / gradient thresholds."""

    @staticmethod
    def next_window(stats, current, initial, max_window=200):
        snr = stats.get("snr", 0)
        grad = abs(stats.get("gradient", 0)) if not np.isnan(stats.get("gradient", np.nan)) else 0

        if snr > 40 or grad > 30:
            return min(current * 8, max_window)
        if snr > 30 or grad > 20:
            return min(current * 4, max_window)
        if snr > 15 or grad > 10:
            return min(current * 2, max_window)
        return initial


# ---------------------------------------------------------------------------
# Main matching loop
# ---------------------------------------------------------------------------

class _FrameMatcher:
    """Drives the frame-by-frame matching loop."""

    def __init__(self, config, matcher):
        self.config = config
        self.matcher = matcher
        self.data = []

    def _nan_record(self, frame_num, search_center, window_length):
        return {
            "query_frame": frame_num,
            "matched_frame": None,
            "search_center": search_center,
            "min_distance": np.nan,
            "max_distance": np.nan,
            "mean_distance": np.nan,
            "std_distance": np.nan,
            "gradient": np.nan,
            "snr": np.nan,
            "frame_diff": np.nan,
            "search_diff": np.nan,
            "window_length": window_length,
        }

    def _find_best(self, query_frame, search_center, window_length):
        current = window_length
        while current <= self.config.max_window_length:
            try:
                with open(os.devnull, "w") as null:
                    with redirect_stdout(null), redirect_stderr(null):
                        _, best_num, _, dists = find_most_similar_frame(
                            self.config.target_video_path,
                            search_center,
                            current,
                            query_frame,
                            self.matcher,
                            self.config.best_percent,
                        )
                dists = np.asarray(dists, dtype=np.float32)
                return best_num, dists, current
            except Exception as e:
                print(f"\nNo match with window {current}: {e}")
                current *= 2

        return None, np.array([np.nan], dtype=np.float32), current

    def _process_frame(self, frame_num, query_frame, search_center, window_length):
        if query_frame is None or query_frame.size == 0:
            return self._nan_record(frame_num, search_center, window_length)
        if np.all(query_frame == 0) or np.all(query_frame == 255):
            return self._nan_record(frame_num, search_center, window_length)

        try:
            best_num, dists, used_window = self._find_best(
                query_frame, search_center, window_length,
            )
        except Exception:
            return self._nan_record(frame_num, search_center, window_length)

        prev_mean = self.data[-1].get("mean_distance") if self.data else None
        stats = _MatchStatistics.calculate(dists, prev_mean)

        return {
            "query_frame": frame_num,
            "matched_frame": best_num,
            "search_center": search_center,
            **stats,
            "frame_diff": abs(frame_num - best_num) if best_num else np.nan,
            "search_diff": abs(best_num - search_center) if best_num else np.nan,
            "window_length": used_window,
        }

    def _generate_paths(self):
        save_dir = os.path.dirname(self.config.query_video_path)
        qname = os.path.splitext(os.path.basename(self.config.query_video_path))[0]
        tname = os.path.splitext(os.path.basename(self.config.target_video_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        atag = "adaptive" if self.config.use_adaptive_window else "fixed"
        base = (
            f"frame_matches_{qname}_to_{tname}_"
            f"start{self.config.query_frame_number}_stop{self.config.stop_frame}_"
            f"center{self.config.search_center}_window{self.config.initial_window_length}_"
            f"{atag}_thresh{self.config.match_threshold}_"
            f"best{self.config.best_percent}_{ts}"
        )
        return {
            "save_dir": save_dir,
            "base": base,
            "csv": os.path.join(save_dir, f"{base}.csv"),
            "checkpoint": os.path.join(save_dir, f"{base}_{atag}_checkpoint.csv"),
            "temp": os.path.join(save_dir, f"{base}_{atag}_temp.csv"),
        }

    def _config_dict(self):
        return {
            "Query Video": self.config.query_video_path,
            "Target Video": self.config.target_video_path,
            "Start Frame": self.config.query_frame_number,
            "Search Center": self.config.search_center,
            "Initial Window Length": self.config.initial_window_length,
            "Match Threshold": self.config.match_threshold,
            "Stop Frame": self.config.stop_frame,
            "Best Percent": self.config.best_percent,
            "Adaptive Window": "Enabled" if self.config.use_adaptive_window else "Disabled",
        }

    def run(self):
        """Execute the matching loop.

        Returns:
            Tuple (DataFrame, csv_path).
        """
        paths = self._generate_paths()
        sig = _SignalHandler()

        # Determine stop frame from video length
        cap = cv2.VideoCapture(self.config.query_video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.config.stop_frame is None:
            self.config.stop_frame = total
        else:
            if self.config.stop_frame > total:
                print(f"Note: stop_frame ({self.config.stop_frame}) > video length ({total}), clamping")
            self.config.stop_frame = min(self.config.stop_frame, total)

        cur_frame = self.config.query_frame_number
        cur_center = self.config.search_center
        next_window = self.config.initial_window_length
        frames_since_save = 0

        n_total = self.config.stop_frame - self.config.query_frame_number
        pbar = tqdm(total=n_total, desc="Matching frames")

        try:
            while cur_frame < self.config.stop_frame and not sig.termination_requested:
                query = extract_frame(self.config.query_video_path, cur_frame)
                if query is None:
                    print(f"\nWarning: Could not extract frame {cur_frame}. Stopping.")
                    break

                window = next_window if self.config.use_adaptive_window else self.config.initial_window_length
                rec = self._process_frame(cur_frame, query, cur_center, window)
                self.data.append(rec)

                if rec["matched_frame"] is not None:
                    cur_center = rec["matched_frame"]
                else:
                    cur_center += 1

                if self.config.use_adaptive_window:
                    next_window = _AdaptiveWindow.next_window(
                        rec, window, self.config.initial_window_length,
                        self.config.max_window_length,
                    )

                cur_frame += 1
                frames_since_save += 1
                pbar.update(1)

                if rec["matched_frame"] is not None:
                    pbar.set_postfix_str(
                        f"Frame {rec['query_frame']} -> {rec['matched_frame']} "
                        f"(window: {rec['window_length']})"
                    )

                # Periodic checkpoint
                if frames_since_save >= self.config.save_interval:
                    df_tmp = pd.DataFrame(self.data)
                    write_matching_csv(paths["temp"], df_tmp, self._config_dict())
                    if os.path.exists(paths["temp"]):
                        if os.path.exists(paths["checkpoint"]):
                            os.remove(paths["checkpoint"])
                        os.rename(paths["temp"], paths["checkpoint"])
                        tqdm.write(f"Checkpoint saved ({len(self.data)} frames)")
                    frames_since_save = 0

                # Periodic GC
                if cur_frame % 50 == 0:
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            traceback.print_exc()
        finally:
            pbar.close()
            sig.cleanup()

        # Save final results
        df = pd.DataFrame(self.data)
        if len(df) == 0:
            return pd.DataFrame(), ""

        completed = not sig.termination_requested
        write_matching_csv(paths["temp"], df, self._config_dict(), completed)

        if sig.termination_requested:
            n_done = cur_frame - self.config.query_frame_number
            final_path = os.path.join(
                paths["save_dir"],
                f"{paths['base']}_partial_{n_done}frames.csv",
            )
        else:
            final_path = paths["csv"]

        if os.path.exists(paths["temp"]):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(paths["temp"], final_path)
            if os.path.exists(paths["checkpoint"]):
                os.remove(paths["checkpoint"])
            print(f"\nSaved {len(df)} frames to: {final_path}")

        return df, final_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_video_frames(query_video_path, query_frame_number, target_video_path,
                       search_center, initial_window_length, matcher,
                       match_threshold=0.2, stop_frame=None, best_percent=50,
                       use_adaptive_window=False, save_interval=50,
                       max_window_length=200):
    """Match frames between a query (warped RGB) and target (thermal) video.

    For each frame in the query video (starting at *query_frame_number*),
    searches a window of frames in the target video to find the best
    SuperGlue feature match.  Results are saved as a CSV compatible with
    the thermal pipeline's ``load_frame_matching()``.

    Args:
        query_video_path: Path to the query video (warped RGB).
        query_frame_number: Starting frame in the query video.
        target_video_path: Path to the target video (thermal).
        search_center: Initial center frame in the target video.
        initial_window_length: Initial half-width of the search window.
        matcher: A SuperGlueMatcher instance.
        match_threshold: SuperGlue threshold (informational, recorded in CSV).
        stop_frame: Frame to stop at (None = end of query video).
        best_percent: Percentage of best matches for distance calculation.
        use_adaptive_window: Dynamically adjust window based on SNR/gradient.
        save_interval: Checkpoint every N frames.
        max_window_length: Maximum search window half-width.

    Returns:
        Tuple (DataFrame, csv_path).
    """
    config = MatchingConfig(
        query_video_path=query_video_path,
        query_frame_number=query_frame_number,
        target_video_path=target_video_path,
        search_center=search_center,
        initial_window_length=initial_window_length,
        match_threshold=match_threshold,
        stop_frame=stop_frame,
        best_percent=best_percent,
        use_adaptive_window=use_adaptive_window,
        save_interval=save_interval,
        max_window_length=max_window_length,
    )

    fm = _FrameMatcher(config, matcher)
    return fm.run()


def resume_matching(csv_path, matcher, stop_frame=None, save_interval=50,
                    max_window_length=200):
    """Resume a previously interrupted matching run from a partial CSV.

    Reads the partial CSV, determines where it left off, and continues
    matching from the next frame.

    Args:
        csv_path: Path to a partial/checkpoint CSV from a previous run.
        matcher: A SuperGlueMatcher instance.
        stop_frame: New stop frame (None = end of query video).
        save_interval: Checkpoint every N frames.
        max_window_length: Maximum search window half-width.

    Returns:
        Tuple (DataFrame, csv_path) with the full (old + new) results.
    """
    # Read the partial CSV and extract parameters from the comment headers
    params = {}
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            if ":" in line:
                key, _, value = line[2:].partition(":")
                params[key.strip()] = value.strip()

    df_existing = read_matching_csv(csv_path)
    if len(df_existing) == 0:
        raise ValueError(f"No data found in {csv_path}")

    # Determine resume point
    last_query = int(df_existing["query_frame"].iloc[-1])
    last_matched = df_existing["matched_frame"].iloc[-1]
    resume_frame = last_query + 1
    resume_center = int(last_matched) if pd.notna(last_matched) else last_query + 1

    query_video = params.get("Query Video", "")
    target_video = params.get("Target Video", "")
    if not query_video or not target_video:
        raise ValueError(
            "Could not extract video paths from CSV header. "
            "Ensure the CSV was generated by this pipeline."
        )

    window = int(params.get("Initial Window Length", 20))
    threshold = float(params.get("Match Threshold", 0.2))
    best_pct = int(params.get("Best Percent", 50))
    adaptive = params.get("Adaptive Window", "Disabled") == "Enabled"

    print(f"Resuming from frame {resume_frame} (last matched: {last_matched})")
    print(f"  Query: {query_video}")
    print(f"  Target: {target_video}")

    df_new, new_path = match_video_frames(
        query_video_path=query_video,
        query_frame_number=resume_frame,
        target_video_path=target_video,
        search_center=resume_center,
        initial_window_length=window,
        matcher=matcher,
        match_threshold=threshold,
        stop_frame=stop_frame,
        best_percent=best_pct,
        use_adaptive_window=adaptive,
        save_interval=save_interval,
        max_window_length=max_window_length,
    )

    if len(df_new) > 0:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Overwrite the new CSV with the combined data
        config_dict = {
            "Query Video": query_video,
            "Target Video": target_video,
            "Start Frame": int(df_existing["query_frame"].iloc[0]),
            "Search Center": params.get("Search Center", resume_center),
            "Initial Window Length": window,
            "Match Threshold": threshold,
            "Stop Frame": stop_frame,
            "Best Percent": best_pct,
            "Adaptive Window": "Enabled" if adaptive else "Disabled",
        }
        write_matching_csv(new_path, df_combined, config_dict, completed=True)
        print(f"Combined {len(df_existing)} + {len(df_new)} = {len(df_combined)} frames -> {new_path}")
        return df_combined, new_path

    return df_existing, csv_path

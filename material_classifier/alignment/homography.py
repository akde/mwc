"""Homography computation pipeline for RGB-thermal alignment.

Computes the 3x3 homography matrix that maps RGB frame coordinates
to thermal frame coordinates for a single experiment, using SuperPoint-
SuperGlue feature matching with RANSAC.
"""

import gc
import os

import cv2
import joblib
import numpy as np

from material_classifier.alignment.experiment_metadata import get_experiment_params


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------

def extract_frame(video_path, frame_number):
    """Extract a single frame from a video file.

    Includes retry logic and memory management for robust extraction.

    Args:
        video_path: Path to the video file.
        frame_number: 0-based frame index to extract.

    Returns:
        BGR numpy array, or None on failure.
    """
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"Error: Failed to open video file: {e}")
        return None

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number >= total_frames:
            print(f"Error: Frame {frame_number} exceeds video length ({total_frames})")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        max_retries = 3
        for retry in range(max_retries):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_copy = frame.copy()
                del frame
                gc.collect()
                return frame_copy

            if retry < max_retries - 1:
                print(f"Warning: Could not read frame {frame_number}, retry {retry + 1}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        print(f"Error: Could not read frame {frame_number} after {max_retries} attempts")
        return None

    except Exception as e:
        print(f"Error extracting frame {frame_number}: {e}")
        return None
    finally:
        cap.release()
        gc.collect()


def crop_and_rotate_eo_frame(frame, x, y, w, h):
    """Apply the EO preprocessing pipeline: rotate 90 CW, crop ROI, rotate 180.

    This sequence matches the preprocessing applied to each RGB frame before
    warping it into thermal space.

    Args:
        frame: Input BGR frame from OpenCV.
        x, y, w, h: ROI coordinates in the rotated-90-CW frame.

    Returns:
        Processed BGR frame, or None on failure.
    """
    if frame is None:
        return None

    rotated_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    rh, rw = rotated_90.shape[:2]

    if y + h > rh or x + w > rw or y < 0 or x < 0:
        y1, y2 = max(0, y), min(rh, y + h)
        x1, x2 = max(0, x), min(rw, x + w)
        if y1 >= y2 or x1 >= x2:
            print(f"Error: Invalid crop region after clamping")
            return None
        cropped = rotated_90[y1:y2, x1:x2]
    else:
        cropped = rotated_90[y:y + h, x:x + w]

    if cropped.size == 0:
        return None

    return cv2.rotate(cropped, cv2.ROTATE_180)


def extract_eo_frame(frame_number, video_path, x, y, w, h):
    """Extract, crop, and rotate an EO (RGB) frame.

    Combines extract_frame + crop_and_rotate_eo_frame into a single call.

    Args:
        frame_number: Frame index in the video.
        video_path: Path to the EO video.
        x, y, w, h: ROI coordinates for crop_and_rotate_eo_frame.

    Returns:
        Processed BGR frame, or None on failure.
    """
    frame = extract_frame(video_path, frame_number)
    if frame is None:
        return None
    return crop_and_rotate_eo_frame(frame, x, y, w, h)


def extract_thermal_frame(frame_number, video_path, x1=0, y1=55, x2=270, y2=220):
    """Extract and crop a frame from a thermal video.

    Args:
        frame_number: Frame index in the video.
        video_path: Path to the thermal video.
        x1, y1, x2, y2: Crop coordinates (left, top, right, bottom).

    Returns:
        Cropped BGR frame, or None on failure.
    """
    frame = extract_frame(video_path, frame_number)
    if frame is None:
        return None
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def calculate_homography(im_src, im_dst, matcher, ransac_threshold=10.0,
                         ransac_max_iters=100000, ransac_confidence=0.95,
                         debug=False):
    """Calculate the homography matrix between two images via SuperGlue + RANSAC.

    Args:
        im_src: Source image (BGR or grayscale numpy array).
        im_dst: Destination image (BGR or grayscale numpy array).
        matcher: A SuperGlueMatcher instance.
        ransac_threshold: RANSAC reprojection threshold in pixels.
        ransac_max_iters: Maximum RANSAC iterations.
        ransac_confidence: RANSAC confidence level.
        debug: Print matching statistics.

    Returns:
        Tuple (H, valid_matches, kpts_src, kpts_dst) where H is the 3x3
        homography matrix (or None if matching failed).
    """
    if im_src is None or im_dst is None:
        print("Error: One or both input images are None")
        return None, None, None, None

    if im_src.size == 0 or im_dst.size == 0:
        print("Error: One or both input images are empty")
        return None, None, None, None

    # Convert to grayscale if needed
    try:
        gray_src = im_src if len(im_src.shape) == 2 else cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
        gray_dst = im_dst if len(im_dst.shape) == 2 else cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error converting images to grayscale: {e}")
        return None, None, None, None

    # Feature matching
    try:
        kpts_src, kpts_dst, matches = matcher.match(gray_src, gray_dst)
    except Exception as e:
        print(f"Error during SuperGlue matching: {e}")
        return None, None, None, None

    if debug:
        print(f"num kpts_src : {len(kpts_src) if kpts_src else 0}")
        print(f"num kpts_dst : {len(kpts_dst) if kpts_dst else 0}")

    if matches is None or kpts_src is None or kpts_dst is None:
        print("Error: Invalid matching results")
        return None, None, None, None

    if len(matches) < 4:
        print(f"Insufficient matches to compute homography. Found: {len(matches)}")
        return None, None, None, None

    # Extract valid point correspondences
    src_pts, dst_pts = [], []
    for m in matches:
        if m.queryIdx < len(kpts_src) and m.trainIdx < len(kpts_dst):
            src_pts.append(kpts_src[m.queryIdx].pt)
            dst_pts.append(kpts_dst[m.trainIdx].pt)

    if len(src_pts) < 4:
        print(f"Error: Not enough valid points after filtering: {len(src_pts)}")
        return None, None, None, None

    src_pts = np.float64(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float64(dst_pts).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=ransac_max_iters,
        confidence=ransac_confidence,
    )

    # Filter to RANSAC inliers
    valid_matches = np.array(matches)[np.all(mask > 0, axis=1)]
    valid_matches = sorted(valid_matches, key=lambda m: m.distance)

    if debug:
        print(f"Number of inlier matches: {len(valid_matches)}")

    return H, valid_matches, kpts_src, kpts_dst


def warp_image(src_img, target_dims_or_img, H):
    """Apply perspective warp using a homography matrix.

    Args:
        src_img: Source image to warp.
        target_dims_or_img: Either a (height, width) tuple or an image whose
            shape determines the output size.
        H: 3x3 homography matrix.

    Returns:
        Warped image with target dimensions.
    """
    if isinstance(target_dims_or_img, tuple) and len(target_dims_or_img) == 2:
        dsize = (target_dims_or_img[1], target_dims_or_img[0])
    else:
        h, w = target_dims_or_img.shape[:2]
        dsize = (w, h)

    return cv2.warpPerspective(src_img, np.float32(H), dsize, flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_homography(exp_id, base_dir, matcher, output_dir=None, debug=False):
    """Compute and save the homography matrix for a single experiment.

    Looks up experiment metadata, extracts registration frames from the EO and
    thermal videos, computes the homography, and saves it as a joblib file.

    Args:
        exp_id: Experiment ID string (e.g. "0798").
        base_dir: Base directory containing experiment folders (e.g. ~/Downloads).
        matcher: A SuperGlueMatcher instance.
        output_dir: Where to save the H_{exp_id}.joblib file.  Defaults to
            the experiment directory under base_dir.
        debug: Print matching statistics.

    Returns:
        Tuple (H, output_path) where H is the 3x3 homography matrix and
        output_path is the path to the saved joblib file.

    Raises:
        FileNotFoundError: If video files cannot be found.
        RuntimeError: If homography computation fails.
    """
    exp_id = str(exp_id)
    params = get_experiment_params(exp_id)
    roi = params["roi"]
    reg = params["registration"]

    exp_dir = os.path.join(os.path.expanduser(base_dir), exp_id)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Locate video files
    eo_video = os.path.join(exp_dir, f"IMG_{exp_id}_synched_cropped.mp4")
    th_video = os.path.join(exp_dir, f"FLIR{exp_id}_synched_cropped.mp4")

    if not os.path.exists(eo_video):
        raise FileNotFoundError(f"EO video not found: {eo_video}")
    if not os.path.exists(th_video):
        raise FileNotFoundError(f"Thermal video not found: {th_video}")

    # Extract registration frames
    print(f"[{exp_id}] Extracting EO registration frame {reg['eo_frame']}...")
    eo_frame = extract_eo_frame(
        reg["eo_frame"], eo_video,
        roi["x"], roi["y"], roi["w"], roi["h"],
    )
    if eo_frame is None:
        raise RuntimeError(f"Failed to extract EO registration frame for {exp_id}")

    print(f"[{exp_id}] Extracting thermal registration frame {reg['th_frame']}...")
    th_frame = extract_thermal_frame(reg["th_frame"], th_video)
    if th_frame is None:
        raise RuntimeError(f"Failed to extract thermal registration frame for {exp_id}")

    # Compute homography
    print(f"[{exp_id}] Computing homography ({len(eo_frame.shape)}D src -> {len(th_frame.shape)}D dst)...")
    H, valid_matches, kpts_src, kpts_dst = calculate_homography(
        eo_frame, th_frame, matcher, debug=debug,
    )
    if H is None:
        raise RuntimeError(f"Homography computation failed for {exp_id}")

    print(f"[{exp_id}] Homography computed with {len(valid_matches)} inlier matches")

    # Save
    if output_dir is None:
        output_dir = exp_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"H_{exp_id}.joblib")
    joblib.dump(H, output_path)
    print(f"[{exp_id}] Homography saved to {output_path}")

    return H, output_path

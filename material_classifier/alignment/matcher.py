"""SuperPoint-SuperGlue feature matcher wrapper.

Wraps the superpoint_superglue_deployment package as a class, replacing
the global-state pattern from the original phd_library.py.
"""

import cv2
import numpy as np

try:
    from superpoint_superglue_deployment.matcher import Matcher
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    Matcher = None


class SuperGlueMatcher:
    """SuperPoint-SuperGlue feature matcher.

    Wraps the Matcher class from superpoint_superglue_deployment, providing
    a stateful instance that can be passed to alignment functions instead of
    relying on a global variable.

    Args:
        match_threshold: SuperGlue match confidence threshold (default: 0.2).
        use_gpu: Whether to use GPU acceleration (default: True).
        keypoint_threshold: SuperPoint keypoint detection threshold (default: 0.0001).
    """

    def __init__(self, match_threshold=0.2, use_gpu=True, keypoint_threshold=0.0001):
        if not SUPERGLUE_AVAILABLE:
            raise ImportError(
                "SuperPoint-SuperGlue is not available. "
                "Install with: pip install superpoint-superglue-deployment"
            )

        self.match_threshold = match_threshold
        self.use_gpu = use_gpu
        self.keypoint_threshold = keypoint_threshold

        self._matcher = Matcher({
            "superpoint": {
                "input_shape": (-1, -1),
                "keypoint_threshold": keypoint_threshold,
            },
            "superglue": {
                "match_threshold": match_threshold,
            },
            "use_gpu": use_gpu,
        })

        print(
            f"SuperPoint-SuperGlue matcher initialized with "
            f"match_threshold={match_threshold}, use_gpu={use_gpu}"
        )

    def match(self, img_gray_src, img_gray_dst):
        """Run SuperPoint-SuperGlue matching on two grayscale images.

        Args:
            img_gray_src: Grayscale source image (H, W) uint8.
            img_gray_dst: Grayscale destination image (H, W) uint8.

        Returns:
            Tuple of (kpts_src, kpts_dst, matches) where kpts are lists of
            cv2.KeyPoint and matches is a list of cv2.DMatch.
        """
        kpts_src, kpts_dst, _, _, matches = self._matcher.match(
            img_gray_src, img_gray_dst
        )
        return kpts_src, kpts_dst, matches

    @staticmethod
    def filter_matches_by_roi(kpts_src, kpts_dst, matches, roi_points=None):
        """Filter matches to keep only those within an ROI polygon.

        Args:
            kpts_src: Source keypoints.
            kpts_dst: Destination keypoints.
            matches: List of cv2.DMatch objects.
            roi_points: Nx2 numpy array of polygon vertices. If None, uses the
                default conveyor-belt ROI from the original pipeline.

        Returns:
            List of cv2.DMatch objects that fall within the ROI.
        """
        if roi_points is None:
            roi_points = np.array([
                [18, 20],
                [185, 0],
                [269, 0],
                [265, 142],
                [16, 112],
            ], dtype=np.int32)

        def in_roi(point):
            return cv2.pointPolygonTest(roi_points, (float(point[0]), float(point[1])), False) >= 0

        filtered = []
        for m in matches:
            src_pt = kpts_src[m.queryIdx].pt
            dst_pt = kpts_dst[m.trainIdx].pt
            if in_roi(src_pt) and in_roi(dst_pt):
                filtered.append(m)
        return filtered

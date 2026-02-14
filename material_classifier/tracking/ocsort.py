"""
OC-SORT Object Tracking Implementation
---------------------------------------
Observation-Centric SORT (OC-SORT) tracker for multi-object tracking.

Based on the official implementation: https://github.com/noahcao/OC_SORT
Paper: "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking"
       https://arxiv.org/abs/2203.14360 (CVPR 2023)

Adapted from: /home/akde/phd/spatio_temporal_alignment/object_tracker_ocsort.py
Removed: parse_box, calculate_mask_centroid, calculate_mask_area, process_experiment,
         main, pandas/argparse/tqdm/cv2 imports.
Added: mask attribute on KalmanBoxTracker, carried through metadata.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_assignment


# ===============================================================================
# Utility Functions: Bounding Box Conversions
# ===============================================================================

def convert_bbox_to_z(bbox):
    """Convert bounding box [x1, y1, x2, y2] to measurement vector [x, y, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Convert state vector [x, y, s, r, ...] back to bounding box [x1, y1, x2, y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    if score is None:
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0,
            score
        ]).reshape((1, 5))


# ===============================================================================
# Utility Functions: Velocity and Direction
# ===============================================================================

def speed_direction(bbox1, bbox2):
    """Calculate normalized velocity direction between two bounding boxes."""
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0

    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_batch(dets, tracks):
    """Compute velocity direction from previous track observations to current detections."""
    CY1, CX1 = (tracks[:, 1] + tracks[:, 3]) / 2.0, (tracks[:, 0] + tracks[:, 2]) / 2.0
    CY2, CX2 = (dets[:, 1] + dets[:, 3]) / 2.0, (dets[:, 0] + dets[:, 2]) / 2.0

    dx = CX2[np.newaxis, :] - CX1[:, np.newaxis]
    dy = CY2[np.newaxis, :] - CY1[:, np.newaxis]

    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm

    return dy, dx


# ===============================================================================
# Utility Functions: IoU Calculations
# ===============================================================================

def iou_batch(bboxes1, bboxes2):
    """Compute IoU between two sets of bounding boxes."""
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - wh

    iou = wh / (union + 1e-7)
    return iou


def giou_batch(bboxes1, bboxes2):
    """Compute Generalized IoU (GIoU) between two sets of bounding boxes."""
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - wh
    iou = wh / (union + 1e-7)

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    area_enclose = wc * hc + 1e-7

    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.0) / 2.0
    return giou


# ===============================================================================
# Association Functions
# ===============================================================================

def linear_assignment(cost_matrix):
    """Solve linear assignment problem (Hungarian algorithm)."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        x, y = scipy_linear_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    """Associate detections to trackers using IoU and velocity direction cost."""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y, X = speed_direction_batch(detections, previous_obs)

    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)

    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = iou_batch(detections, trackers)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ===============================================================================
# Kalman Filter Implementation
# ===============================================================================

class KalmanFilter:
    """
    Simple Kalman Filter for OC-SORT.
    State vector: [x, y, s, r, vx, vy, vs] (7D)
    Measurement vector: [x, y, s, r] (4D)
    """

    def __init__(self):
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1))

        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.R = np.eye(self.dim_z)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        self.Q = np.eye(self.dim_x)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)


# ===============================================================================
# KalmanBoxTracker: Single Object Tracker
# ===============================================================================

class KalmanBoxTracker:
    """OC-SORT Kalman Filter tracker for a single object."""

    count = 0

    def __init__(self, bbox, delta_t=3):
        self.kf = KalmanFilter()
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.observations = {}
        self.history = []
        self.velocity = np.array([0, 0])
        self.delta_t = delta_t

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.age = 0

        self.last_observation = np.array(bbox)
        self.observations[self.age] = bbox

        self.mask = None

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1

        self.observations[self.age] = bbox
        self.last_observation = np.array(bbox)

        if self.age - self.delta_t in self.observations:
            previous_box = self.observations[self.age - self.delta_t]
            self.velocity = speed_direction(previous_box, bbox)
        elif self.age > 0:
            prev_ages = [a for a in self.observations.keys() if a < self.age]
            if prev_ages:
                previous_box = self.observations[max(prev_ages)]
                self.velocity = speed_direction(previous_box, bbox)

        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


# ===============================================================================
# OCSort: Main Tracker Class
# ===============================================================================

class OCSort:
    """OC-SORT Multi-Object Tracker."""

    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, inertia=0.2,
                 merge_iou_threshold=0.7, merge_patience=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia

        self.merge_iou_threshold = merge_iou_threshold
        self.merge_patience = merge_patience
        self._merge_counts = {}  # {(id_low, id_high): consecutive_frame_count}

        KalmanBoxTracker.count = 0

    def _merge_overlapping_tracks(self):
        """
        Merge tracks that consistently co-exist with high spatial overlap.

        After each frame's association, checks all pairs of recently-updated
        tracks for IoU overlap. If a pair maintains IoU >= merge_iou_threshold
        for merge_patience consecutive active frames, the newer track is absorbed
        into the older one (lower ID kept).

        This addresses duplicate detections from the detector (e.g., NMS failures
        for small objects at frame edges) that create competing tracks for the
        same physical object.

        Algorithm:
            1. Collect all tracks updated this frame (time_since_update == 0)
            2. Compute pairwise IoU between their Kalman state estimates
            3. For each pair with IoU >= threshold, increment a consecutive counter
            4. For each pair with IoU < threshold, reset the counter
            5. When a counter reaches merge_patience, merge the pair:
               - Keep the track with the lower ID (older, more established)
               - If the removed track was more recently updated, transfer its
                 state and mask to the keeper
               - Remove the newer track from the tracker list
        """
        if len(self.trackers) < 2:
            return

        # Prune stale entries for tracks no longer alive
        alive_ids = {trk.id for trk in self.trackers}
        self._merge_counts = {
            k: v for k, v in self._merge_counts.items()
            if k[0] in alive_ids and k[1] in alive_ids
        }

        # Collect tracks that received a detection this frame
        active = []
        for trk in self.trackers:
            if trk.time_since_update == 0:
                bbox = trk.get_state()[0]
                active.append((trk, bbox[:4]))

        if len(active) < 2:
            return

        bboxes = np.array([b for _, b in active])
        iou_mat = iou_batch(bboxes, bboxes)

        # Track which pairs were checked this frame (to reset absent pairs)
        checked_pairs = set()
        to_remove_ids = set()

        for i in range(len(active)):
            if active[i][0].id in to_remove_ids:
                continue
            for j in range(i + 1, len(active)):
                if active[j][0].id in to_remove_ids:
                    continue

                trk_i, trk_j = active[i][0], active[j][0]
                pair_key = (min(trk_i.id, trk_j.id), max(trk_i.id, trk_j.id))
                checked_pairs.add(pair_key)

                if iou_mat[i, j] >= self.merge_iou_threshold:
                    self._merge_counts[pair_key] = self._merge_counts.get(pair_key, 0) + 1

                    if self._merge_counts[pair_key] >= self.merge_patience:
                        # Merge: keep older track (lower ID)
                        if trk_i.id < trk_j.id:
                            keeper, removed = trk_i, trk_j
                        else:
                            keeper, removed = trk_j, trk_i

                        # Transfer state if removed track has fresher observation
                        if removed.time_since_update < keeper.time_since_update:
                            keeper.update(removed.last_observation[:4])
                            keeper.mask = removed.mask

                        to_remove_ids.add(removed.id)
                        self._merge_counts.pop(pair_key, None)
                else:
                    # IoU dropped — reset consecutive counter
                    self._merge_counts.pop(pair_key, None)

        if to_remove_ids:
            self.trackers = [t for t in self.trackers if t.id not in to_remove_ids]
            # Clean up any merge counts referencing removed tracks
            self._merge_counts = {
                k: v for k, v in self._merge_counts.items()
                if k[0] not in to_remove_ids and k[1] not in to_remove_ids
            }

    def update(self, dets, metadata_list=None):
        """
        Update tracker with new detections.

        Args:
            dets: [N, 5] array of [x1, y1, x2, y2, score]
            metadata_list: Optional list of metadata dicts per detection.
                           Expected key: 'mask' (binary numpy array).

        Returns:
            List of dicts: {'bbox': [x1,y1,x2,y2], 'track_id': int, 'mask': np.ndarray or None}
        """
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity for trk in self.trackers])
        if len(velocities) == 0:
            velocities = np.zeros((0, 2))

        previous_obs = []
        for trk in self.trackers:
            if len(trk.observations) > 0:
                last_obs = trk.last_observation
                previous_obs.append([last_obs[0], last_obs[1], last_obs[2], last_obs[3], 1])
            else:
                previous_obs.append([0, 0, 0, 0, -1])

        previous_obs = np.array(previous_obs) if previous_obs else np.zeros((0, 5))

        if len(dets) > 0:
            matched, unmatched_dets, unmatched_trks = associate(
                dets, trks, self.iou_threshold,
                velocities, previous_obs, self.inertia
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.array([])
            unmatched_trks = np.arange(len(self.trackers))

        for m in matched:
            det_idx, trk_idx = int(m[0]), int(m[1])
            self.trackers[trk_idx].update(dets[det_idx, :4])
            if metadata_list and det_idx < len(metadata_list):
                self.trackers[trk_idx].mask = metadata_list[det_idx].get('mask')

        # --- Observation-Centric Recovery (OCR) ---
        # Second association: re-match remaining unmatched detections against
        # unmatched trackers using LAST OBSERVATION (not Kalman prediction).
        # This recovers tracks where Kalman prediction drifted but the object
        # is still near its last observed position.
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = dets[unmatched_dets]
            left_trks = np.array([self.trackers[t].last_observation for t in unmatched_trks])
            iou_left = iou_batch(left_dets, left_trks)

            matched_indices_2 = linear_assignment(-(iou_left))

            to_remove_det = []
            to_remove_trk = []
            for m in matched_indices_2:
                det_idx = unmatched_dets[m[0]]
                trk_idx = unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] >= self.iou_threshold:
                    self.trackers[trk_idx].update(dets[det_idx, :4])
                    if metadata_list and det_idx < len(metadata_list):
                        self.trackers[trk_idx].mask = metadata_list[det_idx].get('mask')
                    to_remove_det.append(det_idx)
                    to_remove_trk.append(trk_idx)

            unmatched_dets = np.array([d for d in unmatched_dets if d not in to_remove_det])
            unmatched_trks = np.array([t for t in unmatched_trks if t not in to_remove_trk])

        # --- Duplicate Detection Suppression ---
        # Before creating new tracks, discard unmatched detections that overlap
        # with already-matched tracks. These are duplicate detections from the
        # detector (e.g., NMS failures) for the same physical object.
        if len(unmatched_dets) > 0 and self.merge_iou_threshold > 0:
            matched_bboxes = []
            for trk in self.trackers:
                if trk.time_since_update == 0:  # matched this frame
                    matched_bboxes.append(trk.last_observation[:4])

            if matched_bboxes:
                matched_bboxes = np.array(matched_bboxes)
                dup_dets = dets[unmatched_dets]
                iou_dup = iou_batch(dup_dets, matched_bboxes)

                keep = []
                for d_local in range(len(unmatched_dets)):
                    if np.max(iou_dup[d_local]) < self.merge_iou_threshold:
                        keep.append(unmatched_dets[d_local])
                unmatched_dets = np.array(keep, dtype=int) if keep else np.array([], dtype=int)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], delta_t=self.delta_t)
            if metadata_list and i < len(metadata_list):
                trk.mask = metadata_list[i].get('mask')
            self.trackers.append(trk)

        # Merge duplicate tracks before generating output
        self._merge_overlapping_tracks()

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                ret.append({
                    'bbox': [d[0], d[1], d[2], d[3]],
                    'track_id': trk.id + 1,  # 1-indexed
                    'mask': trk.mask,
                })
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return ret

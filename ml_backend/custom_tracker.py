"""
VectraTrack Custom SORT Tracker
================================
A custom implementation of the SORT (Simple Online and Realtime Tracking)
algorithm with Kalman Filter state estimation and Hungarian Algorithm
bipartite matching — built from scratch.

Algorithms Used:
  1. Kalman Filter        — Constant-velocity motion model to estimate/predict
                            the next bounding box position even when a detection
                            is temporarily lost (dead-reckoning).
  2. IoU (Intersection    — Used as the affinity metric to associate detected
     over Union)           bounding boxes with existing tracks.
  3. Hungarian Algorithm  — Optimal linear assignment (via scipy) to solve
     (Bipartite Match)     the assignment problem between detections & tracks.

State Vector:
  x = [cx, cy, s, r, cx_dot, cy_dot, s_dot]
  where:
    cx, cy       : Bounding-box center coordinates
    s            : Scale (area = width × height)
    r            : Aspect ratio (width / height), assumed constant
    cx_dot,
    cy_dot, s_dot: Respective velocities
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def bbox_to_z(bbox):
    """Convert [x1, y1, x2, y2] bbox to Kalman state vector [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h        # scale (area)
    r = w / float(h) # aspect ratio
    return np.array([[cx], [cy], [s], [r]])


def z_to_bbox(z, score=None):
    """Convert Kalman state vector to [x1, y1, x2, y2, (score)]."""
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).flatten()
    return np.array([x1, y1, x2, y2, score]).flatten()


def iou(bb_test, bb_gt):
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / (union + 1e-9)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to existing trackers using IoU affinity + Hungarian matching.
    Returns: matches, unmatched_dets, unmatched_trks
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    det_indices, trk_indices = linear_sum_assignment(-iou_matrix)

    matches, unmatched_dets, unmatched_trks = [], [], []

    for d in range(len(detections)):
        if d not in det_indices:
            unmatched_dets.append(d)

    for t in range(len(trackers)):
        if t not in trk_indices:
            unmatched_trks.append(t)

    for d, t in zip(det_indices, trk_indices):
        if iou_matrix[d, t] < iou_threshold:
            unmatched_dets.append(d)
            unmatched_trks.append(t)
        else:
            matches.append((d, t))

    return matches, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Kalman Filter Tracker  (single-object state estimator)
# ---------------------------------------------------------------------------

class KalmanFilterTracker:
    """
    Maintains the state estimate of a single tracked object using a
    constant-velocity Kalman Filter.

    State vector  (7D): [cx, cy, s, r, cx', cy', s']
    Measurement   (4D): [cx, cy, s, r]
    """

    count = 0

    def __init__(self, bbox):
        dim_x, dim_z = 7, 4
        self.F = np.eye(dim_x)
        for i in range(dim_z):
            self.F[i, i + dim_z - 1] = 1
        self.H = np.eye(dim_z, dim_x)
        self.R = np.eye(dim_z) * np.array([1, 1, 10, 10])
        self.Q = np.eye(dim_x) * np.array([1, 1, 1, 1, 0.01, 0.01, 0.0001])
        self.P = np.eye(dim_x) * np.array([10, 10, 10, 10, 1e4, 1e4, 1e4])
        self.x = np.zeros((dim_x, 1))
        self.x[:dim_z] = bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanFilterTracker.count
        KalmanFilterTracker.count += 1
        self.history = []
        self.hit_streak = 0
        self.age = 0
        self.velocity = np.array([0.0, 0.0])
        self.cls = -1
        self.conf = 0.0

    def predict(self):
        """Advance the state estimate one time step."""
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        cx, cy = float(self.x[0]), float(self.x[1])
        if len(self.history) == 0 or (cx, cy) != self.history[-1]:
            self.history.append((cx, cy))
            if len(self.history) > 60:
                self.history.pop(0)
        return z_to_bbox(self.x)

    def update(self, bbox, cls=-1, conf=0.0):
        """Correct the state estimate using a new detection."""
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P
        self.velocity = np.array([float(self.x[4]), float(self.x[6])])
        self.time_since_update = 0
        self.hit_streak += 1
        self.cls = cls
        self.conf = conf

    def get_state(self):
        return z_to_bbox(self.x)

    def get_speed(self):
        return float(np.linalg.norm(self.velocity))

    def predict_future(self, steps=15):
        """Project future bounding-box centres using the current velocity."""
        future = []
        cx, cy = float(self.x[0]), float(self.x[1])
        vx, vy = float(self.x[4]), float(self.x[5])
        for _ in range(steps):
            cx += vx
            cy += vy
            future.append((cx, cy))
        return future


# ---------------------------------------------------------------------------
# Custom SORT Tracker  (multi-object orchestrator)
# ---------------------------------------------------------------------------

class CustomSortTracker:
    """
    Multi-object tracker implementing SORT:
      - Kalman Filter for state estimation and prediction
      - Hungarian Algorithm for optimal detection-track assignment
      - Track lifecycle: creation → confirmation (min_hits) → timeout (max_age)
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanFilterTracker.count = 0

    def update(self, detections):
        """
        Run one tracking step.
        Args:
            detections: np.ndarray shape (N, 6) — [x1, y1, x2, y2, conf, class_id]
        Returns:
            active_tracks: list of dicts {id, bbox, cls, conf, speed, history, future, age}
        """
        self.frame_count += 1

        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                to_del.append(i)
            else:
                predicted_boxes.append(box)
        for i in reversed(to_del):
            self.trackers.pop(i)

        det_boxes = detections[:, :4] if len(detections) > 0 else []
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.iou_threshold
        )

        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                cls=int(detections[det_idx, 5]) if detections.shape[1] > 5 else -1,
                conf=float(detections[det_idx, 4]) if detections.shape[1] > 4 else 1.0,
            )

        for det_idx in unmatched_dets:
            new_trk = KalmanFilterTracker(detections[det_idx, :4])
            if detections.shape[1] > 5:
                new_trk.cls = int(detections[det_idx, 5])
                new_trk.conf = float(detections[det_idx, 4])
            self.trackers.append(new_trk)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        active = []
        for trk in self.trackers:
            if trk.time_since_update > 0:
                continue
            if trk.hit_streak < self.min_hits and self.frame_count <= self.min_hits:
                continue
            bbox = trk.get_state()
            active.append({
                "id":      trk.id + 1,
                "bbox":    [int(v) for v in bbox],
                "cls":     trk.cls,
                "conf":    round(trk.conf, 2),
                "speed":   round(trk.get_speed(), 2),
                "history": list(trk.history),
                "future":  trk.predict_future(steps=15),
                "age":     trk.age,
            })
        return active

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanFilterTracker.count = 0

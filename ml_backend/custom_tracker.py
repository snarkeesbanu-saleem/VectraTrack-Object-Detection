"""
VectraTrack Custom Tracker — v3.0
===================================
Algorithms:
  1. Kalman Filter          — 7D constant-velocity state estimation
  2. Hungarian Algorithm    — Optimal bipartite assignment (scipy)
  3. ByteTrack-style        — TWO-STAGE association:
     Two-Stage Association    Stage 1: High-conf detections ↔ all tracks (IoU)
                              Stage 2: Low-conf detections ↔ unmatched tracks
                              This rescues tracks through occlusion without
                              spawning noisy ghost tracks.

State Vector:
  x = [cx, cy, s, r, cx', cy', s']
  where cx,cy = centre, s = area, r = aspect ratio, primes = velocities
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_to_z(bbox):
    """[x1,y1,x2,y2] → Kalman measurement [cx, cy, s, r]"""
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s  = w * h
    r  = w / float(h + 1e-9)
    return np.array([[cx], [cy], [s], [r]])


def z_to_bbox(z):
    """Kalman state → [x1, y1, x2, y2]"""
    w  = np.sqrt(max(z[2] * z[3], 0))
    h  = z[2] / (w + 1e-9)
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0,
    ]).flatten()


def iou_batch(dets, trks):
    """
    Vectorised IoU matrix: dets (M,4) × trks (N,4) → (M,N)
    All boxes in [x1,y1,x2,y2] format.
    """
    dets = np.expand_dims(dets, 1)   # (M,1,4)
    trks = np.expand_dims(trks, 0)   # (1,N,4)

    xx1 = np.maximum(dets[..., 0], trks[..., 0])
    yy1 = np.maximum(dets[..., 1], trks[..., 1])
    xx2 = np.minimum(dets[..., 2], trks[..., 2])
    yy2 = np.minimum(dets[..., 3], trks[..., 3])

    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_d = (dets[..., 2] - dets[..., 0]) * (dets[..., 3] - dets[..., 1])
    area_t = (trks[..., 2] - trks[..., 0]) * (trks[..., 3] - trks[..., 1])

    union  = area_d + area_t - inter
    return inter / (union + 1e-9)


def _hungarian(cost_matrix, threshold):
    """
    Run Hungarian on cost_matrix (already positive = reward).
    Returns matches, unmatched_rows, unmatched_cols.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matches, unmatched_rows, unmatched_cols = [], [], []
    matched_r, matched_c = set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            matches.append((r, c))
            matched_r.add(r)
            matched_c.add(c)

    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in matched_r]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in matched_c]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Kalman Filter — single-object state estimator
# ---------------------------------------------------------------------------

class KalmanFilterTracker:
    """
    Constant-velocity Kalman Filter for a single bounding-box track.

    State (7D): [cx, cy, s, r, cx', cy', s']
    Obs   (4D): [cx, cy, s, r]
    """
    count = 0

    def __init__(self, bbox):
        dim_x, dim_z = 7, 4

        # Transition: position += velocity * dt (dt=1 frame)
        self.F = np.eye(dim_x)
        for i in range(dim_z):
            self.F[i, i + dim_z - 1] = 1

        self.H = np.eye(dim_z, dim_x)          # measurement matrix
        self.R = np.diag([1., 1., 10., 10.])   # measurement noise
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001])  # process noise
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])   # uncertainty

        self.x = np.zeros((dim_x, 1))
        self.x[:dim_z] = bbox_to_z(bbox)

        self.id               = KalmanFilterTracker.count
        KalmanFilterTracker.count += 1

        self.time_since_update = 0
        self.hit_streak        = 0
        self.age               = 0
        self.history           = []        # (cx,cy) trail — last 60 positions
        self.velocity          = np.zeros(2)
        self.cls               = -1
        self.conf              = 0.0

    # ---- Kalman predict step -----------------------------------------------
    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age               += 1
        self.time_since_update += 1

        cx, cy = float(self.x[0]), float(self.x[1])
        if not self.history or (cx, cy) != self.history[-1]:
            self.history.append((cx, cy))
            if len(self.history) > 60:
                self.history.pop(0)

        return z_to_bbox(self.x)

    # ---- Kalman update step ------------------------------------------------
    def update(self, bbox, cls=-1, conf=0.0):
        z    = bbox_to_z(bbox)
        y    = z - self.H @ self.x
        S    = self.H @ self.P @ self.H.T + self.R
        K    = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

        vx, vy = float(self.x[4]), float(self.x[5])
        self.velocity          = np.array([vx, vy])
        self.time_since_update = 0
        self.hit_streak       += 1
        self.cls               = cls
        self.conf              = conf

    # ---- Accessors ---------------------------------------------------------
    def get_state(self):
        return z_to_bbox(self.x)

    def get_speed(self):
        return float(np.linalg.norm(self.velocity))

    def predict_future(self, steps=15):
        """Project next `steps` centres using current velocity."""
        cx, cy = float(self.x[0]), float(self.x[1])
        vx, vy = float(self.x[4]), float(self.x[5])
        future = []
        for _ in range(steps):
            cx += vx; cy += vy
            future.append((cx, cy))
        return future

    def centre(self):
        return (float(self.x[0]), float(self.x[1]))


# ---------------------------------------------------------------------------
# ByteTrack-style Two-Stage Association
# ---------------------------------------------------------------------------

def bytetrack_associate(detections, trackers,
                        high_thresh=0.5, low_thresh=0.1,
                        iou_high=0.3,    iou_low=0.2):
    """
    ByteTrack two-stage detection-to-track association.

    Stage 1 — High-confidence detections vs ALL existing tracks (strict IoU).
    Stage 2 — Low-confidence detections vs UNMATCHED tracks (relaxed IoU).
              These are likely occluded objects that produced weak detections.

    Args:
        detections : np.ndarray (N,6) — [x1,y1,x2,y2,conf,cls]
        trackers   : list of KalmanFilterTracker
        high_thresh: confidence boundary (≥ high_thresh → high det)
        low_thresh : minimum conf to keep (< low_thresh → discard)
        iou_high   : IoU gate for stage-1
        iou_low    : IoU gate for stage-2

    Returns:
        matches_1       : [(det_idx, trk_idx), ...]  — stage-1 matches
        matches_2       : [(det_idx, trk_idx), ...]  — stage-2 matches
        unmatched_high  : unmatched high-conf det indices
        unmatched_low   : unmatched low-conf  det indices
        unmatched_trks  : track indices still unmatched after both stages
    """
    if len(detections) == 0:
        return [], [], [], [], list(range(len(trackers)))

    confs      = detections[:, 4]
    high_mask  = confs >= high_thresh
    low_mask   = (confs >= low_thresh) & (~high_mask)

    high_dets  = detections[high_mask]
    low_dets   = detections[low_mask]
    high_idx   = np.where(high_mask)[0]
    low_idx    = np.where(low_mask)[0]

    predicted  = np.array([t.predict() for t in trackers]) if trackers else np.empty((0, 4))

    # ---- Stage 1: high-conf ↔ all tracks -----------------------------------
    if len(high_dets) > 0 and len(trackers) > 0:
        iou_m1 = iou_batch(high_dets[:, :4], predicted)
        m1, unm_high, unm_trk1 = _hungarian(iou_m1, iou_high)
    else:
        m1 = []
        unm_high  = list(range(len(high_dets)))
        unm_trk1  = list(range(len(trackers)))

    matches_1 = [(int(high_idx[r]), c) for r, c in m1]

    # ---- Stage 2: low-conf ↔ unmatched tracks ------------------------------
    if len(low_dets) > 0 and len(unm_trk1) > 0:
        pred_unm = predicted[unm_trk1]
        iou_m2   = iou_batch(low_dets[:, :4], pred_unm)
        m2, unm_low, unm_trk2 = _hungarian(iou_m2, iou_low)
        matches_2      = [(int(low_idx[r]), unm_trk1[c]) for r, c in m2]
        unmatched_low  = [int(low_idx[r]) for r in unm_low]
        unmatched_trks = [unm_trk1[c] for c in unm_trk2]
    else:
        matches_2      = []
        unmatched_low  = [int(i) for i in low_idx]
        unmatched_trks = unm_trk1

    unmatched_high = [int(high_idx[r]) for r in unm_high]

    return matches_1, matches_2, unmatched_high, unmatched_low, unmatched_trks


# ---------------------------------------------------------------------------
# Custom SORT + ByteTrack Orchestrator
# ---------------------------------------------------------------------------

class CustomSortTracker:
    """
    Multi-object tracker combining:
      • Kalman Filter     — smooth state estimation + occlusion dead-reckoning
      • Hungarian         — globally optimal assignment
      • ByteTrack 2-stage — rescues low-confidence / occluded detections
    """

    def __init__(self, max_age=30, min_hits=3,
                 iou_threshold=0.3,
                 high_conf=0.5, low_conf=0.1):
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.iou_threshold  = iou_threshold
        self.high_conf      = high_conf
        self.low_conf       = low_conf
        self.trackers       = []
        self.frame_count    = 0
        KalmanFilterTracker.count = 0

    def update(self, detections):
        """
        Run one tracking step (ByteTrack 2-stage).

        Args:
            detections: np.ndarray (N,6) — [x1,y1,x2,y2,conf,cls]
                        OR np.empty((0,6)) for empty frames.

        Returns:
            active_tracks: list of dicts —
              {id, bbox, cls, conf, speed, history, future, age, occluded}
        """
        self.frame_count += 1

        # 1. Predict all existing tracks
        pred_boxes = []
        to_del     = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                to_del.append(i)
            else:
                pred_boxes.append(box)
        for i in reversed(to_del):
            self.trackers.pop(i)

        # 2. ByteTrack two-stage association
        m1, m2, unm_high, unm_low, unm_trk = bytetrack_associate(
            detections, self.trackers,
            high_thresh=self.high_conf,
            low_thresh=self.low_conf,
            iou_high=self.iou_threshold,
            iou_low=max(0.1, self.iou_threshold - 0.1),
        )

        # 3. Update stage-1 matches (high-conf)
        for det_idx, trk_idx in m1:
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                cls=int(detections[det_idx, 5])   if detections.shape[1] > 5 else -1,
                conf=float(detections[det_idx, 4]) if detections.shape[1] > 4 else 1.0,
            )

        # 4. Update stage-2 matches (low-conf / occluded)
        for det_idx, trk_idx in m2:
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                cls=int(detections[det_idx, 5])   if detections.shape[1] > 5 else -1,
                conf=float(detections[det_idx, 4]) if detections.shape[1] > 4 else 0.3,
            )

        # 5. Spawn new tracks for unmatched HIGH-conf detections only
        #    (low-conf orphans are NOT promoted to new tracks — ByteTrack rule)
        for det_idx in unm_high:
            trk = KalmanFilterTracker(detections[det_idx, :4])
            if detections.shape[1] > 5:
                trk.cls  = int(detections[det_idx, 5])
                trk.conf = float(detections[det_idx, 4])
            self.trackers.append(trk)

        # 6. Prune dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 7. Collect output tracks
        active = []
        for trk in self.trackers:
            # Tracks still in occlusion (matched via low-conf) are included
            # but flagged; fully unmatched tracks are skipped for display
            occluded = trk.time_since_update > 0

            if trk.hit_streak < self.min_hits and self.frame_count <= self.min_hits:
                continue

            bbox = trk.get_state()
            active.append({
                "id":       trk.id + 1,
                "bbox":     [int(v) for v in bbox],
                "cls":      trk.cls,
                "conf":     round(trk.conf, 2),
                "speed":    round(trk.get_speed(), 2),
                "history":  list(trk.history),
                "future":   trk.predict_future(steps=15),
                "age":      trk.age,
                "occluded": occluded,
                "centre":   trk.centre(),
            })
        return active

    def reset(self):
        self.trackers    = []
        self.frame_count = 0
        KalmanFilterTracker.count = 0

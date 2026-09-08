"""
VectraTrack Analytics Engine — v3.0
=====================================
Provides:
  1. LineCrossingCounter  — Virtual counting line with direction detection
  2. HeatmapAccumulator   — Spatial density heatmap from track trajectories
  3. TelemetryLogger      — Frame-level CSV log builder
"""

import numpy as np
import cv2
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Line Crossing Counter
# ---------------------------------------------------------------------------

class LineCrossingCounter:
    """
    Counts objects crossing a user-defined virtual line and tracks direction.

    The line is defined by two points (x1,y1)→(x2,y2).
    Crossing is detected by sign change of the cross-product of the line
    vector with the vector from line-start to the track centre.

    Attributes:
        counts : dict  {cls_name: {"in": N, "out": N}}
        events : list  of crossing event dicts
    """

    def __init__(self, pt1, pt2, cooldown_frames=10):
        """
        Args:
            pt1, pt2       : Line endpoints as (x, y) tuples (pixel coords).
            cooldown_frames: Minimum frames between two crossings for same ID.
        """
        self.pt1 = np.array(pt1, dtype=float)
        self.pt2 = np.array(pt2, dtype=float)
        self.cooldown = cooldown_frames

        self._prev_side   = {}   # {track_id: side (+1 or -1)}
        self._last_crossed = {}  # {track_id: frame_number}

        self.counts = defaultdict(lambda: {"in": 0, "out": 0})
        self.events = []         # list of {frame, id, cls, direction}
        self.total_in  = 0
        self.total_out = 0

    def _side(self, point):
        """Return +1 or -1 based on which side of the line the point is."""
        lv  = self.pt2 - self.pt1              # line vector
        pv  = np.array(point) - self.pt1       # point vector from line start
        cross = lv[0] * pv[1] - lv[1] * pv[0] # z-component of cross product
        return 1 if cross >= 0 else -1

    def update(self, tracks, frame_idx, coco_classes):
        """
        Call once per frame with the current active tracks.

        Args:
            tracks       : list of track dicts (output from CustomSortTracker.update)
            frame_idx    : current frame number
            coco_classes : list mapping cls_id → name

        Returns:
            new_events: list of crossing events detected this frame
        """
        new_events = []
        for trk in tracks:
            tid   = trk["id"]
            cx, cy = trk["centre"]
            side  = self._side((cx, cy))

            prev  = self._prev_side.get(tid)
            last  = self._last_crossed.get(tid, -999)

            if prev is not None and side != prev:
                if (frame_idx - last) >= self.cooldown:
                    # Determine direction relative to line normal
                    direction = "in" if side == 1 else "out"
                    cls_name  = (coco_classes[trk["cls"]]
                                 if 0 <= trk.get("cls", -1) < len(coco_classes)
                                 else "object")

                    self.counts[cls_name][direction] += 1
                    if direction == "in":
                        self.total_in  += 1
                    else:
                        self.total_out += 1

                    event = {
                        "frame":     frame_idx,
                        "id":        tid,
                        "cls":       cls_name,
                        "direction": direction,
                        "cx":        round(cx, 1),
                        "cy":        round(cy, 1),
                    }
                    self.events.append(event)
                    new_events.append(event)
                    self._last_crossed[tid] = frame_idx

            self._prev_side[tid] = side

        return new_events

    def draw(self, frame, new_events=None):
        """
        Draw the counting line + live count HUD onto `frame` (in-place).

        Args:
            frame      : BGR OpenCV frame
            new_events : events from this frame (flash effect)
        """
        h, w = frame.shape[:2]
        p1 = tuple(np.clip(self.pt1, 0, [w-1, h-1]).astype(int))
        p2 = tuple(np.clip(self.pt2, 0, [w-1, h-1]).astype(int))

        # Flash yellow if a crossing just happened, else neon cyan
        flash = new_events and len(new_events) > 0
        colour = (0, 255, 255) if flash else (0, 220, 200)
        thickness = 3 if flash else 2

        # Dashed line
        dash_len, gap_len = 18, 8
        lv  = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(lv)
        if dist < 1:
            return frame
        step = (lv / dist)
        pos  = np.array(p1, dtype=float)
        drawing = True
        seg_len = 0
        while np.linalg.norm(pos - np.array(p1)) < dist:
            nxt = pos + step * (dash_len if drawing else gap_len)
            if np.linalg.norm(nxt - np.array(p1)) > dist:
                nxt = np.array(p2, dtype=float)
            if drawing:
                cv2.line(frame, tuple(pos.astype(int)), tuple(nxt.astype(int)),
                         colour, thickness, cv2.LINE_AA)
            pos     = nxt
            drawing = not drawing

        # Endpoint circles
        cv2.circle(frame, p1, 6, colour, -1)
        cv2.circle(frame, p2, 6, colour, -1)

        # Direction arrows at mid-point
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        normal = np.array([-lv[1], lv[0]])
        if np.linalg.norm(normal) > 0:
            normal = (normal / np.linalg.norm(normal) * 20).astype(int)
        cv2.arrowedLine(frame, mid, (mid[0]+normal[0], mid[1]+normal[1]),
                        (0,255,100), 2, cv2.LINE_AA, tipLength=0.4)

        # Count HUD box
        label = f"  IN: {self.total_in}   OUT: {self.total_out}   "
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        box_x = max(0, mid[0] - tw//2)
        box_y = max(th + 8, mid[1] - 30)
        cv2.rectangle(frame, (box_x-4, box_y-th-6), (box_x+tw+4, box_y+4),
                      (10, 5, 30), cv2.FILLED)
        cv2.rectangle(frame, (box_x-4, box_y-th-6), (box_x+tw+4, box_y+4),
                      colour, 1)
        cv2.putText(frame, label, (box_x, box_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        return frame

    def reset(self):
        self._prev_side.clear()
        self._last_crossed.clear()
        self.counts    = defaultdict(lambda: {"in": 0, "out": 0})
        self.events    = []
        self.total_in  = 0
        self.total_out = 0


# ---------------------------------------------------------------------------
# 2. Heatmap Accumulator
# ---------------------------------------------------------------------------

class HeatmapAccumulator:
    """
    Accumulates a spatial density heatmap from all track centroid positions.

    Usage:
        acc = HeatmapAccumulator(width, height)
        for frame, tracks in ...:
            acc.update(tracks)
        overlay = acc.render_overlay(frame)
    """

    def __init__(self, width, height, blur_radius=31):
        self.w      = width
        self.h      = height
        self.blur_r = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.canvas = np.zeros((height, width), dtype=np.float32)
        self.total_points = 0

    def update(self, tracks):
        """Add centroid positions of current active tracks."""
        for trk in tracks:
            cx, cy = trk.get("centre", (None, None))
            if cx is None:
                bbox = trk.get("bbox", [])
                if len(bbox) >= 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
            if cx is None:
                continue
            xi = int(np.clip(cx, 0, self.w - 1))
            yi = int(np.clip(cy, 0, self.h - 1))
            self.canvas[yi, xi] += 1.0
            self.total_points += 1

    def render_overlay(self, frame, alpha=0.55):
        """
        Return a copy of `frame` with the neon heatmap blended in.

        Args:
            frame : BGR OpenCV frame (H×W×3)
            alpha : overlay opacity

        Returns:
            blended frame (BGR)
        """
        if self.total_points == 0:
            return frame.copy()

        # Gaussian blur → smooth density
        blurred = cv2.GaussianBlur(self.canvas, (self.blur_r, self.blur_r), 0)

        # Normalise to [0, 255]
        mn, mx = blurred.min(), blurred.max()
        if mx - mn < 1e-6:
            return frame.copy()
        norm = ((blurred - mn) / (mx - mn) * 255).astype(np.uint8)

        # Apply TURBO colormap (blue→green→red hot)
        heatmap_rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

        # Mask: only blend where density > 0
        mask = (norm > 8).astype(np.float32)
        mask_3ch = np.stack([mask]*3, axis=-1)

        frame_float = frame.astype(np.float32)
        heat_float  = heatmap_rgb.astype(np.float32)

        blended = (frame_float * (1 - alpha * mask_3ch) +
                   heat_float  * (alpha * mask_3ch)).astype(np.uint8)

        # Add label
        cv2.putText(blended, f"HEATMAP ({self.total_points} pts)",
                    (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return blended

    def get_png_bytes(self):
        """Return the raw heatmap as PNG bytes for download."""
        if self.total_points == 0:
            blank = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".png", blank)
            return buf.tobytes()

        blurred = cv2.GaussianBlur(self.canvas, (self.blur_r, self.blur_r), 0)
        mn, mx  = blurred.min(), blurred.max()
        norm    = ((blurred - mn) / (max(mx - mn, 1e-6)) * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        _, buf   = cv2.imencode(".png", coloured)
        return buf.tobytes()

    def reset(self):
        self.canvas = np.zeros((self.h, self.w), dtype=np.float32)
        self.total_points = 0


# ---------------------------------------------------------------------------
# 3. Telemetry Logger
# ---------------------------------------------------------------------------

class TelemetryLogger:
    """
    Frame-level telemetry log builder.
    Captures per-track data and crossing events, exports to CSV.
    """

    def __init__(self, coco_classes):
        self.rows        = []
        self.coco        = coco_classes
        self.cross_rows  = []

    def log_frame(self, frame_idx, tracks, fps=0.0):
        for trk in tracks:
            cls_name = (self.coco[trk["cls"]]
                        if 0 <= trk.get("cls", -1) < len(self.coco)
                        else "unknown")
            bbox = trk.get("bbox", [0, 0, 0, 0])
            cx, cy = trk.get("centre", ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
            self.rows.append({
                "frame":              frame_idx,
                "fps":                round(fps, 1),
                "track_id":           trk["id"],
                "class":              cls_name,
                "conf":               trk.get("conf", 0.0),
                "x1":                 bbox[0],
                "y1":                 bbox[1],
                "x2":                 bbox[2],
                "y2":                 bbox[3],
                "cx":                 round(cx, 1),
                "cy":                 round(cy, 1),
                "speed_px_per_frame": trk.get("speed", 0.0),
                "age_frames":         trk.get("age", 0),
                "occluded":           int(trk.get("occluded", False)),
            })

    def log_crossings(self, events):
        for ev in events:
            self.cross_rows.append(ev)

    def get_tracks_csv(self):
        """Return per-frame tracking CSV as bytes."""
        import io, csv
        if not self.rows:
            return b""
        fieldnames = list(self.rows[0].keys())
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(self.rows)
        return buf.getvalue().encode("utf-8")

    def get_crossings_csv(self):
        """Return line-crossing events CSV as bytes."""
        import io, csv
        if not self.cross_rows:
            return b""
        fieldnames = list(self.cross_rows[0].keys())
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(self.cross_rows)
        return buf.getvalue().encode("utf-8")

    def reset(self):
        self.rows       = []
        self.cross_rows = []

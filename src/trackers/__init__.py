"""
Trackers behind one interface:

    tracker = create_tracker(tracker_cfg, fps)
    tracks = tracker.update(detections, frame)   # detections: Nx5 [x1, y1, x2, y2, conf]
    # tracks: list of [x1, y1, x2, y2, track_id] for tracks matched in this frame

"deepsort": Kalman + appearance (MobileNetV2) + IoU matching.
"botsort":  Kalman + camera-motion compensation + two-stage (high/low score)
            IoU matching, no appearance model (Ultralytics BoT-SORT).
"""
from types import SimpleNamespace

import numpy as np


class DeepSortTracker:

    def __init__(self, cfg):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(
            max_age=cfg["max_age"],
            n_init=cfg["n_init"],
            max_iou_distance=cfg["max_iou_distance"],
            max_cosine_distance=cfg.get("max_cosine_distance", 0.2),
            nn_budget=cfg.get("nn_budget"),
            embedder_gpu=False,
        )

    def update(self, detections, frame):
        # DeepSort expects [left, top, width, height].
        raw = [
            ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(conf), "person")
            for x1, y1, x2, y2, conf in detections
        ]
        tracks = self.tracker.update_tracks(raw, frame=frame)
        return [
            [*t.to_ltrb(), int(t.track_id)]
            for t in tracks
            if t.is_confirmed() and t.time_since_update == 0
        ]


def _scale_boxes(xyxy, factor):
    """Scale boxes around their centers (factor > 1 enlarges)."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    half = (xyxy[:, 2:] - xyxy[:, :2]) / 2 * factor
    return np.concatenate([center - half, center + half], axis=1)


class _Detections:
    """Minimal stand-in for an Ultralytics Results.boxes object."""

    def __init__(self, dets):
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
        self.xyxy = dets[:, :4]
        self.conf = dets[:, 4]
        self.cls = np.zeros(len(dets), dtype=np.float32)
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        self.xywh = np.concatenate([self.xyxy[:, :2] + wh / 2, wh], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return _Detections(np.column_stack([self.xyxy, self.conf])[index])


class BoTSORTTracker:

    def __init__(self, cfg, fps):
        from ultralytics.trackers.bot_sort import BOTSORT

        args = SimpleNamespace(
            track_high_thresh=cfg["track_high_thresh"],
            track_low_thresh=cfg["track_low_thresh"],
            new_track_thresh=cfg["new_track_thresh"],
            track_buffer=cfg["track_buffer"],
            match_thresh=cfg["match_thresh"],
            fuse_score=cfg.get("fuse_score", True),
            gmc_method=cfg.get("gmc_method", "sparseOptFlow"),
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        self.tracker = BOTSORT(args, frame_rate=int(round(fps)))
        # Camera motion is estimated on a downscaled frame; global motion
        # doesn't need full detail, and this is the main tracking cost.
        self.tracker.gmc.downscale = max(1, int(cfg.get("gmc_downscale", 2)))

        # Buffered IoU (C-BIoU, Yang et al. WACV 2023): boxes are enlarged by
        # (1 + 2*buffer) for matching only, so tiny boxes that moved a few
        # pixels still overlap. Output boxes are shrunk back.
        self.box_scale = 1.0 + 2.0 * float(cfg.get("match_buffer", 0.0))
        # Report a track only after it was matched in min_hits frames
        # (same role as DeepSORT's n_init), hiding one-frame flickers.
        self.min_hits = int(cfg.get("min_hits", 1))
        self.hits = {}

    def update(self, detections, frame):
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 5)
        if self.box_scale != 1.0 and len(dets):
            dets = np.column_stack([_scale_boxes(dets[:, :4], self.box_scale), dets[:, 4]])
        out = self.tracker.update(_Detections(dets), frame)
        # rows: x1, y1, x2, y2, track_id, score, cls, det_index
        tracks = []
        for row in out:
            track_id = int(row[4])
            self.hits[track_id] = self.hits.get(track_id, 0) + 1
            if self.hits[track_id] >= self.min_hits:
                box = _scale_boxes(row[:4], 1.0 / self.box_scale)[0]
                tracks.append([*box, track_id])
        return tracks


def create_tracker(tracker_cfg, fps):
    tracker_type = tracker_cfg.get("type", "deepsort")
    if tracker_type == "botsort":
        return BoTSORTTracker(tracker_cfg["botsort"], fps)
    if tracker_type == "deepsort":
        return DeepSortTracker(tracker_cfg["deepsort"])
    raise ValueError(f"Unknown tracker type: {tracker_type}")

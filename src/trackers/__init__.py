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


class _Relinker:
    """
    Re-link a NEW track to a recently LOST one (person went behind a bush,
    or the detector missed them for a while), so it keeps the old ID.

    Each lost track keeps a "ghost" that is moved every frame by
    (1) the camera motion measured by BoT-SORT and (2) the person's own
    velocity, which slowly decays (people often stop behind cover).
    A new track takes over the ghost's ID if it passes all gates:
        distance to the ghost   <= gate_px + growth * gap (uncertainty grows)
        direction               consistent with the walking direction
        box size                similar
        appearance (optional)   CNN embedding close enough
    Among candidates, the lowest combined cost wins.
    """

    def __init__(self, cfg, fps):
        self.max_gap = int(cfg.get("max_gap_s", 5.0) * fps)
        self.gate_px = float(cfg.get("gate_px", 40))
        self.gate_growth = float(cfg.get("gate_growth_px_per_s", 60)) / fps
        self.decay = float(cfg.get("velocity_decay", 0.97))
        self.min_cos = float(cfg.get("min_direction_cos", -0.2))
        self.w_dir = float(cfg.get("direction_weight", 0.5))
        self.max_size_ratio = float(cfg.get("max_size_ratio", 2.5))
        self.w_app = float(cfg.get("appearance_weight", 1.0))
        self.max_app = float(cfg.get("max_appearance_dist", 0.35))
        self.embedder = None
        if cfg.get("appearance", "none") == "mobilenet":
            from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
            self.embedder = MobileNetv2_Embedder(half=False, gpu=False)
        self.embed_every = int(cfg.get("embed_every", 5))

        # Target of interest (the walking person) -> always displayed as ID 1.
        # Selection cues: moving relative to the GROUND (camera motion removed;
        # bystanders and hot objects are static), persistence, near image center
        # (the drone keeps the target in view).
        tcfg = cfg.get("target", {})
        self.target_enabled = bool(tcfg.get("enabled", False))
        self.t_min_hits = int(tcfg.get("min_hits", 10))
        self.t_min_speed = float(tcfg.get("min_speed_px_s", 40)) / fps
        self.t_lock_score = float(tcfg.get("lock_score", 1.0))
        self.t_reacquire = int(float(tcfg.get("reacquire_after_s", 3)) * fps)
        # The camera follows the target, so its lost "ghost" stays fixed in the
        # IMAGE (not on the ground) and gets wider gates and a longer gap.
        self.t_gate_px = float(tcfg.get("gate_px", 150))
        self.t_gate_growth = float(tcfg.get("gate_growth_px_per_s", 60)) / fps
        self.t_max_gap = int(float(tcfg.get("max_gap_s", 15)) * fps)

        self.raw_to_id = {}
        self.next_id = 1
        self.tracks = {}  # internal id -> state dict
        self.frame = 0
        self.frame_size = None
        self.relinks = []  # (frame, old_id, raw_id, gap) for inspection
        self.target = None  # internal id of the target
        self.target_locks = []  # (frame, internal id) each time a target is (re)locked
        self.display = {}  # internal id -> displayed id
        self.next_display = 2 if self.target_enabled else 1

    # ---- helpers -------------------------------------------------------
    def _embed(self, frame, boxes):
        crops = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1, 16)  # 2x box, includes some context
            a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
            c, d = int(min(cx + half, w)), int(min(cy + half, h))
            crops.append(frame[b:d, a:c])
        feats = self.embedder.predict(crops)
        return [f / (np.linalg.norm(f) + 1e-9) for f in feats]

    def _warp(self, H):
        A, t = H[:, :2], H[:, 2]
        for tr in self.tracks.values():
            tr["hist"] = [(f, A @ p + t) for f, p in tr["hist"]]
            tr["pred"] = A @ tr["pred"] + t
            tr["vel"] = A @ tr["vel"]

    def _cost(self, tid, tr, center, wh, emb):
        gap = self.frame - tr["last_frame"]
        if gap < 1 or gap > self.max_gap:
            return None
        ratio = max(wh.prod(), 1) / max(tr["wh"].prod(), 1)
        if max(ratio, 1 / ratio) > self.max_size_ratio:
            return None
        dist = np.linalg.norm(center - tr["pred"])
        gate = self.gate_px + self.gate_growth * gap
        if dist > gate:
            return None
        cost = dist / gate
        # Direction: does the new position lie ahead of where the person was walking?
        last = tr["hist"][-1][1]
        move = center - last
        speed = np.linalg.norm(tr["vel"])
        if speed * gap > 10 and np.linalg.norm(move) > 10:
            cos = float(move @ tr["vel"]) / (np.linalg.norm(move) * speed)
            if cos < self.min_cos:
                return None
            cost += self.w_dir * (1 - cos) / 2
        if emb is not None and tr.get("emb") is not None:
            app = 1.0 - float(emb @ tr["emb"])
            if app > self.max_app:
                return None
            cost += self.w_app * app
        return cost

    def _update_track(self, tid, center, wh, emb):
        tr = self.tracks.setdefault(tid, {"hist": [], "vel": np.zeros(2), "emb": None,
                                          "hits": 0, "speed_ema": 0.0, "center_ema": None})
        tr["hist"] = (tr["hist"] + [(self.frame, center)])[-10:]
        tr["wh"], tr["last_frame"], tr["pred"] = wh, self.frame, center.copy()
        tr["pred_img"] = center.copy()  # not moved with the camera (target ghost)
        tr["hits"] += 1
        h, w = self.frame_size
        cdist = float(np.linalg.norm((center - [w / 2, h / 2]) / [w / 2, h / 2]))
        tr["center_ema"] = cdist if tr["center_ema"] is None else 0.9 * tr["center_ema"] + 0.1 * cdist
        (f0, p0), (f1, p1) = tr["hist"][0], tr["hist"][-1]
        if f1 - f0 >= 2:
            tr["vel"] = (p1 - p0) / (f1 - f0)  # px/frame, camera motion removed
            speed = float(np.linalg.norm(tr["vel"]))
            first = tr.setdefault("speed_n", 0) == 0
            tr["speed_ema"] = speed if first else 0.9 * tr["speed_ema"] + 0.1 * speed
            tr["speed_n"] += 1
        if emb is not None:
            tr["emb"] = emb if tr["emb"] is None else 0.8 * tr["emb"] + 0.2 * emb
            tr["emb"] /= np.linalg.norm(tr["emb"]) + 1e-9

    def _display_id(self, tid):
        """1 = target; 0 = tentative (not enough evidence yet, drawn as '?'); else 2, 3, ..."""
        if tid == self.target:
            return 1
        if tid not in self.display:
            if self.target_enabled and self.tracks[tid]["hits"] < self.t_min_hits:
                return 0
            self.display[tid] = self.next_display
            self.next_display += 1
        return self.display[tid]

    def _select_target(self, active):
        """
        Lock the target, or re-acquire it after it was lost.
        A candidate needs >= min_hits (so its ground speed is known) and must be
        moving. While the old target is lost for less than reacquire_after_s, the
        candidate must also be near the target's last IMAGE position (the camera
        follows the target). After that, any moving candidate may take ID 1.
        """
        current = self.tracks.get(self.target)
        if current is not None and current["last_frame"] == self.frame:
            return  # target visible: keep it
        gap = self.frame - current["last_frame"] if current is not None else None
        best, best_score = None, None
        for tid in active:
            tr = self.tracks[tid]
            if tid == self.target or tr["hits"] < self.t_min_hits:
                continue  # too new: ground speed not known yet
            if gap is not None and gap <= self.t_reacquire:
                center = tr["hist"][-1][1]
                gate = self.t_gate_px + self.t_gate_growth * gap
                if np.linalg.norm(center - current["pred_img"]) > gate:
                    continue
            # Each cue 0..1. A static object far from center scores ~0.7; the
            # walking, camera-centered target ~1.1+ even while it starts slowly.
            score = (min(tr["hits"] / 60.0, 1.0)                                   # persistence
                     + (1.0 - min(tr["center_ema"], 1.0))                          # near center
                     + min(tr["speed_ema"] / (3 * self.t_min_speed), 1.0))         # ground speed
            if score < self.t_lock_score:
                continue
            if best_score is None or score > best_score:
                best, best_score = tid, score
        if best is not None:
            if self.target is not None and self.target in self.tracks:
                self.display[self.target] = self.next_display  # old target loses ID 1
                self.next_display += 1
            self.target = best
            self.target_locks.append((self.frame, best))

    # ---- main step -----------------------------------------------------
    def step(self, H, outputs, frame):
        """outputs: list of (box[4], raw_id). Returns [[x1, y1, x2, y2, id], ...]."""
        self.frame += 1
        self.frame_size = frame.shape[:2]
        self._warp(H)
        for tr in self.tracks.values():  # lost ghosts keep walking
            if tr["last_frame"] < self.frame - 1:
                tr["pred"] = tr["pred"] + tr["vel"]
                tr["vel"] = tr["vel"] * self.decay

        boxes = [np.asarray(b, dtype=float) for b, _ in outputs]
        centers = [(b[:2] + b[2:]) / 2 for b in boxes]
        whs = [b[2:] - b[:2] for b in boxes]
        need_emb = [
            self.embedder is not None
            and (raw not in self.raw_to_id or self.frame % self.embed_every == 0)
            for _, raw in outputs
        ]
        embs = [None] * len(outputs)
        idx = [i for i, n in enumerate(need_emb) if n]
        if idx:
            for i, e in zip(idx, self._embed(frame, [boxes[i] for i in idx])):
                embs[i] = e

        active, results, new = set(), [], []
        for i, (_, raw) in enumerate(outputs):
            tid = self.raw_to_id.get(raw)
            if tid is not None and tid not in active:
                active.add(tid)
                results.append((i, tid))
            else:
                new.append(i)

        for i in new:
            raw = outputs[i][1]
            best, best_cost = None, None
            for tid, tr in self.tracks.items():
                if tid in active:
                    continue
                c = self._cost(tid, tr, centers[i], whs[i], embs[i])
                if c is not None and (best_cost is None or c < best_cost):
                    best, best_cost = tid, c
            if best is None:
                best = self.next_id
                self.next_id += 1
            else:
                self.relinks.append((self.frame, best, raw, self.frame - self.tracks[best]["last_frame"]))
            self.raw_to_id[raw] = best
            active.add(best)
            results.append((i, best))

        for i, tid in results:
            self._update_track(tid, centers[i], whs[i], embs[i])
        if self.target_enabled:
            self._select_target([tid for _, tid in results])
        for tid in list(self.tracks):
            limit = self.t_max_gap if tid == self.target else self.max_gap
            if self.frame - self.tracks[tid]["last_frame"] > limit:
                del self.tracks[tid]
        return [[*boxes[i], self._display_id(tid)] for i, tid in results]


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

        # Capture the camera-motion transform BoT-SORT computes each frame,
        # so the re-linker can move lost tracks with the camera for free.
        self.last_warp = np.eye(2, 3)
        gmc_apply = self.tracker.gmc.apply

        def apply_and_record(raw_frame, dets=None):
            self.last_warp = gmc_apply(raw_frame, dets)
            return self.last_warp

        self.tracker.gmc.apply = apply_and_record
        relink_cfg = cfg.get("relink", {})
        self.relinker = _Relinker(relink_cfg, fps) if relink_cfg.get("enabled", False) else None

    def update(self, detections, frame):
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 5)
        if self.box_scale != 1.0 and len(dets):
            dets = np.column_stack([_scale_boxes(dets[:, :4], self.box_scale), dets[:, 4]])
        self.last_warp = np.eye(2, 3)
        out = self.tracker.update(_Detections(dets), frame)
        # rows: x1, y1, x2, y2, track_id, score, cls, det_index
        reported = []
        for row in out:
            track_id = int(row[4])
            self.hits[track_id] = self.hits.get(track_id, 0) + 1
            if self.hits[track_id] >= self.min_hits:
                box = _scale_boxes(row[:4], 1.0 / self.box_scale)[0]
                reported.append((box, track_id))
        if self.relinker is not None:
            return self.relinker.step(np.asarray(self.last_warp, dtype=float), reported, frame)
        return [[*box, track_id] for box, track_id in reported]


def create_tracker(tracker_cfg, fps):
    tracker_type = tracker_cfg.get("type", "deepsort")
    if tracker_type == "botsort":
        return BoTSORTTracker(tracker_cfg["botsort"], fps)
    if tracker_type == "deepsort":
        return DeepSortTracker(tracker_cfg["deepsort"])
    raise ValueError(f"Unknown tracker type: {tracker_type}")

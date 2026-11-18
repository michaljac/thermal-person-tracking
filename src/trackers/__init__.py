"""
Trackers behind one interface:

    tracker = create_tracker(tracker_cfg, fps)
    tracks = tracker.update(detections, frame)   # detections: Nx5 [x1, y1, x2, y2, conf]
    # -> [[x1, y1, x2, y2, id], ...] for tracks matched in this frame

"botsort" (default) — "detect everyone, track everyone, follow one":
    1. BoT-SORT (Ultralytics): Kalman prediction corrected by CAMERA-MOTION
       compensation, two-stage matching (low-score boxes may only extend
       tracks), plus buffered IoU for tiny boxes.            -> BoTSORTTracker
    2. Re-linking: a lost track's "ghost" keeps moving with the camera; a new
       track that fits it (distance, direction, size, embedding) keeps the
       old ID — e.g. a person coming out of a bush.          -> _Relinker
    3. Target selection: the walking person is always shown as ID 1;
       ID 0 = tentative track ("?"), others = 2, 3, ...     -> _Relinker._select_target

"deepsort" — previous tracker (Kalman + MobileNetV2 appearance + IoU),
kept for comparison runs.                                    -> DeepSortTracker
"""
from types import SimpleNamespace

import numpy as np


def create_tracker(tracker_cfg, fps):
    """Build the tracker selected by tracker_cfg["type"]."""
    tracker_type = tracker_cfg.get("type", "deepsort")
    if tracker_type == "botsort":
        return BoTSORTTracker(tracker_cfg["botsort"], fps)
    if tracker_type == "deepsort":
        return DeepSortTracker(tracker_cfg["deepsort"])
    raise ValueError(f"Unknown tracker type: {tracker_type}")


# ---------------------------------------------------------------------------
# DeepSORT (comparison baseline)
# ---------------------------------------------------------------------------
class DeepSortTracker:
    """Thin wrapper around deep_sort_realtime.DeepSort with the common interface."""

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


# ---------------------------------------------------------------------------
# BoT-SORT (default)
# ---------------------------------------------------------------------------
def _scale_boxes(xyxy, factor):
    """Scale boxes around their centers (factor > 1 enlarges)."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    half = (xyxy[:, 2:] - xyxy[:, :2]) / 2 * factor
    return np.concatenate([center - half, center + half], axis=1)


class _Detections:
    """Minimal stand-in for an Ultralytics Results.boxes object (what BOTSORT expects)."""

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
    """
    Ultralytics BoT-SORT plus three additions for tiny targets from a drone:
        - buffered IoU: boxes enlarged for matching only
        - min_hits: a track is reported only after a few matches
        - re-linking + target selection (_Relinker), using the camera-motion
          transform that BoT-SORT already computes each frame
    """

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
            # BoT-SORT's own ReID is off (useless on 10-20 px blobs); these
            # values are required by its constructor but not used.
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        self.tracker = BOTSORT(args, frame_rate=int(round(fps)))
        # Camera motion is estimated on a downscaled frame: global motion
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

        # Record the camera-motion transform BoT-SORT computes each frame, so
        # the re-linker can move lost tracks with the camera at no extra cost.
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

        self.last_warp = np.eye(2, 3)  # identity if camera compensation is off
        out = self.tracker.update(_Detections(dets), frame)

        # out rows: x1, y1, x2, y2, track_id, score, cls, det_index
        reported = []
        for row in out:
            track_id = int(row[4])
            self.hits[track_id] = self.hits.get(track_id, 0) + 1
            if self.hits[track_id] >= self.min_hits:
                box = _scale_boxes(row[:4], 1.0 / self.box_scale)[0]  # undo the buffer
                reported.append((box, track_id))

        if self.relinker is not None:
            return self.relinker.step(np.asarray(self.last_warp, dtype=float), reported, frame)
        return [[*box, track_id] for box, track_id in reported]


# ---------------------------------------------------------------------------
# Re-linking across long gaps + target selection (ID 1)
# ---------------------------------------------------------------------------
class _Relinker:
    """
    Sits on top of BoT-SORT's short-term tracks and assigns the final IDs.

    Re-linking: a NEW track can take over a recently LOST one (person went
    behind a bush, or the detector missed them for a while). Each lost track
    keeps a "ghost", moved every frame by (1) the camera motion and (2) the
    person's own velocity, which slowly decays (people often stop behind
    cover). A new track takes over the ghost's ID if it passes all gates:
        distance to the ghost   <= gate_px + growth * gap (uncertainty grows)
        direction               consistent with the walking direction
        box size                similar
        appearance (optional)   MobileNetV2 embedding close enough (a guard
                                against merging different people)
    Among candidates, the lowest combined cost wins.

    Target selection: see _select_target.

    Coordinates: all positions are in the CURRENT image; every frame the
    camera-motion transform moves stored positions into the new image.
    """

    def __init__(self, cfg, fps):
        # --- re-linking gates ---
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

        # --- target selection (the walking person -> ID 1) ---
        tcfg = cfg.get("target", {})
        self.target_enabled = bool(tcfg.get("enabled", False))
        self.t_min_hits = int(tcfg.get("min_hits", 10))
        self.t_min_speed = float(tcfg.get("min_speed_px_s", 40)) / fps
        self.t_lock_score = float(tcfg.get("lock_score", 1.0))
        max_center = tcfg.get("max_center_dist")  # None = no hard center gate
        self.t_max_center = float(max_center) if max_center is not None else float("inf")
        self.t_max_speed = float(tcfg.get("max_speed_px_s", 400)) / fps
        self.t_reacquire = int(float(tcfg.get("reacquire_after_s", 3)) * fps)
        # The camera follows the target, so right after a loss it is expected
        # near its last IMAGE position (not where the ground moved).
        self.t_gate_px = float(tcfg.get("gate_px", 150))
        self.t_gate_growth = float(tcfg.get("gate_growth_px_per_s", 60)) / fps
        self.t_max_gap = int(float(tcfg.get("max_gap_s", 15)) * fps)

        # --- state ---
        self.raw_to_id = {}  # BoT-SORT track id -> internal id
        self.next_id = 1
        self.tracks = {}  # internal id -> state dict (see _update_track)
        self.frame = 0
        self.frame_size = None
        self.target = None  # internal id of the target
        self.display = {}  # internal id -> displayed id (1 is reserved for the target)
        self.next_display = 2 if self.target_enabled else 1

    # ---- per-track bookkeeping -----------------------------------------
    def _embed(self, frame, boxes):
        """L2-normalized MobileNetV2 embeddings of crops 2x the box size (some context)."""
        crops = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1, 16)
            a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
            c, d = int(min(cx + half, w)), int(min(cy + half, h))
            crops.append(frame[b:d, a:c])
        feats = self.embedder.predict(crops)
        return [f / (np.linalg.norm(f) + 1e-9) for f in feats]

    def _warp(self, H):
        """Move stored positions/velocities into the current image (camera motion)."""
        A, t = H[:, :2], H[:, 2]
        for tr in self.tracks.values():
            tr["hist"] = [(f, A @ p + t) for f, p in tr["hist"]]
            tr["pred"] = A @ tr["pred"] + t
            tr["vel"] = A @ tr["vel"]

    def _update_track(self, tid, center, wh, emb):
        """Update a matched track: history, velocity, ground speed, center distance, embedding."""
        tr = self.tracks.setdefault(tid, {"hist": [], "vel": np.zeros(2), "emb": None,
                                          "hits": 0, "speed_ema": 0.0, "center_ema": None})
        tr["hist"] = (tr["hist"] + [(self.frame, center)])[-10:]  # last 10 positions
        tr["wh"], tr["last_frame"], tr["pred"] = wh, self.frame, center.copy()
        tr["pred_img"] = center.copy()  # last image position, NOT moved with the camera
        tr["hits"] += 1
        # Distance from the image center: 0 = center, 1 = edge (smoothed)
        h, w = self.frame_size
        cdist = float(np.linalg.norm((center - [w / 2, h / 2]) / [w / 2, h / 2]))
        tr["center_ema"] = cdist if tr["center_ema"] is None else 0.9 * tr["center_ema"] + 0.1 * cdist
        # Velocity over the history; camera motion is already removed by _warp,
        # so this is the speed relative to the GROUND (bystanders ~0).
        (f0, p0), (f1, p1) = tr["hist"][0], tr["hist"][-1]
        if f1 - f0 >= 2:
            tr["vel"] = (p1 - p0) / (f1 - f0)  # px/frame
            speed = float(np.linalg.norm(tr["vel"]))
            first = tr.setdefault("speed_n", 0) == 0
            tr["speed_ema"] = speed if first else 0.9 * tr["speed_ema"] + 0.1 * speed
            tr["speed_n"] += 1
        if emb is not None:  # slowly updated appearance template
            tr["emb"] = emb if tr["emb"] is None else 0.8 * tr["emb"] + 0.2 * emb
            tr["emb"] /= np.linalg.norm(tr["emb"]) + 1e-9

    # ---- re-linking ------------------------------------------------------
    def _cost(self, tr, center, wh, emb):
        """Cost of linking a new track to lost track `tr`, or None if any gate fails."""
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
            app = 1.0 - float(emb @ tr["emb"])  # cosine distance
            if app > self.max_app:
                return None
            cost += self.w_app * app
        return cost

    # ---- target selection ------------------------------------------------
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
        Lock the target (ID 1), or re-acquire it after it was lost.

        Score = persistence + near image center + ground speed (each 0..1);
        a candidate needs >= min_hits (so its ground speed is known) and a
        score >= lock_score. A static object scores ~0.7, the walking target
        ~1.1+ even while it starts slowly.

        Re-acquisition (a target existed and is lost) is stricter: the
        candidate must be WALKING, and for the first reacquire_after_s it must
        also be near the target's last image position (the camera follows the
        target). If nobody qualifies, ID 1 stays empty instead of going to a
        guess.
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
            if tr["center_ema"] > self.t_max_center:
                continue  # optional hard center gate (off by default)
            if tr["speed_ema"] > self.t_max_speed:
                continue  # implausibly fast for a walking person
            if gap is not None and tr["speed_ema"] < self.t_min_speed:
                continue  # re-acquisition must be a WALKING person (first lock may be slow)
            if gap is not None and gap <= self.t_reacquire:
                center = tr["hist"][-1][1]
                gate = self.t_gate_px + self.t_gate_growth * gap
                if np.linalg.norm(center - current["pred_img"]) > gate:
                    continue
            score = (min(tr["hits"] / 60.0, 1.0)                            # persistence
                     + (1.0 - min(tr["center_ema"], 1.0))                   # near center
                     + min(tr["speed_ema"] / (3 * self.t_min_speed), 1.0))  # ground speed
            if score < self.t_lock_score:
                continue
            if best_score is None or score > best_score:
                best, best_score = tid, score
        if best is not None:
            if self.target is not None and self.target in self.tracks:
                self.display[self.target] = self.next_display  # old target loses ID 1
                self.next_display += 1
            self.target = best

    # ---- main step ---------------------------------------------------------
    def step(self, H, outputs, frame):
        """
        H: 2x3 camera-motion transform (previous -> current frame).
        outputs: list of (box[4], BoT-SORT track id) reported this frame.
        Returns [[x1, y1, x2, y2, display_id], ...].
        """
        self.frame += 1
        self.frame_size = frame.shape[:2]

        # 1. Move everything with the camera; lost ghosts keep walking (decaying)
        self._warp(H)
        for tr in self.tracks.values():
            if tr["last_frame"] < self.frame - 1:
                tr["pred"] = tr["pred"] + tr["vel"]
                tr["vel"] = tr["vel"] * self.decay

        boxes = [np.asarray(b, dtype=float) for b, _ in outputs]
        centers = [(b[:2] + b[2:]) / 2 for b in boxes]
        whs = [b[2:] - b[:2] for b in boxes]

        # 2. Embeddings only for new tracks and every embed_every frames (CPU cost)
        embs = [None] * len(outputs)
        if self.embedder is not None:
            idx = [i for i, (_, raw) in enumerate(outputs)
                   if raw not in self.raw_to_id or self.frame % self.embed_every == 0]
            if idx:
                for i, e in zip(idx, self._embed(frame, [boxes[i] for i in idx])):
                    embs[i] = e

        # 3. Known BoT-SORT tracks keep their internal id
        active, results, new = set(), [], []
        for i, (_, raw) in enumerate(outputs):
            tid = self.raw_to_id.get(raw)
            if tid is not None and tid not in active:
                active.add(tid)
                results.append((i, tid))
            else:
                new.append(i)

        # 4. New tracks: re-link to the best-fitting lost ghost, else a new id
        for i in new:
            raw = outputs[i][1]
            best, best_cost = None, None
            for tid, tr in self.tracks.items():
                if tid in active:
                    continue
                c = self._cost(tr, centers[i], whs[i], embs[i])
                if c is not None and (best_cost is None or c < best_cost):
                    best, best_cost = tid, c
            if best is None:
                best = self.next_id
                self.next_id += 1
            self.raw_to_id[raw] = best
            active.add(best)
            results.append((i, best))

        # 5. Update matched tracks, choose the target, forget old ghosts
        for i, tid in results:
            self._update_track(tid, centers[i], whs[i], embs[i])
        if self.target_enabled:
            self._select_target([tid for _, tid in results])
        for tid in list(self.tracks):
            limit = self.t_max_gap if tid == self.target else self.max_gap
            if self.frame - self.tracks[tid]["last_frame"] > limit:
                del self.tracks[tid]

        return [[*boxes[i], self._display_id(tid)] for i, tid in results]

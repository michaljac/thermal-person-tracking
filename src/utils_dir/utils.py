import os

import cv2


def build_output_video_path(output_dir, model_path, image_size, confidence,
                            max_frames=None):
    """
    Build an output video path from the model name, image size and confidence.
    Test runs (max_frames set) get a "_test<N>" suffix so they never
    overwrite a full-video result.

    Example:
        models/best_uav_fast.onnx, 512, 0.5
        -> <output_dir>/best_uav_fast_onnx_img512_conf0.5.mp4
    """
    model_stem, extension = os.path.splitext(os.path.basename(model_path))
    backend = extension.lstrip(".")
    if isinstance(image_size, (list, tuple)):
        image_size = "x".join(str(s) for s in image_size)  # [640, 384] -> 640x384
    # Show the image size once: skip "_img<size>" if the model name already has it
    size_part = "" if str(image_size) in model_stem else f"_img{image_size}"
    filename = f"{model_stem}_{backend}{size_part}_conf{confidence}"
    if max_frames is not None:
        filename += f"_test{max_frames}"
    return os.path.join(output_dir, filename + ".mp4")


def save_tracks_mot(path, track_log):
    """
    Save tracks in MOTChallenge text format:
    frame, id, left, top, width, height, conf, x, y, z  (one row per box)
    """
    with open(path, "w", encoding="utf-8") as file:
        for frame, track_id, left, top, width, height in track_log:
            file.write(
                f"{frame},{track_id},{left:.1f},{top:.1f},"
                f"{width:.1f},{height:.1f},1,-1,-1,-1\n"
            )


def compute_track_stats(track_log, total_frames, fps, short_track_s=0.5):
    """
    Label-free tracking metrics from a track log [(frame, id, ...), ...].

    unique_ids:      number of distinct IDs (fewer = less fragmentation)
    mean_track_s:    mean time span of a track, first to last frame (s)
    short_tracks:    share of IDs living < short_track_s (likely false alarms or fragments)
    gaps_recovered:  times a track disappeared and came back with the SAME ID
                     (identity kept through a miss / occlusion)
    coverage:        share of frames with at least one confirmed track
    """
    spans = {}
    frames_by_id = {}
    for frame, track_id, *_ in track_log:
        frames_by_id.setdefault(track_id, []).append(frame)
    for track_id, frames in frames_by_id.items():
        frames.sort()
        spans[track_id] = frames
    if not spans:
        return {"unique_ids": 0, "mean_track_s": 0.0, "short_tracks": 0.0,
                "gaps_recovered": 0, "coverage": 0.0, "id1_frames": 0, "id1_span_s": 0.0}

    lengths_s = [(f[-1] - f[0] + 1) / fps for f in spans.values()]
    gaps = sum(
        sum(1 for a, b in zip(f, f[1:]) if b - a > 1) for f in spans.values()
    )
    covered = len({frame for frame, *_ in track_log})
    target_frames = spans.get(1, [])
    return {
        "unique_ids": len(spans),
        "mean_track_s": sum(lengths_s) / len(lengths_s),
        "short_tracks": sum(l < short_track_s for l in lengths_s) / len(lengths_s),
        "gaps_recovered": gaps,
        "coverage": covered / max(total_frames, 1),
        # Target (ID 1): frames where it is shown, and its first-to-last span
        "id1_frames": len(target_frames),
        "id1_span_s": (target_frames[-1] - target_frames[0] + 1) / fps if target_frames else 0.0,
    }


def load_tracks_mot(path):
    """Read a MOT track log back into [(frame, id, left, top, width, height), ...]."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            v = line.strip().split(",")
            if len(v) >= 6:
                rows.append((int(v[0]), int(v[1]), *map(float, v[2:6])))
    return rows


def get_max_frames(run_cfg, max_frames=None):
    """
    Decide how many frames to process.

    Priority: explicit `max_frames` argument > config test mode > whole video.
    Returns an int, or None for the whole video.
    """
    if max_frames is not None:
        return max_frames
    if run_cfg.get("test_mode", False):
        return int(run_cfg.get("test_num_frames", 50))
    return None


def draw_detections(frame, detections, vis_cfg):
    """
    Draw person detections on a frame.

    detections: Nx5 array
        [x1, y1, x2, y2, confidence]

    Returns:
        Annotated frame.
    """
    annotated_frame = frame.copy()

    # OpenCV uses BGR, so red = (0, 0, 255)
    bbox_color = (0, 0, 255)

    for detection in detections:
        x1, y1, x2, y2, confidence = detection

        x1, y1, x2, y2 = map(
            int, (x1, y1, x2, y2)
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            bbox_color,
            vis_cfg["bbox_thickness"]
        )

        cv2.putText(
            annotated_frame,
            f"Person {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            vis_cfg["font_scale"],
            bbox_color,
            vis_cfg["text_thickness"]
        )

    return annotated_frame


def draw_tracks(frame, tracks):
    """
    Draw confirmed tracks on a frame.

    tracks: list of [x1, y1, x2, y2, track_id]

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2, track_id = map(
            int, [x1, y1, x2, y2, track_id]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"ID {track_id}" if track_id > 0 else "?",  # 0 = tentative track
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated

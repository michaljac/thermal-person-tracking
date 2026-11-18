"""
Small helpers for the pipeline in detect.py:

    Run setup      get_max_frames, build_output_video_path
    Drawing        draw_detections (red: detector boxes), draw_tracks (green: track IDs)
    Track log      save_tracks_mot (MOTChallenge text format)
    Metrics        compute_track_stats (label-free tracking metrics)
"""
import os

import cv2


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------
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


def build_output_video_path(output_dir, model_path, image_size, confidence,
                            max_frames=None):
    """
    Build the output video path from the model name, image size and confidence.

    The image size appears once: it is skipped if the model name already has it.
    Test runs (max_frames set) get a "_test<N>" suffix so they never overwrite
    a full-video result.

    Examples:
        models/best_uav_fast_640x384.onnx, [640, 384], 0.1
            -> <output_dir>/best_uav_fast_640x384_onnx_conf0.1.mp4
        models/best_uav_fast.onnx, 512, 0.3
            -> <output_dir>/best_uav_fast_onnx_img512_conf0.3.mp4
    """
    model_stem, extension = os.path.splitext(os.path.basename(model_path))
    backend = extension.lstrip(".")
    if isinstance(image_size, (list, tuple)):
        image_size = "x".join(str(s) for s in image_size)  # [640, 384] -> 640x384
    size_part = "" if str(image_size) in model_stem else f"_img{image_size}"
    filename = f"{model_stem}_{backend}{size_part}_conf{confidence}"
    if max_frames is not None:
        filename += f"_test{max_frames}"
    return os.path.join(output_dir, filename + ".mp4")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(frame, detections, vis_cfg):
    """
    Draw raw detector boxes (red) with their confidence.

    detections: Nx5 [x1, y1, x2, y2, confidence]
    Returns a new annotated frame.
    """
    annotated = frame.copy()
    color = (0, 0, 255)  # OpenCV uses BGR: red

    for x1, y1, x2, y2, confidence in detections:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, vis_cfg["bbox_thickness"])
        cv2.putText(
            annotated, f"Person {confidence:.2f}", (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, vis_cfg["font_scale"], color, vis_cfg["text_thickness"],
        )
    return annotated


def draw_tracks(frame, tracks):
    """
    Draw tracks (green) with their ID. ID 1 is the target; ID 0 is a
    tentative track (not enough evidence yet) and is labelled "?".

    tracks: list of [x1, y1, x2, y2, track_id]
    Returns a new annotated frame.
    """
    annotated = frame.copy()
    color = (0, 255, 0)  # green

    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2, track_id = map(int, (x1, y1, x2, y2, track_id))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}" if track_id > 0 else "?"
        cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


# ---------------------------------------------------------------------------
# Track log and metrics
# ---------------------------------------------------------------------------
def save_tracks_mot(path, track_log):
    """
    Save tracks in MOTChallenge text format, one row per box:
        frame, id, left, top, width, height, conf, x, y, z
    (conf/x/y/z are unused placeholders: 1, -1, -1, -1). This is the format
    standard evaluation tools (e.g. TrackEval) read.
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
    (No ground truth exists for the video, so these are proxies.)

    unique_ids:      number of distinct IDs (fewer = less fragmentation)
    mean_track_s:    mean time span of a track, first to last frame (s)
    short_tracks:    share of IDs living < short_track_s (false alarms or fragments)
    gaps_recovered:  times a track disappeared and came back with the SAME ID
    coverage:        share of frames with at least one reported track
    id1_frames:      frames where the target (ID 1) is shown
    id1_span_s:      time from the target's first to last appearance (s)
    """
    frames_by_id = {}
    for frame, track_id, *_ in track_log:
        frames_by_id.setdefault(track_id, []).append(frame)
    if not frames_by_id:
        return {"unique_ids": 0, "mean_track_s": 0.0, "short_tracks": 0.0,
                "gaps_recovered": 0, "coverage": 0.0, "id1_frames": 0, "id1_span_s": 0.0}

    for frames in frames_by_id.values():
        frames.sort()
    lengths_s = [(f[-1] - f[0] + 1) / fps for f in frames_by_id.values()]
    gaps = sum(sum(1 for a, b in zip(f, f[1:]) if b - a > 1) for f in frames_by_id.values())
    covered = len({frame for frame, *_ in track_log})
    target = frames_by_id.get(1, [])
    return {
        "unique_ids": len(frames_by_id),
        "mean_track_s": sum(lengths_s) / len(lengths_s),
        "short_tracks": sum(l < short_track_s for l in lengths_s) / len(lengths_s),
        "gaps_recovered": gaps,
        "coverage": covered / max(total_frames, 1),
        "id1_frames": len(target),
        "id1_span_s": (target[-1] - target[0] + 1) / fps if target else 0.0,
    }

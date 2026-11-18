"""
Thermal person tracking — main pipeline.

Per frame:
    read frame -> (optional) pre-processing -> detector -> tracker -> draw -> write video

The tracker ("botsort", see trackers/) follows one target of interest, shown as
ID 1; other people are tracked as distractors (IDs 2, 3, ...).

Outputs (in paths.output_dir):
    <model>_<backend>[_img<size>]_conf<conf>[_test<N>]_<tracker>.mp4   annotated video
    ..._tracks.txt                                                     MOT-format track log
Console: per-stage timing report, FPS, and label-free tracking metrics.

Run from the project root:  python src/detect.py
All settings are in config/config.yaml.
"""
import os
import time

import cv2
import yaml
from tqdm import tqdm

from detectors.factory import create_detector
from preprocess import ThermalPreprocessor
from time_manager import TimeManager
from trackers import create_tracker
from utils_dir.utils import (
    build_output_video_path,
    compute_track_stats,
    draw_detections,
    draw_tracks,
    get_max_frames,
    save_tracks_mot,
)


def main(config_path=os.path.join("config", "config.yaml"), model_override=None,
         max_frames=None, benchmark=False, warmup_override=None):
    """
    Run the full pipeline on the configured video.

    model_override / max_frames / benchmark / warmup_override let other scripts
    (e.g. src/compare_backends.py) reuse the pipeline without editing the config.
    Returns the per-stage timing statistics.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if model_override is not None:
        config["paths"]["model"] = model_override
    if benchmark:
        config.setdefault("profiling", {})["enabled"] = True
    if warmup_override is not None:
        config.setdefault("profiling", {})["warmup_frames"] = warmup_override

    paths = config["paths"]
    detector_cfg = config["detector"]
    tracker_cfg = config["tracker"]
    video_cfg = config["video"]
    vis_cfg = config["visualization"]
    profile_cfg = config.get("profiling", {})
    max_frames = get_max_frames(config.get("run", {}), max_frames)  # None = whole video

    os.makedirs(paths["output_dir"], exist_ok=True)

    timer = TimeManager(
        enabled=profile_cfg.get("enabled", False),
        warmup_frames=profile_cfg.get("warmup_frames", 0),
    )

    # BoT-SORT's second matching stage uses low-score detections, so the
    # detector must output everything down to track_low_thresh. Only boxes
    # above track_high_thresh are drawn, to keep the video readable.
    tracker_type = tracker_cfg.get("type", "deepsort")
    draw_min_conf = 0.0
    if tracker_type == "botsort":
        low = tracker_cfg["botsort"]["track_low_thresh"]
        draw_min_conf = tracker_cfg["botsort"]["track_high_thresh"]
        for key in ("confidence_threshold", "hae-s-ir_confidence_threshold"):
            detector_cfg[key] = min(detector_cfg[key], low)

    # ------------------------------------------------------------------
    # Pipeline blocks
    # ------------------------------------------------------------------
    detector = create_detector(model_path=paths["model"], detector_cfg=detector_cfg)
    detector.timer = timer  # HAE-S-IR reports its internal timings through it

    preprocessor = ThermalPreprocessor(config.get("preprocessing", {}))  # mode "none" = no-op

    output_video = build_output_video_path(
        paths["output_dir"],
        paths["model"],
        detector_cfg["image_size"],
        detector_cfg["confidence_threshold"],
        max_frames,
    )
    # Tracker name in the filename, so DeepSORT and BoT-SORT runs don't overwrite each other
    stem, ext = os.path.splitext(output_video)
    output_video = f"{stem}_{tracker_type}{ext}"

    # ------------------------------------------------------------------
    # Video input / output
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(paths["input_video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {paths['input_video']}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = video_cfg["default_fps"]

    # The tracker needs the FPS: its time limits are given in seconds
    tracker = create_tracker(tracker_cfg, fps)
    print(f"Tracker: {tracker_type}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*video_cfg["output_codec"]), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    total_frames = video_frames
    if max_frames is not None:
        total_frames = min(max_frames, video_frames) if video_frames else max_frames
        print(f"Test mode: processing {total_frames} frames")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    track_log = []  # MOT rows: (frame, id, left, top, width, height)
    frame_count = 0
    total_detections = 0
    progress = tqdm(total=total_frames, desc="Tracking", unit="frame")
    start_time = time.perf_counter()

    try:
        while max_frames is None or frame_count < max_frames:
            with timer.measure("total_frame"):
                with timer.measure("video_read"):
                    success, frame = cap.read()
                if not success:
                    break

                with timer.measure("preprocess"):
                    model_input = preprocessor.process(frame)

                with timer.measure("detection"):
                    detections = detector.predict(model_input)  # Nx5: x1, y1, x2, y2, conf
                total_detections += len(detections)

                with timer.measure("tracking"):
                    tracks = tracker.update(detections, frame)  # [x1, y1, x2, y2, id]
                    for x1, y1, x2, y2, track_id in tracks:
                        if track_id == 0:
                            continue  # tentative track (not enough evidence yet): not logged
                        track_log.append((frame_count + 1, track_id, x1, y1, x2 - x1, y2 - y1))

                with timer.measure("drawing"):
                    shown = [d for d in detections if d[4] >= draw_min_conf]
                    annotated_frame = draw_detections(frame, shown, vis_cfg)
                    annotated_frame = draw_tracks(annotated_frame, tracks)

                with timer.measure("video_write"):
                    writer.write(annotated_frame)

                frame_count += 1
                progress.update(1)
                progress.set_postfix(detections=total_detections, tracks=len(tracks))
            timer.next_frame()

    finally:
        progress.close()
        cap.release()
        writer.release()
        elapsed = time.perf_counter() - start_time

        # The failed end-of-video read is not a processed frame.
        for stage in ("total_frame", "video_read"):
            if timer.timings[stage] and timer.timings[stage][-1][0] == timer.frame_index:
                timer.timings[stage].pop()
        if timer.enabled:
            timer.print_report()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Processing FPS: {frame_count / elapsed:.2f}")

    track_log_path = os.path.splitext(output_video)[0] + "_tracks.txt"
    save_tracks_mot(track_log_path, track_log)

    # Label-free metrics (no ground truth available), see utils.compute_track_stats
    track_stats = compute_track_stats(track_log, frame_count, fps)
    print(
        f"Tracks: {track_stats['unique_ids']} IDs | "
        f"mean length {track_stats['mean_track_s']:.1f}s | "
        f"short (<0.5s) {track_stats['short_tracks']:.0%} | "
        f"gaps recovered {track_stats['gaps_recovered']} | "
        f"coverage {track_stats['coverage']:.0%}"
    )
    print(
        f"Target (ID 1): shown in {track_stats['id1_frames']} frames | "
        f"span {track_stats['id1_span_s']:.1f}s"
    )
    print(f"Output video saved to: {os.path.abspath(output_video)}")
    print(f"Track log saved to: {os.path.abspath(track_log_path)}")
    return timer.get_stats()


if __name__ == "__main__":
    main()

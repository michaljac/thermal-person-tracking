
import os
import time
import numpy as np

import cv2
import yaml
from tqdm import tqdm
from time_manager import TimeManager
from preprocess import ThermalPreprocessor
from detectors.factory import create_detector
from trackers import create_tracker
from utils_dir.utils import *

def main(config_path=os.path.join("config", "config.yaml"), model_override=None,
         max_frames=None, benchmark=False, warmup_override=None):
    
    # Load configuration
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
    video_cfg = config["video"]
    vis_cfg = config["visualization"]
    profile_cfg = config.get("profiling", {})
    max_frames = get_max_frames(config.get("run", {}), max_frames)
    model_path = paths["model"]
    input_video = paths["input_video"]

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Profiling timer
    timer = TimeManager(
        enabled=profile_cfg.get("enabled", False),
        warmup_frames=profile_cfg.get("warmup_frames", 0),
    )

    if benchmark:
        # Get model filename without extension, plus backend suffix
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_name += "_" + os.path.splitext(model_path)[1].lstrip(".")

    # BoT-SORT's second stage needs low-score detections, so the detector must
    # pass everything down to track_low_thresh. Only confident ones are drawn.
    tracker_cfg = config["tracker"]
    tracker_type = tracker_cfg.get("type", "deepsort")
    draw_min_conf = 0.0
    if tracker_type == "botsort":
        low = tracker_cfg["botsort"]["track_low_thresh"]
        draw_min_conf = tracker_cfg["botsort"]["track_high_thresh"]
        for key in ("confidence_threshold", "hae-s-ir_confidence_threshold"):
            detector_cfg[key] = min(detector_cfg[key], low)

    # Initialize detector
    detector = create_detector(
        model_path=model_path,
        detector_cfg=detector_cfg
    )
    # Build output filename
    output_video = build_output_video_path(
        output_dir,
        model_path,
        detector_cfg["image_size"],
        detector_cfg["confidence_threshold"],
        max_frames,
    )
    # Tracker name in the filename, so DeepSORT and BoT-SORT runs don't overwrite each other
    stem, ext = os.path.splitext(output_video)
    output_video = f"{stem}_{tracker_type}{ext}"

    # Share the timer for detailed HAE preprocessing/inference timings.
    detector.timer = timer

    # Optional thermal contrast enhancement (mode "none" = no-op)
    preprocessor = ThermalPreprocessor(config.get("preprocessing", {}))

    track_log = []  # MOT format rows: frame, id, left, top, width, height

    # Open input video
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {input_video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = video_cfg["default_fps"]

    # Initialize tracker (needs fps: lost-track buffer is defined at 30 FPS)
    tracker = create_tracker(tracker_cfg, fps)
    print(f"Tracker: {tracker_type}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(
        *video_cfg["output_codec"]
    )

    writer = cv2.VideoWriter(
        output_video,
        codec,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create output video: {output_video}"
        )

    frame_count = 0
    total_detections = 0

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    total_frames = video_frames
    if max_frames is not None:
        total_frames = min(max_frames, video_frames) if video_frames else max_frames
        print(f"Test mode: processing {total_frames} frames")
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
                    detections = detector.predict(model_input)
                total_detections += len(detections)

                with timer.measure("tracking"):
                    tracks = tracker.update(detections, frame)
                    for x1, y1, x2, y2, track_id in tracks:
                        if track_id == 0:
                            continue  # tentative track (not enough evidence yet)
                        track_log.append(
                            (frame_count + 1, track_id, x1, y1, x2 - x1, y2 - y1)
                        )

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

    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {frame_count / elapsed:.2f}"
        )

    track_log_path = os.path.splitext(output_video)[0] + "_tracks.txt"
    save_tracks_mot(track_log_path, track_log)

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

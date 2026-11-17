
import os
import time
import numpy as np

import cv2
import yaml
from tqdm import tqdm
from time_manager import TimeManager
from preprocess import ThermalPreprocessor
from detectors.factory import create_detector
# from deep_sort.deep_sort.tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils_dir.utils import *
# from deep_sort.deep_sort.tracker import DeepSortTracker

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

    # Share the timer for detailed HAE preprocessing/inference timings.
    detector.timer = timer

    # Optional thermal contrast enhancement (mode "none" = no-op)
    preprocessor = ThermalPreprocessor(config.get("preprocessing", {}))

    # Initalize tracker
    tracker_cfg = config["tracker"]
    tracker = DeepSort(
        max_age=tracker_cfg["max_age"],
        n_init=tracker_cfg["n_init"],
        max_iou_distance=tracker_cfg["max_iou_distance"],
        max_cosine_distance=tracker_cfg.get("max_cosine_distance", 0.2),
        nn_budget=tracker_cfg.get("nn_budget"),
        embedder_gpu=False,
    )
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
                    # DeepSort expects [left, top, width, height].
                    raw_detections = [
                        ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         float(confidence), "person")
                        for x1, y1, x2, y2, confidence in detections
                    ]
                    updated_tracks = tracker.update_tracks(raw_detections, frame=frame)
                    tracks = [
                        [*track.to_ltrb(), int(track.track_id)]
                        for track in updated_tracks
                        if track.is_confirmed() and track.time_since_update == 0
                    ]
                    for x1, y1, x2, y2, track_id in tracks:
                        track_log.append(
                            (frame_count + 1, track_id, x1, y1, x2 - x1, y2 - y1)
                        )

                with timer.measure("drawing"):
                    annotated_frame = draw_detections(frame, detections, vis_cfg)
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

    print(f"Output video saved to: {os.path.abspath(output_video)}")
    print(f"Track log saved to: {os.path.abspath(track_log_path)}")
    return timer.get_stats()


if __name__ == "__main__":
    main()

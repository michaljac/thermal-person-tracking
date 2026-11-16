
import os
import time
import numpy as np

import cv2
import yaml
from time_manager import TimeManager
from detectors.factory import create_detector
# from deep_sort.deep_sort.tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
# from deep_sort.deep_sort.tracker import DeepSortTracker

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
            f"ID {track_id}",
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated

def main():
    # Load configuration
    config_path = os.path.join("config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = config["paths"]
    detector_cfg = config["detector"]
    video_cfg = config["video"]
    vis_cfg = config["visualization"]
    log_cfg = config["logging"]
    profile_cfg = config.get("profiling", {})
    timer = TimeManager(
        enabled=profile_cfg.get("enabled", False),
        warmup_frames=profile_cfg.get("warmup_frames", 0),
    )
    model_path = paths["model"]
    model_name = os.path.basename(model_path).lower()
    input_video = paths["input_video"]
    output_dir = os.path.dirname(paths["output_video"])

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Get model filename without extension
    model_name = os.path.splitext(
        os.path.basename(paths["model"])
    )[0]

    # Generate output filename based on model
    output_video = os.path.join(
        output_dir,
        f"{model_name}.mp4"
    )

    # Initialize detector
    detector = create_detector(
        model_path=model_path,
        detector_cfg=detector_cfg
    )
    # Share the timer for detailed HAE preprocessing/inference timings.
    detector.timer = timer

    # Initalize tracker
    tracker = DeepSort(
        max_age=config["tracker"]["max_age"],
        n_init=config["tracker"]["n_init"],
        max_iou_distance=config["tracker"]["max_iou_distance"],
        embedder_gpu=False,
    )

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
    start_time = time.perf_counter()

    try:
        while True:
            with timer.measure("total_frame"):
                with timer.measure("video_read"):
                    success, frame = cap.read()

                if not success:
                    break

                with timer.measure("detection"):
                    detections = detector.predict(frame)
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

                with timer.measure("drawing"):
                    annotated_frame = draw_detections(frame, detections, vis_cfg)
                    annotated_frame = draw_tracks(annotated_frame, tracks)

                with timer.measure("video_write"):
                    writer.write(annotated_frame)

                frame_count += 1
                if frame_count % log_cfg["print_every_n_frames"] == 0:
                    print(
                        f"Processed {frame_count} frames | "
                        f"Detections: {total_detections}"
                    )
            timer.next_frame()

    finally:
        cap.release()
        writer.release()
        if timer.enabled:
            timer.print_report()
            timer.save_figures(
                profile_cfg.get("output_dir", "outputs/profiling"),
                dpi=profile_cfg.get("figure_dpi", 150),
            )

    elapsed = time.perf_counter() - start_time

    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {frame_count / elapsed:.2f}"
        )

    print(f"Output: {os.path.abspath(output_video)}")


if __name__ == "__main__":
    main()

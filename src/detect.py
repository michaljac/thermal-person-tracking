
import os
import time

import cv2
import yaml
from src.detectors.factory import create_detector

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
    model_path = paths["model"]
    model_name = os.path.basename(model_path).lower()
    input_video = paths["input_video"]
    output_video = os.path.join(
    output_dir,
    f"{model_name}.mp4"
)

    if not os.path.isfile(input_video):
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize detector once
    detector = create_detector(
        model_path=model_path,
        detector_cfg=detector_cfg
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
            success, frame = cap.read()

            if not success:
                break

            # 1. Detect persons
            detections = detector.predict(frame)
            total_detections += len(detections)

            # 2. Draw detections
            annotated_frame = draw_detections(
                frame,
                detections,
                vis_cfg
            )

            # 3. Save annotated frame
            writer.write(annotated_frame)

            frame_count += 1

            if frame_count % log_cfg["print_every_n_frames"] == 0:
                print(
                    f"Processed {frame_count} frames | "
                    f"Detections: {total_detections}"
                )

    finally:
        cap.release()
        writer.release()

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

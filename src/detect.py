
import os
import time
import cv2
import yaml
from ultralytics import YOLO

#from src.time_manager import TimeManager

def main():
    # Load configuration directly from YAML
    config_path = os.path.join("config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = config["paths"]
    detector_cfg = config["detector"]
    video_cfg = config["video"]
    vis_cfg = config["visualization"]
    log_cfg = config["logging"]
    profiling_cfg = config["profiling"]
    input_video = paths["input_video"]
    output_video = paths["output_video"]

    if not os.path.isfile(input_video):
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    os.makedirs(
        os.path.dirname(output_video),
        exist_ok=True
    )

#     profiler = TimeManager(
#     enabled = profiling_cfg["enabled"],
#     warmup_frames = profiling_cfg["warmup_frames"]
# )


    # Initialize YOLO detector
    model = YOLO(paths["model"])

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
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Detect persons using YOLO
            result = model.predict(
                frame,
                classes=detector_cfg["classes"],
                conf=detector_cfg["confidence_threshold"],
                imgsz=detector_cfg["image_size"],
                device=detector_cfg["device"],
                verbose=False
            )[0]

            # Draw bounding boxes
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    vis_cfg["bbox_color"],
                    vis_cfg["bbox_thickness"]
                )

                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    vis_cfg["font_scale"],
                    vis_cfg["bbox_color"],
                    vis_cfg["text_thickness"]
                )

            writer.write(frame)
            frame_count += 1
            # profiler.next_frame()

            if frame_count % log_cfg["print_every_n_frames"] == 0:
                print(f"Processed {frame_count} frames")

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start_time

    print(f"Total frames: {frame_count}")
    print(f"Total time: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {frame_count / elapsed:.2f}"
        )

    print(f"Output: {os.path.abspath(output_video)}")

    # profiler.print_report()

    # profiler.save_figures(
    #     output_dir=profiling_cfg["output_dir"],
    #     dpi=profiling_cfg["figure_dpi"]
    # )

if __name__ == "__main__":
    main()


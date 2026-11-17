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
    filename = f"{model_stem}_{backend}_img{image_size}_conf{confidence}"
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
            f"ID {track_id}",
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated

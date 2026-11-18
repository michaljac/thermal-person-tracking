import numpy as np
from ultralytics import YOLO

from detectors.base import BaseDetector


class YOLODetector(BaseDetector):
    """
    Person detector based on Ultralytics YOLO (.pt or .onnx).

    Default model: YOLOv8s fine-tuned on drone views of people, exported to
    ONNX at 640x384 (the portrait video's aspect ratio, no square padding).
    A static ONNX model must be run at exactly its export size (image_size).
    """

    def __init__(self, model_path, detector_cfg):
        self.model = YOLO(model_path)
        self.cfg = detector_cfg

    def predict(self, frame):
        result = self.model.predict(
            frame,
            classes=self.cfg["classes"],  # [0] = person
            conf=self.cfg["confidence_threshold"],
            imgsz=self.cfg["image_size"],
            device=self.cfg["device"],
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return np.column_stack((boxes, scores)).astype(np.float32)

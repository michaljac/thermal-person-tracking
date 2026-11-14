
import numpy as np
from ultralytics import YOLO

from src.detectors.base import BaseDetector


class YOLODetector(BaseDetector):

    def __init__(self, model_path, detector_cfg):
        self.model = YOLO(model_path)
        self.cfg = detector_cfg

    def predict(self, frame):
        result = self.model.predict(
            frame,
            classes=self.cfg["classes"],
            conf=self.cfg["confidence_threshold"],
            imgsz=self.cfg["image_size"],
            device=self.cfg["device"],
            verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        return np.column_stack((boxes, scores)).astype(
            np.float32
        )


import os

from src.detectors.yolo_detector import YOLODetector
from src.detectors.hae_sir_detector import HAESIRDetector


def create_detector(model_path, detector_cfg):

    filename = os.path.basename(model_path).lower()
    extension = os.path.splitext(filename)[1]

    if "hae-s-ir" in filename or "hae_s_ir" in filename:
        if extension != ".onnx":
            raise ValueError("HAE-S-IR requires an ONNX model")
        return HAESIRDetector(model_path, detector_cfg)

    elif extension in (".pt", ".onnx", ".engine"):
        return YOLODetector(model_path, detector_cfg)

    else:
        raise ValueError(
            f"Unsupported model: {model_path}"
        )

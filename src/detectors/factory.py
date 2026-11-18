import os

from detectors.hae_sir_detector import HAESIRDetector
from detectors.yolo_detector import YOLODetector


def create_detector(model_path, detector_cfg):
    """
    Pick the detector from the model filename:
        *hae-s-ir*.onnx         -> HAESIRDetector (uses the hae-s-ir_* settings)
        .pt / .onnx / .engine   -> YOLODetector
    """
    filename = os.path.basename(model_path).lower()
    extension = os.path.splitext(filename)[1]

    if "hae-s-ir" in filename or "hae_s_ir" in filename:
        if extension != ".onnx":
            raise ValueError("HAE-S-IR requires an ONNX model")
        # HAE-S-IR has its own threshold and size; copy them into the common keys
        detector_cfg["confidence_threshold"] = detector_cfg["hae-s-ir_confidence_threshold"]
        detector_cfg["image_size"] = detector_cfg["hae-s-ir_image_size"]
        return HAESIRDetector(model_path, detector_cfg)

    if extension in (".pt", ".onnx", ".engine"):
        return YOLODetector(model_path, detector_cfg)

    raise ValueError(f"Unsupported model: {model_path}")

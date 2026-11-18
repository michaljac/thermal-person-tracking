from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Common detector interface, so the pipeline can swap YOLO / HAE-S-IR."""

    @abstractmethod
    def predict(self, frame):
        """Return detections as an Nx5 float32 array: [x1, y1, x2, y2, confidence]."""

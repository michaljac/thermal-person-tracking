
from abc import ABC, abstractmethod


class BaseDetector(ABC):

    @abstractmethod
    def predict(self, frame):
        """Return detections as Nx5 array:
        [x1, y1, x2, y2, confidence]
        """
        pass

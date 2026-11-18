
import cv2


class ThermalPreprocessor:
    """
    Optional thermal contrast enhancement applied before detection.

    Off by default (mode "none"): on this video CLAHE halved detection recall
    (0.58 -> 0.30), because it brightens the background around the warm person
    and the detector was trained on unprocessed frames.

    params (config "preprocessing" section):
        mode: none | clahe | inverted | clahe_inverted
        clahe_clip_limit: float
        clahe_tile_grid_size: [w, h]
    """

    def __init__(self, params):
        self.mode = params.get("mode", "none")

        self.clahe = cv2.createCLAHE(
            clipLimit=params.get("clahe_clip_limit", 2.0),
            tileGridSize=tuple(
                params.get("clahe_tile_grid_size", [8, 8])
            )
        )

    def process(self, frame):
        if self.mode == "none":
            return frame

        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        if self.mode in ("clahe", "clahe_inverted"):
            gray = self.clahe.apply(gray)

        if self.mode in ("inverted", "clahe_inverted"):
            gray = cv2.bitwise_not(gray)

        return cv2.cvtColor(
            gray, cv2.COLOR_GRAY2BGR
        )

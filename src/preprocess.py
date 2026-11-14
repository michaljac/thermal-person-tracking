
import os

import cv2
import yaml


class ThermalPreprocessor:
    def __init__(self):
        config_path = os.path.join("config", "config.yaml")

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        params = config["preprocessing"]
        self.mode = params["mode"]

        self.clahe = cv2.createCLAHE(
            clipLimit=params["clahe_clip_limit"],
            tileGridSize=tuple(
                params["clahe_tile_grid_size"]
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

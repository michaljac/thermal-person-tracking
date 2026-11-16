
import cv2
import numpy as np
import onnxruntime as ort
from contextlib import nullcontext

from detectors.base import BaseDetector


class HAESIRDetector(BaseDetector):

    def __init__(self, model_path, detector_cfg):
        self.cfg = detector_cfg
        self.conf_threshold = float(
            detector_cfg["confidence_threshold"]
        )
        self.scale = float(detector_cfg.get("hae_scale", 1.5))
        self.stride = 4

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()
        self.output_names = [out.name for out in self.output_info]

        if not {"cls", "box"}.issubset(self.output_names):
            raise ValueError(
                f"Unexpected HAE outputs: {self.output_names}"
            )

        print("HAE-S-IR loaded")
        print("Input:", self.input_info.name,
              self.input_info.shape)
        print("Outputs:", [
            (out.name, out.shape)
            for out in self.output_info
        ])

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def preprocess(self, frame):
        """
        BGR thermal frame -> normalized float32 NCHW tensor.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if gray.dtype != np.uint8:
            raise ValueError("HAE-S-IR expects uint8 thermal frames")

        original_h, original_w = gray.shape

        scaled_w = int(round(original_w * self.scale))
        scaled_h = int(round(original_h * self.scale))

        scaled = cv2.resize(
            gray,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Pad to multiples of 32.
        pad_h = (-scaled_h) % 32
        pad_w = (-scaled_w) % 32

        padded = cv2.copyMakeBorder(
            scaled,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_REPLICATE
        )

        # Normalize on the 0..255 intensity scale.
        image = padded.astype(np.float32)

        mean = float(image.mean())
        std = float(image.std())

        image = (image - mean) / max(std, 1.0)

        tensor = image[None, None, :, :]

        return tensor, (original_h, original_w), (
            scaled_h, scaled_w
        )

    def postprocess(self, cls, box, original_size, scaled_size):
        """
        Decode HAE classification logits and box distances.

        Returns Nx5 float32:
        [x1, y1, x2, y2, confidence]
        """
        original_h, original_w = original_size
        scaled_h, scaled_w = scaled_size

        scores = self.sigmoid(cls[0, 0])
        distances = self.softplus(box[0]) * self.stride

        # 3x3 local maximum filtering.
        local_max = cv2.dilate(
            scores.astype(np.float32),
            np.ones((3, 3), dtype=np.uint8)
        )

        peak_mask = (
            (scores >= self.conf_threshold)
            & (scores == local_max)
        )

        ys, xs = np.where(peak_mask)

        if len(xs) == 0:
            return np.empty((0, 5), dtype=np.float32)

        confidence = scores[ys, xs]

        cx = (xs + 0.5) * self.stride
        cy = (ys + 0.5) * self.stride

        # Distances assumed to be left, top, right, bottom.
        left = distances[0, ys, xs]
        top = distances[1, ys, xs]
        right = distances[2, ys, xs]
        bottom = distances[3, ys, xs]

        x1 = cx - left
        y1 = cy - top
        x2 = cx + right
        y2 = cy + bottom

        # Remove detections centered in padded regions.
        valid = (cx < scaled_w) & (cy < scaled_h)

        # Map back to original frame coordinates.
        scale_x = scaled_w / original_w
        scale_y = scaled_h / original_h

        x1 = np.clip(x1 / scale_x, 0, original_w)
        y1 = np.clip(y1 / scale_y, 0, original_h)
        x2 = np.clip(x2 / scale_x, 0, original_w)
        y2 = np.clip(y2 / scale_y, 0, original_h)

        detections = np.column_stack(
            (x1, y1, x2, y2, confidence)
        ).astype(np.float32)

        valid &= (x2 > x1) & (y2 > y1)

        return detections[valid]

    def predict(self, frame):
        # A standalone detector also works without a profiling timer.
        timer = getattr(self, "timer", None)
        measure = timer.measure if timer is not None else lambda stage: nullcontext()

        with measure("hae_preprocess"):
            tensor, original_size, scaled_size = self.preprocess(frame)

        with measure("hae_inference"):
            cls, box = self.session.run(
                ["cls", "box"],
                {self.input_info.name: tensor}
            )

        with measure("hae_postprocess"):
            return self.postprocess(cls, box, original_size, scaled_size)

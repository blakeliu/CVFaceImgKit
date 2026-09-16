from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from faceimagekit.utils import rersize_points, resize_image

from .face_detectors import scrfd_model
from .face_landmarks import rtmpose_model


class _BasePipeline(ABC):
    @abstractmethod
    def prepare(self, x):
        raise NotImplementedError()

    @abstractmethod
    def predict(x, *args, **kwargs):
        raise NotImplementedError()

    def release(self):
        """Release underlying pipeline resources."""

    def __del__(self):
        self.release()


EngineType = Literal["ONNXInfer", "OpencvInfer", "RKNNInfer"]
DeviceType = Literal["cpu", "gpu", "npu", "rk3588", "rk3588s", "rk3576", "rk3568"]


def clip_box(landmarks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    x1 = max(0, min(int(np.floor(landmarks[:, 0].min())), width - 1))
    y1 = max(0, min(int(np.floor(landmarks[:, 1].min())), height - 1))
    x2 = max(x1 + 1, min(int(np.ceil(landmarks[:, 0].max())), width))
    y2 = max(y1 + 1, min(int(np.ceil(landmarks[:, 1].max())), height))
    return np.asarray([x1, y1, x2, y2], dtype=np.int32)


class FaceLandmarkPipeline(_BasePipeline):
    def __init__(
        self,
        det_weight: str,
        landmark_weight: str,
        det_backend: EngineType,
        landmark_backend: EngineType,
        det_input_shape: tuple = (3, 640, 640),
        landmark_input_shape: tuple = (3, 256, 256),
        device: DeviceType = "cpu",
    ) -> None:
        """landmark检测pipeline: face detection -> face landmark

        Args:
            det_weight (str): detection weight file
            landmark_weight (str): landmark weight file
            det_backend (EngineType): detection backend
            landmark_backend (EngineType): landmark backend
            det_input_shape (tuple, optional): c h w. Defaults to (3, 640, 640).
            landmark_input_shape (tuple, optional): c h w. Defaults to (3, 256, 256).
            device (DeviceType, optional): 'cpu' or 'gpu'. Defaults to 'cpu'.
        """
        self.det_input_shape = det_input_shape
        self.landmark_input_shape = landmark_input_shape
        self._det_infer = scrfd_model(
            det_weight, det_backend, input_shape=det_input_shape
        )
        self._ld_infer = rtmpose_model(
            landmark_weight, landmark_backend, input_shape=landmark_input_shape
        )
        self.device = device

    def prepare(self, **kwargs):
        self._det_infer.prepare(device=self.device, **kwargs)
        self._ld_infer.prepare(device=self.device, **kwargs)

    def release(self):
        """Release pipeline models in reverse initialization order."""
        if hasattr(self, "_ld_infer") and self._ld_infer is not None:
            if hasattr(self._ld_infer, "release"):
                self._ld_infer.release()
            elif hasattr(getattr(self._ld_infer, "session", None), "release"):
                self._ld_infer.session.release()
            self._ld_infer = None
        if hasattr(self, "_det_infer") and self._det_infer is not None:
            if hasattr(self._det_infer, "release"):
                self._det_infer.release()
            elif hasattr(getattr(self._det_infer, "session", None), "release"):
                self._det_infer.session.release()
            self._det_infer = None

    def __del__(self):
        self.release()

    def lds_infer(self, img, boxes: list | None = None):
        keypoints, _scores = self._ld_infer.predict(img, boxes)
        results = []
        for kps in keypoints:
            results.append(
                {
                    "landmarks": kps[0],
                }
            )
        return results

    def predict(
        self, img: np.ndarray, score_threshold: float = 0.5, nms_threshold: float = 0.4
    ):
        res_img, scale_factor, pad = resize_image(img, self.det_input_shape[1:])
        height, width = img.shape[0:2]
        dets_list, kpss_list = self._det_infer.predict(
            res_img, score_threshold, nms_threshold
        )

        face_list = []
        for dets, kps in zip(dets_list[0], kpss_list[0]):
            det_box = rersize_points(dets[0:4], scale_factor, pad)  # xyxy
            det_box = det_box.astype(np.int32)
            det_box[0::2] = np.clip(det_box[0::2], 0, width - 1)
            det_box[1::2] = np.clip(det_box[1::2], 0, height - 1)
            prob = dets[4]
            kps = rersize_points(kps, scale_factor, pad)

            lds_info = self.lds_infer(img, [det_box])[0]
            lds_info["landmarks"] = lds_info["landmarks"].astype(np.int32)
            lds_info["landmarks"][:, 0] = np.clip(
                lds_info["landmarks"][:, 0], 0, width - 1
            )
            lds_info["landmarks"][:, 1] = np.clip(
                lds_info["landmarks"][:, 1], 0, height - 1
            )
            lds_info["det_box"] = det_box
            lds_info["bbox"] = clip_box(lds_info["landmarks"], img.shape)
            lds_info["prob"] = prob
            lds_info["kps"] = kps

            face_list.append(lds_info)
        return face_list

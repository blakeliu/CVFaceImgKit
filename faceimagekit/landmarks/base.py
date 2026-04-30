from typing import Optional, Union, List, Tuple
from abc import ABCMeta

import numpy as np


class Landmarker(metaclass=ABCMeta):
    def __init__(self, infer_backend, version=1) -> None:
        """
        Args:
            infer_backend (): _description_
            version (int, optional): _description_. Defaults to 1.
        """
        self.session = infer_backend
        self.score_thr = 0.4
        self.masks = False
        self.version = version
        self.input_shape = None
        self.out_shapes = None
        self.infer_shape = None
        self.score_list = None
        self.bbox_list = None
        self.kps_list = None
        self._num_anchors = 2
        self.stream = None
        self.input_ptr = None

    def prepare(self, score_thr: float = 0.4, **kwargs):
        """
        Read network params and populate class paramters.
        Args:
            score_thr (float, optional): _description_. Defaults to 0.4.
        """
        self.score_thr = score_thr
        self.session.prepare(**kwargs)
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

    def _preprocess(self, img: np.ndarray):
        pass

    def _forward(self, blob):
        pass

    def _postprocess(self, net_outputs, score_thr):
        pass

    def predict(self, imgs, score_thr=0.5):
        raise NotImplementedError

    def classes(self):
        raise NotImplementedError(f"must implement classes property!")

import os.path as osp
import sys
import logging
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
from faceimagekit.core.exception import OpencvDNNRunException

if not module_available("cv2"):
    raise ModuleNotFoundError(
        "onnxtuntime package not found! please 'pip install opencv-python'")
import cv2


class OpencvInfer:
    def __init__(self, weight_file,
                 input_shape=None,
                 output_order=None, **kwargs):

        self._model = None
        logging.info('OpencvInfer started')
        self.input = None
        self.input_shape = input_shape  # c h w
        self.output_order = output_order
        self.out_shapes = None
        self._weight_file = weight_file
        if not osp.exists(self._weight_file):
            raise FileNotFoundError(
                f"onnx file: {self._weight_file} not found!")
        self.providers = {
            'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

            # You need to manually build OpenCV through cmake
            'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
        }
        self.__dict__.update(**kwargs)

    def __del__(self):
        self._model = None

    # warmup
    def prepare(self, device: str = 'cpu'):
        
        self._model = cv2.dnn.readNetFromONNX(self._weight_file)
        self._model.setPreferableBackend(self.providers[device][0])
        self._model.setPreferableTarget(self.providers[device][1])

        self.input_dtype = np.float32

        logging.info("Warming up Opencv Runtime engine...")

        self.out_names = self.output_order or self._model.getUnconnectedOutLayersNames()

        # build input to (1, 3, H, W)
        dummy_input = np.zeros(self.input_shape, self.input_dtype)
        dummy_input = dummy_input[None, :, :, :]
        self.run(dummy_input)

    def run(self, input):
        self._model.setInput(input)
        try:
            net_out = self._model.forward(self.out_names)
        except Exception as e:
            raise OpencvDNNRunException(f"Opencv DNN run error: {str(e)}")
        return net_out


def regsiter_opencv_backend(register: Registry):
    register(fn=OpencvInfer, name=OpencvInfer.__name__,
             namespace="engine_backend", type="opencv")

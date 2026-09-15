import logging
import os.path as osp

import numpy as np

from faceimagekit.core import Registry, module_available
from faceimagekit.core.exception import ONNXRunException

logger = logging.getLogger(__name__)


def _is_onnxruntime_available() -> bool:
    return module_available("onnxruntime")


def get_onnxruntime_providers(device: str = "cpu"):
    if not _is_onnxruntime_available():
        raise ModuleNotFoundError(
            "onnxruntime package not found! Please install with 'pip install onnxruntime' or 'pip install .[cpu]'"
        )
    import onnxruntime

    if device.lower() not in ("gpu", "cuda"):
        return ["CPUExecutionProvider"]

    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        logger.warning(
            "CUDAExecutionProvider is not available; falling back to CPUExecutionProvider"
        )
        return ["CPUExecutionProvider"]

    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class ONNXInfer:
    def __init__(self, weight_file, input_shape=None, output_order=None, **kwargs):
        if not _is_onnxruntime_available():
            raise ModuleNotFoundError(
                "onnxruntime package not found! Please install with 'pip install onnxruntime' or 'pip install .[cpu]'"
            )
        self._model = None
        logger.info("ONNXInfer started")
        self.input = None
        self.input_dtype = None
        self.input_shape = input_shape  # c h w
        self.output_order = output_order
        self.out_shapes = None
        self._weight_file = weight_file
        if not osp.exists(self._weight_file):
            raise FileNotFoundError(f"onnx file: {self._weight_file} not found!")
        self.__dict__.update(**kwargs)

    def __del__(self):
        self._model = None

    # warmup
    def prepare(self, device: str = "cpu"):
        if not _is_onnxruntime_available():
            raise ModuleNotFoundError(
                "onnxruntime package not found! Please install with 'pip install onnxruntime' or 'pip install .[cpu]'"
            )
        import onnxruntime

        providers = get_onnxruntime_providers(device)
        self._model = onnxruntime.InferenceSession(
            self._weight_file, providers=providers
        )
        meta = self._model.get_modelmeta().custom_metadata_map  # metadata
        self.input = self._model.get_inputs()[0]
        self.input_dtype = self.input.type
        if self.input_dtype == "tensor(float)":
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.uint8
        if self.output_order is None:
            self.output_order = [e.name for e in self._model.get_outputs()]
        if "stride" in meta:
            stride, names = int(meta["stride"]), eval(meta["names"])
            self.stride = stride
            self.names = names
        logger.info("Warming up ONNX Runtime engine...")

        self.out_shapes = [e.shape for e in self._model.get_outputs()]
        if not isinstance(self.input.shape[0], int):
            bs = 1
        else:
            bs = self.input.shape[0]
        self.input_shape = (bs, *self.input_shape)
        self.run(np.zeros(self.input_shape, self.input_dtype))

    def run(self, input):
        try:
            net_out = self._model.run(self.output_order, {self.input.name: input})
        except Exception as e:
            raise ONNXRunException(f"onnx run error: {e!s}") from e
        return net_out


def regsiter_onnx_backend(register: Registry):
    register(
        fn=ONNXInfer, name=ONNXInfer.__name__, namespace="engine_backend", type="onnx"
    )

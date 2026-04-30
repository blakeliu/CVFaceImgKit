from faceimagekit.core import Registry
from .onnx_backend import regsiter_onnx_backend
from .opencv_backend import regsiter_opencv_backend

ENGINE_BACKENDS = Registry("engine_backends")
regsiter_onnx_backend(ENGINE_BACKENDS)
regsiter_opencv_backend(ENGINE_BACKENDS)

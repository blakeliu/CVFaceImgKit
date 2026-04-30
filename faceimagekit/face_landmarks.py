from typing import Union
from pathlib import Path
from .landmarks import LANDMARKERS
from .engine_backends import ENGINE_BACKENDS


def _normalize_rtmpose_input_shape(input_shape):
    if input_shape is None or len(input_shape) != 3:
        return input_shape
    if input_shape[-1] == 3 and input_shape[0] != 3:
        return [input_shape[2], input_shape[0], input_shape[1]]
    return input_shape


def rtmpose_model(model_path: Union[str, Path], backend: str, **kwargs):
    if "input_shape" in kwargs:
        kwargs["input_shape"] = _normalize_rtmpose_input_shape(kwargs["input_shape"])
    inference_backend = ENGINE_BACKENDS.get(backend)(weight_file=model_path, **kwargs)
    model = LANDMARKERS.get("RTMPose")(infer_backend=inference_backend)
    return model

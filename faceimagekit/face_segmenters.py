from pathlib import Path
from typing import Union

from .engine_backends import ENGINE_BACKENDS
from .segments import SEGMENTERS

pplitesegface_outputs = ["519"]
pplitesegface12_outputs = ["741"]


def pplitesegface_model(
    model_path: Union[str, Path], backend: str = "ONNXInfer", **kwargs
):
    if backend == "ONNXInfer":
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, **kwargs
        )
    else:
        raise ValueError(f"backend must be 'ONNXInfer', but got {backend!s}")
    model = SEGMENTERS.get("PPLiteSeg")(infer_backend=inference_backend)
    return model


def pplitesegface12_model(
    model_path: Union[str, Path], backend: str = "ONNXInfer", **kwargs
):
    if backend == "ONNXInfer":
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, **kwargs
        )
    else:
        raise ValueError(f"backend must be 'ONNXInfer', but got {backend!s}")
    model = SEGMENTERS.get("PPLiteSeg")(infer_backend=inference_backend)
    return model

from __future__ import annotations

import logging
import os.path as osp

import numpy as np

from faceimagekit.core import Registry, module_available
from faceimagekit.core.exception import RKNNRunException

logger = logging.getLogger(__name__)


def _is_rknn_available() -> bool:
    return module_available("rknn.api") or module_available("rknnlite.api")


CORE_MASK_MAP: dict[str, int] = {
    "auto": 0x0,
    "0": 0x1,
    "1": 0x2,
    "2": 0x4,
    "0_1": 0x3,
    "0_1_2": 0x7,
    "all": 0xFFFF,
}


class RKNNInfer:
    """RKNN Inference Engine Backend.

    Supports both PC RKNN Simulator (via rknn-toolkit2 with .onnx model)
    and RKNN NPU hardware (via rknn-toolkit2 with ADB target or rknn-toolkit-lite2 with .rknn model).
    """

    def __init__(
        self,
        weight_file: str,
        input_shape: tuple[int, ...] | None = None,
        output_order: list[str] | None = None,
        mean_values: list[list[float]] | None = None,
        std_values: list[list[float]] | None = None,
        target_platform: str = "rk3588",
        verbose: bool = False,
        **kwargs,
    ):
        if not _is_rknn_available():
            raise ModuleNotFoundError(
                "RKNN package not found! Please install faceimagekit with rknn extra: "
                "'pip install .[rknn]' (requires 'rknn-toolkit2' or 'rknn-toolkit-lite2')."
            )

        self._model = None
        logger.info("RKNNInfer started")
        self._weight_file = weight_file
        self.input_shape = input_shape  # c h w or (1, c, h, w)
        self.output_order = output_order
        self.out_shapes = None
        self.input_dtype = np.float32
        self._is_lite = False
        norm_platform = target_platform.lower().strip()
        if norm_platform in ("rk3588s", "3588s", "3588"):
            norm_platform = "rk3588"
        self.target_platform = norm_platform
        self.verbose = verbose

        if not osp.exists(self._weight_file):
            raise FileNotFoundError(
                f"Model weight file: {self._weight_file} not found!"
            )

        # Default mean/std values based on model type if not explicitly supplied
        lower_weight = self._weight_file.lower()
        if mean_values is not None:
            self.mean_values = mean_values
        elif "scrfd" in lower_weight:
            self.mean_values = [[127.5, 127.5, 127.5]]
        elif "rtm" in lower_weight or "end2end" in lower_weight:
            self.mean_values = [[123.675, 116.28, 103.53]]
        else:
            self.mean_values = [[0.0, 0.0, 0.0]]

        if std_values is not None:
            self.std_values = std_values
        elif "scrfd" in lower_weight:
            self.std_values = [[128.0, 128.0, 128.0]]
        elif "rtm" in lower_weight or "end2end" in lower_weight:
            self.std_values = [[58.395, 57.12, 57.375]]
        else:
            self.std_values = [[1.0, 1.0, 1.0]]

        self.__dict__.update(**kwargs)

    def release(self):
        """Release underlying RKNN/RKNNLite C and NPU runtime resources."""
        if self._model is not None:
            if hasattr(self._model, "release"):
                try:
                    self._model.release()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to release model resources: %s", exc)
            self._model = None

    def __del__(self):
        self.release()

    def prepare(self, device: str = "cpu", warmup: bool = False, **kwargs):
        """Initialize RKNN runtime environment.

        Args:
            device: 'cpu' / 'simulator' for PC host simulation,
                    or hardware platform like 'rk3588', 'rk3588s', 'rk3576', 'rk3568', 'npu'.
            warmup: Whether to execute dummy warmup inference. Defaults to False.
                    In multi-model pipelines on-device (RKNNLite), all models must complete
                    init_runtime before ANY model executes inference to avoid NPU context conflict.
            **kwargs: Extra runtime parameters like core_mask (e.g. 'all', 'auto', '0', '0_1_2', 7).
        """
        dev = device.lower().strip()
        if dev in ("rk3588s", "3588s", "3588"):
            dev = "rk3588"

        is_hardware = dev in (
            "gpu",
            "npu",
            "rk3588",
            "rk3576",
            "rk3568",
            "rk3566",
            "rk3562",
            "rv1106",
            "rv1103",
            "rv1126b",
        )
        target = (
            dev
            if is_hardware and dev not in ("gpu", "npu")
            else (self.target_platform if dev == "npu" else None)
        )

        runtime_kwargs = kwargs.copy()
        if "core_mask" in runtime_kwargs and isinstance(
            runtime_kwargs["core_mask"], str
        ):
            mask_key = runtime_kwargs["core_mask"].lower().replace("core", "").strip()
            if mask_key in CORE_MASK_MAP:
                runtime_kwargs["core_mask"] = CORE_MASK_MAP[mask_key]

        # 1. Edge Board Mode (rknn-toolkit-lite2)
        if module_available("rknnlite.api"):
            self._is_lite = True
            logger.info(
                "Using RKNNLite on-device runtime (core_mask=%s)...",
                runtime_kwargs.get("core_mask", "auto"),
            )
            from rknnlite.api import RKNNLite

            self._model = RKNNLite(verbose=self.verbose)
            ret = self._model.load_rknn(self._weight_file)
            if ret != 0:
                raise RKNNRunException(
                    f"RKNNLite load_rknn failed on {self._weight_file} (ret={ret})"
                )
            ret = self._model.init_runtime(**runtime_kwargs)
            if ret != 0:
                raise RKNNRunException(f"RKNNLite init_runtime failed (ret={ret})")

        # 2. Host Mode (rknn-toolkit2)
        else:
            self._is_lite = False
            from rknn.api import RKNN

            self._model = RKNN(verbose=self.verbose)

            host_kwargs = runtime_kwargs.copy()
            # In PC simulator mode (target is None), core_mask is ignored
            if target is None and "core_mask" in host_kwargs:
                host_kwargs.pop("core_mask")

            if self._weight_file.endswith(".rknn"):
                if target is None:
                    raise RKNNRunException(
                        f"RKNN binary model '{self._weight_file}' cannot run on PC simulator without target hardware. "
                        "Please specify an NPU target (e.g. device='rk3588' via ADB), "
                        "or specify the source .onnx model directly for PC simulation."
                    )
                logger.info(
                    "Loading RKNN model: %s for target: %s...",
                    self._weight_file,
                    target,
                )
                ret = self._model.load_rknn(self._weight_file)
                if ret != 0:
                    raise RKNNRunException(
                        f"RKNN load_rknn failed on {self._weight_file} (ret={ret})"
                    )
                ret = self._model.init_runtime(target=target, **host_kwargs)
                if ret != 0:
                    raise RKNNRunException(
                        f"RKNN init_runtime with target={target} failed (ret={ret})"
                    )
            elif self._weight_file.endswith(".onnx"):
                logger.info(
                    "Building and initializing RKNN runtime from ONNX: %s (target=%s)...",
                    self._weight_file,
                    target,
                )
                self._build_rknn_from_onnx(self._weight_file)
                ret = self._model.init_runtime(target=target, **host_kwargs)
                if ret != 0:
                    raise RKNNRunException(f"RKNN init_runtime failed (ret={ret})")
            else:
                raise RKNNRunException(
                    f"Unsupported model file format: '{self._weight_file}'. Expected '.rknn' or '.onnx'."
                )

        # Warmup and out shapes calculation (only if explicitly requested and not on lite)
        if warmup and not self._is_lite:
            logger.info("Warming up RKNN Runtime engine...")
            if self.input_shape is not None:
                if len(self.input_shape) == 3:
                    full_shape = (1, *self.input_shape)
                else:
                    full_shape = tuple(self.input_shape)
                dummy_input = np.zeros(full_shape, dtype=np.float32)
                warmup_outs = self.run(dummy_input)
                self.out_shapes = [out.shape for out in warmup_outs]

    def _build_rknn_from_onnx(self, onnx_file: str):
        """Configure, load ONNX model and build in-memory RKNN graph for simulation."""
        self._model.config(
            mean_values=self.mean_values,
            std_values=self.std_values,
            target_platform=self.target_platform,
        )

        load_kwargs = {}
        if self.output_order is not None:
            load_kwargs["outputs"] = self.output_order

        # Set input dimensions if specified
        if self.input_shape is not None:
            if len(self.input_shape) == 3:
                in_shape = [1, *self.input_shape]
            else:
                in_shape = list(self.input_shape)
            if "rtm" in onnx_file.lower() or "end2end" in onnx_file.lower():
                load_kwargs["inputs"] = ["input"]
                load_kwargs["input_size_list"] = [in_shape]

        ret = self._model.load_onnx(model=onnx_file, **load_kwargs)
        if ret != 0:
            raise RKNNRunException(f"rknn.load_onnx failed on {onnx_file} (ret={ret})")

        ret = self._model.build(do_quantization=False)
        if ret != 0:
            raise RKNNRunException(f"rknn.build failed for {onnx_file} (ret={ret})")

    def run(
        self, input: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]
    ) -> list[np.ndarray]:
        """Execute inference.

        Supports both raw uint8 images and pre-normalized float32 tensors.
        For models with baked-in mean_values and std_values, pre-normalized float32
        inputs are automatically denormalized back to uint8 NHWC so the hardware NPU
        handles mean/std normalization natively without buffer type misinterpretation.
        """
        if isinstance(input, (list, tuple)):
            inputs = list(input)
        else:
            inputs = [input]

        converted_inputs = []
        pass_through_list = []
        for idx, x in enumerate(inputs):
            if isinstance(x, np.ndarray):
                # 1. Ensure 4D (batch dimension)
                if x.ndim == 3:
                    if x.shape[0] in (1, 3, 4) and x.shape[0] < x.shape[2]:
                        x = x[None, ...]
                    else:
                        x = x[None, ...]

                # 2. Transpose NCHW -> NHWC if needed (RKNPU native layout is NHWC)
                if x.ndim == 4 and x.shape[1] in (1, 3, 4) and x.shape[1] < x.shape[3]:
                    x = x.transpose(0, 2, 3, 1)

                # 3. Check mean/std configuration
                mean_list = (
                    self.mean_values[idx]
                    if idx < len(self.mean_values)
                    else self.mean_values[0]
                )
                std_list = (
                    self.std_values[idx]
                    if idx < len(self.std_values)
                    else self.std_values[0]
                )
                mean = np.array(mean_list, dtype=np.float32)
                std = np.array(std_list, dtype=np.float32)
                has_norm = not (np.allclose(mean, 0.0) and np.allclose(std, 1.0))

                # 4. Handle dtypes & normalization
                if np.issubdtype(x.dtype, np.floating) and has_norm:
                    # Model has mean/std baked in -> denormalize float [-1..1] back to uint8 [0..255]
                    # The NPU hardware handles normalization automatically at 0 CPU cost.
                    x = np.clip(np.round(x * std + mean), 0, 255).astype(np.uint8)
                    x = np.ascontiguousarray(x)
                    pass_through = 0
                elif np.issubdtype(x.dtype, np.integer):
                    # Already raw uint8 / integer image
                    x = np.ascontiguousarray(x)
                    pass_through = 0
                else:
                    # Float input without mean/std: pass through directly
                    x = np.ascontiguousarray(x.astype(np.float32))
                    pass_through = 1

                converted_inputs.append(x)
                pass_through_list.append(pass_through)
            else:
                converted_inputs.append(x)
                pass_through_list.append(0)

        all_pt_zero = all(pt == 0 for pt in pass_through_list)

        if self._is_lite:
            # On-device RKNNLite:
            # - When all pass_through are 0 (standard uint8 path), data_format=None defaults
            #   to NHWC and avoids librknnrt's C-level buffer reallocation/free crash.
            # - When float pass_through is needed, explicit data_type='float32' ensures
            #   librknnrt allocates 4 bytes/element instead of default 1 byte uint8.
            try:
                if all_pt_zero:
                    net_out = self._model.inference(inputs=converted_inputs)
                else:
                    net_out = self._model.inference(
                        inputs=converted_inputs,
                        data_type="float32",
                        data_format=None,
                        inputs_pass_through=pass_through_list,
                    )
            except Exception as exc:
                raise RKNNRunException(f"RKNNLite inference error: {exc!s}") from exc
        else:
            # PC Simulator (rknn-toolkit2):
            try:
                if all_pt_zero:
                    net_out = self._model.inference(
                        inputs=converted_inputs,
                        data_format=["nhwc"] * len(converted_inputs),
                    )
                else:
                    net_out = self._model.inference(
                        inputs=converted_inputs,
                        data_format=["nhwc"] * len(converted_inputs),
                        inputs_pass_through=pass_through_list,
                    )
            except Exception as exc:
                raise RKNNRunException(f"RKNN inference error: {exc!s}") from exc

        if net_out is None:
            raise RKNNRunException(
                "RKNN inference returned None (inference failed, check logs)"
            )

        return net_out


def regsiter_rknn_backend(register: Registry):
    register(
        fn=RKNNInfer, name=RKNNInfer.__name__, namespace="engine_backend", type="rknn"
    )

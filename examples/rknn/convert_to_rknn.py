"""RKNN Model Conversion Script for FaceLandmarkPipeline models:
- scrfd_10g_gnkps_shape640x640.onnx
- scrfd_2.5g_gnkps_shape640x640.onnx
- rtmface-m/mmdeploy/end2end.onnx

Supports both FP16 and INT8 quantization modes across Rockchip NPU platforms:
rk3588, rk3576, rk3568, rk3566, rk3562, rv1106, rv1103, rv1126b.
"""

import argparse
import os
import os.path as osp
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rknn.api import RKNN

# Model Configurations
MODEL_CONFIGS = {
    "scrfd_2.5g": {
        "onnx_path": "/home/tf/PycharmProjects/face/weights/scrfd/onnx/scrfd_2.5g_gnkps_shape640x640.onnx",
        "output_name": "scrfd_2.5g_gnkps_shape640x640",
        "mean_values": [[127.5, 127.5, 127.5]],
        "std_values": [[128.0, 128.0, 128.0]],
        "input_names": None,  # automatic from ONNX
        "input_size_list": None,
    },
    "scrfd_10g": {
        "onnx_path": "/home/tf/PycharmProjects/face/weights/scrfd/onnx/scrfd_10g_gnkps_shape640x640.onnx",
        "output_name": "scrfd_10g_gnkps_shape640x640",
        "mean_values": [[127.5, 127.5, 127.5]],
        "std_values": [[128.0, 128.0, 128.0]],
        "input_names": None,
        "input_size_list": None,
    },
    "rtmface_m": {
        "onnx_path": "/home/tf/PycharmProjects/face/weights/rtmface-m/mmdeploy/end2end.onnx",
        "output_name": "rtmface_m_134_256x256",
        "mean_values": [[123.675, 116.28, 103.53]],
        "std_values": [[58.395, 57.12, 57.375]],
        # Dynamic batch size in ONNX: freeze to batch 1
        "input_names": ["input"],
        "input_size_list": [[1, 3, 256, 256]],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX models to RKNN format")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "scrfd_2.5g", "scrfd_10g", "rtmface_m"],
        help="Which model to convert (default: all)",
    )
    parser.add_argument(
        "--target_platform",
        type=str,
        default="rk3588",
        choices=[
            "rk3588",
            "rk3576",
            "rk3568",
            "rk3566",
            "rk3562",
            "rv1106",
            "rv1103",
            "rv1126b",
        ],
        help="Target hardware platform (default: rk3588)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "i8"],
        help="Quantization dtype: fp16 (unquantized) or i8 (int8 quantized)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset.txt file for INT8 quantization calibration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=osp.join(PROJECT_ROOT, "weights/rknn"),
        help="Directory to save exported .rknn models",
    )
    return parser.parse_args()


def convert_single_model(
    model_key: str,
    cfg: dict,
    target_platform: str,
    dtype: str,
    dataset: str,
    output_dir: str,
):
    onnx_path = cfg["onnx_path"]
    if not osp.exists(onnx_path):
        print(f"[Error] ONNX model not found: {onnx_path}")
        return False

    print("\n" + "=" * 60)
    print(
        f"Converting [{model_key}] -> RKNN ({dtype.upper()}, Platform={target_platform})"
    )
    print(f"ONNX Path: {onnx_path}")
    print("=" * 60)

    rknn = RKNN(verbose=False)

    # 1. Config
    print("--> 1. Configuring model...")
    rknn.config(
        mean_values=cfg["mean_values"],
        std_values=cfg["std_values"],
        target_platform=target_platform,
    )

    # 2. Load ONNX
    print("--> 2. Loading ONNX model...")
    load_kwargs = {"model": onnx_path}
    if cfg["input_names"] and cfg["input_size_list"]:
        load_kwargs["inputs"] = cfg["input_names"]
        load_kwargs["input_size_list"] = cfg["input_size_list"]

    ret = rknn.load_onnx(**load_kwargs)
    if ret != 0:
        print(f"[Error] Load ONNX failed for {model_key}!")
        rknn.release()
        return False

    # 3. Build model
    do_quantization = dtype == "i8"
    print(f"--> 3. Building RKNN model (do_quantization={do_quantization})...")
    if do_quantization and not dataset:
        print(
            "[Warning] INT8 quantization requested but --dataset is not provided! Using unquantized FP16 build."
        )
        do_quantization = False

    ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
    if ret != 0:
        print(f"[Error] Build RKNN failed for {model_key}!")
        rknn.release()
        return False

    # 4. Export RKNN
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{target_platform}_{dtype}.rknn"
    export_filename = cfg["output_name"] + suffix
    export_path = osp.join(output_dir, export_filename)

    print(f"--> 4. Exporting RKNN to: {export_path}")
    ret = rknn.export_rknn(export_path)
    rknn.release()

    if ret != 0:
        print(f"[Error] Export RKNN failed for {model_key}!")
        return False

    file_size_mb = osp.getsize(export_path) / (1024 * 1024)
    print(f"[Success] Exported {export_filename} ({file_size_mb:.2f} MB)")
    return True


def main():
    args = parse_args()
    models_to_convert = MODEL_CONFIGS.keys() if args.model == "all" else [args.model]

    success_count = 0
    for key in models_to_convert:
        ok = convert_single_model(
            model_key=key,
            cfg=MODEL_CONFIGS[key],
            target_platform=args.target_platform,
            dtype=args.dtype,
            dataset=args.dataset,
            output_dir=args.output_dir,
        )
        if ok:
            success_count += 1

    print("\n" + "=" * 60)
    print(
        f"Conversion summary: {success_count}/{len(models_to_convert)} models converted successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

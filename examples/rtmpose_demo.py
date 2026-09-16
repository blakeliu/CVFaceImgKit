from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp

import cv2
import numpy as np

from faceimagekit.face_landmarks import rtmpose_model
from faceimagekit.utils import Timer, draw_face

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test rtmpose face landmark detection"
    )
    # Basic
    _default_weight_dir = osp.join(osp.dirname(__file__), "..", "weights")
    _default_onnx = osp.join(
        _default_weight_dir, "rtmface-m", "mmdeploy", "end2end.onnx"
    )
    _default_rknn = osp.join(
        _default_weight_dir,
        "rknn",
        "rtmface_m_134_256x256_rk3588_fp16.rknn",
    )
    _default_weight = _default_rknn if osp.exists(_default_rknn) else _default_onnx
    _default_img = osp.join(osp.dirname(__file__), "..", "pics", "10216.jpg")
    _default_out = osp.join(osp.dirname(__file__), "..", "outputs", "rtmpose")

    parser.add_argument(
        "-weight",
        "--weight_path",
        type=str,
        default=None,
        help="Model weight file (.onnx or .rknn). Defaults to RKNN model for RKNNInfer, ONNX model otherwise.",
    )
    parser.add_argument(
        "-hd",
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "npu", "rk3588", "rk3588s", "rk3576", "rk3568"],
        default="cpu",
        help="hardware type.",
    )
    parser.add_argument(
        "--core_mask",
        type=str,
        default="auto",
        choices=["auto", "0", "1", "2", "0_1", "0_1_2", "all"],
        help="NPU core mask for multi-core platforms like RK3588/RK3588S (e.g. 'all', 'auto', '0').",
    )
    parser.add_argument(
        "-engine",
        "--engine_type",
        type=str,
        choices=["ONNXInfer", "OpencvInfer", "RKNNInfer"],
        default="ONNXInfer",
        help="engine type.",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[3, 256, 256],
        help="resize input shape: c, h, w",
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        default=None,
        help="Face bounding box x1 y1 x2 y2. Defaults to [241, 169, 779, 932] for 10216.jpg.",
    )
    parser.add_argument(
        "-files",
        "--file_list",
        type=str,
        nargs="+",
        default=[_default_img] if osp.exists(_default_img) else [],
        help="file path list",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=_default_out,
        help="path to save generation result",
    )
    parser.add_argument("--imshow", action="store_true", help="show image with opencv")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    if args.weight_path is None:
        _weight_dir = osp.join(osp.dirname(__file__), "..", "weights")
        if args.engine_type == "RKNNInfer":
            args.weight_path = osp.join(
                _weight_dir,
                "rknn",
                "rtmface_m_134_256x256_rk3588_fp16.rknn",
            )
        else:
            args.weight_path = osp.join(
                _weight_dir, "rtmface-m", "mmdeploy", "end2end.onnx"
            )

    infer = rtmpose_model(
        args.weight_path, backend=args.engine_type, input_shape=args.input_shape
    )
    prepare_kwargs = {}
    if args.accelerator.lower() in ("npu", "rk3588", "rk3588s", "rk3576"):
        prepare_kwargs["core_mask"] = args.core_mask

    try:
        infer.prepare(device=args.accelerator, **prepare_kwargs)

        for fp in args.file_list:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"opencv read {fp!s} failed")

            if args.bbox:
                input_bboxes = [np.array(args.bbox)]
            elif "10216" in osp.basename(fp):
                input_bboxes = [np.array([241, 169, 779, 932])]
            else:
                input_bboxes = [np.array([0, 0, img.shape[1], img.shape[0]])]

            print(
                f"Processing {fp}, input bbox: {input_bboxes[0].tolist()}"
            )

            t_infer = Timer()
            poses_list, scores_list = infer.predict(img, bboxes=input_bboxes)
            print(
                f"Model: {osp.basename(args.weight_path)}, shape: {args.input_shape}, infer time: {t_infer.time():.4f} s"
            )

            results = []
            for i, (kps, sc, box) in enumerate(zip(poses_list, scores_list, input_bboxes)):
                pts = kps[0]
                mean_score = float(np.mean(sc)) if len(sc) > 0 else 1.0
                results.append({"landmarks": pts, "score": mean_score, "bbox": box})
                print(
                    f"--> Face [{i}]: detected {len(pts)} landmarks (mean score: {mean_score:.4f}), sample pt0={pts[0].astype(int).tolist()}"
                )

            show_img = draw_face(img, results, draw_bbox=True, draw_lanamrk=True)
            if args.imshow:
                cv2.imshow(f"{osp.basename(fp)}", show_img)
                cv2.waitKey(0)
            if args.save_path:
                os.makedirs(args.save_path, exist_ok=True)
                out_img_path = osp.join(args.save_path, osp.basename(fp))
                cv2.imwrite(out_img_path, show_img)
                print(f"--> Saved annotated image to: {out_img_path}")

                json_path = osp.join(
                    args.save_path, f"{osp.splitext(osp.basename(fp))[0]}_rtmpose.json"
                )
                json_data = [
                    {
                        "box": r["bbox"].astype(int).tolist(),
                        "score": r["score"],
                        "landmarks": r["landmarks"].astype(int).tolist(),
                    }
                    for r in results
                ]
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"--> Saved landmark json to: {json_path}")
    finally:
        if hasattr(infer, "release"):
            infer.release()
        elif hasattr(getattr(infer, "session", None), "release"):
            infer.session.release()


if __name__ == "__main__":
    main()

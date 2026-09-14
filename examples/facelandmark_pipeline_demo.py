from typing import Tuple, List, Dict, Sequence
import os
import os.path as osp
import sys
import argparse
import json
import pathlib
import numpy as np
import cv2
from faceimagekit.pipelines import FaceLandmarkPipeline
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points


def parse_args():
    parser = argparse.ArgumentParser(description="Test FaceLandmarkPipeline")
    # Basic
    _default_weight_dir = "/home/tf/PycharmProjects/face/weights"
    _default_det = osp.join(
        _default_weight_dir, "scrfd/onnx/scrfd_2.5g_gnkps_shape640x640.onnx"
    )
    _default_ld = osp.join(_default_weight_dir, "rtmface-m/mmdeploy/end2end.onnx")
    _default_img = osp.join(osp.dirname(__file__), "..", "pics", "10216.jpg")
    _default_out = osp.join(osp.dirname(__file__), "..", "outputs")

    parser.add_argument(
        "-det_weight",
        "--det_weight_path",
        type=str,
        default=_default_det,
        help="det weight file.",
    )
    parser.add_argument(
        "-ld_weight",
        "--ld_weight_path",
        type=str,
        default=_default_ld,
        help="landmark weight file.",
    )
    parser.add_argument(
        "-hd",
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "npu", "rk3588", "rk3576", "rk3568"],
        default="cpu",
        help="hardware type.",
    )
    parser.add_argument(
        "-det_engine",
        "--det_engine_type",
        type=str,
        choices=["ONNXInfer", "OpencvInfer", "RKNNInfer"],
        default="OpencvInfer",
        help="detection engine type.",
    )
    parser.add_argument(
        "-ld_engine",
        "--ld_engine_type",
        type=str,
        choices=["ONNXInfer", "OpencvInfer", "RKNNInfer"],
        default="OpencvInfer",
        help="landmark engine type.",
    )
    parser.add_argument(
        "--det_input_shape",
        type=int,
        nargs="+",
        default=[3, 640, 640],
        help="detector input shape: c, h, w",
    )
    parser.add_argument(
        "--ld_input_shape",
        type=int,
        nargs="+",
        default=[3, 256, 256],
        help="landmarker input shape: c, h, w",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="score threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="nms threshold")
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


def _face_area(face: Dict[str, np.ndarray]) -> int:
    x1, y1, x2, y2 = face["det_box"].astype(int).tolist()
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_box(
    landmarks: np.ndarray, image_shape: Sequence[int]
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = max(0, min(int(np.floor(landmarks[:, 0].min())), width - 1))
    y1 = max(0, min(int(np.floor(landmarks[:, 1].min())), height - 1))
    x2 = max(x1 + 1, min(int(np.ceil(landmarks[:, 0].max())), width))
    y2 = max(y1 + 1, min(int(np.ceil(landmarks[:, 1].max())), height))
    return x1, y1, x2, y2


def _face_to_json(
    face: Dict[str, np.ndarray], image_shape: Sequence[int]
) -> Dict[str, List[int] | List[List[int]]]:
    landmarks = face["landmarks"]
    return {
        "box": list(clip_box(landmarks, image_shape)),
        "det_box": face["det_box"].astype(int).tolist(),
        "lds": landmarks.astype(int).tolist(),
    }


def draw_pipeline_faces(
    image: np.ndarray, faces: List[Dict[str, np.ndarray]]
) -> np.ndarray:
    show_img = image.copy()
    for face in faces:
        det_box = face["det_box"].astype(int)
        ld_box = face["bbox"].astype(int)
        cv2.rectangle(
            show_img,
            tuple(det_box[:2]),
            tuple(det_box[2:4]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            show_img,
            f"det {face['prob']:.3f}",
            (det_box[0], max(0, det_box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            show_img,
            tuple(ld_box[:2]),
            tuple(ld_box[2:4]),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            show_img,
            "landmarks box",
            (ld_box[0], min(show_img.shape[0] - 1, ld_box[1] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        landmarks = face["landmarks"].astype(int)
        for x, y in landmarks:
            cv2.circle(show_img, (x, y), 1, (0, 0, 255), 2)
        kps = face["kps"].astype(int)
        for x, y in kps:
            cv2.circle(show_img, (x, y), 3, (255, 0, 0), -1)
    return show_img


def _json_save_path(save_dir: str, image_path: str, total_files: int) -> str:
    if total_files == 1:
        return osp.join(save_dir, "face_box.json")
    stem = osp.splitext(osp.basename(image_path))[0]
    return osp.join(save_dir, f"{stem}_face_box.json")


def main():
    args = parse_args()
    infer = FaceLandmarkPipeline(
        args.det_weight_path,
        args.ld_weight_path,
        args.det_engine_type,
        args.ld_engine_type,
        args.det_input_shape,
        args.ld_input_shape,
        args.accelerator,
    )
    try:
        infer.prepare()
    except Exception as e:
        raise RuntimeError(f"FaceLandmarkPipeline infer error: {str(e)}")

    for fp in args.file_list:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileExistsError(f"opencv read {str(fp)} failed")

        t_infer = Timer()
        face_list = infer.predict(
            img, score_threshold=args.threshold, nms_threshold=args.nms
        )
        print(f"FaceLandmarkPipeline infer time: {t_infer.time()} s")
        show_img = draw_pipeline_faces(img, face_list)
        if args.imshow:
            show_name = osp.basename(fp)
            if min(show_img.shape[0:2]) > 1080:
                h, w = show_img.shape[0:2]
                resize = (int(w * 0.5), int(h * 0.5))
                resize_img = cv2.resize(show_img, resize, interpolation=cv2.INTER_AREA)
                cv2.imshow(show_name, resize_img)
            else:
                cv2.imshow(show_name, show_img)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)), show_img)
            json_path = _json_save_path(args.save_path, fp, len(args.file_list))
            if face_list:
                face = max(face_list, key=_face_area)
                face_data = _face_to_json(face, img.shape)
            else:
                face_data = {"box": [], "lds": []}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(face_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

from typing import Tuple, List, Dict
import os
import os.path as osp
import sys
import platform
import argparse
import pathlib
import cv2
from faceimagekit.face_landmarks import rtmpose_model
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points


def parse_args():
    parser = argparse.ArgumentParser(description="Test rtmpose face landamrk detection")
    # Basic
    parser.add_argument("-weight", "--weight_path", type=str, help="mmdeploy dir file.")
    parser.add_argument(
        "-hd",
        "--accelerator",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="hardware type.",
    )
    parser.add_argument(
        "-engine",
        "--engine_type",
        type=str,
        choices=["ONNXInfer", "OpencvInfer"],
        default="OpencvInfer",
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
        "-files", "--file_list", type=str, nargs="+", default=[], help="file path list"
    )
    parser.add_argument("--save_path", type=str, help="path to save generation result")
    parser.add_argument("--imshow", action="store_true", help="show image with opencv")
    return parser.parse_args()


def main():
    args = parse_args()
    # sysstr = platform.system()
    # if(sysstr =="Windows"):
    #     weight_path = pathlib.WindowsPath(args.weight_path)
    # else:
    #     weight_path = pathlib.Path(args.weight_path)

    infer = rtmpose_model(
        args.weight_path, backend=args.engine_type, input_shape=args.input_shape
    )
    infer.prepare(device=args.accelerator)

    for fp in args.file_list:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileExistsError(f"opencv read {str(fp)} failed")

        t_infer = Timer()
        poses_list, scores = infer.predict(img, bboxes=[])
        print(
            f"model name: {args.weight_path}, input_shape: {args.input_shape}, infer time: {t_infer.time()} s"
        )
        results = []
        for kps in poses_list:
            results.append({"landmarks": kps[0]})
        show_img = draw_face(img, results, draw_bbox=False, draw_lanamrk=True)
        if args.imshow:
            cv2.imshow(f"{osp.basename(fp)}", show_img)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)), show_img)


if __name__ == "__main__":
    main()

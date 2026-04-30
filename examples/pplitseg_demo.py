from typing import Tuple, List, Dict
import os
import os.path as osp
import sys
import platform
import argparse
import pathlib
import numpy as np
import cv2
from faceimagekit.face_segmenters import pplitesegface12_model
from faceimagekit.pipelines import FaceLandmarkPipeline
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test face segmentation detection')
    # detection
    parser.add_argument('-det_weight', '--det_weight_path',
                        type=str, help="det weight file.")
    parser.add_argument('-ld_weight', '--ld_weight_path',
                        type=str, help="landmark weight file.")
    parser.add_argument('-det_engine', '--det_engine_type', type=str,
                        choices=['ONNXInfer', 'OpencvInfer'], default='ONNXInfer', help="detection engine type.")
    parser.add_argument('-ld_engine', '--ld_engine_type', type=str,
                        choices=['ONNXInfer', 'OpencvInfer'], default='OpencvInfer', help="landmark engine type.")
    parser.add_argument('--det_input_shape', type=int, nargs='+',
                        default=[3, 640, 640], help='detector input shape: c, h, w')
    parser.add_argument('--ld_input_shape', type=int, nargs='+',
                        default=[3, 256, 256], help='landmarker input shape: c, h, w')
    parser.add_argument('--det_threshold', type=float,
                        default=0.5, help='det score threshold')
    parser.add_argument('--det_nms', type=float,
                        default=0.4, help='det nms threshold')

    # Basic
    parser.add_argument('-weight', '--weight_path',
                        type=str, help="onnx weight file.")
    parser.add_argument('-hd', '--accelerator', type=str,
                        choices=['cpu', 'gpu'], default='cpu', help="hardware type.")
    parser.add_argument('-engine', '--engine_type', type=str,
                        choices=['ONNXInfer'], default='ONNXInfer', help="engine type.")
    parser.add_argument('--input_shape', type=int, nargs='+',
                        default=[3, 512, 512], help='resize input shape: c, h, w')
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='score threshold')
    parser.add_argument('-files', '--file_list',
                        type=str, nargs='+', default=[], help="file path list")
    parser.add_argument('--save_path', type=str,
                        help='path to save generation result')
    parser.add_argument('--imshow', action='store_true',
                        help="show image with opencv")
    return parser.parse_args()


def head_position(image: np.ndarray, landmark: np.ndarray):
    height, width = image.shape[0: 2]
    # get head img
    x1 = np.clip(landmark[:, 0].min(), 0, width - 1)
    x2 = np.clip(landmark[:, 0].max(), 0, width - 1)
    y1 = np.clip(landmark[:, 1].min(), 0, height - 1)
    y2 = np.clip(landmark[:, 1].max(), 0, height - 1)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2

    head_size = int(max([w, h]) * 1.3)
    x1 = cx - head_size // 2
    x2 = x1 + head_size
    y1 = cy - head_size // 2
    y2 = y1 + head_size

    left = 0
    top = 0
    bottom = 0
    right = 0
    if x1 < 0:
        left = -x1
    if y1 < 0:
        top = -y1
    if x2 >= width:
        right = x2 - width + 1
    if y2 >= height:
        bottom = y2 - height + 1

    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    cropped_img = image[y1:y2+1, x1:x2+1]
    cropped_box = [
        x1, y1, x2, y2
    ]

    return cropped_img, cropped_box


def main():
    args = parse_args()

    ld_infer = FaceLandmarkPipeline(
        args.det_weight_path,
        args.ld_weight_path,
        args.det_engine_type,
        args.ld_engine_type,
        args.det_input_shape,
        args.ld_input_shape,
        args.accelerator
    )
    try:
        ld_infer.prepare()
    except Exception as e:
        raise RuntimeError(f"FaceLandmarkPipeline infer error: {str(e)}")

    seg_infer = pplitesegface12_model(
        args.weight_path, backend=args.engine_type, input_shape=args.input_shape)
    seg_infer.prepare(device=args.accelerator)

    for fp in args.file_list:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileExistsError(f"opencv read {str(fp)} failed")

        t_im = Timer()
        try:
            face_list: List[Dict[str, np.ndarray]] = ld_infer.predict(
                img, score_threshold=0.5, nms_threshold=0.4)
        except Exception as e:
            print(f"predict face landmark err: {str(e)}")
            continue
        if len(face_list) == 0:
            print("image can't found any face!")
            continue
        print(f"ld infer time: {t_im.time()} s")

        # get the largest face
        face_list.sort(key=lambda x: (
            x["bbox"][2] - x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True)
        face = face_list[0]

        # crop image for face seg mask
        cropped_img, cropped_box = head_position(img, face["landmarks"])

        t_infer = Timer()
        seg_mask = seg_infer.predict(cropped_img, palette=True)
        print(
            f"model name: {args.weight_path}, input_shape: {args.input_shape}, infer time: {t_infer.time()} s")

        seg_color_mask = np.zeros(img.shape, dtype=np.uint8)
        seg_color_mask[cropped_box[1]: cropped_box[3]+1,
                       cropped_box[0]: cropped_box[2]+1] = seg_mask

        if args.imshow:
            seg_color_mask = np.hstack((img, seg_color_mask))
            show_name = osp.basename(fp)
            if min(seg_color_mask.shape[0: 2]) > 1080:
                h, w = seg_color_mask.shape[0: 2]
                resize = (int(w*0.5), int(h*0.5))
                seg_color_mask = cv2.resize(
                    seg_color_mask, resize, interpolation=cv2.INTER_LINEAR)

            cv2.imshow(show_name, seg_color_mask)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)),
                        seg_color_mask)


if __name__ == "__main__":
    main()

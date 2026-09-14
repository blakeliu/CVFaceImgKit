"""Precision and Accuracy Comparison Script: ONNX vs RKNN.

Compares:
1. Tensor-level numerical fidelity: Cosine Similarity, MAE, Max Difference.
2. Task-level business metrics:
   - SCRFD face detection: bounding box IoU & score delta.
   - RTMPose 134 landmarks: pixel Euclidean distance (Max, Mean, Median).
3. Optional RKNN built-in layer-wise accuracy analysis (`rknn.accuracy_analysis`).
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rknn.api import RKNN

from faceimagekit.face_detectors import scrfd_model
from faceimagekit.face_landmarks import rtmpose_model
from faceimagekit.utils import Timer, rersize_points, resize_image


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compare_scrfd_precision(
    onnx_path: str,
    test_img_path: str,
    target_platform: str = "rk3588",
    run_layer_analysis: bool = False,
):
    print("\n" + "=" * 65)
    print(f"1. Precision Comparison: SCRFD [{osp.basename(onnx_path)}]")
    print("=" * 65)

    img = cv2.imread(test_img_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {test_img_path}")

    # --- A. ONNX Inference ---
    det_onnx = scrfd_model(onnx_path, backend="ONNXInfer", input_shape=[3, 640, 640])
    det_onnx.prepare(device="cpu")

    res_img, scale_factor, pad = resize_image(img, (640, 640))
    t_onnx = Timer()
    dets_onnx, _kpss_onnx = det_onnx.predict(
        res_img, score_threshold=0.5, nms_threshold=0.4
    )
    onnx_time = t_onnx.time()

    # --- B. RKNN Build & Simulator Inference ---
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[128.0, 128.0, 128.0]],
        target_platform=target_platform,
    )
    ret = rknn.load_onnx(model=onnx_path)
    assert ret == 0, "rknn.load_onnx failed"
    ret = rknn.build(do_quantization=False)
    assert ret == 0, "rknn.build failed"
    ret = rknn.init_runtime()
    assert ret == 0, "rknn.init_runtime failed"

    # Feed RGB image to RKNN
    rgb_input = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    rgb_input = np.expand_dims(rgb_input, 0)
    t_rknn = Timer()
    rknn_raw_outs = rknn.inference(inputs=[rgb_input], data_format=["nhwc"])
    rknn_time = t_rknn.time()

    # --- C. Raw Tensor Output Comparison ---
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    norm_img = res_img[..., ::-1].transpose(2, 0, 1).astype(np.float32)
    norm_img = (norm_img - 127.5) / 128.0
    norm_img = np.expand_dims(norm_img, 0)
    out_names = [o.name for o in sess.get_outputs()]
    onnx_raw_outs = sess.run(out_names, {sess.get_inputs()[0].name: norm_img})

    print(f"\n[Raw Tensor Numerical Comparison] (Total {len(out_names)} output heads):")
    print(
        f" {'Output Head':<14} | {'Shape':<14} | {'Cosine Sim':<12} | {'MAE':<12} | {'Max Diff':<12}"
    )
    print("-" * 72)
    cosine_sims, maes, max_diffs = [], [], []
    for i, name in enumerate(out_names):
        o_v = onnx_raw_outs[i].flatten()
        r_v = rknn_raw_outs[i].flatten()
        cos = np.dot(o_v, r_v) / (np.linalg.norm(o_v) * np.linalg.norm(r_v) + 1e-9)
        mae = np.mean(np.abs(o_v - r_v))
        diff = np.max(np.abs(o_v - r_v))
        cosine_sims.append(cos)
        maes.append(mae)
        max_diffs.append(diff)
        print(
            f" {name:<14} | {list(rknn_raw_outs[i].shape)!s:<14} | {cos:<12.6f} | {mae:<12.4e} | {diff:<12.4e}"
        )

    print(
        f"\n>> Tensor Summary: Avg Cosine Sim = {np.mean(cosine_sims):.6f}, Avg MAE = {np.mean(maes):.4e}"
    )

    from faceimagekit.detectors.scrfd import filter as scrfd_filter

    bboxes_rknn, kpss_rknn, scores_rknn = det_onnx._postprocess(
        rknn_raw_outs, 640, 640, threshold=0.5
    )
    det_boxes_rknn, _det_kps_rknn = scrfd_filter(
        bboxes_rknn[0], kpss_rknn[0], scores_rknn[0], nms_threshold=0.4
    )

    print("\n[Face Detection Metric Comparison]:")
    print(
        f"  ONNX Detected Faces: {len(dets_onnx[0])}  (Infer time: {onnx_time * 1000:.2f} ms)"
    )
    print(
        f"  RKNN Detected Faces: {len(det_boxes_rknn)}  (Simulator time: {rknn_time * 1000:.2f} ms)"
    )

    if len(dets_onnx[0]) > 0 and len(det_boxes_rknn) > 0:
        box_o = rersize_points(dets_onnx[0][0][:4], scale_factor, pad)
        box_r = rersize_points(det_boxes_rknn[0][:4], scale_factor, pad)
        iou = compute_iou(box_o, box_r)
        score_diff = abs(dets_onnx[0][0][4] - det_boxes_rknn[0][4])
        print(f"  Face #1 Box IoU:       {iou * 100:.2f} %")
        print(f"  Face #1 Score Delta:   {score_diff:.6f}")
        print(f"  ONNX Box:              {box_o.astype(int).tolist()}")
        print(f"  RKNN Box:              {box_r.astype(int).tolist()}")

    # --- E. Optional Layer-wise Analysis ---
    if run_layer_analysis:
        snap_dir = osp.join(PROJECT_ROOT, "outputs/snapshot_scrfd")
        os.makedirs(snap_dir, exist_ok=True)
        print(f"\n--> Running layer-wise rknn.accuracy_analysis() to: {snap_dir}")
        rknn.accuracy_analysis(inputs=[test_img_path], output_dir=snap_dir)
        print("    Layer-wise analysis completed. See error_analysis.txt.")

    rknn.release()


def compare_rtmpose_precision(
    onnx_path: str,
    test_img_path: str,
    target_platform: str = "rk3588",
    run_layer_analysis: bool = False,
):
    print("\n" + "=" * 65)
    print(f"2. Precision Comparison: RTMPose [{osp.basename(onnx_path)}]")
    print("=" * 65)

    img = cv2.imread(test_img_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {test_img_path}")

    # Face box for test image
    face_box = [29, 71, 517, 770]

    # --- A. ONNX Inference ---
    ld_model = rtmpose_model(onnx_path, backend="ONNXInfer", input_shape=[3, 256, 256])
    ld_model.prepare(device="cpu")
    t_onnx = Timer()
    onnx_kpts, _onnx_scores = ld_model.predict(img, [face_box])
    onnx_time = t_onnx.time()
    onnx_pts = onnx_kpts[0][0]  # (134, 2)

    # --- B. RKNN Build & Simulator Inference ---
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform=target_platform,
    )
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["input"],
        input_size_list=[[1, 3, 256, 256]],
    )
    assert ret == 0, "rknn.load_onnx failed"
    ret = rknn.build(do_quantization=False)
    assert ret == 0, "rknn.build failed"
    ret = rknn.init_runtime()
    assert ret == 0, "rknn.init_runtime failed"

    # Preprocess face crop
    crop_img, center, scale = ld_model.preprocess(img, face_box)
    crop_nchw = crop_img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    t_rknn = Timer()
    rknn_outs = rknn.inference(
        inputs=[crop_nchw], data_format=["nchw"], inputs_pass_through=[1]
    )
    rknn_time = t_rknn.time()

    rknn_kpts, _rknn_score = ld_model.postprocess(rknn_outs, center, scale)
    rknn_pts = rknn_kpts[0]  # (134, 2)

    # --- C. Raw Heatmap Output Comparison ---
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    onnx_outs = sess.run(out_names, {sess.get_inputs()[0].name: crop_nchw})

    print("\n[Raw Tensor Numerical Comparison] (simcc_x, simcc_y 1D heatmaps):")
    print(
        f" {'Output Head':<14} | {'Shape':<14} | {'Cosine Sim':<12} | {'MAE':<12} | {'Max Diff':<12}"
    )
    print("-" * 72)
    for i, name in enumerate(out_names):
        o_v = onnx_outs[i].flatten()
        r_v = rknn_outs[i].flatten()
        cos = np.dot(o_v, r_v) / (np.linalg.norm(o_v) * np.linalg.norm(r_v) + 1e-9)
        mae = np.mean(np.abs(o_v - r_v))
        diff = np.max(np.abs(o_v - r_v))
        print(
            f" {name:<14} | {list(rknn_outs[i].shape)!s:<14} | {cos:<12.6f} | {mae:<12.4e} | {diff:<12.4e}"
        )

    # --- D. Landmark Coordinate Metric Comparison ---
    pixel_errors = np.linalg.norm(onnx_pts - rknn_pts, axis=1)
    print("\n[134 Facial Landmark Point Metric Comparison]:")
    print(f"  ONNX Landmark Infer Time: {onnx_time * 1000:.2f} ms")
    print(f"  RKNN Landmark Infer Time: {rknn_time * 1000:.2f} ms")
    print(f"  Max Landmark Pixel Error:    {pixel_errors.max():.4f} px")
    print(f"  Mean Landmark Pixel Error:   {pixel_errors.mean():.4f} px")
    print(f"  Median Landmark Pixel Error: {np.median(pixel_errors):.4f} px")
    pct_under_1px = np.mean(pixel_errors < 1.0) * 100
    print(f"  Points with Error < 1.0 px:  {pct_under_1px:.1f} %")

    if run_layer_analysis:
        snap_dir = osp.join(PROJECT_ROOT, "outputs/snapshot_rtmpose")
        os.makedirs(snap_dir, exist_ok=True)
        print(f"\n--> Running layer-wise rknn.accuracy_analysis() to: {snap_dir}")
        # Save crop for accuracy_analysis input
        crop_save_path = osp.join(snap_dir, "crop_input.npy")
        np.save(crop_save_path, crop_nchw)
        rknn.accuracy_analysis(inputs=[crop_save_path], output_dir=snap_dir)
        print("    Layer-wise analysis completed. See error_analysis.txt.")

    rknn.release()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ONNX vs RKNN accuracy and precision"
    )
    parser.add_argument(
        "--target_platform",
        type=str,
        default="rk3588",
        choices=["rk3588", "rk3588s", "rk3576", "rk3568", "rk3566", "rk3562", "rv1106"],
        help="Target platform for RKNN compiler",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=osp.join(PROJECT_ROOT, "asserts/rtmface_134.png"),
        help="Path to test image",
    )
    parser.add_argument(
        "--layer_analysis",
        action="store_true",
        help="Run rknn.accuracy_analysis() to dump layer-by-layer error analysis",
    )
    args = parser.parse_args()

    platform = "rk3588" if args.target_platform.lower() in ("rk3588s", "3588s") else args.target_platform

    # 1. Compare SCRFD 2.5G
    scrfd_2_5g = "/home/tf/PycharmProjects/face/weights/scrfd/onnx/scrfd_2.5g_gnkps_shape640x640.onnx"
    compare_scrfd_precision(
        onnx_path=scrfd_2_5g,
        test_img_path=args.test_image,
        target_platform=platform,
        run_layer_analysis=args.layer_analysis,
    )

    # 2. Compare SCRFD 10G
    scrfd_10g = "/home/tf/PycharmProjects/face/weights/scrfd/onnx/scrfd_10g_gnkps_shape640x640.onnx"
    compare_scrfd_precision(
        onnx_path=scrfd_10g,
        test_img_path=args.test_image,
        target_platform=platform,
        run_layer_analysis=False,
    )

    # 3. Compare RTMPose 134 landmarks
    rtmpose_onnx = (
        "/home/tf/PycharmProjects/face/weights/rtmface-m/mmdeploy/end2end.onnx"
    )
    compare_rtmpose_precision(
        onnx_path=rtmpose_onnx,
        test_img_path=args.test_image,
        target_platform=platform,
        run_layer_analysis=args.layer_analysis,
    )

    print("\n" + "=" * 65)
    print("All comparisons completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()

from __future__ import annotations

import time

import cv2
import numpy as np


def draw_face(
    image: np.ndarray,
    faces: list[dict[str, np.ndarray]],
    draw_bbox: bool = True,
    draw_socre: bool = False,
    draw_kps: bool = False,
    draw_lanamrk: bool = False,
):
    for face in faces:
        w = None
        if draw_bbox:
            box = face["bbox"].astype(int)
            pt1 = tuple(box[0:2])
            pt2 = tuple(box[2:4])
            x, y = pt1
            r, _ = pt2
            w = r - x
            color = (0, 255, 0)
            cv2.rectangle(image, pt1, pt2, color, 1)

        if draw_socre:
            text = f"{face['prob']:.3f}"
            pos = (x + 3, y - 5)
            textcolor = (0, 0, 0)
            thickness = 1
            border = int(thickness / 2)
            cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
            cv2.putText(image, text, pos, 0, 0.5, color, 3, 16)
            cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

        if draw_lanamrk:
            lms = face["landmarks"].astype(int)
            if w is None:
                w = image.shape[1]
            pt_size = int(w * 0.005)
            for i in range(lms.shape[0]):
                cv2.circle(image, (lms[i][0], lms[i][1]), 1, (0, 0, 255), pt_size)

        if draw_kps:
            lms = face["kps"].astype(int)
            if w is None:
                w = image.shape[1]
            pt_size = int(w * 0.01)
            for i in range(lms.shape[0]):
                cv2.circle(image, (lms[i][0], lms[i][1]), 1, (255, 0, 0), pt_size)

    return image


def rersize_points(dets, scale: float, pad: tuple[int, int] = (0, 0)):
    if pad != (0, 0):
        dets = dets.copy()
        dets[..., 0::2] -= pad[0]
        dets[..., 1::2] -= pad[1]
    if scale != 1.0:
        dets = dets / scale
    return dets


def resize_image(image, new_shape: list[int], value: tuple[int, int, int] = (0, 0, 0)):
    """letter_box style resize

    Args:
        image (_type_): _description_
        new_shape (List[int]): (h, w)
        value (Tuple[int], optional): _description_. Defaults to (0, 0, 0).

    Returns:
        _type_: _description_
    """
    target_h = new_shape[0]
    target_w = new_shape[1]
    h, w = image.shape[:2]

    scale_factor = min(target_w / w, target_h / h)
    new_unpad = round(w * scale_factor), round(h * scale_factor)
    dw = (target_w - new_unpad[0]) / 2
    dh = (target_h - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)
    transformed_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=value,
    )

    return transformed_image, scale_factor, (left, top)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def time(self):
        return time.time() - self.start_time


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_image(
    image,
    scale: list | float,
    return_scale=False,
    interpolation=cv2.INTER_LINEAR,
):
    """resize image keep ratio

    Args:
        image (_type_): _description_
        scale (list): w, h
        return_scale (bool, optional): return scale factor . Defaults to False.
        interpolation (_type_, optional): _description_. Defaults to cv2.INTER_AREA.
    """
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale: {scale}, must be positive.")
        scale_factor = scale
    else:
        h, w = image.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_size = _scale_size((w, h), scale_factor)

    resized_img = cv2.resize(image, new_size, interpolation=interpolation)
    if return_scale:
        return resized_img, scale_factor
    else:
        return resized_img

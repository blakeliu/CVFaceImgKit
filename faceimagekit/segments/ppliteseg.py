import numpy as np
from faceimagekit.core import Registry, module_available
import cv2
from scipy.special import softmax
from faceimagekit.utils import rescale_image
from .base import Segmenter


def normalize_on_np(input: np.ndarray):
    """
    rgb image
    Args:
        input (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    img = input[..., ::-1].copy()  # bgr to rgb
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    img = np.subtract(img, [0.485, 0.456, 0.406])
    img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))  # hwc to chw
    img = np.ascontiguousarray(img)
    return img.astype(np.float32)


class PPLiteSeg(Segmenter):
    # rgb
    color_map = [[0, 0, 0],
                 [255, 255, 255],
                 [81, 243, 218],
                 [252, 119, 61],
                 [192, 237, 215],
                 [83, 201, 95],
                 [96, 126, 4],
                 [144, 3, 190],
                 [104, 111, 5],
                 [156, 226, 149],
                 [247, 232, 203],
                 [218, 159, 173],
                 [98, 119, 254],
                 [69, 210, 136],
                 [212, 92, 44],
                 [125, 170, 135],
                 [120, 88, 54],
                 [37, 31, 174],
                 [25, 118, 98],
                 [77, 10, 58],
                 [250, 139, 146],
                 [19, 245, 33],
                 [66, 47, 72],
                 [169, 240, 248],
                 [164, 113, 99],
                 [24, 100, 221],
                 [6, 247, 155],
                 [79, 170, 93],
                 [243, 186, 164],
                 [230, 27, 157],
                 [185, 126, 86],
                 [167, 235, 42]]

    # label_map = {0: 'background',
    #              1: 'skin',
    #              2: 'cheek',
    #              3: 'chin',
    #              4: 'ear',
    #              5: 'helix',
    #              6: 'lobule',
    #              7: 'bottom_lid',
    #              8: 'pupil',
    #              9: 'iris',
    #              10: 'sclera',
    #              11: 'tear_duct',
    #              12: 'top_lid',
    #              13: 'eyebrow',
    #              14: 'forhead',
    #              15: 'frown',
    #              16: 'hair',
    #              17: 'temple',
    #              18: 'jaw',
    #              19: 'beard',
    #              20: 'inferior_lip',
    #              21: 'oral comisure',
    #              22: 'superior_lip',
    #              23: 'teeth',
    #              24: 'neck',
    #              25: 'nose',
    #              26: 'ala_nose',
    #              27: 'bridge',
    #              28: 'nose_tip',
    #              29: 'nostril',
    #              30: 'DU26',
    #              31: 'sideburns'}
    label_map = {
        0: "background",  # 0.背景
        1: "skin",  # 1.皮肤
        2: "eye",  # 2.眼睛
        3: "pupil",  # 3.瞳孔
        4: "bottom_lid",  # 4.下眼皮
        5: "top_lid",  # 5.上眼皮
        6: "eyebrow",  # 6.眉毛
        7: "hair",  # 7.头发
        8: "superior_lip",  # 8上嘴唇
        9: "teeth",  # 9.牙齿
        10: "inferior_lip",  # 10.下嘴唇
        11: "nose",  # 11.鼻子
    }

    def __init__(self, infer_backend, version=1) -> None:
        """PPLitseg
        site:
        https://github.com/tfrbt/FaceSeg.git
        默认是multi-class分割,即一个pixle只能是一个class
        Args:
            infer_backend (_type_): _description_
            version (int, optional): _description_. Defaults to 1.
        """
        super().__init__(infer_backend, version)
        self.input_shape = (1, 3, 512, 512)
        self.out_shapes = None
        self.infer_shape = None

    def prepare(self, **kwargs):
        """
        Read network params and populate class paramters.
        Args:
        """
        self.session.prepare(**kwargs)
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

    def _transform(self, img: np.ndarray):
        """norm img to tensor

        Args:
            img (np.ndarray): bgr uint8 image 
        Returns:
            _type_: float32 ndarray
        """
        blob = normalize_on_np(img)
        return blob

    def _preprocess(self, img: np.ndarray):
        """resize and norm img to tensor

        Args:
            img (np.ndarray): bgr uint8 image 
        Returns:
            _type_: float32 ndarray
        """
        new_shape = self.input_shape[2:]  # hw
        res_img, scale_factor = rescale_image(
            img, new_shape[::-1], return_scale=True)
        shape = res_img.shape[:2]  # hw

        r = min(new_shape[1] / shape[1], new_shape[0] / shape[0])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # wh
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            res_img = cv2.resize(res_img, new_unpad,
                                 interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        res_img = cv2.copyMakeBorder(res_img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=0)  # add border

        return res_img, scale_factor, (dw, dh)

    def predict(self, img, score_thr: float = 0.5, palette: bool = False):
        """pipeline

        Args:
            img (_type_): bgr img
            score_thr (float, optional): score threshold. Defaults to 0.5.
            palette (bool, optional): pixle to color_map. Defaults to False.
        """
        h, w = img.shape[0: 2]
        res_img, scale_factor, pad = self._preprocess(img)
        net_outputs = self._forward(
            np.expand_dims(self._transform(res_img), 0))

        seg_pred = self._postprocess(net_outputs, (w, h), pad)

        if seg_pred.shape[0] == 1:
            seg_pred = seg_pred[0]
        if palette:
            color_seg = np.zeros(
                (h, w, 3), dtype=np.uint8)
            for label, name in self.label_map.items():
                color_seg[seg_pred == label, :] = self.color_map[label]
            color_seg = color_seg[..., ::-1]  # rgb to bgr
            return color_seg
        else:
            return seg_pred

    def _postprocess(self, net_output, new_size, pad):
        """尺寸缩放和pad

        Args:
            net_output (_type_): chw
            scale_factor (_type_): _description_
            pad (_type_): _description_
        """
        dw, dh = pad
        _, oh, ow = net_output.shape
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        pad_img = net_output[:, top: oh-bottom, left: ow-right]
        c, nh, nw = pad_img.shape
        if new_size != (nw, nh):
            # new_size = int((nw+0.5) / float(scale_factor)), int((nh+0.5) / float(scale_factor))
            res_tensors = np.zeros((c, *new_size[::-1]), dtype=np.uint8)
            for i in range(c):
                res_tensors[i] = cv2.resize(pad_img[i].astype(
                    np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
            return np.ascontiguousarray(res_tensors)
        return pad_img.astype(np.uint8)

    def _forward(self, blob):
        """
        send input data to infer backend
        Args:
            blob (_type_): _description_
        """
        assert blob is not None
        output = self.session.run(blob)
        seg_logit: np.ndarray = softmax(output[0], axis=1)
        seg_pred = seg_logit.argmax(axis=1)
        return seg_pred


def regsiter_ppliteseg_segment(register: Registry):
    register(fn=PPLiteSeg, name=PPLiteSeg.__name__,
             namespace="segment", type="face_segment")

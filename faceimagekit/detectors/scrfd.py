import logging
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available

if not module_available("numba"):
    raise ModuleNotFoundError("numba package not found! please 'pip install numba'")
from numba import njit
from .base import Detector

logging.getLogger("numba").setLevel(logging.WARNING)


@njit(cache=True)
def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def normalize_on_np(input):
    img = np.asarray(input)
    img = img[..., ::-1]
    img = np.transpose(img, (0, 3, 1, 2))
    img = np.subtract(img, 127.5, dtype=np.float32)
    img = np.multiply(img, 1 / 128)
    return img


@njit(fastmath=True, cache=True)
def single_distance2bbox(point, distance, stride):
    """
    Fast conversion of single bbox distances to coordinates

    :param point: Anchor point
    :param distance: Bbox distances from anchor point
    :param stride: Current stride scale
    :return: bbox
    """
    distance[0] = point[0] - distance[0] * stride
    distance[1] = point[1] - distance[1] * stride
    distance[2] = point[0] + distance[2] * stride
    distance[3] = point[1] + distance[3] * stride
    return distance


@njit(fastmath=True, cache=True)
def single_distance2kps(point, distance, stride):
    """
    Fast conversion of single keypoint distances to coordinates

    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :param stride: Current stride scale
    :return: keypoint
    """
    for ix in range(0, distance.shape[0], 2):
        distance[ix] = distance[ix] * stride + point[0]
        distance[ix + 1] = distance[ix + 1] * stride + point[1]
    return distance


@njit(fastmath=True, cache=True)
def generate_proposals(
    score_blob,
    bbox_blob,
    kpss_blob,
    stride,
    anchors,
    threshold,
    score_out,
    bbox_out,
    kpss_out,
    offset,
):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated np.ndarrays for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores np.ndarray
    :param bbox_out: Output bbox np.ndarray
    :param kpss_out: Output key points np.ndarray
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset

    for ix in range(0, anchors.shape[0]):
        if score_blob[ix, 0] > threshold:
            score_out[total] = score_blob[ix]
            bbox_out[total] = single_distance2bbox(anchors[ix], bbox_blob[ix], stride)
            kpss_out[total] = single_distance2kps(anchors[ix], kpss_blob[ix], stride)
            total += 1

    return score_out, bbox_out, kpss_out, total


@njit(fastmath=True, cache=True)
def filter(
    bboxes_list: np.ndarray,
    kpss_list: np.ndarray,
    scores_list: np.ndarray,
    nms_threshold: float = 0.4,
):
    """
    Filter postprocessed network outputs with NMS

    :param bboxes_list: List of bboxes (np.ndarray)
    :param kpss_list: List of keypoints (np.ndarray)
    :param scores_list: List of scores (np.ndarray)
    :return: Face bboxes with scores [t,l,b,r,score], and key points
    """

    pre_det = np.hstack((bboxes_list, scores_list))
    keep = nms(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    det = pre_det[keep, :]
    kpss = kpss_list[keep, :]
    kpss = kpss.reshape((kpss.shape[0], -1, 2))

    return det, kpss


class SCRFD(Detector):
    def __init__(self, infer_backend, version=1) -> None:
        """
        Sample and Computation Redistribution for Efficient Face Detection
        参考:
        https://github.com/deepinsight/insightface/tree/master/detection/scrfd
        https://github.com/SthPhoenix/InsightFace-REST/src/api_trt/modules/model_zoo/detectors/scrfd.py
        Args:
            infer_backend (): _description_
            version (int, optional): _description_. Defaults to 1.
        """
        super().__init__(infer_backend, version)
        self.session = infer_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.version = version
        self.input_shape = None
        self.out_shapes = None
        self.infer_shape = None
        self.score_list = None
        self.bbox_list = None
        self.kps_list = None
        self._acnhor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def prepare(self, **kwargs):
        """
        Read network params and populate class paramters.
        Args:
            nms_threshold (float, optional): _description_. Defaults to 0.4.
        """
        self.session.prepare(**kwargs)
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape
        if len(self.infer_shape) == 3:
            self.infer_shape = (1, *self.infer_shape)
        # compute proposal length
        max_proposal_len = self._get_max_prop_len(
            self.infer_shape, self._feat_stride_fpn, self._num_anchors
        )
        self.score_list = np.zeros((max_proposal_len, 1), dtype=np.float32)
        self.bbox_list = np.zeros((max_proposal_len, 4), dtype=np.float32)
        self.kps_list = np.zeros((max_proposal_len, 10), dtype=np.float32)

    def predict(self, imgs, score_threshold: float = 0.5, nms_threshold: float = 0.4):
        """Run detection pipeline for input imgs

        Args:
            imgs (_type_): _description_
            score_threshold (float, optional): score threshold. Defaults to 0.5.
            nms_threshold (float, optional): nms threshold. Defaults to 0.4.
        Returns:
            _type_: _description_
        """
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            if len(imgs) == 1:
                imgs = np.expand_dims(imgs[0], 0)
            else:
                imgs = np.stack(imgs)
        elif isinstance(imgs, np.ndarray):
            imgs = np.expand_dims(imgs, 0)

        input_height = imgs[0].shape[0]
        input_width = imgs[0].shape[1]
        blob = self._preprocess(imgs)
        net_outs = self._forward(blob)

        dets_list = []
        kpss_list = []
        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(
            net_outs, input_height, input_width, score_threshold
        )

        # nms
        for e in range(self.infer_shape[0]):
            det, kpss = filter(
                bboxes_by_img[e], kpss_by_img[e], scores_by_img[e], nms_threshold
            )
            dets_list.append(det)
            kpss_list.append(kpss)
        return dets_list, kpss_list

    def _preprocess(self, img: np.ndarray):
        blob = normalize_on_np(img)
        return blob

    def _forward(self, blob):
        """
        send input data to infer backend
        Args:
            blob (_type_): _description_
        """
        assert blob is not None
        output = self.session.run(blob)
        return output

    def _postprocess(self, net_outputs, input_height, input_width, threshold):
        key = (input_height, input_width)
        if not self.center_cache.get(key):
            self.center_cache[key] = self._build_anchors(
                input_height, input_width, self._feat_stride_fpn, self._num_anchors
            )
        anchor_centers = self.center_cache[key]

        reshape_net_outputs = [
            np.expand_dims(out, 0) if len(out.shape) == 2 else out
            for out in net_outputs
        ]

        bboxes, kpss, scores = self._process_strides(
            reshape_net_outputs, threshold, anchor_centers
        )
        return bboxes, kpss, scores

    @staticmethod
    def _get_max_prop_len(input_shape, feat_strides, num_anchors):
        """
        Estimate maximum possible number of proposals returned by network

        :param input_shape: maximum input shape of model (i.e (1, 3, 640, 640))
        :param feat_strides: model feature strides (i.e. [8, 16, 32])
        :param num_anchors: model number of anchors (i.e 2)
        :return:
        """

        ln = 0
        pixels = input_shape[2] * input_shape[3]
        for e in feat_strides:
            ln += pixels / (e * e) * num_anchors
        return int(ln)

    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
            centers.append(anchor_centers)
        return centers

    def _process_strides(self, net_outs, threshold, anchor_centers):
        """
        Process network outputs by strides and return results proposals filtered by threshold

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """

        offset = 0

        batch_size = self.infer_shape[0]
        bboxes_by_img = []
        kpss_by_img = []
        scores_by_img = []

        for n_img in range(batch_size):
            for idx, stride in enumerate(self._feat_stride_fpn):
                score_blob = net_outs[idx][n_img]
                bbox_blob = net_outs[idx + self.fmc][n_img]
                kpss_blob = net_outs[idx + self.fmc * 2][n_img]
                stride_anchors = anchor_centers[idx]
                self.score_list, self.bbox_list, self.kps_list, total = (
                    generate_proposals(
                        score_blob,
                        bbox_blob,
                        kpss_blob,
                        stride,
                        stride_anchors,
                        threshold,
                        self.score_list,
                        self.bbox_list,
                        self.kps_list,
                        offset,
                    )
                )
                offset = total

            bboxes_by_img.append(self.bbox_list[:offset])
            kpss_by_img.append(self.kps_list[:offset])
            scores_by_img.append(self.score_list[:offset])

        return bboxes_by_img, kpss_by_img, scores_by_img


def regsiter_scrfd_detector(register: Registry):
    register(fn=SCRFD, name=SCRFD.__name__, namespace="detectors", type="face_det")

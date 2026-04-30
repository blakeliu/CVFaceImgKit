from faceimagekit.core import Registry
from .ppliteseg import regsiter_ppliteseg_segment

SEGMENTERS = Registry("segmenters")
regsiter_ppliteseg_segment(SEGMENTERS)
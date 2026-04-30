from faceimagekit.core import Registry
from .scrfd import regsiter_scrfd_detector

DETECTORS = Registry("detectors")
regsiter_scrfd_detector(DETECTORS)
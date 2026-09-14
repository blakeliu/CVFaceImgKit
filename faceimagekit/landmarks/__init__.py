from faceimagekit.core import Registry

from .rtmpose import regsiter_rtmpose_landmarks

LANDMARKERS = Registry("landmarkers")
regsiter_rtmpose_landmarks(LANDMARKERS)

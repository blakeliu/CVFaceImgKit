from __future__ import annotations

import sys

MIN_PY_VERSION: tuple[int, int] = (
    3,
    8,
)
__version__: str = "1.1.0"


def _versify(py_version_info: tuple[int, ...]) -> str:
    return ".".join([str(x) for x in py_version_info])


def _check_version() -> None:
    """check FaceImagekit version

    Raises:
        RuntimeError: _description_
    """
    py_version_info: tuple[int, int] = sys.version_info[:2]

    if py_version_info < MIN_PY_VERSION:
        error_msg = (
            "This version of pytextrank requires Python {} or later ({} detected)\n"
        )
        raise RuntimeError(
            error_msg.format(_versify(MIN_PY_VERSION), _versify(py_version_info))
        )

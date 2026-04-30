import functools
import importlib
import operator
import types
from importlib.util import find_spec
from typing import List, Union

from importlib.metadata import PackageNotFoundError

try:
    from packaging.version import Version
except (ModuleNotFoundError, PackageNotFoundError):
    Version = None


def module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False
    except ValueError:
        # Sometimes __spec__ can be None and gives a ValueError
        return True


def compare_version(package: str, op, version) -> bool:
    """Compare package version with some requirements.

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, PackageNotFoundError, ValueError):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True to generate all summaries
        return True
    return op(pkg_version, Version(version))

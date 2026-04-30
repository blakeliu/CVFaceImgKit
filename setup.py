import typing
import pathlib
import importlib
import setuptools

KEYWORDS = [
    "face detection",
    "face landmark",
    "opencv",
    "onnxruntime",
]

def parse_requirements_file (filename: str) -> typing.List:
    """read and parse a Python `requirements.txt` file, returning as a list of str"""
    results: list = []

    with pathlib.Path(filename).open() as f:
        for l in f.readlines():
            results.append(l.strip().replace(" ", "").split("#")[0])

    return results

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("faceimagekit.version", "faceimagekit/version.py")
    pytr_version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pytr_version)
    pytr_version._check_version()  # pylint: disable=W0212

    _packages = parse_requirements_file("requirements.txt")

    setuptools.setup(
        name="faceimagekit",
        version = pytr_version.__version__,

        python_requires = ">=" + pytr_version._versify(pytr_version.MIN_PY_VERSION),  # pylint: disable=W0212
        packages = setuptools.find_packages(exclude=[ "converters", "examples", ".vscode" ]),
        install_requires = _packages,

        author="blakeliu",
        author_email="blake120386@163.com",
        license="AGPL-3.0-only OR Commercial License",

        description="Python implementation of face detection, face landmark, face segmentation and so on.",
        long_description = pathlib.Path("README.md").read_text(encoding='utf-8'),
        long_description_content_type = "text/markdown",
        include_package_data=True,
        package_data={
            "weights/scrfd/onnx": ["*.onnx"],
            },
        

        keywords = ", ".join(KEYWORDS),
        classifiers = [
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: GNU Affero General Public License v3",
            "Operating System :: OS Independent",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Computer Vision",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Computer Vision",
            "Topic :: Face Analysis :: General",
            ],

        # url = "",
        project_urls = {
            "Source": "https://github.com/blakeliu/CVFaceImgKit.git",
            "onnxruntime": "https://github.com/microsoft/onnxruntime",
            },

        zip_safe=False,
        )

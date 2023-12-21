import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from os import path

from setuptools import find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "ipython>=8.14.0",
    "matplotlib>=3.5.2",
    "networkx>=2.6.3",
    "numpy>=1.20.3",
    "requests>=2.26.0",
    "scipy>=1.8.1",
    "setuptools>=58.0.4",
    "sparse>=0.13.0",
    "scikit-build>=0.16.1",
    "pybind11>=2.10.3",
    "graphviz>=0.14.2",
    "ply~=3.11",
]

setup(
    name="pyquafu",
    version="0.4.0",
    author="ssli",
    author_email="ssli@iphy.ac.cn",
    url="https://github.com/ScQ-Cloud/pyquafu",
    description="Python toolkit for Quafu-Cloud",
    install_requires=requirements,
    packages=find_packages(exclude=["test*"]),
    cmake_install_dir="quafu/simulators/",
    include_package_data=True,
    package_data={"quafu": ["qfasm/*.inc"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={"test": ["pytest"]},
    python_requires=">=3.8",
    zip_safe=False,
    setup_cfg=True,
    license="Apache-2.0 License",
)

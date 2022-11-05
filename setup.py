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

from setuptools import find_packages

requirements = [
    "matplotlib>=3.5.2",
    "networkx>=2.6.3",
    "numpy>=1.20.3",
    "requests>=2.26.0",
    "scipy>=1.8.1",
    "sparse>=0.13.0"
]

setup(
    name="pyquafu",
    version="0.2.8",
    author="ssli",
    author_email="ssli@iphy.ac.cn",
    url="https://github.com/ScQ-Cloud/pyquafu",
    description="Python toolkit for Quafu-Clound",
    install_requires=requirements,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/quafu/simulators/",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.9",
    zip_safe=False,
    setup_cfg=True,
    license="Apache-2.0 License"
)
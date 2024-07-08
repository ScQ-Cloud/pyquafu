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

def parse_requirements(filename):
     with open(filename, 'r') as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

requirements = parse_requirements("requirements.txt")

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
    cmake_args=["-DCMAKE_BUILD_TYPE:STRING=Debug"],
    python_requires=">=3.8",
    zip_safe=False,
    setup_cfg=True,
    license="Apache-2.0 License",
)

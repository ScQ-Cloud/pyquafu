# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup file for pyquafu."""

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
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]


requirements = parse_requirements("requirements.txt")

setup(
    name="pyquafu",
    version="0.4.3",
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

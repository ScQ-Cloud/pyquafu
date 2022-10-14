from setuptools import find_packages, setup 
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = [
    "matplotlib>=3.5.2",
    "networkx>=2.6.3",
    "numpy>=1.20.3",
    "requests>=2.26.0",
    "scipy>=1.8.1",
    "sparse>=0.13.0"
]

setup(name="pyquafu",
    version="0.2.6",
    author="ssli",
    author_email="ssli@iphy.ac.cn",
    url="https://github.com/ScQ-Cloud/pyquafu",
    description="Python toolkit for Quafu-Clound",
    install_requires=requirements,
    python_requires='>=3.8',
    packages=find_packages('src'),
    package_dir={'':'src'},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_cfg=True,
    license="Apache-2.0 License"
)
from setuptools import find_packages, setup 

setup(name="quafu",
    version="0.2.3",
    author="ssli",
    author_email="ssli@iphy.ac.cn",
    url="https://github.com/ScQ-Cloud/quafu",
    description="Python toolkit for Quafu-Clound",
    install_requires=["numpy", "matplotlib", "sparse"],
    packages=find_packages('src'),
    package_dir={'':'src'},
    include_package_data=True,
    zip_safe=False,
    setup_cfg=True,
    license="Apache-2.0 License"
)
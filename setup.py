from setuptools import find_packages, setup 

setup(name="scqkit",
    version="0.1.1",
    description="Python toolkit for ScQ-Clound",
    install_requires=["numpy", "matplotlib", "qutip"],
    packages=find_packages('src'),
    package_dir={'':'src'},
    include_package_data=True,
    zip_safe=False,
    setup_cfg=True,
    license="apache 3.0"
)
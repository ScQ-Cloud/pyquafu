# PyQuafu
[![License](https://img.shields.io/github/license/ScQ-Cloud/pyquafu.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![](https://github.com/ScQ-Cloud/pyquafu/actions/workflows/unittest.yml/badge.svg)](https://github.com/ScQ-Cloud/pyquafu/actions/workflows/unittest.yml)
[![](https://img.shields.io/github/release/ScQ-Cloud/pyquafu.svg?style=popout-square)](https://github.com/ScQ-Cloud/pyquafu/releases)
[![](https://img.shields.io/pypi/dm/pyquafu?style=popout-square)](https://pypi.org/project/pyquafu/)

A Python toolkit for submitting quantum circuits to the superconducting quantum computing cloud [Quafu](http://quafu.baqis.ac.cn/).

## Introduction

**PyQuafu** is designed for users of [Quafu](http://quafu.baqis.ac.cn/) to construct, compile, and execute quantum circuits on real quantum devices. With PyQuafu, you can interact with various quantum backends provided by the experimental group at [Quafu](http://quafu.baqis.ac.cn/).

## Installation

### Install via PyPI

You can install PyQuafu directly from PyPI:

```bash
pip install pyquafu
```

### Build from Source

Alternatively, you can build PyQuafu from the source:

```bash
pip install -r requirements.txt
python setup.py install
```

### Graphviz Dependency

If you need to visualize Directed Acyclic Graphs (DAGs), ensure that the [Graphviz software](https://graphviz.org/) is installed on your system. Refer to the [graphviz · PyPI](https://pypi.org/project/graphviz/#description) page for installation guidance.

### GPU Support

To install PyQuafu with GPU-based circuit simulation, you need to build from the source and ensure that the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is installed. Use the following command to install the GPU version:

```bash
python setup.py install -DUSE_GPU=ON
```

If you also have [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) installed, you can install PyQuafu with cuQuantum support:

```bash
python setup.py install -DUSE_GPU=ON -DUSE_CUQUANTUM=ON
```

## Documentation

For detailed documentation about usage, please visit the [PyQuafu documentation website](https://scq-cloud.github.io/).

## Note for Apple Silicon Mac Users

If you encounter the error "illegal hardware instruction" on an Apple silicon Mac, ensure that you have updated to the arm64 version of Anaconda. See [this issue](https://github.com/abess-team/abess/issues/310) for more details.

## Examples

### Quantum Reinforcement Learning

This example demonstrates how quantum reinforcement learning interacts with Quafu to solve the CartPole environment. For more details, refer to the [quantum-RL-with-quafu repository](https://github.com/enchanted123/quantum-RL-with-quafu).

## Author

This project is developed by the quantum cloud computing team at the Beijing Academy of Quantum Information Sciences.

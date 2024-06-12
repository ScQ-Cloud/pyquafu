# PyQuafu
[![License](https://img.shields.io/github/license/ScQ-Cloud/pyquafu.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![](https://github.com/ScQ-Cloud/pyquafu/actions/workflows/unittest.yml/badge.svg)](https://github.com/ScQ-Cloud/pyquafu/actions/workflows/unittest.yml)
[![](https://img.shields.io/pypi/dm/pyquafu?style=popout-square)](https://pypi.org/project/pyquafu/)


Python toolkit for submitting quantum circuits on the superconducting quantum computing cloud [Quafu](http://quafu.baqis.ac.cn/).


## Introduction

PyQuafu is developed for the users of [Quafu](http://quafu.baqis.ac.cn/) to construct, compile and execute quantum circuits on real quantum devices. One can use PyQuafu to interact with different quantum backends provides by the experimental group of [Quafu](http://quafu.baqis.ac.cn/).

## Installation

You can directly install via PyPI,

```
pip install pyquafu
```

or build from source

```
pip install -r requirements.txt
python setup.py install
```

Note that we visualize DAG(directed acyclic graph) through python package ``graphviz``. And if you need it, make sure [Graphviz software](https://graphviz.org/) being installed on your system. Refer to [graphviz · PyPI](https://pypi.org/project/graphviz/#description) for installation guidance.

## GPU support
To install PyQuafu with GPU-based circuit simulator, you need build from the source and make sure that [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is installed. You can run

```
python setup.py install -DUSE_GPU=ON
```
to install the GPU version. If you further have [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) installed, you can install PyQuafu with cuQuantum support.
```
python setup.py install -DUSE_GPU=ON -DUSE_CUQUANTUM=ON
```


## Document
Please see the website [docs](https://scq-cloud.github.io/).

## Note
If you are using an Apple silicon Mac and meet the error "illegal hardware instruction", please confirm whether you have updated to the arm64 version of Anaconda (see https://github.com/abess-team/abess/issues/310).

## Examples

### 1.quantum_rl

The example shows quantum reinforcement learning interacts with Quafu to solve CartPole environment.

Refer to https://github.com/enchanted123/quantum-RL-with-quafu for more details.

## Author
This project is developed by the quantum cloud computing team at the Beijing Academy of Quantum Information Sciences.

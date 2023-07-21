# PyQuafu

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

## Authors
This project is developed by the quantum cloud computing team at the Beijing Academy of Quantum Information Sciences.

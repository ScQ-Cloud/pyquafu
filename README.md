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

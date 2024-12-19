# pyquafu

[查看中文](./README_CN.md)

<!-- TOC --->

- [pyquafu](#pyquafu)
  - [What is pyquafu](#what-is-pyquafu)
  - [First experience](#first-experience)
    - [Build parameterized quantum circuit](#build-parameterized-quantum-circuit)
    - [Train quantum neural network](#train-quantum-neural-network)
  - [Installation](#installation)
    - [Confirming System Environment Information](#confirming-system-environment-information)
    - [Install by Source Code](#install-by-source-code)
    - [Install by pip](#install-by-pip)
      - [Install pyquafu](#install-pyquafu)
    - [Build from source](#build-from-source)
  - [Verifying Successful Installation](#verifying-successful-installation)
  - [Install with Docker](#install-with-docker)
  - [Note](#note)
  - [Building binary wheels](#building-binary-wheels)
  - [License](#license)

<!-- /TOC -->

## What is pyquafu

pyquafu is forked from [MindQuantum](https://gitee.com/mindspore/quafu/). Based on MindQuantum, pyquafu can run quantum algorithm directly on the quantum cloud developed by Beijing Academy of Quantum Information Sciences.

## First experience

### Build parameterized quantum circuit

The below example shows how to build a parameterized quantum circuit.

```python
from pyquafu import *
import numpy as np

encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi / 2, 'a1': np.pi / 2}, ket=True))
```

Then you will get,

```bash
      ┏━━━┓ ┏━━━━━━━━━━┓
q0: ──┨ H ┠─┨ RX(2*a0) ┠───
      ┗━━━┛ ┗━━━━━━━━━━┛
      ┏━━━━━━━━┓
q1: ──┨ RY(a1) ┠───────────
      ┗━━━━━━━━┛

-1/2j¦00⟩
-1/2j¦01⟩
-1/2j¦10⟩
-1/2j¦11⟩
```

In jupyter notebook, we can just call `svg()` of any circuit to display the circuit in svg picture (`dark` and `light` mode are also supported).

```python
circuit = (qft(range(3)) + BarrierGate(True)).measure_all()
circuit.svg()  # circuit.svg('light')
```

![circuit_svg](./docs/circuit_svg.png)

### Train quantum neural network

```python
ansatz = CPN(encoder.hermitian(), {'a0': 'b0', 'a1': 'b1'})
sim = Simulator('mqvector', 2)
ham = Hamiltonian(-QubitOperator('Z0 Z1'))
grad_ops = sim.get_expectation_with_grad(
    ham,
    encoder.as_encoder() + ansatz.as_ansatz(),
)

import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
net = MQLayer(grad_ops)
encoder_data = ms.Tensor(np.array([[np.pi / 2, np.pi / 2]]))
opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
train_net = ms.nn.TrainOneStepCell(net, opti)
for i in range(100):
    train_net(encoder_data)
print(dict(zip(ansatz.params_name, net.trainable_params()[0].asnumpy())))
```

The trained parameters are,

```bash
{'b1': 1.5720831, 'b0': 0.006396801}
```

## Installation

### Confirming System Environment Information

- The hardware platform should be CPU with avx2 supported.

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://github.com/ScQ-Cloud/pyquafu.git
```

2.Compiling pyquafu

```bash
cd ~/pyquafu
bash build.sh
cd output
pip install pyquafu-*.whl
```

### Install by pip

#### Install pyquafu

```bash
pip install pyquafu
```

### Build from source

1. Clone source.

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/pyquafu.git
    ```

2. Build pyquafu

    For **linux system**, please make sure your cmake version >= 3.18.3, and then run code:

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee
    ```

    Here `--gitee` is telling the script to download third party from gitee. If you want to download from github, you can ignore this flag. If you want to build under GPU, please make sure you have install CUDA 11.x and the GPU driver, and then run code:

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee --gpu
    ```

    For **windows system**, please make sure you have install MinGW-W64 and CMake >= 3.18.3, and then run:

    ```bash
    cd ~/pyquafu
    ./build.bat /Gitee
    ```

    For **Mac system**, please make sure you have install openmp and CMake >= 3.18.3, and then run:

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee
    ```

3. Install whl

    Please go to output, and install pyquafu wheel package by `pip`.

## Verifying Successful Installation

Successfully installed, if there is no error message such as No module named 'pyquafu' when execute the following command:

```bash
python -c 'import pyquafu'
```

## Install with Docker

Mac or Windows users can install pyquafu through Docker. Please refer to [Docker installation guide](./install_with_docker_en.md)

## Note

Please set the parallel core number before running pyquafu scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.

## Building binary wheels

If you would like to build some binary wheels for redistribution, please have a look to our [binary wheel building guide](./INSTALL.md)

## License

[Apache License 2.0](LICENSE)

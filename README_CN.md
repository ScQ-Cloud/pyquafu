# pyquafu

[View English](./README.md)

<!-- TOC --->

- [pyquafu](#pyquafu)
  - [pyquafu介绍](#pyquafu介绍)
  - [初体验](#初体验)
    - [搭建参数化量子线路](#搭建参数化量子线路)
    - [训练量子神经网络](#训练量子神经网络)
  - [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [pip安装](#pip安装)
      - [安装pyquafu](#安装pyquafu)
    - [源码安装](#源码安装)
  - [验证是否成功安装](#验证是否成功安装)
  - [Docker安装](#docker安装)
  - [注意事项FAQ](#注意事项faq)
  - [构建二进制whl包](#构建二进制whl包)
  - [许可证](#许可证)

<!-- /TOC -->

## pyquafu介绍

pyquafu 从 [MindQuantum](https://gitee.com/mindspore/mindquantum/) fork而来，在此基础上，pyquafu可以直接在北京量子信息科学研究院所搭建的quafu量子计算云平台上执行量子算法。

## 初体验

### 搭建参数化量子线路

通过如下示例可便捷搭建参数化量子线路

```python
from pyquafu import *
import numpy as np

encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi / 2, 'a1': np.pi / 2}, ket=True))
```

你将得到

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

在jupyter notebook中，也可通过线路的`svg()`接口来以svg格式绘制量子线路图（更有`dark`和`light`模式可选）

```python
circuit = (qft(range(3)) + BarrierGate(True)).measure_all()
circuit.svg()  # circuit.svg('light')
```

![circuit_svg](./docs/circuit_svg.png)

### 训练量子神经网络

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

训练得到参数为

```bash
{'b1': 1.5720831, 'b0': 0.006396801}
```

## 安装教程

### 确认系统环境信息

- 硬件平台支持avx2指令集。

### pip安装

#### 安装pyquafu

```bash
pip install pyquafu
```

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://github.com/ScQ-Cloud/pyquafu.git
    ```

2. 编译pyquafu

    **Linux系统**下请确保安装好CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee
    ```

    这里 `--gitee` 让脚本从gitee代码托管平台下载第三方依赖。如果需要编译GPU版本，请先安装好 CUDA 11.x，和对应的显卡驱动，然后执行如下编译指令：

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee --gpu
    ```

    **Windows系统**下请确保安装好MinGW-W64和CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/pyquafu
    ./build.bat /Gitee
    ```

    **Mac系统**下请确保安装好openmp和CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/pyquafu
    bash build.sh --gitee
    ```

3. 安装编译好的whl包

    进入output目录，通过`pip`命令安装编译好的pyquafu的whl包。

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'pyquafu'`，则说明安装成功。

```bash
python -c 'import pyquafu'
```

## Docker安装

通过Docker也可以在Mac系统或者Windows系统中使用pyquafu。具体参考[Docker安装指南](./install_with_docker.md).

## 注意事项FAQ

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。

更多注意事项请查看[FAQ页面](https://gitee.com/mindspore/pyquafu/blob/r0.8/tutorials/0.frequently_asked_questions.ipynb)。

## 构建二进制whl包

如果你想构建用于分发的二进制whl包，请参考[二进制whl包构建指南](./INSTALL_cn.md)

## 许可证

[Apache License 2.0](LICENSE)

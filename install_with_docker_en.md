# quafu Docker Installation GUide

<!-- TOC --->

- [Installing quafu by Docker](#installing-quafu-by-docker)
    - [Obtaining the MindSpore Image](#obtaining-the-mindSpore-image)
    - [Running the MindSpore Image](#running-the-mindspore-image)
    - [MindSpore Installation Verification](#mindspore-installation-verification)
    - [Installing quafu Inside a Docker Container](#installing-quafu-inside-a-docker-container)
    - [quafu Installation Verification](#quafu-installation-verification)

<!-- TOC --->

## Installing quafu by Docker

This document describes how to use Docker to quickly install quafu. First, you need to install MindSpore through Docker. The process is introduced on [MindSpore's official website](https://www.mindspore.cn/install/en). This part will be repeated below.

### Obtaining the MindSpore Image

For the CPU backend, you can directly use the following command to obtain the latest stable image:

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag}
```

In which, {tag} corresponds to the tag in the above table.

### Running the MindSpore Image

Execute the following command to start the Docker container instance:

```shell
docker run -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag} /bin/bash
```

In which, {tag} corresponds to the tag in the above table.

### MindSpore Installation Verification

- If you are installing a container of the specified version x.y.z.

After entering the MindSpore container according to the above steps, to test whether Docker is working properly, please run the following Python code and check the output:

```python
import numpy as np
import mindspore as ms
from mindspore import set_context, ops, Tensor

set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
print(ops.add(x, y))
```

When the code runs successfully, it will output:

```text
[[[[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]],

  [[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]],

  [[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]]]]
```

At this point, you have successfully installed the MindSpore CPU version by Docker.

### Installing quafu Inside a Docker Container

1. Enter the Docker container.

    ```shell
    docker exec -it {docker_container} /bin/bash
    ```

    In which, {docker_container} is the id or name of the docker container.

2. Choose to install by pip or by compiling.

    **Install by compiling:**

    ```shell
    git clone https://gitee.com/mindspore/quafu.git
    cd ~/quafu
    python setup.py install --user
    ```

    **Install by pip:**

    ```shell
    pip install https://hiq.huaweicloud.com/download/quafu/newest/linux/quafu-master-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### quafu Installation Verification

```bash
python -c 'import quafu'
```

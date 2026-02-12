安装说明
============

RLinf 支持多种后端引擎，用于训练和推理。目前支持以下配置：

- **Megatron** 和 **SGLang/vLLM**：用于训练 MATH 任务中的大语言模型（LLM）。
- **FSDP** 和 **Huggingface**：用于训练 LIBERO 和 ManiSkill3 环境下的 VLA 模型。

后端引擎
---------------

1. **训练引擎**

   - **FSDP**：简单高效、适合初学者，兼容性强，使用便捷，支持原生 PyTorch 模块。

   - **Megatron**：为追求极致性能的开发者设计，支持多种并行配置，具备先进的训练速度和可扩展性。

2. **推理引擎**

   - **SGLang/vLLM**：成熟且广泛使用，具备许多高级功能和优化能力。

   - **Huggingface**：简单易用，配套 Huggingface 生态提供的原生 API。

硬件要求
~~~~~~~~~~~~~~~~~~~~~~~

以下是经过充分测试的硬件配置：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 配置
   * - GPU
     - 每个节点 8 块 H100
   * - CPU
     - 每个节点 192 核心
   * - 内存
     - 每个节点 1.8TB
   * - 网络
     - NVLink + RoCE / IB，带宽 3.2 Tbps
   * - 存储
     - | 单节点实验使用 1TB 本地存储  
       | 分布式实验使用 10TB 共享存储（NAS）

软件要求
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 版本
   * - 操作系统
     - Ubuntu 22.04
   * - NVIDIA 驱动
     - 535.183.06
   * - CUDA
     - 12.4
   * - Docker
     - 26.0.0
   * - NVIDIA Container Toolkit
     - 1.17.8

安装方式
--------------------

RLinf 提供两种安装方式。我们 **推荐使用 Docker**，因为这可以提供最快速、最可复现的环境。  
如果你的系统无法使用 Docker 镜像，也可以选择在本地 Python 环境中手动安装。

安装方式1： Docker 镜像
-------------------------

我们提供了两个官方镜像，分别针对不同的实验场景：

- **具身智能镜像：**

  - ``rlinf/rlinf:agentic-rlinf0.1-maniskill_libero`` （适用于Libero或ManiSkill基准测试环境，对于其它的基准测试环境，请参考 :doc:`../examples/index`）
  
- **数学推理镜像：**

  - ``rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1`` （用于增强大语言模型在 MATH 任务中的推理能力）


确认适合你任务的镜像后，拉取镜像：

.. code-block:: bash

  # 对于中国大陆用户，可以使用以下方式加速下载：
  # docker.1ms.run/rlinf/rlinf:CHOSEN_IMAGE
  docker pull rlinf/rlinf:CHOSEN_IMAGE

然后启动容器：

.. warning::

  1. 请确保使用 `-e NVIDIA_DRIVER_CAPABILITIES=all` 启动 docker，以启用 GPU 支持（其中至少包含 `compute`、`utility`、`graphics` 能力，具身实验中的渲染依赖 `graphics`）。

  2. 请勿覆盖容器内的 `/root` 和 `/opt` 目录（通过 `docker run` 的 `-v` 或 `--volume`），因为它们包含重要的资源文件和环境。如果你的平台一定会挂载 `/root`，请在启动容器后在容器内运行 `link_assets` 来恢复 `/root` 目录中的资源链接。

  3. 请避免更改 `$HOME` 环境变量（例如通过 `docker run -e HOME=/new_home` ），该变量默认应为 `/root`。ManiSkill 和其他工具依赖此路径查找需要的资源。如果您在镜像中运行脚本之前更改了 `$HOME`，请执行 `link_assets` 将资源重新链接到新的 `$HOME`。

.. code-block:: bash

  docker run -it --gpus all \
    --shm-size 100g \
    --net=host \
    --name rlinf \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    rlinf/rlinf:CHOSEN_IMAGE /bin/bash

进入容器后，克隆 RLinf 仓库：

.. code-block:: bash

  # 为提高国内下载速度，可以使用：
  # git clone https://ghfast.top/github.com/RLinf/RLinf.git
  git clone https://github.com/RLinf/RLinf.git
  cd RLinf

具身智能镜像中包含多个 Python 虚拟环境（venv），位于 ``/opt/venv`` 目录下，分别对应不同模型，即 ``openvla``、``openvla-oft`` 和 ``openpi``。
默认环境设置为 ``openvla``。
要切换到所需的 venv，可以使用内置脚本 `switch_env`：

.. code-block:: bash

   source switch_env <env_name>
   # source switch_env openvla
   # source switch_env openvla-oft
   # source switch_env openpi

.. note::

  `link_assets` 和 `switch_env` 脚本是我们提供的 Docker 镜像中的内置工具。您可以在 `/usr/local/bin` 中找到它们。

安装方式2：UV 自定义环境
-------------------------------
**如果你已经使用了 Docker 镜像，下面步骤可跳过。**

可以运行 `requirements/install.sh` 脚本安装目标实验所需的依赖。
该脚本通过 *target* 和 *model* 两个维度组织：

- ``embodied`` target（具身智能相关），支持通过 `--model` 参数选择不同模型，例如 ``openvla``， ``openvla-oft`` 或 ``openpi``

  每个模型还需要通过 ``--env`` 参数指定基准测试环境，例如 ``maniskill_libero``、 ``behavior`` 或 ``metaworld``。

- ``reason`` target（推理 / Megatron 等相关）。

- ``docs`` target（用于构建文档）。

例如，要安装 OpenVLA + ManiSkill LIBERO 实验的依赖，可以运行：

.. code-block:: shell

  cd <path_to_RLinf_repository>
  # 对于国内用户，可以在 install.sh 命令中添加 `--use-mirror` 以加速下载。
  bash requirements/install.sh embodied --model openvla --env maniskill_libero

这将在当前路径下创建一个名为 `.venv` 的虚拟环境。
要激活该虚拟环境，可以使用以下命令：

.. code-block:: shell
  
  source .venv/bin/activate

要退出虚拟环境，只需运行：

.. code-block:: shell

  deactivate

如果你希望安装 MATH 推理相关（Megatron + SGLang/vLLM）环境，可以运行：

.. code-block:: shell

  bash requirements/install.sh reason

你也可以通过 ``--venv`` 参数覆盖默认虚拟环境目录，例如：

.. code-block:: shell

  bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv openpi-venv
  source openpi-venv/bin/activate
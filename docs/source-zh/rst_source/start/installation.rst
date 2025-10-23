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

我们提供了两个官方镜像，分别针对不同后端配置进行了优化：

- **基于Megatron + SGLang/vLLM的数学推理镜像**：

  - ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1`` （用于增强大语言模型在 MATH 任务中的推理能力）

- **基于FSDP + Huggingface的具身智能镜像**：

  - ``rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0`` （适用于 OpenVLA/OpenVLA-OFT/OpenPI 模型）

确认适合你任务的镜像后，拉取镜像：

.. code-block:: bash

   docker pull rlinf/rlinf:CHOSEN_IMAGE

然后启动容器：

.. warning::

  1. 请确保使用 `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` 启动 docker，以启用 GPU 支持，尤其是具身实验中渲染所需的 `graphics` 功能。

  2. 请勿覆盖容器内的 `/root` 和 `/opt` 目录（通过 `docker run` 的 `-v` 或 `--volume`），因为它们包含重要的资源文件和环境。如果你的平台一定会挂载 `/root`，请在启动容器后在容器内运行 `link_assets` 来恢复 `/root` 目录中的资源链接。

  3. 请避免更改 `$HOME` 环境变量（例如通过 `docker run -e HOME=/new_home` ），该变量默认应为 `/root`。ManiSkill 和其他工具依赖此路径查找需要的资源。如果您在镜像中运行脚本之前更改了 `$HOME`，请执行 `link_assets` 将资源重新链接到新的 `$HOME`。

.. code-block:: bash

   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --name rlinf \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      rlinf/rlinf:CHOSEN_IMAGE /bin/bash

进入容器后，克隆 RLinf 仓库：

.. code-block:: bash

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

.. tip::

   如果进行多节点训练，请将仓库克隆到共享存储路径，确保每个节点都能访问该代码。

安装方式2：UV 自定义环境
-------------------------------
**如果你已经使用了 Docker 镜像，下面步骤可跳过。**

我们推荐使用 `uv <https://docs.astral.sh/uv/>`_ 工具来安装所需的 Python 包。  
您可以通过 `pip` 安装 ``uv``。

.. code-block:: shell

   pip install --upgrade uv

安装完成后，你可以运行`requirements/install.sh`脚本安装目标实验所需的依赖。
该脚本接受一个参数，指定目标实验，包括 `openvla`、`openvla-oft`、`openpi` 和 `reason`。
例如，要安装 openvla 实验的依赖，可以运行：

.. note:: 

  该脚本需要在 RLinf 仓库的根目录下运行。请确保不要在 `requirements/` 目录下运行该脚本。

.. code-block:: shell
  
  bash requirements/install.sh openvla

这将在当前路径下创建一个名为 `.venv` 的虚拟环境。
要激活该虚拟环境，可以使用以下命令：

.. code-block:: shell
  
  source .venv/bin/activate

要退出虚拟环境，只需运行：

.. code-block:: shell

  deactivate
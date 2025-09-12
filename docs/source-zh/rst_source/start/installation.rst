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

安装方式
--------------------

RLinf 提供两种安装方式。我们 **推荐使用 Docker**，因为这可以提供最快速、最可复现的环境。  
如果你的系统无法使用 Docker 镜像，也可以选择在本地 Python 环境中手动安装。

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

使用 Docker 镜像安装
-------------------------

我们提供了两个官方镜像，分别针对不同后端配置进行了优化：

- **Megatron + SGLang/vLLM**：

  - ``rlinf/rlinf:math-rlinf0.1-torch2.5.1-sglang0.4.4-vllm0.7.1-megatron0.11.0-te2.1`` （用于增强大语言模型在 MATH 任务中的推理能力）

- **FSDP + Huggingface**：

  - ``rlinf/rlinf:agentic-openvla-rlinf0.1-torch2.5.1`` （适用于 OpenVLA 模型）  
  - ``rlinf/rlinf:agentic-openvlaoft-rlinf0.1-torch2.5.1`` （适用于 OpenVLA-OFT 模型）

确认适合你任务的镜像后，拉取镜像：

.. code-block:: bash

   docker pull rlinf/rlinf:CHOSEN_IMAGE

然后启动容器：

.. code-block:: bash

   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      --name rlinf \
      rlinf/rlinf:CHOSEN_IMAGE /bin/bash

进入容器后，克隆 RLinf 仓库：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

.. tip::

   如果进行多节点训练，请将仓库克隆到共享存储路径，确保每个节点都能访问该代码。

自定义环境安装
-------------------------------

根据你的实验类型，安装分为三步进行：

第一步，对于所有实验，请先完成 :ref:`共同依赖 <common-dependencies>` 中的依赖安装，  
这一步已经包括了 **FSDP + Huggingface** 的完整配置。

第二步，如果你的实验使用的是 **Megatron 和 SGLang/vLLM** 后端，  
请参考 :ref:`Megatron 及 SGLang/vLLM 依赖 <megatron-and-sglang-vllm-dependencies>` 安装相应依赖。

第三步，如果你要运行具身智能相关实验（如 OpenVLA、OpenVLA-OFT、Pi0），  
请参考 :ref:`具身智能依赖 <embodied-dependencies>` 安装专用依赖项。

.. _common-dependencies:

通用依赖安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们推荐使用 `uv <https://docs.astral.sh/uv/>`_ 工具来安装所需的 Python 包。  
如果你使用的是 `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_，可以通过 `pip` 安装 ``uv``。

.. code-block:: shell

   conda create -n rlinf python=3.11.10 -y
   conda activate rlinf
   pip install --upgrade uv

安装 ``uv`` 后，创建虚拟环境并安装 PyTorch 与通用依赖：

.. code-block:: shell

   uv venv
   source .venv/bin/activate
   UV_TORCH_BACKEND=auto uv sync

.. _megatron-and-sglang-vllm-dependencies:

Megatron 和 SGLang/vLLM 依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

运行以下命令，安装 Megatron、SGLang/vLLM 及其所需依赖：

.. code-block:: shell

   uv sync --extra sgl_vllm
   mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
   APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -r requirements/megatron.txt --no-build-isolation

使用 Megatron 前，请将其路径加入 ``PYTHONPATH`` 环境变量：

.. code-block:: shell

   export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

SGLang 安装：

.. code-block:: shell

   uv sync --extra sglang

vLLM 安装：

.. code-block:: shell

   uv sync --extra vllm

.. _embodied-dependencies:

具身智能相关依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若你运行的是具身智能实验，首先通过 apt 安装必要的系统依赖（仅支持 Debian/Ubuntu 系统）：

.. code-block:: shell

   bash requirements/install_embodied_deps.sh
   uv sync --extra embodied

接着，根据具体实验类型安装对应的 Python 包：

.. code-block:: shell

   # OpenVLA / OpenVLA-OFT 实验所需依赖
   UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation

   # Pi0 实验所需依赖
   UV_TORCH_BACKEND=auto uv pip install -r requirements/pi0.txt --no-build-isolation

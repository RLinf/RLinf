安装与硬件参考
==============

.. important::

   **想直接运行具体任务？无需阅读本页。** 每个模型、仿真器、真机与 Agent 教程页都自带完整的安装和运行命令，直接前往对应教程即可；首次体验可直接从 :doc:`快速开始 <vla>` 起步。本页仅作为 ``install.sh`` 全局参数、预构建 Docker 镜像、国内加速与硬件档位的参考手册使用。

使用 ``requirements/install.sh`` 在本地 ``uv`` 虚拟环境中安装 RLinf，或直接拉取 Docker Hub 预构建镜像。每个虚拟环境对应一套模型与环境组合，共存多套环境时传入 ``--venv <dir>`` 即可避免覆盖默认环境。

安装步骤
--------

先克隆仓库并进入根目录：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf

再按目标场景选择对应的安装命令：

.. tab-set::

   .. tab-item:: 具身仿真
      :sync: embodied

      按下方表格将 ``--model`` 与 ``--env`` 替换为目标模型和仿真器（轻量 MLP / CNN policy 可省略 ``--model``）：

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env maniskill_libero
         source .venv/bin/activate

   .. tab-item:: 真机机器人
      :sync: realworld

      在机器人控制节点安装驱动与传感器依赖（无需 GPU），在 GPU 节点安装模型与驱动。按下方表格把 ``franka`` 换成你的机型（完整真机列表及 Robbyant R2 的 ROS 2 接入见 :doc:`../examples/real_world_index`）：

      .. code-block:: bash

         # 机器人控制节点（驱动与传感器）
         bash requirements/install.sh embodied --env franka
         source .venv/bin/activate

         # GPU 训练 / 推理节点（或单机同配 GPU 与机器人）
         bash requirements/install.sh embodied --model openpi --env franka
         source .venv/bin/activate

   .. tab-item:: Agent 与 LLM
      :sync: agentic

      .. code-block:: bash

         bash requirements/install.sh agentic --engine sglang
         source .venv/bin/activate

   .. tab-item:: 预构建 Docker 镜像
      :sync: docker

      从 Docker Hub 选取匹配模型与仿真器的镜像标签后，启用 GPU 与 EGL 图形能力启动容器：

      .. code-block:: bash

         docker run -it --rm --gpus all --shm-size=16g --network=host \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -v "$(pwd)":/workspace/RLinf \
           rlinf/rlinf:<tag> bash

      .. grid:: 1 1 1 1
         :gutter: 2

         .. grid-item-card:: Docker Hub — rlinf/rlinf
            :link: https://hub.docker.com/r/rlinf/rlinf

            在 Docker Hub 查看全部预构建镜像标签、容器启动命令与国内镜像站说明。

   .. tab-item:: 文档构建
      :sync: docs

      .. code-block:: bash

         bash requirements/install.sh docs --venv .docs-venv

.. tip::

   **国内镜像加速：** 在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

支持的模型、环境与硬件平台
--------------------------

将下表中的名称直接传入 ``--model``、``--env``、``--engine`` 或 ``--platform``：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 参数
     - 支持的值
   * - ``--model``\ （:doc:`VLA <../examples/vla_wam_index>`）
     - ``abot_m0``、``dexbotic``、``evo1``、``gr00t``、``gr00t_n1d6``、``gr00t_n1d7``、``lingbotvla``、``lingbotvlav2``、``molmoact2``、``openpi``、``openvla``、``openvla-oft``、``pi0_fast``、``psi0``、``starvla``
   * - ``--model``\ （:doc:`WAM / VLM / 生成 <../examples/vla_wam_index>`）
     - ``cosmos3``、``diffusion``、``dreamzero``、``fastwam``、``qwen3_vl``、``sglang``
   * - ``--env``\ （:doc:`仿真器 <../examples/simulators_index>`）
     - ``behavior``、``calvin``、``d4rl``、``embodichain``、``frankasim``、``genesis``、``geniesim``、``habitat``、``isaaclab``、``libero``、``liberoplus``、``liberopro``、``maniskill_libero``、``metaworld``、``mujoco_warp``、``polaris``、``robocasa``、``robocasa365``、``robodojo``、``robolab``、``robotwin``、``roboverse``、``simple``
   * - ``--env``\ （:doc:`世界模型 <../examples/world_model_index>`）
     - ``opensora``、``wan``
   * - ``--env``\ （:doc:`真机 <../examples/real_world_index>`）
     - ``arx5``、``dosw1``、``dummy``、``franka``、``franka-ros``、``gim_arm``、``lekiwi``、``piper``、``so101``、``xsquare_turtle2``、``yam``\ （Robbyant R2 使用 RFrame 与 ROS 2，见 :doc:`../examples/embodied/r2`）
   * - ``--engine``
     - ``sglang``\ （``agentic`` 默认）、``vllm``\ （``agentic``）、``phyai``\ （``embodied`` 搭配 ``--model openpi --env libero``）
   * - ``--platform``
     - ``nvidia``\ （默认）、``amd``、``ascend``、``musa``、``kunlun``、``biren``、``metax``\ （Intel XPU 使用 PyTorch XPU 环境，暂未通过 ``--platform`` 选项安装，详见 :ref:`硬件支持矩阵 <model-hardware-support>`）

.. dropdown:: 更多安装参数（可选）

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 参数
        - 作用
      * - ``--use-mirror``
        - 使用中国大陆镜像源加速依赖与 GitHub 下载（运行 ``hf download`` 前也可设置 ``export HF_ENDPOINT=https://hf-mirror.com``）。
      * - ``--venv <dir>``
        - 指定虚拟环境目录（默认 ``.venv``）。
      * - ``--python <version>``
        - 按需指定 Python 版本（如 ``behavior`` 和 ``d4rl`` 需要 ``3.10``）。
      * - ``--no-root``
        - 跳过 ``sudo`` 系统库安装，适合已预装系统依赖的主机。
      * - ``--no-flash-attn`` / ``--no-apex``
        - 在不需要或无匹配 CUDA 编译工具链的主机上跳过 ``flash-attn`` 或 ``apex`` 编译。

   运行 ``bash requirements/install.sh --help`` 查看全部版本覆盖参数。

.. _hardware-and-software-requirements:

硬件与软件要求
--------------

按任务规模选择对应硬件，从单张 GPU 到多机集群均可复用同一套配置。

.. list-table:: 按场景划分的 GPU 档位
   :header-rows: 1
   :widths: 22 44 34

   * - 档位
     - 适用任务
     - 推荐入口
   * - 单卡（4–24 GB）
     - MLP / CNN 仿真 RL（~4 GB）、VLA 评测（24 GB 起）、LoRA SFT、真机数据采集与部署
     - :doc:`快速开始（具身智能） <vla>`、:doc:`评测快速体验 <../evaluations/get_started/quick_tour>`
   * - 1–2 卡（40–80 GB）
     - 单机 OpenVLA / OpenPI 训练、共置 PPO / GRPO、1.5B Agent / LLM GRPO
     - :doc:`快速开始（Agent 与 LLM） <vla>`、:doc:`具身智能 <../examples/index>`
   * - 4–8 卡（80 GB）
     - VLA 全参 SFT 与基准复现（LIBERO、ManiSkill、RoboTwin）、7B–14B 推理 RL
     - :doc:`具身训练 <../examples/index>`
   * - 多机集群（8×H100 起）
     - 大规模分离式与混合式训练、Megatron + SGLang / vLLM、世界模型
     - :doc:`多节点训练 <../guides/multi_node>`

除 NVIDIA GPU 外，RLinf 还支持 AMD ROCm、华为昇腾 CANN、摩尔线程 MUSA、Intel XPU、昆仑芯 XPU、沐曦 MACA 与壁仞 SUPA。已验证的模型与环境组合见 :ref:`硬件支持矩阵 <model-hardware-support>`。

.. list-table:: 软件环境
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 支持 / 测试版本
   * - 操作系统
     - Linux（已在 Ubuntu 22.04 上测试）
   * - Python
     - 3.10–3.12（由 ``requirements/install.sh`` 自动管理）
   * - NVIDIA 驱动与 CUDA
     - 驱动 535+（仿真渲染需支持图形与 EGL）；CUDA 12.4+
   * - 容器（可选）
     - Docker 26+ 与 NVIDIA Container Toolkit（预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__）

.. dropdown:: 大规模参考集群配置（8×H100 起）

   多机大规模训练与完整基准复现通常在以下集群配置上验证：

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 组件
        - 参考配置
      * - GPU
        - 每节点 8×H100
      * - CPU
        - 每节点 192 核
      * - 内存
        - 每节点 1.8 TB
      * - 网络
        - NVLink + RoCE / IB（3.2 Tbps）
      * - 存储
        - 单机 1 TB 本地盘 / 多机 10 TB 共享存储（NAS）

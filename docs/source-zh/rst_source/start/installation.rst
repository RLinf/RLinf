安装与环境
==========

使用 ``requirements/install.sh`` 在本地 ``uv`` 虚拟环境中一键安装 RLinf，或直接拉取 Docker Hub 预构建镜像。每个虚拟环境对应一套 ``--model`` + ``--env``\ （或 ``agentic --engine``）组合；需要共存多套组合时传入 ``--venv <dir>``，避免覆盖默认的 ``.venv``。各教程页面都附带对应命令，本页作为全局安装参数与硬件环境参考。

安装与验证
----------

选择目标场景，直接复制运行完整安装与验证命令：

.. tab-set::

   .. tab-item:: 具身仿真（Embodied）
      :sync: embodied

      按下方表格将 ``--model`` 与 ``--env`` 替换为目标模型和仿真器（轻量 MLP / CNN policy 可省略 ``--model``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         python -c "import torch, rlinf; print(rlinf.__file__, 'CUDA:', torch.cuda.is_available())" && ray --version

      预期输出：

      .. code-block:: text

         /path/to/RLinf/rlinf/__init__.py CUDA: True
         ray, version 2.x.x

   .. tab-item:: 真机机器人（Real-World）
      :sync: realworld

      在机器人控制节点安装驱动与传感器依赖（无需 GPU），在 GPU 节点安装模型与驱动（按需将 ``franka`` 替换为 ``arx5``、``yam``、``xsquare_turtle2``、``so101``、``piper``、``lekiwi``、``dosw1`` 或 ``gim_arm``；Robbyant R2 通过 RFrame 与 ROS 2 接入，见 :doc:`../examples/embodied/r2`，完整列表见 :doc:`../examples/real_world_index`）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf

         # 机器人控制节点（驱动与传感器）
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         # GPU 训练 / 推理节点（或单机同配 GPU 与机器人）
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

   .. tab-item:: 智能体 / 推理（Agentic）
      :sync: agentic

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate
         python -c "import torch, rlinf; print(rlinf.__file__, 'CUDA:', torch.cuda.is_available())" && ray --version

      预期输出：

      .. code-block:: text

         /path/to/RLinf/rlinf/__init__.py CUDA: True
         ray, version 2.x.x

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

   .. tab-item:: 文档构建（Docs）
      :sync: docs

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
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
     - ``openvla``、``openvla-oft``、``openpi``、``psi0``、``pi0_fast``、``gr00t``、``gr00t_n1d6``、``gr00t_n1d7``、``dexbotic``、``starvla``、``lingbotvla``、``lingbotvlav2``、``abot_m0``、``molmoact2``、``evo1``
   * - ``--model``\ （:doc:`WAM / VLM / 生成 <../examples/vla_wam_index>`）
     - ``dreamzero``、``fastwam``、``cosmos3``、``qwen3_vl``、``diffusion``、``sglang``
   * - ``--env``\ （:doc:`仿真器 <../examples/simulators_index>`）
     - ``maniskill_libero``、``libero``、``liberopro``、``liberoplus``、``behavior``、``metaworld``、``calvin``、``isaaclab``、``robocasa``、``robocasa365``、``frankasim``、``robotwin``、``robodojo``、``roboverse``、``habitat``、``genesis``、``geniesim``、``embodichain``、``d4rl``、``mujoco_warp``、``polaris``、``simple``、``robolab``
   * - ``--env``\ （:doc:`世界模型 <../examples/world_model_index>`）
     - ``opensora``、``wan``
   * - ``--env``\ （:doc:`真机 <../examples/real_world_index>`）
     - ``franka``、``franka-ros``、``xsquare_turtle2``、``dosw1``、``gim_arm``、``so101``、``lekiwi``、``piper``、``arx5``、``yam``、``dummy``\ （Robbyant R2 使用 RFrame 与 ROS 2，见 :doc:`../examples/embodied/r2`）
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

按任务规模选择对应硬件。你可以从单张 GPU 起步，无需修改代码就能将同一套配置扩展到多机集群。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hardware_tiers_zh.svg
   :align: center
   :width: 100%

   RLinf 从单卡到多机集群的硬件档位。

.. list-table:: 按场景划分的 GPU 档位
   :header-rows: 1
   :widths: 22 44 34

   * - 档位
     - 适用任务
     - 推荐入口
   * - 单卡（8–24 GB 起）
     - MLP / CNN 仿真 RL、VLA 评测、LoRA SFT、真机数据采集与部署
     - :doc:`训练快速开始（MLP） <vla>`、:doc:`评测快速体验 <../evaluations/get_started/quick_tour>`
   * - 1–2 卡（40–80 GB）
     - 单机 OpenVLA / OpenPI 快速开始、共置 PPO / GRPO
     - :doc:`训练快速开始（VLA） <vla>`
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

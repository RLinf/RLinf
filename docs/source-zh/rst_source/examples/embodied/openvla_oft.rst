OpenVLA-OFT 强化学习训练
========================

.. figure:: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
   :align: center
   :width: 90%

   原始 OFT 微调研究的 LIBERO 结果（图片来源：`OpenVLA-OFT 项目 <https://openvla-oft.github.io/>`__）。

使用 RLinf 对 OpenVLA-OFT 进行强化学习微调。本页先介绍 NVIDIA 上的 LIBERO + GRPO 训练流程，再说明如何在 AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 上使用同一模型运行 LIBERO 或 ManiSkill。原始 OpenVLA 模型的训练流程见 :doc:`maniskill`。

概览
----

选择与任务匹配的 checkpoint 和配置，在 LIBERO 上训练 OpenVLA-OFT。其他环境的训练流程见下方链接的模拟器页面。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · ManiSkill · RoboTwin · MetaWorld · BEHAVIOR · OpenSora · Wan

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      语言条件操作任务

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <openvla-oft-hardware>` （LIBERO · ManiSkill）

| **你将完成：** 安装 → 下载 LIBERO-Goal checkpoint → 设置模型路径 → 启动 GRPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 所选后端的硬件和驱动。

任务
~~~~

可先运行 LIBERO-Goal。其他模拟器的完整流程保留在各自页面中。

.. list-table::
   :header-rows: 1
   :widths: 18 22 36 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - :doc:`LIBERO <libero>`
     - Goal
     - ``libero_goal_grpo_openvlaoft``
     - 目标条件操作。
   * - :doc:`ManiSkill <maniskill>`
     - Plate-25
     - ``maniskill_ppo_openvlaoft``
     - 桌面操作。
   * - :doc:`RoboTwin <robotwin>`
     - Place empty cup
     - ``robotwin_place_empty_cup_grpo_openvlaoft``
     - 双臂操作。
   * - :doc:`MetaWorld <metaworld>`
     - MT50
     - ``metaworld_50_grpo_openvlaoft``
     - 多任务操作。
   * - :doc:`BEHAVIOR <behavior>`
     - 家居任务
     - ``behavior_ppo_openvlaoft``
     - 长程活动。
   * - :doc:`OpenSora <opensora>` / :doc:`Wan <wan>`
     - LIBERO Spatial
     - ``opensora_libero_spatial_grpo_openvlaoft`` / ``wan_libero_spatial_grpo_openvlaoft``
     - 使用世界模型训练。

观测与动作
~~~~~~~~~~

LIBERO 训练流程根据图像和任务提示生成动作块。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - Observation
     - 模型配置指定的 RGB 图像。
   * - Action
     - 由 7 维末端与夹爪动作组成的动作块。
   * - Reward
     - 按训练配置缩放的 LIBERO 任务成功奖励。
   * - Prompt
     - 当前任务的自然语言指令。

安装
----

默认流程使用下方的 NVIDIA 安装步骤。AMD、昇腾或 MUSA 用户请先完成 :ref:`对应后端的安装 <openvla-oft-hardware>`，再下载模型。

.. include:: _setup_common.rst

在仓库根目录启动 NVIDIA 容器：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero bash
   source switch_env openvla-oft

也可以在本地仅安装 OpenVLA-OFT 和 LIBERO 的依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env libero
   source .venv/bin/activate

下载模型
--------

下载该配置对应的 LIBERO-Goal SFT checkpoint：

.. code-block:: bash

   hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 \
      --local-dir checkpoints/Openvla-oft-SFT-libero-goal-traj1

.. include:: _model_path.rst

在 ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`` 中设置这些路径。保留 ``actor.model.unnorm_key: libero_goal_no_noops``，以匹配 checkpoint 的动作统计数据。其他套件需要对应的 checkpoint 和 key，详见 :doc:`libero`。

运行
----

激活模型环境并设置路径后，启动 LIBERO-Goal 训练：

.. code-block:: bash

   ROBOT_PLATFORM=LIBERO bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

脚本加载指定的 YAML，按 ``cluster.component_placement`` 创建 actor、rollout 和环境 worker，然后启动 GRPO。请按设备资源调整 placement 和 batch size，参见 :doc:`../../concepts/placement` 与 :doc:`../../resources/faq`。

.. _openvla-oft-hardware:

在不同硬件后端上运行
--------------------

NVIDIA 使用上面的安装与启动流程。AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 都通过共用平台安装器、scheduler 设备 API 与模型运行路径支持 OpenVLA-OFT 在 LIBERO 和 ManiSkill 上运行。概览中的其他环境有各自的硬件要求。

.. _openvla-oft-amd:

AMD ROCm
~~~~~~~~

可以启动 ROCm 容器，也可以在已安装 ROCm 的宿主机上直接安装。

.. include:: _amd_libero.rst

发布镜像已经包含该模型环境：

.. code-block:: bash

   source switch_env openvla-oft

若宿主机已安装 ROCm，也可以直接安装依赖：

.. code-block:: bash

   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero
   source .venv/bin/activate

省略 ``--rocm`` 可自动检测已安装的版本；中国大陆用户可添加 ``--use-mirror``。

.. _openvla-oft-ascend:

华为昇腾 CANN
~~~~~~~~~~~~~

可以使用容器，也可以在已具备 CANN 和 NPU 驱动的宿主机上安装。

.. include:: _ascend_libero.rst

进入已发布的 RLinf 容器后，激活模型环境：

.. code-block:: bash

   source switch_env openvla-oft

本地安装时，通过昇腾选项安装 CPU PyTorch 及匹配的 ``torch-npu``：

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero
   source .venv/bin/activate

中国大陆用户可添加 ``--use-mirror``。安装脚本在昇腾上会跳过 CUDA flash-attention 的构建。

.. _openvla-oft-musa:

摩尔线程 MUSA
~~~~~~~~~~~~~

在摩尔线程容器内运行。安装器会复用镜像中的 MUSA 版 PyTorch 与 ``torch_musa``，避免被通用 torch wheel 覆盖。

.. include:: _musa_libero.rst

进入容器后，选择 MUSA 平台安装 OpenVLA-OFT：

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --model openvla-oft --env libero
   source .venv/bin/activate

中国大陆用户可添加 ``--use-mirror``。RLinf 会检测 MUSA 设备，并沿用其他 GPU 后端的 placement 配置完成分配。

.. _openvla-oft-backend-launch:

在 AMD、昇腾或 MUSA 上启动 LIBERO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完成其中一种后端的安装后，按前面的说明下载 LIBERO-Goal checkpoint 并设置模型路径。在同一 shell 中启用软件渲染。

.. include:: _libero_osmesa.rst

启动已配置的 GRPO 训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

在 AMD、昇腾或 MUSA 上运行 ManiSkill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD 与昇腾使用 OpenVLA-OFT 的 ManiSkill + LIBERO 组合环境。根据所选模型 accelerator 执行对应命令：

.. code-block:: bash

   # AMD ROCm
   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env maniskill_libero

   # 华为昇腾 CANN
   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env maniskill_libero

   source .venv/bin/activate

MUSA 需要保留厂商模拟器包，并在厂商镜像中添加 OpenVLA-OFT 模型环境：

.. include:: _musa_maniskill.rst

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --venv openvla-oft --model openvla-oft --env libero
   source openvla-oft/bin/activate

三种非 CUDA 后端均使用以下 CPU simulation 配置：

.. include:: _maniskill_non_cuda.rst

下载 ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` 及其 LoRA adapter，在 ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml`` 中设置 ``model_path`` 与 ``lora_path``，并根据可用设备调整 placement 与 batch size 后启动 PPO：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

可视化与结果
------------

在训练日志中观察 ``env/success_once``。指标定义见 :doc:`训练指标 <../../reference/metrics>`，已发布的训练结果见 :doc:`LIBERO 结果 <libero>`。独立评测请按 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>` 操作。

基于 RoboUniverse-10K 的具身强化学习
====================================

.. figure:: https://raw.githubusercontent.com/Einsia/RoboUniverse-10K/main/assets/task-taxonomy.png
   :align: center
   :width: 90%
   :alt: RoboUniverse-10K task taxonomy

   RoboUniverse-10K 在 Franka Panda 和 AgileX 双臂 Piper 上的任务分类（来源：`Einsia/RoboUniverse-10K <https://github.com/Einsia/RoboUniverse-10K>`__）。

本页介绍如何用 π₀.₅ / OpenPI 在 RoboUniverse-10K 的 10,807 个 Isaac Sim 操作任务上运行具身评估和强化学习训练。

概览
----

使用两路 RGB 图像、关节状态和自然语言指令评估或训练具身 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: 算法
      :text-align: center

      评估、PPO、GRPO

   .. grid-item-card:: 任务
      :text-align: center

      10,307 个训练任务 · 500 个 ``heldout500`` 任务

   .. grid-item-card:: 硬件
      :text-align: center

      Linux x86_64 · NVIDIA RTX GPU（Isaac Sim 5.1.0）

| **步骤：** 安装 → 下载数据集 → 配置 EULA 和路径 → 启动运行 → 查看 ``eval/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 阅读 `NVIDIA Isaac Sim 许可协议 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html>`_。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 机器人平台
     - 任务数量
     - 环境配置
     - 动作空间
   * - Franka Panda
     - 共 4,132 个（``heldout500`` 含 200 个）
     - ``env/robouniverse_franka``
     - 9 维关节位置目标（7 维机械臂 + 2 维夹爪）
   * - AgileX 双臂 Piper
     - 共 6,675 个（``heldout500`` 含 300 个）
     - ``env/robouniverse_agilex``
     - 16 维关节位置目标（双臂各 6 维 + 双夹爪各 2 维）

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 格式
   * - Observation
     - 外部相机与腕部相机两路 640×480 RGB 图像、实测关节位置（Franka 为 9 维，AgileX 为 16 维）以及可选关节速度。
   * - Action
     - 20 Hz 关节位置目标，自动裁剪到各任务的关节限位内。
   * - Reward
     - 回合结束给出稀疏奖励（成功为 1，失败为 0）；仿真异常单独记入 ``infra_error``。
   * - Prompt
     - 每个任务自带的自然语言指令。

安装
----

.. include:: _setup_common.rst

安装支持 OpenPI 与 RoboUniverse 的 RLinf 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robouniverse
   source .venv/bin/activate

准备可用的 Isaac Sim 5.1.0 Python 解释器（例如 ``nvcr.io/nvidia/isaac-sim:5.1.0`` 中的 ``/isaac-sim/python.sh``）。

下载数据集
----------

从 Hugging Face 下载任务索引和 ``heldout500`` 划分文件：

.. code-block:: bash

   hf download Einsia/RoboUniverse-10K catalog.csv splits/heldout500.csv \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/robouniverse-10k

将 ``catalog.csv`` 中列出的任务 ZIP 下载到同一目录。

运行
----

编辑 ``examples/embodiment/config/robouniverse_franka_eval_openpi.yaml``，填写数据集目录、模型路径、仿真器 Python 路径并同意 NVIDIA 许可协议：

.. code-block:: yaml

   env:
     eval:
       init_params:
         data_root: /path/to/robouniverse-10k
         sim_python: /isaac-sim/python.sh
         accept_isaac_eula: true

   rollout:
     model:
       model_path: /path/to/model

启动评估：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robouniverse_franka_eval_openpi

单任务冒烟测试可设置 ``env.eval.init_params.cases: ["000247_rotate_twist"]``；运行 AgileX 双臂任务时，将环境配置改为 ``env/robouniverse_agilex`` 并设置 ``rollout.model.action_dim: 16``。

.. warning::

   每个仿真器 worker 至少分配 8 核 CPU 和 24 GiB 内存。核心数不足会导致 Isaac Sim 冷启动超时。

可视化与结果
------------

在 TensorBoard 中查看 ``eval/success_once``、``eval/return`` 和 ``eval/infra_error``。日志配置与指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

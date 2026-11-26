在 Isaac Lab-Arena 上评测 π₀.₅
=========================================

这个示例用 OpenPI π₀.₅ 控制 DROID 机械臂，统计它把香蕉放到盘子里的成功率。

.. figure:: https://media.githubusercontent.com/media/isaac-sim/IsaacLab-Arena/a72262a0f1896f56bb3a1fdf329a1548ba55aad5/docs/images/openpi_droid_get_started.png
   :width: 80%
   :align: center
   :alt: Isaac Lab-Arena 中的 DROID 桌面操作场景

   DROID 场景图来自 `NVIDIA Isaac Lab-Arena <https://github.com/isaac-sim/IsaacLab-Arena>`__。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI π₀.₅（DROID）

   .. grid-item-card:: 算法
      :text-align: center

      policy 评测与 :ref:`单动作 PPO <isaaclab-arena-single-action-ppo>`

   .. grid-item-card:: 任务
      :text-align: center

      桌面抓放

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA RTX

任务
~~~~

.. list-table::
   :header-rows: 1

   * - Arena 任务
     - 操作目标
   * - 香蕉装盘（默认场景图）
     - 拿起香蕉并放到盘子里。
   * - 已注册的枫木桌任务
     - 将指定物体移到桌上的目标位置。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - 外部相机与腕部相机的 RGB 图像、七个关节的位置和夹爪闭合程度。
   * - 动作
     - 每个动作指定七个关节的绝对目标角度（弧度）和一个夹爪命令。默认评测配置每次推理预测 15 个动作。
   * - 奖励
     - 使用任务定义的奖励。任务没有奖励项时，仅成功返回 1，其余步返回 0。
   * - 指令
     - 使用 Arena 任务自带的自然语言指令。

安装与运行
----------

准备一台配有 RTX GPU 的 x86-64 Linux 主机，确认驱动支持 Isaac Sim 6.2，并能联网下载 NVIDIA 的仿真资源。硬件和驱动要求见 `Isaac Sim 文档 <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>`__。

OpenPI 控制端使用 PyTorch 2.7.1 的 CUDA 12.8 构建，包含 Blackwell RTX GPU 所需的内核。安装脚本会为 Arena 创建独立的 Python 3.12 环境。如果已有对应版本（``a72262a0f1``）的可用环境，可在安装前将 ``ISAACLAB_ARENA_PYTHON`` 设为其解释器的绝对路径，并在启动评测时保留这个环境变量。

在 RLinf 仓库根目录运行以下命令。主机准备和软件源设置见 :doc:`安装指南 <../../start/installation>`。

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (DROID)

      **1. 安装控制端与仿真器**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env isaaclab_arena
         source .venv/bin/activate

      **2. 下载 DROID 权重和归一化统计量**

      .. code-block:: bash

         export MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
             --local-dir "$MODEL_DIR"

      **3. 启动评测**

      接受 Isaac Sim 许可后，运行香蕉装盘任务。

      .. code-block:: bash

         export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR"

可视化与结果
------------

默认并行评测四个回合。在终端或 TensorBoard 中查看 ``eval/success_once``，它表示达成目标的回合比例。切换任务、试跑短回合和保存相机录像的方法见 :doc:`评测指南 <../../evaluations/guides/isaaclab_arena>`。

IsaacLab 强化学习
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab（图片来源：`IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__）。

`IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__ 是 NVIDIA 的 GPU 加速机器人学习仿真器。你将使用 RLinf 在自定义 Franka 方块堆叠任务上，通过 PPO 微调 GR00T N1.5 或 OpenPI π₀.₅。

概览
----------------------------------------

先使用 SFT 检查点，再通过 PPO 在 IsaacLab Franka stack-cube 任务上微调 VLA。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 下载 Isaac Sim + SFT 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim · SFT 检查点（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - 将红色方块堆到蓝色方块上，再将绿色方块堆到红色方块上。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 第三人称相机和腕部相机 RGB（默认 256×256），以及机器人本体状态。
   * - 动作
     - 7 维连续动作：3D 位置（x, y, z）+ 3D 旋转（roll, pitch, yaw）+ 夹爪。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GR00T N1.5
      :sync: gr00t

      **1. 安装环境**

      安装 RLinf（GR00T N1.5 + IsaacLab）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t --env isaaclab && source .venv/bin/activate

      下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         每次在新终端中启动 IsaacLab 前，都需要运行 ``source ./setup_conda_env.sh``。

      **2. 下载模型**

      下载 GR00T N1.5 的 SFT 检查点（基于 |huggingface| `IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__ 人类演示数据微调）：

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir ./RLinf-Gr00t-SFT-Stack-cube

      .. include:: _model_path.rst

      **3. 启动训练**

      修改 ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`` 中的模型路径，启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

      GR00T 默认配置会分离 env、rollout 和 actor placement；按 GPU 显存预算调整 ``cluster.component_placement``、``rollout.pipeline_stage_num`` 和 ``actor.enable_offload``。独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>` 并传入后缀 ``isaaclab_franka_stack_cube_ppo_gr00t``。

   .. tab-item:: OpenPI π₀.₅
      :sync: openpi

      **1. 安装环境**

      安装 RLinf（OpenPI + IsaacLab）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env isaaclab && source .venv/bin/activate

      下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         每次在新终端中启动 IsaacLab 前，都需要运行 ``source ./setup_conda_env.sh``。

      **2. 下载模型**

      下载 OpenPI π₀.₅ 的 SFT 检查点（基于 |huggingface| `IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__ 人类演示数据微调）：

      .. code-block:: bash

         hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir ./RLinf-pi05-SFT-Stack-cube

      .. include:: _model_path.rst

      **3. 启动训练**

      修改 ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`` 中的模型路径，启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

      OpenPI 默认配置使用 ``actor,env,rollout: all`` 共置；按 GPU 显存预算调整 ``cluster.component_placement``、``rollout.pipeline_stage_num`` 和 ``actor.enable_offload``。独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>` 并传入后缀 ``isaaclab_franka_stack_cube_ppo_openpi_pi05``。

.. note::

   如需添加自定义 IsaacLab 任务，按 :doc:`../../extending/new_env` 注册任务后，在 ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` 等环境配置中将 ``init_params.id`` 指向新的 task id。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - 模型阶段
     - 成功率
   * - GR00T N1.5 基础模型（无 SFT）
     - 0.000
   * - GR00T N1.5 SFT 模型
     - 0.654
   * - GR00T N1.5 RL 微调模型（SFT + RL）
     - 0.897
   * - OpenPI π₀.₅ SFT 模型
     - 0.859
   * - OpenPI π₀.₅ RL 微调模型（SFT + RL）
     - 0.953

.. note::

   **致谢：** 感谢 `许明辉 <https://github.com/smallcracker>`__ 和 `杨楠 <https://github.com/AquaSage18>`__ 对 GR00T N1.5 示例的贡献与支持，也感谢 `Yifan Wu <https://github.com/YifWRobotics>`__ 对 OpenPI π₀.₅ 示例的贡献与支持。

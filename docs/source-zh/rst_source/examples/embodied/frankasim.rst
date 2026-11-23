Franka-Sim 强化学习
===================

.. figure:: https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png
   :align: center
   :width: 70%

   RLinf SERL fork 中的 Franka-Sim 资源。

Franka-Sim 是基于 `SERL <https://rail-berkeley.github.io/serl/docs/sim_quick_start.html>`__ 栈构建的轻量级 Franka Panda 仿真环境。你将使用 RLinf 在状态观测上用 PPO 训练 MLP policy，或在 RGB 观测上用异步 SAC 训练 CNN policy。

概览
----------------------------------------

使用状态观测或视觉观测训练 Franka pick-cube 策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP · CNN

   .. grid-item-card:: 算法
      :text-align: center

      PPO · SAC

   .. grid-item-card:: 任务
      :text-align: center

      PickCube state · vision

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1 GPU

| **你将完成：** 安装 → 可选下载 ResNet → 启动训练 → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 Franka-Sim 资源。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``PandaPickCube-v0``
     - 面向 MLP + PPO 配方的状态观测 pick-cube 任务。
   * - ``PandaPickCubeVision-v0``
     - 面向 CNN + 异步 SAC 配方的 RGB 观测 pick-cube 任务。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - ``PandaPickCube-v0`` 使用本体状态与目标位置；``PandaPickCubeVision-v0`` 使用 RGB 图像与状态。
   * - 动作
     - 4 维连续动作：3D 末端执行器位置增量和夹爪控制。
   * - 奖励
     - 稠密任务进度奖励。
   * - 提示词
     - 状态 MLP 配方不使用提示词；视觉策略从 env wrapper 接收任务条件观测。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP / CNN

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env frankasim && source .venv/bin/activate

      **2. 下载模型**

      MLP + PPO 配方可跳过此步。CNN + SAC 配方需要下载预训练 ResNet-10 检查点：

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **3. 启动训练**

      可用训练配置：

      .. list-table::
         :header-rows: 1
         :widths: 24 38 20 18

         * - 配方
           - 配置
           - 入口
           - 命令后缀
         * - MLP + PPO
           - ``examples/embodiment/config/frankasim_ppo_mlp.yaml``
           - ``run_embodiment.sh``
           - ``frankasim_ppo_mlp``
         * - CNN + SAC
           - ``examples/embodiment/config/frankasim_sac_cnn_async.yaml``
           - ``run_async.sh``
           - ``frankasim_sac_cnn_async``

      .. include:: _model_path.rst

      使用对应脚本启动训练：

      .. code-block:: bash

         # 状态观测 MLP + PPO
         bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp

         # 视觉 CNN + 异步 SAC
         bash examples/embodiment/run_async.sh frankasim_sac_cnn_async actor.model.model_path=$PWD/RLinf-ResNet10-pretrained

      .. dropdown:: 进阶配置（可选）

         两个参考配置默认都运行在 GPU ``0``。迁移到多卡机器时，按需调整 ``cluster.component_placement``、``env.train.total_num_envs`` 和 batch size。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配方
     - 结果描述
   * - CNN + 异步 SAC
     - 在原始运行使用的仿真设置中，约一小时内学习到稳定抓取策略。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/frankasim_curve.png
   :align: center
   :width: 90%

   Franka-Sim 异步 SAC + CNN 成功率曲线。

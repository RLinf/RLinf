基于 MuJoCo Warp 的强化学习
============================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   MuJoCo Warp 批量仿真的三个 humanoid（图片来源：`MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__）。

本页介绍如何用 PPO 在五个经典 MuJoCo 运动控制任务上训练 MLP policy，并给出任务维度、安装方法、启动命令和验证结果。`MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ 会在 NVIDIA GPU 上以一次批量物理调用推进 RLinf env worker 中的所有 world。

概述
----

选择任务后，即可通过 RLinf 的 actor、rollout 和 env worker 训练低维 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      half_cheetah · hopper · walker2d · ant · humanoid

   .. grid-item-card:: 硬件
      :text-align: center

      单节点 · 单张 NVIDIA GPU

| **你将完成：** 安装 → 选择任务 → 启动训练 → 观察 ``eval/env/return``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 训练需要 NVIDIA GPU；环境开发可使用 CPU。

任务
~~~~

通过 ``env.train.task_name`` 和 ``env.eval.task_name`` 选择任务，并为 policy 设置对应维度。

.. list-table::
   :header-rows: 1
   :widths: 24 16 16 44

   * - 任务
     - 观测维度
     - 动作维度
     - 终止条件
   * - ``half_cheetah``
     - 17
     - 6
     - 仅由回合步数上限截断。
   * - ``hopper``
     - 11
     - 3
     - 躯干高度或角度异常，或状态包含非有限值。
   * - ``walker2d``
     - 17
     - 6
     - 躯干高度或角度异常，或状态包含非有限值。
   * - ``ant``
     - 27
     - 8
     - 躯干高度异常或状态包含非有限值。
   * - ``humanoid``
     - 45
     - 17
     - 躯干高度异常或状态包含非有限值。

观测与动作
~~~~~~~~~~

观测和奖励遵循 Gymnasium MuJoCo v4 约定。每个回合到达 ``max_episode_steps`` 后截断。

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 规格
   * - 观测
     - ``qpos`` 去掉全局 x 位置（3D 任务同时去掉 y 位置），再与 ``qvel`` 拼接。
   * - 动作
     - 连续执行器控制量，限制在各模型的控制范围内。
   * - 奖励
     - 前进速度加任务对应的存活奖励，再减去控制量平方代价，最后乘以 ``reward_scale``\ （默认 ``1.0``）。
   * - 提示词
     - 不使用；这是低维 policy 控制配方。

.. note::

   MJWarp 不支持 PGS solver。接入层会将 hopper 和 walker2d 切换为 CG，因此它们的接触动力学与 Gymnasium 的 MuJoCo CPU 后端并不完全一致。

安装
----

.. include:: _setup_common.rst

**Docker 镜像**

构建 MuJoCo Warp target，并启动可访问 GPU 的容器：

.. code:: bash

   docker build -f docker/Dockerfile \
       --build-arg BUILD_TARGET=embodied-mujoco_warp \
       -t rlinf:embodied-mujoco_warp .
   docker run -it --rm --gpus all \
       --shm-size 32g --network host --name rlinf \
       -v .:/workspace/RLinf rlinf:embodied-mujoco_warp

**自定义环境**

安装依赖并激活环境：

.. code:: bash

   # 中国大陆用户可添加 --use-mirror。
   bash requirements/install.sh embodied --env mujoco_warp
   source .venv/bin/activate

安装脚本固定 ``mujoco-warp==3.17.0`` 和 ``warp-lang==1.16.0``。该组合已经过 CUDA 12 驱动验证，也避免无驱动的 Docker 构建解析到仅支持 CUDA 13 的 Warp wheel。

运行
----

在仓库根目录启动 half-cheetah 配方：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp

这条命令会：

1. 加载 ``examples/embodiment/config/mujoco_warp_ppo_mlp.yaml`` 和 ``examples/embodiment/config/env/mujoco_warp.yaml``。
2. 在一张 GPU 上启动 actor、rollout 和 env worker，分别使用 512 个训练 world 和 16 个评估 world。
3. 采集 PPO rollout，并更新 MLP policy 和 value head。

训练其他任务时，同时覆盖两个 env 的任务名和 actor 维度。例如：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17

其他任务的维度见上表。可根据 GPU 显存调整 ``env.train.total_num_envs``、``actor.micro_batch_size`` 和 ``actor.global_batch_size``。

可视化与结果
------------

在仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

``eval/env/return`` 是平均评估回合回报，``env/return`` 是训练回合回报。共享指标定义见 :doc:`训练指标 <../../reference/metrics>`；独立评估流程见 Evaluation 章节。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 验证
     - 结果
   * - 单 epoch CI 配方
     - 两个并行 world 在一张 NVIDIA GPU 上完成 rollout、优势计算和一次 PPO 更新。
   * - 20-step PPO 配方
     - 在单张 A800 上，``env/return`` 从第 1 步的 -268.66 增至第 20 步的 867.35；``eval/env/return`` 在第 10 步和第 20 步分别为 726.50 和 1246.81。

视频录制默认关闭。通过以下命令为评估启用录像：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.eval.video_cfg.save_video=true

评估视频保存在 ``<log_path>/video/eval`` 下。物理仿真仍由 MJWarp 执行，录像 wrapper 会将每个 world 的状态复制到原生 MuJoCo 进行 RGB 渲染。录像会增加设备到主机的同步和逐 world 渲染开销，因此测试吞吐量时应保持关闭。可通过 ``render_width`` 和 ``render_height`` 调整单个 world 的分辨率，通过 ``video_cfg.fps`` 覆盖默认控制频率。

首次仿真步需要 JIT 编译 MJWarp kernel，耗时会高于后续步骤。

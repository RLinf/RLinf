基于 MuJoCo Warp 的强化学习
============================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   MuJoCo Warp 批量仿真的三个 humanoid（图片来源：`MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__）。

本页介绍如何用 PPO 在五个经典 MuJoCo 运动控制任务上训练 MLP policy，并说明这些任务如何定义、维度、安装方法、启动命令和验证结果。`MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ 在 NVIDIA GPU 上把一个 RLinf env worker 里的全部 world 一起推进。每个物理子步是一次批量调用，一个控制步会按该任务的 frame skip 重复这次调用：half_cheetah、ant、humanoid 为 5，hopper、walker2d 为 4。

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

上表五个名字是本配方自带的任务。一个 env worker 里的全部 world 共用一份批量物理状态 ``MjWarpWorld``\ ：它可以加载 MJCF，也可以加载已经编译好的 ``MjModel``\ ，每个物理子步把所有 world 一起推进。奖励、终止和初始状态来自任务 wrapper ``TaskWrapper``\ 。内置任务共用同一个运动控制 wrapper ``LocomotionTask``\ ，``task_name`` 选择其中一份 ``TaskSpec``\ 。奖励形式相同的新任务，是再注册一份 ``TaskSpec`` 及其 MJCF。观测或奖励不同时，为该 ``task_name`` 实现新的 ``TaskWrapper``\ ，并由环境构造它。

观测与动作
~~~~~~~~~~

每个回合在 ``max_episode_steps`` 处截断。下表是 policy 读到的观测，以及环境返回的奖励。

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
     - 根部 x 轴速度，加上该任务的存活奖励，再减去 ``ctrl_cost_weight * ||ctrl||^2``\ ，最后乘以 ``reward_scale``\ （默认 ``1.0``）。
   * - 提示词
     - 不使用；这是低维 policy 控制配方。

half_cheetah、hopper、walker2d 的奖励项与 Gymnasium MuJoCo v4 相同。本配方在躯干高度或角度异常、或状态含非有限值时终止 hopper 和 walker2d。Gymnasium v4 还会把关节速度裁到 ``[-10, 10]``\ ，并在其余状态离开 ``(-100, 100)`` 时终止。

ant 的控制代价权重是 ``5e-4``\ ，前向速度取根部 x 轴。Gymnasium Ant-v4 的权重是 ``0.5``\ ，速度取躯干质心。

humanoid 的观测是任务表中的 45 维 ``qpos`` 与 ``qvel``\ 。乘 ``reward_scale`` 之前，奖励为 ``1.25 + root_x_velocity - 0.1 * ||ctrl||^2``\ 。Gymnasium Humanoid-v4 的观测是 376 维，奖励为 ``5.0 + 1.25 * 质心速度 - 0.1 * ||ctrl||^2``\ 。

.. note::

   MJWarp 没有 PGS solver。加载时，solver 已经是 PGS 的模型会改成 CG。随包资源里只有 humanoid 写了 ``solver="PGS"``\ ，因此它的接触动力学与 Gymnasium 的 MuJoCo CPU 后端不同。

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

``eval/env/return`` 是平均评估回合回报，``env/return`` 是训练回合回报。共享指标定义见 :doc:`训练指标 <../../reference/metrics>`；独立评估可通过在 ``run_embodiment.sh`` 中设置 ``runner.only_eval=true`` 和 ``runner.ckpt_path=<path>`` 运行。

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

评估视频保存在 ``<log_path>/video/eval``\ 。物理仿真仍由 MJWarp 执行。环境把每个 world 的 ``qpos``\ 、``qvel`` 和 ``ctrl`` 复制到原生 MuJoCo 并渲染 RGB 帧，录像 wrapper 再把这些帧写成 MP4。录像会增加设备到主机的同步和逐 world 渲染开销，因此测试吞吐量时应保持关闭。可通过 ``render_width`` 和 ``render_height`` 调整单个 world 的分辨率，通过 ``video_cfg.fps`` 覆盖默认控制频率。

首次仿真步需要 JIT 编译 MJWarp kernel，耗时会高于后续步骤。

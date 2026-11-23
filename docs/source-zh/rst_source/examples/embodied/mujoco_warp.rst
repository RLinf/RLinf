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

| **你将完成：** 安装 → 选择任务 → 启动训练 → 观察 ``eval/return``。
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

训练其他任务时，直接用 Python 入口传入任务名和 actor 维度。在仓库根目录调用 Python 前，先设置路径和渲染后端。以 humanoid 为例：

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD"
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17

其他任务的维度见上表。可根据 GPU 显存调整 ``env.train.total_num_envs``、``actor.micro_batch_size`` 和 ``actor.global_batch_size``。

可视化与结果
------------

在仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

默认日志路径下，TensorBoard event 文件位于 ``../results/tensorboard``\ 。如果修改了 ``runner.logger.log_path``\ ，就让 ``--logdir`` 指向修改后的目录。在浏览器打开 ``http://localhost:6006``\ ，选择 ``env``\ 、``eval`` 和 ``time`` 指标组。

用 ``eval/return`` 观察评估回报，用 ``env/return`` 观察训练回报，再结合对应的 ``episode_len`` 判断存活时间。:doc:`训练指标 <../../reference/metrics>` 给出了这些指标及回合计数的定义。它们是已完成回合的均值，不是平均每步奖励。humanoid、hopper、walker2d 和 ant 都有存活奖励，回报变高或存活更久本身并不能证明形成了更快、更自然的步态，还要看评估视频。

训练入口默认每 10 个迭代评估并保存 checkpoint。只评估已保存的 humanoid checkpoint 而不继续训练时，使用独立评估入口，并在 ``rollout.model`` 下指定 policy 维度：

.. code:: bash

   python evaluations/eval_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.eval.task_name=humanoid \
       runner.ckpt_path=<checkpoint>/actor/model_state_dict/full_weights.pt \
       +rollout.model.model_type=mlp_policy \
       +rollout.model.policy_setup=locomotion \
       +rollout.model.obs_dim=45 +rollout.model.action_dim=17 \
       +rollout.model.num_action_chunks=1 +rollout.model.hidden_dim=256 \
       +rollout.model.add_value_head=true +rollout.model.is_lora=false

其中，``<checkpoint>`` 是训练保存的 ``global_step_<N>`` 目录。评估环境按顺序选取重置状态编号，并在每次重置后推进编号，所以不同次评估不一定使用相同的初始状态。需要严格对照 checkpoint 时，设置 ``env.eval.use_fixed_reset_state_ids=true``\ ，让每个 world 复用相同的初始状态。

对照实验
~~~~~~~~

以下实验用同一套 PPO + MLP 配方覆盖全部五个任务。每组实验使用一张 A800-SXM4-80GB GPU、512 个训练 world 和 16 个评估 world。每个训练迭代从每个训练 world 收集 1000 个控制步，再执行默认的 PPO 更新。每 10 个迭代评估并录像，``max_episode_steps=1000``。实验使用代码提交 `461104f <https://github.com/RLinf/RLinf/commit/461104f2e5a106f38add2c794c19b7aa269e31a3>`__，软件版本为 Python 3.11.18、PyTorch 2.11.0+cu128、MuJoCo 3.17.0、MuJoCo Warp 3.17.0 和 Warp 1.16.0。

表中训练回报取自第 1 和第 50 个迭代，评估回报和回合长度取自第 10 和第 50 个迭代。回报是已完成回合未经折扣的奖励总和的均值。环境 seed 控制重置噪声，actor seed 固定为 1234，其他 PPO 配置保持默认值。

.. list-table::
   :header-rows: 1
   :widths: 20 19 19 19 23

   * - 任务（环境 seed）
     - 训练回报（1 -> 50）
     - 评估回报（10 -> 50）
     - 评估长度（10 -> 50）
     - 曲线与视频
   * - ``half_cheetah`` (42)
     - -276.82 -> 2650.85
     - 602.72 -> 3496.31
     - 1000.00 -> 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_iterations_10_50.mp4>`__
   * - ``hopper`` (42)
     - 19.04 -> 357.37
     - 267.06 -> 374.66
     - 107.47 -> 132.96
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_iterations_10_50.mp4>`__
   * - ``walker2d`` (42)
     - 3.17 -> 386.73
     - 263.73 -> 454.09
     - 144.54 -> 144.41
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_iterations_10_50.mp4>`__
   * - ``ant`` (42)
     - 98.99 -> 593.47
     - 1002.27 -> 1191.58
     - 1000.00 -> 1000.00
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_iterations_10_50.mp4>`__
   * - ``humanoid`` (42)
     - 22.01 -> 155.71
     - 114.55 -> 235.88
     - 62.14 -> 110.19
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4>`__
   * - ``humanoid`` (43)
     - 21.95 -> 145.55
     - 156.77 -> 215.50
     - 97.52 -> 111.93
     - `曲线 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_iterations_10_50.mp4>`__

humanoid、hopper、walker2d 和 ant 都包含存活奖励。回报更高或回合更长，不一定代表学会了有效的运动方式。应结合各任务的视频，区分真正的移动和仅仅维持存活的行为。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png
   :width: 100%
   :alt: Humanoid 在 50 个 PPO 迭代中的训练回报和回合长度

   环境 seed 为 42 的 humanoid：训练回报、评估回报和回合长度。回合长度上限为 1000。

下面的视频左侧是第 10 个迭代，右侧是第 50 个迭代。`下载 humanoid 对照视频（MP4） <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4>`__。

.. raw:: html

   <video controls muted playsinline preload="metadata" style="width: 100%; max-width: 960px;">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4" type="video/mp4">
   </video>

视频在左右两侧展示同一个评估 world，用来说明动作变化，不表示 16 个评估 world 的均值。后期的迈步动作有所增加，但 humanoid 仍会跌倒和重置，不能据此认为已经稳定行走或收敛。

第二组 humanoid 实验只把环境 seed 从 42 改为 43，用于比较不同重置条件，不是独立的多随机种子统计实验。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_two_seeds.png
   :width: 100%
   :alt: Humanoid 在环境 seed 42 和 43 下的评估回报与长度

   两种环境重置 seed 下的 humanoid 评估结果。actor seed 和训练超参数不变。

这些曲线只覆盖 50 个训练迭代，不代表收敛结果，也不用于和 Gymnasium 的 MuJoCo CPU 后端比较。上文列出的任务定义和 solver 差异意味着这里的回报不能直接等同于 Gymnasium Humanoid-v4 的回报。本次 humanoid 实验的部分状态触发了 MJWarp solver 迭代上限警告；配方保留任务原有的 solver 设置，没有隐藏这些警告。

复现实验
^^^^^^^^

按结果表选择任务、对应维度和环境 seed。下面的命令在一张 GPU 上复现环境 seed 为 42 的 humanoid 实验：

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD"
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export CUDA_VISIBLE_DEVICES=0

   export TASK=humanoid ENV_SEED=42 OBS_DIM=45 ACTION_DIM=17
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.train.task_name=$TASK env.eval.task_name=$TASK \
       actor.model.obs_dim=$OBS_DIM actor.model.action_dim=$ACTION_DIM \
       env.train.seed=$ENV_SEED env.eval.seed=$ENV_SEED \
       runner.logger.log_path=../results/mujoco_warp_${TASK}_seed${ENV_SEED} \
       runner.max_steps=50 \
       runner.val_check_interval=10 runner.save_interval=10 \
       env.eval.video_cfg.save_video=true

复现环境 seed 为 43 的 humanoid 时，只把 ``ENV_SEED`` 改成 43。每组实验的 TensorBoard event、checkpoint 和评估视频都写入所选的 ``runner.logger.log_path``\ 。录像会增加渲染耗时，测量吞吐量时应关闭录像。`合并的指标 CSV <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_metrics.csv>`__ 和 `实验元数据 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_results.json>`__ 包含绘图数值和软件版本。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 验证
     - 结果
   * - 单 epoch CI 配方
     - 两个并行 world 在一张 NVIDIA GPU 上完成 rollout、优势计算和一次 PPO 更新。

视频录制默认关闭。通过以下命令为评估启用录像：

.. code:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.eval.video_cfg.save_video=true

评估视频保存在 ``<log_path>/video/eval/seed_<seed>``\ 。文件名是从 0 开始的评估序号：``runner.val_check_interval=10`` 时，``0.mp4`` 对应第 10 个迭代，``1.mp4`` 对应第 20 个迭代，依此类推。每段视频展示 16 个评估 world 组成的网格。提前终止的 world 会在同一段视频中重置，因此一段视频可能包含多个回合。

物理仿真仍由 MJWarp 执行。环境把每个 world 的 ``qpos``\ 、``qvel`` 和 ``ctrl`` 复制到原生 MuJoCo 并渲染 RGB 帧，录像 wrapper 再把这些帧写成 MP4。录像会增加设备到主机的同步和逐 world 渲染开销，因此测试吞吐量时应保持关闭。可通过 ``render_width`` 和 ``render_height`` 调整单个 world 的分辨率，通过 ``video_cfg.fps`` 覆盖默认控制频率。

首次仿真步需要 JIT 编译 MJWarp kernel，耗时会高于后续步骤。

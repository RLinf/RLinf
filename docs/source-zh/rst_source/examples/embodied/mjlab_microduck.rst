mjlab（Microduck）强化学习
==========================

使用 PPO 和 `mjlab <https://github.com/mujocolab/mjlab>`__\ （MuJoCo Warp）在 `Microduck <https://github.com/pollen-robotics/microduck_rl>`__ 双足行走与动作任务上训练 50 Hz MLP policy。仿真通过 `BAM <https://github.com/Rhoban/bam>`__ 模拟电压控制的 Dynamixel XL330 执行器。

概览
----

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

      Velocity · VelStand · StandUp · SitStand · GroundPick · BallKick · Roulade · Rollers

   .. grid-item-card:: 硬件
      :text-align: center

      单节点 · 单张 NVIDIA GPU

| **你将完成：**\ 安装 → 选择任务 → 启动训练 → 查看 ``eval/env/return``\ 。
| **前置条件：**\ :doc:`安装 </rst_source/start/installation>` · 支持 CUDA 的 NVIDIA GPU（用于运行 MuJoCo Warp）。

任务
~~~~

用 ``env.train.task_id`` 和 ``env.eval.task_id`` 指定任务。Velocity、VelStand、StandUp、SitStand、GroundPick、BallKick 和 Rollers 任务还提供齿轮间隙变体，在 ``MicroDuck`` 前加上 ``-Backlash`` 即可切换（例如 ``Mjlab-Velocity-Flat-Backlash-MicroDuck``\ ）。

.. list-table::
   :header-rows: 1
   :widths: 46 16 38

   * - 任务 ID
     - 地形
     - 任务目标
   * - ``Mjlab-Velocity-{Flat,Rough}-MicroDuck``
     - 平地 / 崎岖
     - 速度指令与头部姿态跟踪
   * - ``Mjlab-VelStand-{Flat,Rough}-MicroDuck``
     - 平地 / 崎岖
     - 行走与倒地起身一体化 policy
   * - ``Mjlab-StandUp-{Flat,Rough}-MicroDuck``
     - 平地 / 崎岖
     - 倒地起身并保持站姿
   * - ``Mjlab-SitStand-{Flat,Rough}-MicroDuck``
     - 平地 / 崎岖
     - 按指令在坐姿与站姿间切换
   * - ``Mjlab-GroundPick-{Flat,Rough}-MicroDuck``
     - 平地 / 崎岖
     - 下蹲触地后恢复站立
   * - ``Mjlab-BallKick-Flat-MicroDuck``
     - 平地
     - 向前踢出 70 mm 小球
   * - ``Mjlab-Roulade-Flat-MicroDuck``
     - 平地
     - 前滚翻并双脚落地
   * - ``Mjlab-Velocity-Flat-MicroDuck-Rollers``
     - 平地
     - 被动轮滑速度跟踪

观测与动作
~~~~~~~~~~

所有 Microduck policy 共用同一套 61 维 actor 观测与 14 维动作空间，导出的 ONNX 模型可以直接在真机运行时热切换。

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 规格
   * - 观测
     - Actor（61 维）：机体角速度（3）、重力投影（3）、相对关节位置（14）、关节速度（14）、上一步动作（14）以及指令 ``[twist(3), head_pose(4), body_pose(6)]``\ 。Critic（76 维）：无噪声、无偏置的 actor 观测项，附加机体线速度（3）、双足离地高度（2）、腾空时间（2）、接触标志（2）和足底接触力（6）。
   * - 动作
     - 14 维连续关节位置目标（左腿 5、颈部与头部 4、右腿 5），通过 BAM XL330 执行器模型以 50 Hz 执行。
   * - 奖励
     - 由 ``mjlab`` 计算各任务的稠密奖励与正则项。
   * - 提示词
     - 无。

安装
----

.. include:: _setup_common.rst

**Docker 镜像**

构建 ``embodied-mjlab`` 镜像，并启动支持 GPU 的容器：

.. code:: bash

   docker build -f docker/Dockerfile \
       --build-arg BUILD_TARGET=embodied-mjlab \
       -t rlinf:embodied-mjlab .
   docker run -it --rm --gpus all \
       --shm-size 32g --network host --name rlinf \
       -v .:/workspace/RLinf rlinf:embodied-mjlab

**自定义环境**

在 Python 3.12 虚拟环境中安装 ``mjlab`` 依赖并激活环境：

.. code:: bash

   # 中国大陆用户可添加 --use-mirror。
   bash requirements/install.sh --no-flash-attn embodied --env mjlab
   source .venv/bin/activate

.. note::

   安装脚本会将 ``pollen-robotics/microduck_rl`` 克隆到虚拟环境中。代码采用 Apache-2.0 协议，3D 机器人网格采用 CC BY-NC-SA 4.0 协议。

运行
----

在仓库根目录启动平地行走训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mjlab_microduck_ppo_mlp

这条命令加载 ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml`` 和 ``examples/embodiment/config/env/mjlab_microduck.yaml``\ ，默认在单张 GPU 上以 4096 个并行环境训练。

切换到其他 Microduck 任务时，修改 ``examples/embodiment/config/env/mjlab_microduck.yaml`` 中的 ``task_id``\ （例如 ``Mjlab-Velocity-Rough-Backlash-MicroDuck``\ ），再重新运行 ``run_embodiment.sh``\ 。

如果要快速冒烟测试或显存较小，可以在 ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml`` 中同步调小 ``env.train.total_num_envs``\ 、\ ``actor.micro_batch_size`` 和 ``actor.global_batch_size``\ 。

如果要将训练好的 checkpoint 导出为 ONNX 供下游运行时使用，见 ``toolkits/mjlab/README.md``\ 。

可视化与结果
------------

``run_embodiment.sh`` 默认把训练日志和 checkpoint 写入 ``<train_log_path>``\ （\ ``./logs/<timestamp>-mjlab_microduck_ppo_mlp``\ ）。在仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

在 TensorBoard 中查看 ``eval/env/return``\ 、\ ``env/return``\ 、\ ``env/episode_len`` 以及各项 ``Episode_Reward/*`` 指标。共享指标定义见 :doc:`训练指标 <../../reference/metrics>`\ 。

训练期间每 ``runner.val_check_interval`` 步（默认 ``250``\ ）会自动执行一次评估。如需在训练期间录制评估视频，可在 ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml`` 中将 ``env.eval.video_cfg.save_video`` 设为 ``True``\ ；测试训练吞吐时请关闭录像。

下图展示了在单张 RTX 4090、4096 个并行环境下训练 ``Mjlab-Velocity-Flat-MicroDuck`` 5000 步的曲线（\ ``env/return`` 达到 124.8，\ ``env/episode_len`` 达到 967.7 / 1000；中途的小幅回落对应课程难度提升）：

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 12px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/return_curve.png" alt="Microduck 回合回报曲线" style="width: 100%;"/>
       <p><em>回合回报（env/return）</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/episode_len_curve.png" alt="Microduck 回合长度曲线" style="width: 100%;"/>
       <p><em>回合长度（env/episode_len）</em></p>
     </div>
   </div>

评估
----

使用 ``evaluations/run_eval.sh`` 单独评估已保存的 actor checkpoint（完整 CLI 参数见 :doc:`评测 CLI <../../evaluations/reference/cli>`\ ）：

.. code:: bash

   bash evaluations/run_eval.sh mjlab mjlab_microduck_ppo_mlp_eval \
       runner.ckpt_path=<train_log_path>/mjlab_microduck_ppo_mlp/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt

单独评估将结果写入 ``<eval_log_path>``\ （\ ``./logs/<timestamp>-mjlab_microduck_ppo_mlp_eval``\ ），默认开启视频录制（\ ``env.eval.video_cfg.save_video: True``\ ）并保存到 ``<eval_log_path>/video/eval``\ 。追加 ``env.eval.video_cfg.save_video=false`` 可关闭录像，追加 ``env.eval.task_id=<task_id>`` 可评估其他 Microduck 任务，追加 ``env.eval.play_mode=true`` 可从训练扰动配置（平地速度每 3–6 秒随机推力）切换到无头压力测试预设（不打开交互式窗口）：

- **速度 / 坐起 / 踢球任务**\ （Velocity / SitStand / BallKick）：将随机推力间隔缩短至 0.5–1.0 秒，用于测试抗推稳定性。
- **崎岖地形任务**\ （\ ``*-Rough-*``\ ）：关闭地形难度课程，使评估直接覆盖完整地形难度。
- **行走与倒地起身任务**\ （\ ``VelStand``\ ）：倒地后不提前终止回合，并施加更强、更频繁的推倒扰动，用于在单回合内连续测试倒地后起身能力。

下方视频展示了第 5000 步 checkpoint 在 ``Mjlab-Velocity-Flat-MicroDuck`` 上的单独评估效果（\ ``play_mode=false``\ ，16 个并行环境平均 ``eval/return`` 为 149.8，平均 ``eval/episode_len`` 为 903.8 / 1000）：

.. raw:: html

   <div style="text-align: center;">
     <video controls autoplay loop muted playsinline preload="metadata" width="640">
       <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/microduck_walk_eval.mp4" type="video/mp4">
       你的浏览器不支持视频播放。
     </video>
   </div>

致谢
----

本示例基于以下开源项目构建：

- `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`__\ ：提供 Microduck 机器人模型、任务定义、奖励与域随机化配置以及 sim2real 方案（代码采用 Apache-2.0 协议，3D 机器人网格采用 CC BY-NC-SA 4.0 协议）。
- `mujocolab/mjlab <https://github.com/mujocolab/mjlab>`__\ ：提供基于 MuJoCo Warp 的强化学习环境框架。
- `Rhoban/bam <https://github.com/Rhoban/bam>`__\ ：提供 BAM Dynamixel XL330 执行器模型。

DAgger
======

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg
   :align: center
   :width: 75%

   一个 DAgger 训练循环。

**DAgger** （Dataset Aggregation）是一种模仿学习算法：它让学生策略与环境交互，再让专家策略对访问到的状态重新标注，并持续聚合这些带专家标签的轨迹用于后续训练。本页介绍 RLinf 中基于模拟器的具身 DAgger 工作流，支持 MLP 和 Pi0 模型，以及 **同步** 和 **异步** 两种训练流程。

真实世界中 Franka 的 HG-DAgger 全流程请参考 :doc:`hg-dagger`。

概览
----------------------------------------

使用 DAgger 微调具身策略：学生策略与环境交互，专家对访问到的状态重新标注，聚合后的专家数据再训练学生策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      DAgger

   .. grid-item-card:: 模型
      :text-align: center

      MLP · π₀

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill · LIBERO · RoboTwin

   .. grid-item-card:: 训练
      :text-align: center

      Sync · Async

| **你将完成：** 安装 → 设置学生/专家检查点 → 运行 ``run_embodiment.sh``\ （或 ``run_async.sh``）→ 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 学生与专家检查点（见下文步骤）。

支持的配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - 模型
     - 环境
     - 配置
   * - MLP
     - ManiSkill（pick-cube）
     - ``maniskill_dagger_mlp.yaml``
   * - π₀
     - LIBERO-Spatial
     - ``libero_spatial_dagger_openpi.yaml``
   * - π₀
     - RoboTwin（adjust-bottle）
     - ``robotwin_adjust_bottle_dagger_openpi.yaml``
   * - π₀
     - LIBERO-Spatial（在线 LeRobot）
     - ``libero_spatial_dagger_openpi_lerobot.yaml``

DAgger 工作原理
----------------------------------------

**DAgger 训练流程**

1. **混合策略采样**

   - 在训练阶段，rollout worker 以概率 ``beta`` 选择专家动作。
   - 在评估阶段，RLinf 始终只使用学生策略。

2. **专家重标注**

   - 如果环境中执行的是学生动作，RLinf 会在同一观测上额外运行一次专家前向。
   - 专家动作会作为该 step 的监督目标被保存下来。

3. **Replay Buffer 训练**

   - 带专家标签的轨迹会写入 replay buffer。
   - actor 随后在这些样本上优化 ``embodied_dagger`` 损失。

4. **Beta 调度**

   - ``init_beta`` 控制初始的专家执行概率。
   - ``beta_schedule`` 和 ``beta_decay`` 控制从专家逐步切换到学生的速度。
   - ``beta_min`` 为可选项，用于设置 ``beta`` 的下界。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO（Pi0）
      :sync: libero

      安装环境并下载预训练 Pi0 学生权重（专家权重可来自 :doc:`pi0` 的 PPO 训练结果）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

      在 ``examples/embodiment/config/libero_spatial_dagger_openpi.yaml``\ （在线 LeRobot 对应 ``libero_spatial_dagger_openpi_lerobot.yaml``）中填写学生和专家路径（rollout 会自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

         rollout:
           expert_model:
             model_path: /path/to/expert_model

      启动同步或异步训练：

      .. code-block:: bash

         # 同步模式
         bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi
         bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi_lerobot

         # 异步模式
         bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi
         bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi_lerobot

   .. tab-item:: RoboTwin（Pi0）
      :sync: robotwin

      安装环境并下载预训练 Pi0 学生权重（按 :doc:`robotwin` 准备 RoboTwin 代码与资产，专家权重可来自 :doc:`pi0`）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

      在 ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml`` 中填写学生模型、专家模型和 RoboTwin 资产路径：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

         rollout:
           expert_model:
             model_path: /path/to/expert_model

         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      导出 ``ROBOTWIN_PATH``，启动同步或异步训练：

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         # 同步模式（异步模式换为 run_async.sh）
         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dagger_openpi

   .. tab-item:: ManiSkill（MLP）
      :sync: maniskill

      安装环境（复用 ManiSkill / LIBERO 环境）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      在 ``examples/embodiment/config/maniskill_dagger_mlp.yaml`` 中配置专家 checkpoint（可来自 :doc:`mlp` 的 PPO 训练结果）和可选学生 warm start：

      .. code-block:: yaml

         runner:
           ckpt_path: null                       # 可选：学生模型 warm start
           expert_ckpt_path: /path/to/expert_ckpt

      启动同步或异步训练：

      .. code-block:: bash

         # 同步模式（异步模式换为 run_async.sh）
         bash examples/embodiment/run_embodiment.sh maniskill_dagger_mlp

.. dropdown:: 在线 LeRobot DAgger

   三份经典配置将专家标注轨迹存入内存 **replay buffer**。``libero_spatial_dagger_openpi_lerobot.yaml`` 提供 **在线 LeRobot** 路径：以 LeRobot 格式收集完整 episode，经内存流式传给 actor，并按帧级滑动窗口训练。

   **经典 replay buffer vs 在线 LeRobot**

   两种路径优化相同的 DAgger 目标，区别仅在于数据的存储与采样方式：

   - **经典 replay buffer** — 将完整轨迹存入内存 buffer，按轨迹级滑动窗口采样。
   - **在线 LeRobot** — 以 LeRobot 格式收集完整 episode，按帧级 rolling window 采样，原生支持 Pi0 action-chunk 监督，并可选择仅保留成功 episode。

   在线 LeRobot 路径标签更干净、更强调近期数据，并与 SFT 及真机数据管线对齐。训练 Pi0 或其他 chunk-based 模型、需要 success-only 过滤，或希望将在线 DAgger 与离线 SFT / 真机数据衔接时，推荐使用该路径。

   **在线 LeRobot 流程**

   1. **混合 rollout 与专家重标注** — 与经典 DAgger 相同的 ``beta`` 调度与专家重标注。
   2. **Episode 收集** — ``algorithm.dagger.online_lerobot.enabled`` 设为 ``True`` 后，轨迹收集器累积各环境的帧并导出完整 episode。
   3. **Actor 接收** — 完成的 episode 流式传给 actor，追加到滚动 LeRobot 数据集中。
   4. **滑动窗口训练** — actor 从滚动数据集采样（可选开启解码帧缓存），优化 ``embodied_dagger`` 损失。

   **离线 vs 在线采集**

   - **离线落盘** — ``env.train.data_collection.enabled`` 将 LeRobot shard 写入磁盘，供后续 SFT 或 HG-DAgger 使用。见 :doc:`../../guides/data_collection` 与 :doc:`hg-dagger`。
   - **在线内存采集** — ``algorithm.dagger.online_lerobot.enabled`` 在 DAgger 训练过程中直接将 episode 流式传给 actor。启用 online LeRobot 时，**不要** 在同一 train env 上同时开启 ``env.train.data_collection``。

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置好 beta 调度、replay buffer 大小以及在线 LeRobot 参数，仅在需要调优时再覆写。

   **经典 DAgger 与 replay buffer 参数**

   .. code-block:: yaml

      algorithm:
        dagger:
          only_save_expert: False   # 经典 DAgger：全部样本都由专家重标注后保存
          init_beta: 1.0
          beta_schedule: "exponential"
          beta_decay: 0.99
          beta_min: 0.05            # 可选；代码默认值为 0.05

        replay_buffer:
          enable_cache: True
          cache_size: 2000
          min_buffer_size: 16
          sample_window_size: 2000

   MLP ManiSkill 示例默认使用更大的 replay buffer，且 ``beta_decay: 0.98``。实际取值请以启动时使用的 YAML 为准。

   **在线 LeRobot 参数**

   在线 LeRobot 配置不使用 ``algorithm.replay_buffer``，请调整 ``algorithm.dagger.online_lerobot`` 字段：

   .. code-block:: yaml

      algorithm:
        dagger:
          online_lerobot:
            enabled: True
            only_success: True
            robot_type: "panda"
            fps: 10
            finalize_interval: 8
            data_path: ${runner.logger.log_path}/physical-intelligence/libero
            rolling_lerobot_window_size: 50000
            enable_decoded_cache: true
            decoded_cache_capacity: 25000
            cache_ingest_mode: new_shards   # 或 last_n / both
            lerobot_num_workers: 0          # 开启 cache 时推荐为 0

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 配置项
        - 含义
      * - ``online_lerobot.enabled``
        - 开启 DAgger 训练过程中的在线 LeRobot episode 收集。
      * - ``online_lerobot.only_success``
        - 仅保留成功 episode；失败 rollout 会被丢弃。
      * - ``online_lerobot.data_path``
        - rolling dataset 写入 LeRobot shard 的根目录。
      * - ``online_lerobot.finalize_interval``
        - 每 N 个 episode finalize 一次 LeRobot 元数据。
      * - ``rolling_lerobot_window_size``
        - 滑动训练窗口保留的最大帧数。
      * - ``enable_decoded_cache``
        - 缓存解码后的帧，加速 actor DataLoader 采样。

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 推荐关注的监控指标**

指标含义见 :doc:`训练指标 <../../reference/metrics>`。DAgger 专属指标：

- ``env/success_once``：推荐用于监控具身 DAgger 训练效果的成功率指标。
- ``train/dagger/actor_loss``：基于专家标注样本计算的 DAgger 监督损失。
- ``train/actor/lr``：学习率。
- ``train/actor/grad_norm``：梯度范数。
- ``train/replay_buffer/num_trajectories``：replay buffer 中轨迹数量。
- ``train/replay_buffer/total_samples``：replay buffer 中可训练样本总数。
- ``train/replay_buffer/cache_size``：当前缓存的展平轨迹数量。

**3. 实验结果**

.. csv-table::
   :header: "配置", "学生初始成功率", "专家成功率", "训练时间", "学生最终成功率"

   "MLP + ManiSkill", "0%", "100%", "20min", "100%"
   "Pi0 + LIBERO", "60%", "95%", "17h", "93%"

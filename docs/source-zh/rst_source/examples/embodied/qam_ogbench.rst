在 OGBench 上训练离线 QAM
================================

用 RLinf 的原生 PyTorch QAM 从 OGBench 离线数据集训练 flow policy，再到模拟器中评估成功率。本示例采用 AntMaze 导航任务，每个训练 seed 使用一张 GPU。

概览
----

训练只使用固定的离线数据，评估则运行完整 episode，检查 policy 的任务完成情况。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      QAM

   .. grid-item-card:: 模型
      :text-align: center

      Flow MLP policy

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      OGBench AntMaze

   .. grid-item-card:: 训练方式
      :text-align: center

      离线

| **你将完成：** 安装 → 使用一个随机种子训练 → 恢复训练或评估 checkpoint → 观察 ``eval/success``\ 。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一张 NVIDIA GPU · 首次运行时下载 OGBench 数据。

任务
~~~~

示例配置对应以下任务及观测、动作维度：

.. list-table::
   :header-rows: 1

   * - 任务
     - 状态 / 动作维度
     - 配置
   * - ``antmaze-large-navigate-singletask-task1-v0``
     - 29 / 8
     - ``ogbench_qam_offline.yaml``

评估默认每个 worker 运行一个环境。设置 ``env.eval.auto_reset=false`` 和 ``env.eval.early_stop_on_done=false`` 后，也可以并行评估多个环境。每个环境独立保存 episode seed、回报、长度、成功标记和终点状态。本示例的动作 chunk 长度为 1。训练也支持较长的动作 chunk，并在 episode 边界补齐数据；其他任务和 chunk 长度仍需单独测试。

QAM 的工作方式
--------------

多个 critic 为候选动作打分。Slow flow policy 学习数据集中的动作分布，fast flow policy 则利用沿 flow 反向传播的价值梯度改进动作。评估使用 fast policy。算法参考见 `QAM 官方实现 <https://github.com/ColinQiyangLi/qam/tree/2726d767c9a0a7a46d49693f0391f73dc2cf58ac>`_\ 。本示例覆盖离线训练，在线微调不在支持范围内。

安装
----

在 RLinf 仓库根目录安装 OGBench 所需的具身训练依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --env ogbench
   source .venv/bin/activate

安装器使用 OGBench 1.2.1 和 MuJoCo 3.3.4。启动脚本会为无显示器评估配置 EGL。本示例已在 NVIDIA A800 上验证训练，其他加速后端尚无 benchmark 结果。

首次训练会将 ``antmaze-large-navigate-v0.npz`` 及其验证集下载到 ``~/.ogbench/data``\ ，共约 255 MiB（267 MB）。已有数据放在其他磁盘上时，在各条启动命令后追加 ``data.dataset_dir=/path/to/ogbench-data`` 即可复用。评估产生的交互数据不会加入训练集。

默认模型的每个 checkpoint 约占 146 MiB，每个 seed 保存四份，共约 583 MiB。除 Python 环境和包缓存外，建议为每个 seed 至少预留 1 GiB，用于数据、checkpoint 和日志。下方实验在共享 A800 上，每个 seed 完成训练约需 8–11 小时，包含中途评估，不含最终评估。实际耗时取决于硬件和 GPU 负载。

运行
----

使用标准离线启动脚本训练一个 seed：

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001

默认训练 1,000,000 次更新，batch size 为 256。每 50,000 次更新评估 50 个 episode，每 250,000 次更新保存 checkpoint。控制台日志位于 ``logs/<timestamp>-ogbench_qam_offline/run_offline_rl.log``\ ，其中会输出训练和评估指标。

下面的命令依次训练三个独立 seed：

.. code-block:: bash

   for seed in 10001 20002 30003; do
     bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
       actor.seed="$seed" \
       runner.logger.experiment_name="qam-antmaze-$seed"
   done

每次启动都会创建独立日志目录。要使用 GPU 1，在命令后追加 ``cluster.component_placement.actor=1-1``\ 、\ ``cluster.component_placement.env=1-1`` 和 ``cluster.component_placement.rollout=1-1``\ 。同时运行多个训练任务时，为每个任务分配不同的空闲 GPU。

恢复训练与评估
--------------

上方单 seed 命令的 checkpoint 位于 ``logs/<timestamp>-ogbench_qam_offline/ogbench_qam_offline/checkpoints/``\ 。将下面的路径替换为该次运行的第 250000 步目录，即可恢复训练：

.. code-block:: bash

   QAM_CHECKPOINT=/absolute/path/to/checkpoints/global_step_250000
   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT"

恢复时保持原来的 seed、模型维度与架构、优化器设置、数据集和 batch size。原命令中的自定义参数也要保留，尤其是小模型测试所用的参数。Checkpoint 会恢复模型参数、优化器状态、训练噪声生成器和数据采样器。继续训练的目标步数必须大于已保存步数。恢复训练后，评估使用新的随机数序列。即使训练从保存的状态继续，评估 episode 也可能与原运行不同。

追加 ``runner.only_eval=true`` 可以只评估 checkpoint：

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT" \
     runner.only_eval=true

这条命令评估 50 个 episode，并在新日志目录中按 checkpoint 的步数记录评估指标。完成后应能看到 ``eval/num_trajectories=50``\ 。

通过标准 EnvWorker 和 RolloutWorker 并行评估两个环境，可以运行：

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir="$QAM_CHECKPOINT" \
     runner.only_eval=true \
     env.eval.total_num_envs=2 \
     env.eval.early_stop_on_done=false

使用一个评估 worker 时，每轮评估会在两个环境中各运行一个 episode。默认 ``runner.eval_episodes=50`` 因此共完成 100 个 episode，并记录 ``eval/num_trajectories=100``\ 。worker 会运行到设定的 episode 步数上限，避免一个环境提前成功后中断另一个环境。

可视化与结果
------------

启动 TensorBoard 查看各次独立运行：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

每次运行的事件文件和实际使用的配置位于 ``<log_directory>/tensorboard/``\ 。可重点观察下表中的指标，通用日志约定见 :doc:`训练指标 <../../reference/metrics>`\ 。

.. list-table::
   :header-rows: 1

   * - 指标
     - 观察内容
   * - ``train/critic_loss``\ 、\ ``train/flow_loss``\ 、\ ``train/adj_loss``
     - Critic、slow policy 和 fast policy 的损失是否为有限值
   * - ``eval/success``
     - 成功 episode 的比例
   * - ``eval/return``\ 、\ ``eval/episode_len``
     - 平均原始 episode 回报与长度
   * - ``eval/num_trajectories``
     - 已完成的评估 episode 数量

以下结果由作者在 `PR #1811 <https://github.com/RLinf/RLinf/pull/1811>`_ 中报告，来自提交 ``569ee9c`` 的离线接入版本，使用 A800 GPU。三个 seed 均采用默认 AntMaze 模型，训练至 1,000,000 次更新后各评估 50 个 episode。后续版本尚未重新验证这些结果。

.. list-table::
   :header-rows: 1

   * - Seed
     - 成功率
     - 平均 Episode 长度
   * - 10001
     - 0.96
     - 531.38
   * - 20002
     - 0.74
     - 730.88
   * - 30003
     - 0.92
     - 545.38
   * - 均值 ± 样本标准差
     - 0.8733 ± 0.1172
     - —

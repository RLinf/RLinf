Bench2Dex
=========

使用 RLinf 在 `Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__\ （基于 Isaac Sim 的灵巧手操作基准）并行环境中训练和评测 π₀.₅ policy。Bench2Dex 通过 ``none``、``cov``、``inv`` 和 ``inv_cov`` 四个泛化通道评测 policy；本页以 Task22 和 Task64 为例介绍完整流程，同时支持通过配置扩展到其他 Bench2Dex 场景和机器人。

概览
----

policy 根据双目和腕部 RGB 图像以及关节位置，输出各关节的目标绝对位置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO · 异步 PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA GPU

| **你将完成：** 安装依赖 → 下载资产、场景锚点并转换 π₀.₅ 权重 → 运行 critic 预热与 PPO → 完成四通道评测 → 观察 ``env/success_once``。
| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · NVIDIA GPU（Task22 训练与 8 卡评测使用 8 × 80 GB GPU，Task64 训练与评测使用 4 × 80 GB GPU）。

任务、观测与动作
~~~~~~~~~~~~~~~~

示例配置涵盖两项双臂灵巧手操作任务：

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - 场景
     - 机器人
     - 动作维度
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 双目与腕部 RGB 图像，加机器人受控关节位置。
   * - 动作
     - 对应机器人受控关节的目标绝对位置。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - 来自 Bench2Dex 场景定义的任务描述。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Task22（Critic 预热 + PPO）
      :sync: bench2dex-task22

      先从转换后的 π₀.₅ 权重运行 10 轮 critic 预热（1,400 次优化器更新），再在固定白底场景中使用 8 × 80 GB GPU 运行 190 轮同步 PPO（每 10 轮评测并保存一次 checkpoint）。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env bench2dex \
           --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
         source .venv-bench2dex-openpi/bin/activate

      **2. 下载资产、场景锚点并转换权重**

      从 `Bench2Dex/Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`__ 下载场景资产，从 `Bench2Dex/teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`__ 下载 Task22 锚点，从 `Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`__ 下载官方 π₀.₅ checkpoint：

      .. code-block:: bash

         hf download Bench2Dex/Assets --repo-type dataset \
           --local-dir ./data/bench2dex/assets/dex2bench_dataset
         hf download Bench2Dex/teleopdata --repo-type dataset \
           --include "dataset/22_tool_box_loading/origin-generalization/**" \
           --local-dir ./data/bench2dex/teleopdata
         hf download Bench2Dex/policy_ckpt \
           --include "22/multi_panda_with_orca/pi05/pi05_base_dex2bench_full/22_pi05_256/2000/**" \
           --local-dir ./data/bench2dex/jax

      ``BENCH2DEX_ASSETS_PATH`` 需指向包含 ``dex2bench_dataset/`` 的父目录；激活 ``.venv-bench2dex-openpi`` 时会自动导出 ``BENCH2DEX_PATH``。

      导出本地路径（Task22 的训练与评测配置直接读取 ``BENCH2DEX_ASSETS_PATH``、``BENCH2DEX_ANCHOR_DIR`` 和 ``BENCH2DEX_PI05_PATH``）并将下载的 checkpoint 转换为 PyTorch 格式：

      .. code-block:: bash

         export BENCH2DEX_ASSETS_PATH="$PWD/data/bench2dex/assets"
         export BENCH2DEX_ANCHOR_DIR="$PWD/data/bench2dex/teleopdata/dataset/22_tool_box_loading/origin-generalization"
         export BENCH2DEX_PI05_PATH="$PWD/data/bench2dex/pi05_task22"
         CKPT_SRC=./data/bench2dex/jax/22/multi_panda_with_orca/pi05/pi05_base_dex2bench_full/22_pi05_256/2000
         JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT_SRC" --input-norm-stats "$CKPT_SRC/assets/dex2bench/norm_stats.json" \
           --output-model "$BENCH2DEX_PI05_PATH" \
           --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
           --action-dim 40 --action-horizon 20 --max-token-len 280

      **3. 启动 Critic 预热与 PPO**

      用转换后的 π₀.₅ 权重为 Task22 预热 critic，共 10 轮（1,400 次优化器更新）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task22_critic_warmup_openpi_pi05 \
           runner.logger.log_path="$PWD/results/task22-warmup"

      预热结束后，通过 ``runner.ckpt_path`` 加载本地保存的 ``full_weights.pt`` 启动 PPO，训练会重新初始化优化器：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05 \
           runner.ckpt_path="$PWD/results/task22-warmup/bench2dex_task22_critic_warmup_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"

      **4. 四通道评测**

      使用 8 张 GPU 在四个泛化通道上评测 Task22，每个通道运行 32 个回合。``rollout.model.model_path`` 可填 ``$BENCH2DEX_PI05_PATH``，也可填训练保存的 ``full_weights.pt`` 路径：

      .. code-block:: bash

         python -m evaluations.bench2dex.run_all_channels \
           bench2dex_task22_openpi_pi05_eval --output-root ./results/task22-eval \
           --gpu-id 0,1,2,3,4,5,6,7 --num-envs 32 --num-episodes 32 \
           "rollout.model.model_path=$BENCH2DEX_PI05_PATH" \
           "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

   .. tab-item:: Task64（PPO）
      :sync: bench2dex-task64

      在 ``64_sports_ball_cup_sort`` 双臂 Panda 与 Allegro 灵巧手环境（32 维动作，4 × 80 GB GPU，16 个并行环境）上训练或评测 π₀.₅ policy。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env bench2dex \
           --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
         source .venv-bench2dex-openpi/bin/activate

      **2. 下载资产、场景锚点并转换权重**

      从 `Bench2Dex/Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`__ 下载资产，从 `Bench2Dex/teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`__ 下载 Task64 锚点，从 `Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`__ 下载 Task64 的 π₀.₅ checkpoint：

      .. code-block:: bash

         hf download Bench2Dex/Assets --repo-type dataset \
           --local-dir ./data/bench2dex/assets/dex2bench_dataset
         hf download Bench2Dex/teleopdata --repo-type dataset \
           --include "dataset/64_sports_ball_cup_sort/origin-generalization/**" \
           --local-dir ./data/bench2dex/teleopdata
         hf download Bench2Dex/policy_ckpt \
           --include "64/multi_panda_with_allegro/pi05/pi05_base_dex2bench_full/64_pi05_256/2000/**" \
           --local-dir ./data/bench2dex/jax

      ``BENCH2DEX_ASSETS_PATH`` 需指向包含 ``dex2bench_dataset/`` 的父目录；激活 ``.venv-bench2dex-openpi`` 时会自动导出 ``BENCH2DEX_PATH``。

      导出本地路径并将下载的 checkpoint 转换为 PyTorch 格式（Task64 训练与评测均读取 ``BENCH2DEX_ASSETS_PATH`` 和 ``BENCH2DEX_PI05_PATH``，``BENCH2DEX_ANCHOR_DIR`` 仅用于验证和四通道评测）：

      .. code-block:: bash

         export BENCH2DEX_ASSETS_PATH="$PWD/data/bench2dex/assets"
         export BENCH2DEX_ANCHOR_DIR="$PWD/data/bench2dex/teleopdata/dataset/64_sports_ball_cup_sort/origin-generalization"
         export BENCH2DEX_PI05_PATH="$PWD/data/bench2dex/pi05_task64"
         CKPT_SRC=./data/bench2dex/jax/64/multi_panda_with_allegro/pi05/pi05_base_dex2bench_full/64_pi05_256/2000
         JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT_SRC" --input-norm-stats "$CKPT_SRC/assets/dex2bench/norm_stats.json" \
           --output-model "$BENCH2DEX_PI05_PATH" \
           --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
           --action-dim 32 --action-horizon 20 --max-token-len 280

      **3. 启动 PPO**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

      **4. 四通道评测**

      使用 4 张 GPU 与 16 个并行环境在四个通道上评测 Task64，每个通道运行 50 个回合：

      .. code-block:: bash

         python -m evaluations.bench2dex.run_all_channels \
           bench2dex_task64_openpi_pi05_eval --output-root ./results/task64-eval \
           --gpu-id 0,1,2,3 --num-envs 16 --num-episodes 50 \
           "rollout.model.model_path=$BENCH2DEX_PI05_PATH" \
           "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

.. list-table:: 评测通道
   :header-rows: 1
   :widths: 20 80

   * - 通道
     - 场景变化
   * - ``none``
     - 原始锚点场景。
   * - ``cov``
     - 物体位姿和桌面高度。
   * - ``inv``
     - 外观、杂物和相机设置。
   * - ``inv_cov``
     - 组合变化，并使用未见过的资产。

.. dropdown:: 其他训练配置（异步 PPO 与 GRPO）

   将启动命令中的配置名替换为下表对应项即可切换算法；异步 PPO 使用 ``examples/embodiment/run_async.sh``，GRPO 使用 ``examples/embodiment/run_embodiment.sh``。各配置均直接读取 ``BENCH2DEX_PI05_PATH``：

   .. list-table::
      :header-rows: 1

      * - 任务
        - 算法
        - 配置名
      * - Task22
        - 异步 PPO
        - ``bench2dex_task22_async_ppo_openpi_pi05``
      * - Task64
        - 异步 PPO
        - ``bench2dex_task64_async_ppo_openpi_pi05``
      * - Task64
        - GRPO
        - ``bench2dex_task64_grpo_openpi_pi05``

.. dropdown:: 适配其他 Bench2Dex 任务与录制视频（可选）

   运行其他 Bench2Dex 任务和机器人构型时，通过下表中的 Hydra 配置项指定场景、机器人与对应权重、锚点：

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - Hydra 配置项
        - 作用
      * - ``env.train.init_params.task`` / ``env.train.init_params.embodiment``
        - 选择 Bench2Dex 场景名和机器人构型（开启验证时同步设置 ``env.eval``）。
      * - ``env.train.init_params.anchor_dir``
        - 指向该任务的 HDF5 场景锚点目录。
      * - ``actor.model.action_dim`` / ``actor.model.openpi.model_action_dim``
        - 设置为 policy 控制的机器人受控关节数。
      * - ``env.train.max_episode_steps`` / ``env.train.max_steps_per_rollout_epoch``
        - 分别设置为任务控制步上限以及按 ``actor.model.num_action_chunks`` 向上取整后的单轮 rollout 步数上限。
      * - ``env.train.skip_intermediate_renders`` / ``env.eval.skip_intermediate_renders``
        - 开启 ``env.train.video_cfg.save_video=true`` 或 ``env.eval.video_cfg.save_video=true`` 录制视频时，需将其设为 ``false``。

   调整 GPU 分配见 :doc:`Placement </rst_source/concepts/placement>`，恢复已保存的训练见 :doc:`断点续训 </rst_source/guides/resume>`。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 与 ``eval/success_once``\ （查看上文 Task22 critic 预热日志时请传 ``--logdir ./results/task22-warmup``），指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

四通道评测会在 ``--output-root`` 下生成汇总各通道成功率的 ``success_rates.tsv``，并在各通道子目录写入逐回合的 ``per_episode.jsonl``；通用单配置评测命令行用法见 :doc:`评测 CLI 参考 </rst_source/evaluations/reference/cli>`。

Task22 固定白底同步 PPO
~~~~~~~~~~~~~~~~~~~~~~~

Task22 同步 PPO 从预热权重加载并新建优化器，在固定白底场景下每轮收集 128 个训练回合，并在第 10 轮和第 20 轮后各运行 32 个回合的独立评测。参考配置共 190 轮；验证实验在独立评测达到 100% 后于第 22 轮停止：

.. list-table::
   :header-rows: 1
   :widths: 38 20 42

   * - 阶段
     - 轮次
     - 成功率
   * - 训练 rollout
     - 1
     - 86.72%（111/128）
   * - 训练 rollout
     - 22
     - 96.88%（124/128）
   * - 独立评测
     - 10
     - 96.88%（31/32）
   * - 独立评测
     - 20
     - 100.00%（32/32）

Task64 四通道评测与并行吞吐
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 NVIDIA A800 80 GB GPU 上测试官方 ``2000`` 步 π₀.₅ checkpoint 在 ``64_sports_ball_cup_sort``\ （双臂 Panda 与 Allegro 灵巧手）上的 RL 前评测一致性与端到端物理步吞吐。成功率取 4 卡 16 环境 3 次运行、每通道 50 回合的均值，吞吐对比官方单卡单环境实现：

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 19 20 16

   * - 通道
     - 官方 π₀.₅
     - RLinf π₀.₅（3 次均值）
     - 官方吞吐（1 卡 / 1 环境）
     - RLinf 吞吐（4 卡 / 16 环境）
     - 相对官方单卡倍率
   * - ``none``
     - 42.00%
     - 34.00%
     - 11.94 steps/s
     - 65.30 steps/s
     - 5.47×
   * - ``cov``
     - 18.00%
     - 16.67%
     - 11.41 steps/s
     - 65.89 steps/s
     - 5.78×
   * - ``inv``
     - 4.00%
     - 16.00%
     - 11.88 steps/s
     - 54.52 steps/s
     - 4.59×
   * - ``inv_cov``
     - 22.00%
     - 18.67%
     - 11.83 steps/s
     - 65.80 steps/s
     - 5.56×
   * - **平均**
     - **21.50%**
     - **21.33%**
     - —
     - —
     - —

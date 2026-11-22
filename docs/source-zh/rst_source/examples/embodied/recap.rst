RECAP
=====

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png
   :align: center
   :width: 80%

   The RECAP offline pipeline.

使用 **RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）** 在离线数据上优化策略。RECAP 不需要在线环境交互：先离线计算 return、训练 value model、估计 advantage，再通过 **Classifier-Free Guidance（CFG）训练** 优化策略，适合真实机器人等难以大规模在线采样的场景。

概览
----------------------------------------

离线提升 π₀.₅ 策略（无需新采样）：用价值模型为已有数据打分，再以无分类器引导（CFG）进行优化。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      RECAP (CFG)

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      LeRobot 数据集

   .. grid-item-card:: 训练
      :text-align: center

      离线 · 4 阶段

| **你将完成：** 计算回报 → SFT 价值模型 → 计算优势 → CFG 训练策略 → 评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SigLIP2 + Gemma3 + π₀.₅ 检查点 · LeRobot 格式数据集（见下文步骤）。

RECAP 工作原理
----------------------------------------

RECAP 全程离线运行，包含四个阶段：

1. **计算回报（Compute Returns）**：对每条轨迹逆序计算折扣累积回报（默认 ``gamma: 1.0``；SFT 成功轨迹每步奖励 ``-1``、终止步 ``0``，失败 rollout 轨迹终止步惩罚为 ``failure_reward``，如 ``-300``），结果以 sidecar 文件保存，不修改原始数据。
2. **训练价值模型（Value Model SFT）**：训练价值模型（SigLIP2 视觉编码器 + Gemma3 语言模型 + 可学习 Critic Expert），输出归一化回报空间 ``[-1, 0]`` 上 201 个 bin 的分类价值分布。
3. **计算优势（Compute Advantages）**：用训练好的价值模型在多卡上分布式推理，计算 N 步前瞻优势，并将排名前 ``positive_quantile``\ （默认前 30%）的样本标记为正样本。
4. **CFG 训练**：用优势标签对 OpenPI π₀.₅ 做无分类器引导（CFG）训练——正样本作为条件输入（按 ``unconditional_prob`` 概率随机丢弃为无条件，默认 ``0.1``），负样本始终作为无条件输入；推理时通过 ``cfgrl_guidance_scale`` 引导模型生成高优势动作。

安装
----------------------------------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI（LIBERO / ManiSkill）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate


下载模型
----------------------------------------

RECAP 流程需要以下预训练模型：

- **SigLIP2-so400m**：视觉编码器，用于 Step 2 价值模型训练
- **Gemma3-270M**：语言模型，用于 Step 2 价值模型训练
- **pi0.5 base (PyTorch)**：策略模型，用于 Step 4 CFG 训练。参考 `openpi <https://github.com/Physical-Intelligence/openpi>`_ 获取模型权重并转换为 PyTorch 格式

.. tab-set::

   .. tab-item:: SigLIP2 + Gemma3（价值模型）

      .. code-block:: bash

         hf download google/siglip2-so400m-patch14-224 --local-dir ./siglip2-so400m-patch14-224
         hf download google/gemma-3-270m --local-dir ./gemma-3-270m

下载完成后，在 Step 2 和 Step 4 的配置文件中指定模型路径：

.. code-block:: yaml

   # Step 2 价值模型配置
   actor:
     model:
       siglip_path: ./siglip2-so400m-patch14-224
       gemma3_path: ./gemma-3-270m
       tokenizer_path: ./gemma-3-270m

   # Step 4 策略模型配置
   actor:
     model:
       model_path: /path/to/pi05_base_pytorch


数据准备
----------------------------------------

RECAP 流程使用 LeRobot 格式的数据集。数据集分为两类：

- **SFT 数据集**：来自人类演示或已训练策略的成功轨迹
- **Rollout 数据集**：在线交互采集的轨迹，包含成功和失败

数据集配置示例：

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft                # 全部成功轨迹
         weight: 1.0
       - dataset_path: /path/to/rollout_dataset
         type: rollout            # 含失败轨迹
         weight: 1.0

.. note::

   ``train_data_paths`` 是一个列表。如果需要混合多个数据集，可以继续新增 item；如果只想使用单个数据集训练，也可以只保留一个 item。
   所有 step 中的 ``train_data_paths`` 应保持一致，确保回报、价值和优势的计算基于同一批数据。

Pipeline Tag 机制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RECAP 通过 **tag** 实现各步骤之间的数据传递和版本管理：

- **returns_tag**：Step 1 生成，Step 2 和 Step 3 读取。请确保 Step 1 的 ``data.tag``、Step 2 的 ``data.tag`` 和 Step 3 的 ``advantage.returns_tag`` 保持一致。
- **advantage_tag**：Step 3 生成，Step 4 读取。请确保 Step 3 的 ``advantage.tag`` 和 Step 4 的 ``data.advantage_tag`` 保持一致。

.. list-table:: **Tag 在各步骤中的流转**
   :header-rows: 1

   * - Step
     - 配置字段
     - 说明
   * - 1
     - ``data.tag``
     - 写入 ``meta/returns_{tag}.parquet``
   * - 2
     - ``data.tag``
     - 读取 ``meta/returns_{tag}.parquet``
   * - 3
     - ``advantage.returns_tag``
     - 读取 ``meta/returns_{tag}.parquet``
   * - 3
     - ``advantage.tag``
     - 写入 ``meta/advantages_{tag}.parquet``
   * - 4
     - ``data.advantage_tag``
     - 读取 ``meta/advantages_{tag}.parquet``


Step 1：计算回报（Compute Returns）
----------------------------------------

本步骤对数据集中每条轨迹逆序计算折扣累积回报，结果以 sidecar 文件形式保存，不修改原始数据。

**配置文件**

在 ``examples/offline_rl/config/recap_compute_returns.yaml`` 中填写数据集路径和输出 tag：

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout
     eval_data_paths:          # 需覆盖 Step 2 的 eval_data_paths
       - dataset_path: /path/to/eval_dataset
         type: rollout
     tag: "fail300"

**启动命令**

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置 ``gamma: 1.0``、``failure_reward: -300.0`` 和 ``num_workers: 128``：

   .. code-block:: yaml

      data:
        gamma: 1.0              # 折扣因子
        failure_reward: -300.0   # 失败轨迹终止奖励
        tag: "fail300"           # 输出文件标签
        num_workers: 128         # 并行处理线程数

   .. list-table::
      :header-rows: 1
      :widths: 30 15 55

      * - 参数
        - 默认值
        - 说明
      * - ``data.gamma``
        - ``1.0``
        - 折扣因子。``1.0`` 表示无折扣，即简单累加未来奖励
      * - ``data.failure_reward``
        - ``-300.0``
        - 失败轨迹终止步的惩罚值。值越大（绝对值），成功/失败的回报区分度越高
      * - ``data.eval_data_paths``
        - ``[]``
        - eval 数据集，需包含 Step 2 ``data.eval_data_paths`` 中的所有数据集
      * - ``data.tag``
        - ``null``
        - 输出文件标签，生成 ``meta/returns_{tag}.parquet``
      * - ``data.num_workers``
        - ``128``
        - 并行处理 parquet 文件的线程数

**输出文件**

- ``meta/returns_{tag}.parquet``：每行包含 ``episode_index``、``frame_index``、``return``、``reward``、``prompt``
- ``meta/stats.json``：更新回报统计信息（均值、标准差、最小值、最大值）

**验证方法**

.. code-block:: bash

   python3 -c "
   import json
   stats = json.load(open('./datasets/rollout/meta/stats.json'))
   assert 'return' in stats
   print('return stats:', stats['return'])
   "


Step 2：训练价值模型（Value Model SFT）
-------------------------------------------

使用 Step 1 计算的回报作为监督信号，训练价值模型学习从观察（图像 + 语言指令）预测归一化回报。

**模型架构**

价值模型由三部分组成：

- **视觉编码器**：SigLIP2-so400m（1152 维）—— 处理 RGB 图像输入
- **语言模型**：Gemma3-270M（640 维）—— 处理语言指令
- **Critic Expert**：可学习专家头，将多模态表征映射到价值预测

输出为分类分布（Categorical Value Distribution），覆盖 ``[-1, 0]`` 区间的 201 个 bin。

**配置文件**

在 ``examples/offline_rl/config/recap_value_model_sft.yaml`` 中填写数据集路径、Step 1 的 ``tag`` 和上文下载的模型路径：

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
         weight: 1.0
         robot_type: "libero"
         model_type: "pi05"
     tag: "fail300"                     # 对应 Step 1 的 tag

**启动命令**

训练脚本会自动初始化 Ray 集群：

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置评估采样、batch size、优化器超参和保存间隔默认值：

   .. code-block:: yaml

      data:
        action_horizon: 10
        normalize_to_minus_one_zero: true  # 归一化到 [-1, 0]
        eval_data_paths:                   # 可选，推荐配置
          - dataset_path: /path/to/eval_dataset
            max_samples: 10000
            robot_type: "libero"
            model_type: "pi05"

      actor:
        micro_batch_size: 32
        global_batch_size: 256
        model:
          freeze_vlm: false         # 是否冻结 VLM backbone
          value_dropout: 0.0        # Value Head dropout
        optim:
          lr: 5.0e-5               # VLM backbone 学习率
          value_lr: 1.0e-4          # Value Head 学习率
          weight_decay: 1.0e-10
          lr_warmup_steps: 500

      runner:
        max_epochs: 30000
        save_interval: 3000         # checkpoint 保存间隔

   .. list-table::
      :header-rows: 1
      :widths: 35 15 50

      * - 参数
        - 默认值
        - 说明
      * - ``data.tag``
        - ``null``
        - 与 Step 1 相同的 tag，用于读取对应的 ``returns_{tag}.parquet``
      * - ``data.normalize_to_minus_one_zero``
        - ``true``
        - 归一化模式。``true`` 映射到 ``[-1, 0]``；``false`` 映射到 ``[0, 1]``
      * - ``actor.model.freeze_vlm``
        - ``false``
        - 冻结视觉编码器。设为 ``true`` 时需同时设置 ``actor.fsdp_config.use_orig_params: True``
      * - ``actor.model.value_dropout``
        - ``0.0``
        - Value Head 前的 dropout 率
      * - ``actor.optim.lr``
        - ``5.0e-5``
        - VLM backbone 学习率
      * - ``actor.optim.value_lr``
        - ``1.0e-4``
        - Value Head 学习率

**输出**

- 模型 checkpoint 保存在 ``logs/value_sft/{config_name}-{timestamp}/value_sft/checkpoints/``
- TensorBoard 日志

**关键监控指标**

- ``train/actor/loss``：价值模型的总训练损失
- ``train/actor/grad_norm``：梯度范数
- ``eval/spearman_correlation``：Spearman 相关系数，衡量预测值与真实回报的排序一致性

.. note::

   训练完成后，需要记录 checkpoint 路径用于 Step 3。Checkpoint 位于：

   .. code-block:: text

      logs/value_sft/{config_name}-{timestamp}/value_sft/checkpoints/global_step_{N}/actor/model_state_dict


Step 3：计算优势（Compute Advantages）
-------------------------------------------

使用 Step 2 训练好的价值模型，按 ``advantage_lookahead_step`` 与 ``gamma`` 计算数据集中每个时间步的 N 步前瞻优势值，并根据分位数阈值将样本标记为正/负。

**配置文件**

在 ``examples/offline_rl/config/recap_compute_advantages.yaml`` 中填写 Step 2 checkpoint、tag 和数据集路径：

.. code-block:: yaml

   advantage:
     value_checkpoint: /path/to/value_checkpoint
     tag: "fail300_N10_ckpt18000_q30"
     returns_tag: "fail300"       # 对应 Step 1 的 tag

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         robot_type: "libero"
         type: "sft"
         weight: 1.0

**启动命令**

支持多 GPU 分布式推理：

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh recap_compute_advantages

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置 ``positive_quantile: 0.3``、``batch_size: 1024``、``advantage_lookahead_step: 10`` 和 ``gamma: 1.0``：

   .. code-block:: yaml

      advantage:
        positive_quantile: 0.3      # top 30% 标记为正样本
        batch_size: 1024

      data:
        advantage_lookahead_step: 10   # N 步前瞻
        gamma: 1.0

   .. list-table::
      :header-rows: 1
      :widths: 35 15 50

      * - 参数
        - 默认值
        - 说明
      * - ``advantage.value_checkpoint``
        - 必填
        - Step 2 训练好的价值模型 checkpoint 路径
      * - ``advantage.positive_quantile``
        - ``0.3``
        - 正样本比例。``0.3`` 表示优势值 top 30% 的样本标记为正
      * - ``data.advantage_lookahead_step``
        - ``10``
        - 前瞻步数 N，即考虑未来多少步的奖励
      * - ``advantage.returns_tag``
        - ``null``
        - 读取 Step 1 生成的回报文件标签
      * - ``advantage.tag``
        - ``null``
        - 输出优势文件标签，生成 ``meta/advantages_{tag}.parquet``

**输出文件**

- ``meta/advantages_{tag}.parquet``：包含 ``advantage``\ （布尔）、``advantage_continuous``\ （浮点）等列
- 更新 ``mixture_config.yaml``：记录全局阈值和归一化统计

**验证方法**

.. code-block:: bash

   python3 -c "
   import pandas as pd
   df = pd.read_parquet('./datasets/rollout/meta/advantages_fail300_N10_ckpt18000_q30.parquet')
   print(f'samples={len(df)}, columns={list(df.columns)}')
   print(df[['advantage_continuous']].describe())
   "


Step 4：CFG Training
----------------------------------------

使用 Step 3 计算的优势标签，对 OpenPI 策略模型进行 classifier-free guidance 训练。

**训练机制**

- **正样本**\ （``advantage=True``\ ）：作为条件输入传入模型
- **负样本**\ （``advantage=False``\ ）：作为无条件输入传入模型
- 当 ``positive_only_conditional`` 开启时，正样本以 ``unconditional_prob`` 概率随机 dropout 为无条件，实现正则化
- 推理时通过引导尺度 ``cfgrl_guidance_scale`` 放大正负样本之间的差异，引导模型生成高优势动作

**配置文件**

在 ``examples/offline_rl/config/cfg_rl_openpi.yaml`` 中填写 ``data.advantage_tag``、数据集路径和 ``actor.model.model_path``：

.. code-block:: yaml

   data:
     advantage_tag: "fail300_N10_ckpt18000_q30"   # 对应 Step 3 的 advantage.tag
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
         weight: 1.0
       - dataset_path: /path/to/rollout_dataset
         type: rollout
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch

**启动命令**

.. code-block:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置 CFG 条件参数、batch size 与优化器默认值：

   .. code-block:: yaml

      data:
        balance_dataset_weights: true

      actor:
        micro_batch_size: 32
        global_batch_size: 512
        model:
          openpi:
            config_name: "pi05_libero"
            positive_only_conditional: true
            unconditional_prob: 0.1
            cfgrl_guidance_scale: 1.0
        optim:
          lr: 1.0e-5
          lr_scheduler: cosine
          lr_warmup_steps: 5000
          total_training_steps: 30000

   .. list-table::
      :header-rows: 1
      :widths: 40 15 45

      * - 参数
        - 默认值
        - 说明
      * - ``data.advantage_tag``
        - ``null``
        - 对应 Step 3 的 ``advantage.tag``，用于读取优势标签
      * - ``actor.model.openpi.positive_only_conditional``
        - ``true``
        - 仅正样本做条件输入。``false`` 时正负样本均做条件输入
      * - ``actor.model.openpi.unconditional_prob``
        - ``0.1``
        - 样本 dropout 为无条件的概率。``positive_only_conditional`` 开启时仅作用于正样本，否则作用于所有样本
      * - ``actor.model.openpi.cfgrl_guidance_scale``
        - ``1.0``
        - 推理时的引导尺度。越大越偏向高优势动作
      * - ``actor.model.openpi.config_name``
        - ``pi05_libero``
        - 数据变换配置。LIBERO 用 ``pi05_libero``，Franka 用 ``pi05_maniskill_sim_real_co_training``
      * - ``data.balance_dataset_weights``
        - ``true``
        - 按数据集大小平衡采样权重

**关键监控指标**

- ``train/actor/loss``：策略训练损失
- ``train/actor/grad_norm``：梯度范数

.. dropdown:: 可视化优势分布（可选）

   Step 3 完成后，运行 ``visualize_advantage_dataset.py`` 生成综合分布面板（``advantage_distribution.png``）、逐 episode 汇总图（``episode_{N}_summary.png``）和带优势标注的回放视频（``episode_{N}.mp4``）。传入 ``--no-video`` 仅输出静态图，传入 ``--episodes 0 5 10`` 指定具体 episode：

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/recap/process/visualize_advantage_dataset.py \
          --dataset ./datasets/rollout \
          --output ./outputs/advantage_viz \
          --tag "fail300_N10_ckpt18000_q30" \
          --num-episodes 10

.. dropdown:: 进阶用法（阈值重标与迭代优化）

   **优势阈值重标**

   无需重跑 GPU 推理，在 CPU 上直接复用 ``advantage_continuous`` 调整分位数阈值；传入 ``--dataset_root`` 与 ``--advantage_lookahead_step`` 可对目录下多个数据集重算优势：

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/recap/process/relabel_advantages.py \
          --dataset_paths ./datasets/sft ./datasets/rollout \
          --source_tag "fail300_N10_ckpt18000_q30" \
          --new_tag "fail300_N10_ckpt18000_q20" \
          --positive_quantile 0.2

   **迭代优化**

   用 Step 4 训练出的 policy 采集新数据后，更换 tag 从 Step 1 重新迭代：

   .. code-block:: text

      Iter 1: tag="fail300"          → 训练 Value Model → tag="fail300_N10_ckpt18000_q30"     → CFG Training
      Iter 2: tag="fail300_iter2"    → 训练 Value Model → tag="fail300_iter2_N10_ckpt6000_q20" → CFG Training

可视化与结果
----------------------------------------

使用 TensorBoard（``tensorboard --logdir ./logs --port 6006``）查看 Step 2（``logs/value_sft/``）和 Step 4（``logs/cfg_sft/``）的训练日志。指标含义和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

我们在 `LIBERO-10 <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ Task 0 上使用 `RECAP-Libero10-Task0-48succ-Data <https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data/tree/main>`_ 数据集（包含专家 SFT 示范、few-shot π\ :sub:`0.5` policy 采集的 4,096 条混合 rollout 轨迹与留出验证集）复现了 RECAP 流程。执行一轮 RECAP 迭代后，成功率从 **48.8%**\ （该紧凑实验设置下的 few-shot SFT 基线）提升至 **66.5%**\ （+17.7%）。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap_libero10_task0.png
   :align: center
   :width: 55%

   LIBERO-10 Task 0 上的 RECAP 实验结果。

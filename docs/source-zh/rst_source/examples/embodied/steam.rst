STEAM
=====

在 RLinf 中运行 **STEAM** 流程。STEAM 是一种离线策略优化方法：用\ **成对分类的进度评论器（progress critic）**\ 配合\ **深度集成（deep ensemble）**\ 为已有数据打分，将保守的 worst-of-N 集成估计转化为逐帧的优势标签，再用这些标签驱动与 :doc:`RECAP <recap>` 相同的 **无分类器引导（Classifier-Free Guidance, CFG）训练**。

与 RECAP 一样，STEAM 无需在线环境交互，适合难以大规模在线采样的真实机器人场景。区别在于价值信号：STEAM 不回归折扣回报，而是从帧对中学习\ **时间进度（temporal progress）**\ 评论器，并通过集成抑制单一预测器在分布外 rollout 上对优势的高估。

概览
----------------------------------------

离线提升策略（无需新采样）：用集成进度评论器为已有数据打分，再以无分类器引导（CFG）进行优化。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      STEAM（worst-of-N 集成）

   .. grid-item-card:: 模型
      :text-align: center

      SigLIP + Gemma3 评论器 · π₀.₅

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      LeRobot 数据集

   .. grid-item-card:: 训练
      :text-align: center

      离线 · 3 阶段

| **你将完成：** SFT 一个集成进度评论器 → 计算集成优势 → CFG 训练策略 → 评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SigLIP + Gemma3 + π₀.₅ 检查点 · LeRobot 格式数据集（见下文步骤）。

STEAM 工作原理
----------------------------------------

STEAM（Self-supervised Temporal Ensemble Advantage Modeling，自监督时序集成优势建模）全程离线运行，包含三个阶段：

1. **价值模型 SFT（进度评论器集成）**：在专家演示帧对上训练一组进度评论器（SigLIP 视觉编码器 + Gemma3 语言模型 + 分类头）。每个成员预测按轨迹长度归一化的有符号帧步幅（归入 ``num_bins`` 个 bin，反向帧对作为负向进度样本），无需奖励函数或人工标注即可度量时间效率。
2. **计算集成优势（worst-of-N）**：用全部集成成员为每一帧打分，取最小值（worst-of-N）抑制分布外 rollout 状态上的高估。得到的连续分数 ``advantage_continuous``\ （范围 ``[-1, 1]``）按数据源通过 ``label_mode: threshold``\ （rollout 帧高于 ``positive_threshold`` 为正，SFT 帧恒为 ``True``）或 ``label_mode: quantile``\ （取 rollout 帧前 ``rollout_quantile`` 与可选 SFT 帧前 ``expert_quantile``）转为布尔 ``advantage`` 标签。
3. **CFG 训练**：将优势标签送入 OpenPI π₀.₅ 的无分类器引导（CFG）训练阶段，正样本作为条件输入、负样本作为无条件输入。完整 CFG 参数（``positive_only_conditional``、``unconditional_prob``、``cfgrl_guidance_scale``）见 :doc:`RECAP <recap>`。

安装
----------------------------------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

下载模型
----------------------------------------

下载 STEAM 价值模型所需的两个预训练 backbone（SigLIP-so400m 视觉编码器与 Gemma3-270M 语言模型）：

.. tab-set::

   .. tab-item:: 价值模型 Backbone

      .. code-block:: bash

         hf download google/siglip-so400m-patch14-384 --local-dir ./siglip-so400m-patch14-384
         hf download google/gemma-3-270m --local-dir ./gemma-3-270m

在模型配置（``examples/offline_rl/config/model/steam_value_model.yaml``\ ）中设置路径：

.. code-block:: yaml

   actor:
     model:
       vision_repo_id: ./siglip-so400m-patch14-384
       language_repo_id: ./gemma-3-270m
       tokenizer_path: ./gemma-3-270m

数据准备
----------------------------------------

STEAM 使用 LeRobot 格式数据集，分为两类：

- **SFT 数据集**：专家级演示（成功的专家轨迹）。
- **Rollout 数据集**：在线交互采集的轨迹（同时包含成功与失败），外加人工介入（human intervention）数据。

示例数据配置：

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

.. note::

   Step 1 与 Step 2 的 ``train_data_paths``、``data.k`` 和 ``data.camera_keys`` 必须保持一致：优势计算必须使用评论器训练时相同的数据集、时间步幅和相机视角。

流程 Tag 系统
~~~~~~~~~~~~~~~~~~~~~

STEAM 用 **advantage tag** 在各步骤间传递数据。与 RECAP 不同，STEAM 没有计算回报（compute returns）这一步，因此没有 ``returns_tag``——唯一的 tag 就是 **advantage_tag**：由 Step 2 写出、由 Step 3 读取。请确保 Step 2 的 ``advantage.tag`` 与 Step 3 的 ``data.advantage_tag`` 一致，CFG 即可读取 ``meta/advantages_{tag}.parquet``。

.. list-table:: **各步骤的 Tag 流转**
   :header-rows: 1

   * - 步骤
     - 配置字段
     - 说明
   * - 2
     - ``advantage.tag``
     - 写入 ``meta/advantages_{tag}.parquet``
   * - 3
     - ``data.advantage_tag``
     - 读取 ``meta/advantages_{tag}.parquet``

Step 1：价值模型 SFT
----------------------------------------

使用 ``examples/offline_rl/config/steam_value_model_sft.yaml`` 训练集成进度评论器。配置好 ``data.train_data_paths``\ （相机名不同时同步调整 ``data.camera_keys``）后直接启动：

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft

.. dropdown:: 进阶配置（可选）

   默认配置已设好常用参数（``data.k=32``、``actor.model.num_bins=32``、``actor.model.ensemble_size=3``、``runner.max_steps=16000``），需要调参时可在 YAML 或命令行覆盖（如 ``data.k=8``）：

   .. code-block:: yaml

      runner:
        max_epochs: -1
        max_steps: 16000
        save_interval: 1000

      data:
        train_data_paths:
          - dataset_path: /path/to/sft_dataset
            type: sft
        k: 32                       # 最大有符号步幅 K；多 bin 模式下 2*K 须为 num_bins 整数倍
        camera_keys: [face_view, left_wrist_view, right_wrist_view]

      actor:
        micro_batch_size: 32
        global_batch_size: 512
        model:
          precision: fp32           # fp32 master 参数，FSDP forward 使用 bf16
          num_bins: 32              # 2 = 二分类进度；>2 = 多 bin（偶数）
          ensemble_size: 3          # >1 启用 worst-of-N 聚合与不确定性统计
          fusion_hidden_dim: 512
          freeze_vision_encoder: false
          freeze_language_model: false
          use_gradient_checkpointing: true
        optim:
          lr: 5.0e-5
          value_lr: 5.0e-5

   如需在留出集上筛选检查点，设置 ``data.eval_data_paths`` 与正整数 ``runner.val_check_interval``，按 ``eval/loss`` 和 ``eval/accuracy`` 择优。

检查点保存在 ``logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor``。

Step 2：计算集成优势
----------------------------------------

在 ``examples/offline_rl/config/steam_compute_advantages_ensemble.yaml`` 中将 ``advantage.value_checkpoint`` 指向 Step 1 的 ``actor`` 检查点目录（并保持 ``data.train_data_paths`` 与 ``data.camera_keys`` 一致），然后启动：

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble

.. dropdown:: 进阶配置（可选）

   默认使用 ``label_mode: quantile``\ （``rollout_quantile: 0.3``、``expert_quantile: 0.8``、``tag: steam_k32_ensemble3_q30``）。可追加 ``--nproc 4`` 指定 GPU 数，或切换为 ``label_mode: threshold`` 并设置 ``advantage.positive_threshold``\ （范围 ``[-1, 1]``）：

   .. code-block:: yaml

      advantage:
        value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
        batch_size: 256
        label_mode: quantile        # "threshold" 或 "quantile"
        rollout_quantile: 0.3       # rollout 帧最高的 30% 标为 True
        expert_quantile: 0.8        # 可选：sft 帧最高的 80% 标为 True
        tag: steam_k32_ensemble3_q30

      data:
        k: 32                       # 必须与 Step 1 的 data.k 一致
        camera_keys: [face_view, left_wrist_view, right_wrist_view]

运行后写出 ``meta/advantages_{tag}.parquet`` 与 ``meta/mixture_config.yaml``。

Step 3：CFG 训练
----------------------------------------

在 ``examples/offline_rl/config/cfg_rl_openpi.yaml`` 中配置 ``data.train_data_paths`` 与 ``actor.model.model_path``\ （确认 ``openpi.config_name`` 与 ``norm_stats.json`` 匹配），再传入 Step 2 的 tag 启动：

.. code-block:: yaml

   data:
     advantage_tag: steam_k32_ensemble3_q30
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

.. code-block:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi \
       data.advantage_tag=steam_k32_ensemble3_q30

使用 :doc:`RLT <rlt>` 中的 ManiSkill 关节控制数据时，设置 ``actor.model.openpi.config_name: pi05_rlt_maniskill_joint`` 与 ``actor.model.action_dim: 8``。完整 CFG 参数见 :doc:`RECAP <recap>`，策略评测见 :doc:`评测 </rst_source/evaluations/index>`。

.. dropdown:: 进阶用法（合并、重标注与可视化）

   **合并集成 checkpoint**

   将独立训练的单模型（或用 ``PATH:idx`` 从已有集成中抽取第 ``idx`` 个成员）合并为一个集成推理 checkpoint：

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/steam/process/merge_steam_ensemble.py \
          --member ./logs/seed1/checkpoints/global_step_5000/actor \
          --member ./logs/seed2/checkpoints/global_step_5000/actor \
          --member ./logs/ensemble/checkpoints/global_step_6000/actor:2 \
          --output ./merged_steam_actor

   **阈值 / 分位数重标注**

   无需重跑 GPU 推理，直接在 CPU 上复用 ``advantage_continuous`` 更改标注规则：

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py \
          --dataset_paths ./datasets/sft ./datasets/rollout \
          --source_tag steam_k32_ensemble3_q30 \
          --new_tag steam_k32_ensemble3_q20 \
          --mode quantile --rollout_quantile 0.2

   **可视化优势**

   从优势 parquet 渲染分布、逐成员、不确定性与 episode 时间线诊断图：

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/steam/process/visualize_advantage.py \
          --dataset ./datasets/rollout \
          --tag steam_k32_ensemble3_q30 \
          --output ./outputs/steam_viz

可视化与结果
----------------------------------------

使用 ``tensorboard --logdir ./logs --port 6006`` 查看日志，完整指标定义见 :doc:`训练指标 <../../reference/metrics>`。

我们在四个真机操作任务上将 STEAM 与行为克隆（**BC**）、**HG-DAgger** 以及 :doc:`RECAP <recap>` 对比（↑ 表示相对 BC 基线的绝对提升）：

.. list-table:: 成功率（%）——越高越好
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - 任务
     - BC
     - HG-DAgger
     - RECAP
     - STEAM
   * - Towel Folding
     - 33.3
     - 40
     - 55.6
     - **92.3** (↑59)
   * - Chips Checkout
     - 39.5
     - 53.3
     - 53.3
     - **93.8** (↑54.3)
   * - Pick-and-Place
     - 63.8
     - —
     - 53.8
     - **80** (↑16.2)
   * - Cola Restocking
     - 52
     - —
     - 52.9
     - **75** (↑23)

.. list-table:: 吞吐量（每小时成功 episode 数）——越高越好
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - 任务
     - BC
     - HG-DAgger
     - RECAP
     - STEAM
   * - Towel Folding
     - 42
     - 48
     - 39
     - **58**
   * - Chips Checkout
     - 16.3
     - 22.0
     - 23.9
     - **47.5**
   * - Pick-and-Place
     - 230
     - —
     - 161
     - **254**
   * - Cola Restocking
     - 71
     - —
     - 46
     - **90**

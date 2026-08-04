STEAM：用于离线策略优化的集成优势建模
========================================================

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

流程
----------------------------------------

一次 STEAM 运行包含两个 STEAM 特有阶段，外加一个 CFG 训练阶段：

.. code-block:: text

   ┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
   │  Step 1                │     │  Step 2                │     │  Step 3              │
   │  STEAM Value Model SFT │────▶│  Compute Ensemble      │────▶│  CFG Training        │
   │                        │     │  Advantages            │     │                      │
   │  Train an ensemble of  │     │  Worst-of-N ensemble   │     │  Train the policy    │
   │  pair-classification   │     │  signed score -> bool  │     │  with classifier-    │
   │  progress critics      │     │  advantage labels      │     │  free guidance       │
   └────────────────────────┘     └────────────────────────┘     └──────────────────────┘

**核心思路**

1. **Value Model SFT**：训练一组进度评论器（SigLIP + Gemma3 backbone + 分类头）。每个成员接收帧对 :math:`(o_t, o_{t+k})`，将有符号的帧步幅分类到若干 bin，因此预测的是\ **时间进度**\ 而非回归回报。

2. **Compute Ensemble Advantages**：对每一帧，让所有集成成员在帧对 :math:`(o_t, o_{t+k})` 上推理，并以 **worst-of-N** 规则（:math:`A = \min_m A_m`）聚合，得到有符号分数 ``advantage_continuous`` :math:`\in [-1, 1]`，再按阈值或分位数规则将帧标记为正/负。

3. **CFG Training**：将优势标签交给 CFG 阶段——正样本（高优势）作为条件输入，负样本作为无条件输入，实现 classifier-free guidance 策略优化。

STEAM 工作原理
----------------------------------------

**STEAM 核心组件**

1. **优势建模（advantage modeling）**

   STEAM（Self-supervised Temporal Ensemble Advantage Modeling，自监督时序集成优势建模）仅从专家演示的\ **时序顺序**\ 中学习优势——无需奖励、人工标注或外部价值模型。对专家轨迹中的帧对 :math:`(f_i, f_j)`\ ，**时序偏移（temporal offset）** 就是有符号帧步幅 :math:`j - i`：把一帧与未来帧配对监督前向进度，而把帧对反向输入则给出负偏移，从而仅用成功演示就暴露回退行为。偏移还会\ **按轨迹长度归一化**\ （:math:`\propto L_{\max}/L_\tau`\ ），使目标衡量的是\ **时序效率（temporal efficiency）**\ 而非原始步数——更短、更高效的执行得分更高，更慢或次优的执行得分更低。

   每个预测器（SigLIP 视觉编码器 + Gemma3 语言模型 + 任务相关的预测头）将帧对与语言指令映射为 :math:`N`\ （``num_bins``\ ）个时序偏移 bin 上的类别分布，并以交叉熵损失对分箱后的偏移目标进行训练。逐成员的优势会从预测的期望 bin 中\ **减去一个固定的基线（baseline）偏移**\ ，因此衡量的是相对“期望进度速度”的进展：

   .. math::

      A_m = \frac{2}{N}\left( E_{b \sim p_{\theta_m}}[b] - b_{\mathrm{ref}} \right) \in [-1, 1]

   其中 :math:`E_{b}[b]` 是预测器 :math:`m` 分布的期望 bin 索引，:math:`b_{\mathrm{ref}}` 是确定性的参考基线——即在最长 episode 上、固定前瞻 :math:`H` 对应的、经长度归一化的真实偏移。:math:`A_m` 在高效进展处高、在停滞或回退处低（甚至为负）。（``num_bins == 2`` 退化为二分类进度判别器。）

2. **优势估计（advantage estimation）**

   单个预测器在分布外 rollout 状态上可能高估。成员在分布内一致、但在陌生状态上发散，因此 STEAM 用保守的 **worst-of-N** 规则聚合 :math:`M` 个预测器——通过惩罚高方差来抑制假阳性：

   .. math::

      A_{\text{STEAM}} = \min_{m \in \{1, \dots, M\}} A_m

   :math:`A_{\text{STEAM}}` 写入 ``advantage_continuous``\ ；逐成员的均值 / 最小值 / 方差作为诊断量记录下来。由于不同数据源的优势分布不同，``advantage_continuous`` 会按数据源、通过两种 ``label_mode`` 规则之一转化为布尔 ``advantage``：

   - ``threshold``：对 rollout 帧 ``advantage = advantage_continuous > positive_threshold``\ （:math:`[-1, 1]` 内的有符号分数阈值）；sft 帧恒为 True（按构造是成功演示）。
   - ``quantile``：将 rollout 帧中分数最高的 ``rollout_quantile`` 比例标为 True；当设置了 ``expert_quantile`` 时，再将 sft 帧中最高的 ``expert_quantile`` 比例标为 True——两个池独立打分。

3. **无分类器引导（Classifier-Free Guidance, CFG）训练**

   STEAM 的优势标签驱动 OpenPI（π₀.₅）策略上的 CFG 阶段：正样本（高优势）作为条件输入，负样本作为无条件输入，实现 classifier-free guidance 策略优化。完整的 CFG 机制（``positive_only_conditional``\ 、``unconditional_prob``\ 、``cfgrl_guidance_scale``\ ）见 :doc:`CFG 训练阶段 <recap>`。

安装
----------------------------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 中国大陆用户可使用以下镜像以获得更快下载速度：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~

STEAM 与 RECAP 共用 OpenPI 环境。

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 为提高国内下载速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后，切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：自建环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

下载模型
----------------------------------------

STEAM 价值模型由两个预训练 backbone 构成：

- **SigLIP-so400m**\ （``google/siglip-so400m-patch14-384``\ ）：视觉编码器
- **Gemma3-270M**\ （``google/gemma-3-270m``\ ）：语言模型与分词器

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/google/siglip-so400m-patch14-384
   git clone https://huggingface.co/google/gemma-3-270m

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
   hf download google/gemma-3-270m --local-dir gemma-3-270m

在模型配置（``examples/offline_rl/config/model/steam_value_model.yaml``\ ）中设置路径：

.. code:: yaml

   actor:
     model:
       vision_repo_id: /path/to/siglip-so400m-patch14-384
       language_repo_id: /path/to/gemma-3-270m
       tokenizer_path: /path/to/gemma-3-270m

数据准备
----------------------------------------

STEAM 使用 LeRobot 格式数据集，分为两类：

- **SFT 数据集**：专家级演示（成功的专家轨迹）。
- **Rollout 数据集**：在线交互采集的轨迹（同时包含成功与失败），外加人工介入（human intervention）数据。

示例数据配置：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

.. note::

   Step 1 与 Step 2 的 ``train_data_paths`` 和 ``data.k`` 必须保持一致：优势计算必须以评论器训练时相同的时间步幅对帧对打分。

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

训练集成进度评论器。每个成员是 SigLIP + Gemma3 backbone 加一个分类头；成员从共享 backbone 克隆而来，并对其 value head 重新设种子，使集成方差成为有意义的认知不确定性信号。

**配置**

配置文件位于 ``examples/offline_rl/config/steam_value_model_sft.yaml``；模型默认值在 ``examples/offline_rl/config/model/steam_value_model.yaml``。关键字段：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
     k: 32                       # 最大有符号步幅 K（帧对时间尺度）
     # 评论器逐帧加载的图像（视角）名称；必须与检查点训练时的视角一致。
     # 缺失的视角会被补成零占位（mask=False）。
     camera_keys: [face_view, left_wrist_view, right_wrist_view]

   actor:
     micro_batch_size: 32
     global_batch_size: 512
     model:
       num_bins: 32              # 2 = 二分类进度；>2 = 多 bin（偶数）
       ensemble_size: 3          # 集成中评论器数量
       fusion_hidden_dim: 512
       freeze_vision_encoder: false
       freeze_language_model: false
       use_gradient_checkpointing: true
     optim:
       lr: 5.0e-5
       value_lr: 5.0e-5

**关键参数**

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - 参数
     - 默认值
     - 说明
   * - ``data.k``
     - ``required``
     - 最大有符号步幅 :math:`K`。多 bin 模式下 ``2*K`` 必须是 ``num_bins`` 的整数倍。
   * - ``actor.model.num_bins``
     - ``2``
     - bin 数量。``2`` 为二分类进度；``> 2``\ （偶数）为多 bin 有符号步幅分类。
   * - ``actor.model.ensemble_size``
     - ``1``
     - 集成成员数。``> 1`` 启用 worst-of-N 聚合与不确定性统计。

**启动命令**

.. code:: bash

   bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft

   # 命令行覆盖配置字段：
   bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft data.k=8

**输出**

- 检查点位于 ``logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor``
- TensorBoard 日志

**关键指标**

- ``train/actor/loss``：有符号步幅 bin 上的交叉熵
- ``train/actor/accuracy``：最优 bin 分类准确率
- ``train/actor/grad_norm``：梯度范数

Step 2：计算集成优势
----------------------------------------

用训练好的集成对每一帧推理，并写出逐帧优势标签。

**配置**

配置文件为 ``examples/offline_rl/config/steam_compute_advantages_ensemble.yaml``：

.. code:: yaml

   advantage:
     value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
     batch_size: 256
     label_mode: quantile        # 必填："threshold" 或 "quantile"
     rollout_quantile: 0.3       # rollout 帧最高的 30% 标为 True
     expert_quantile: 0.8        # 可选：sft 帧最高的 80% 标为 True
     tag: steam_k32_ensemble3_q30

   data:
     k: 32                       # 必须与 Step 1 的 data.k 一致
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

**关键参数**

``label_mode`` 决定哪些旋钮生效。在 ``threshold`` 模式下，只有 ``advantage.positive_threshold`` 起作用——它是 :math:`[-1, 1]` 内的有符号分数阈值；rollout 帧分数高于它即为正样本，sft 帧恒为正。在 ``quantile`` 模式下，``positive_threshold`` 被忽略，改由 ``rollout_quantile`` / ``expert_quantile`` 两个比例分别在各自池中独立选取分数最高的帧（省略 ``expert_quantile`` 则把所有 sft 帧都标为正）。

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - 参数
     - 默认值
     - 说明
   * - ``advantage.value_checkpoint``
     - ``required``
     - Step 1 集成检查点路径（``actor`` 目录）。
   * - ``advantage.label_mode``
     - ``required``
     - ``threshold`` 或 ``quantile``\ （无默认值，必须显式设置）。
   * - ``advantage.positive_threshold``
     - ``null``
     - :math:`[-1, 1]` 内的有符号分数阈值（仅 ``label_mode=threshold``\ ）。
   * - ``advantage.rollout_quantile``
     - ``null``
     - rollout 帧标为 True 的最高比例（``label_mode=quantile``\ ，必填）。
   * - ``advantage.expert_quantile``
     - ``null``
     - sft 帧标为 True 的最高比例（``label_mode=quantile``\ ，可选）。
   * - ``advantage.tag``
     - ``required``
     - 输出 tag；写入 ``meta/advantages_{tag}.parquet``。
   * - ``data.k``
     - ``required``
     - 帧对步幅；必须与 Step 1 训练的 ``data.k`` 一致。

**启动命令**

.. code:: bash

   # 自动检测 GPU 数；单卡与 torchrun 多卡均支持。
   bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble

   # 指定 GPU 数：
   bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble --nproc 4

**输出文件**

- ``meta/advantages_{tag}.parquet``：逐帧的 ``advantage``\ （布尔）、``advantage_continuous``\ （有符号分数）、``ensemble_signed_score``\ 、逐成员值，以及集成熵 / 方差等诊断量。
- ``meta/mixture_config.yaml``：每个 tag 一条记录，记录 ``label_mode``\ 、所用阈值、``ensemble_size``\ 、``num_bins`` 和正样本计数。

Step 3：CFG 训练
----------------------------------------

策略优化直接在 STEAM 优势 parquet 上运行共享的 CFG 阶段。将 CFG 配置的 ``data.advantage_tag`` 指向 Step 2 的 ``advantage.tag`` 并启动：

.. code:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi \
       data.advantage_tag=steam_k32_ensemble3_q30

完整的 CFG 配置与参数见 :doc:`CFG 训练阶段 <recap>`。

STEAM 实验结果
----------------------------------------

我们在四个真实机器人操作任务上，将 STEAM 与行为克隆（**BC**）、**HG-DAgger** 以及 :doc:`RECAP <recap>` 进行对比。STEAM 在每个任务上都显著提升了任务成功率（相对 BC 基线的绝对提升以 ↑ 标注）：

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

在四个任务上，STEAM 将成功率提升至 75–93.8%，并取得最高吞吐量，其中 Towel Folding（↑59）与 Chips Checkout（↑54.3）的成功率提升最大。（↑ 表示相对 BC 基线的绝对提升。）

进阶用法
----------------------------------------

合并集成检查点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为独立单模型训练（或从已有集成中抽取）的成员，可以融合为一个集成推理检查点。每个 ``--member`` 是一个检查点路径，或用 ``PATH:idx`` 从集成中抽取第 ``idx`` 个成员：

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/merge_steam_ensemble.py \
       --member /path/to/seed1/checkpoints/global_step_5000/actor \
       --member /path/to/seed2/checkpoints/global_step_5000/actor \
       --member /path/to/ensemble/checkpoints/global_step_6000/actor:2 \
       --output /path/to/merged/actor

合并逻辑位于
``rlinf.models.embodiment.value_model.steam.checkpoint_merge.merge_ensemble_checkpoints``。

阈值 / 分位数重标注
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若要在不重跑 GPU 推理的情况下改变标注阈值，可对已有优势 parquet 重标注（纯 CPU——复用 ``advantage_continuous``\ ）：

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py \
       --dataset_paths /path/to/sft_ds /path/to/rollout_ds \
       --source_tag steam_k32_ensemble3_q30 \
       --new_tag steam_k32_ensemble3_q20 \
       --mode quantile --rollout_quantile 0.2

重标注逻辑位于 ``examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py``。

可视化优势
~~~~~~~~~~~~~~~~~~~~~~

从优势 parquet 渲染分布、逐成员、不确定性、逐 episode 以及 episode 时间线等诊断图：

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/visualize_advantage.py \
       --dataset /path/to/dataset \
       --tag steam_k32_ensemble3_q30 \
       --output outputs/steam_viz

可视化与结果
----------------------------------------

指标定义见 :doc:`训练指标 <../../reference/metrics>`。

.. code:: bash

   tensorboard --logdir ./logs --port 6006

文件结构
----------------------------------------

STEAM 将各步骤脚本自包含在 ``examples/`` 下（绑定模型的推理 + 标注逻辑），模型 / 数据集代码在 ``rlinf/models``、``rlinf/data/datasets`` 下，与 RECAP 共享的模型无关后处理在 ``rlinf/data/process/`` 下：

.. code-block:: text

   examples/offline_rl/
   ├── config/                                  # 共享生产配置
   │   ├── steam_value_model_sft.yaml           # Step 1
   │   ├── steam_compute_advantages_ensemble.yaml   # Step 2
   │   ├── cfg_rl_openpi.yaml                   # Step 3（CFG，与 RECAP 共用）
   │   └── model/
   │       └── steam_value_model.yaml           # 价值模型架构默认配置
   ├── advantage_labeling/
   │   └── steam/
   │       ├── train_steam.py                   # Step 1：价值模型 SFT 入口
   │       ├── run_steam_sft.sh                 # Step 1 启动脚本
   │       └── process/                         # Step 2：自包含入口脚本（与 recap 一致）
   │           ├── compute_advantages_ensemble.py     # Step 2：集成推理 + 标注（Hydra）
   │           ├── relabel_advantages.py              # CLI：重标注优势（CPU）
   │           ├── merge_steam_ensemble.py            # CLI：合并集成检查点
   │           ├── visualize_advantage.py             # 优势可视化
   │           └── run_compute_advantages_ensemble.sh # Step 2 启动脚本
   └── policy_optimization/
       └── cfg_rl/
           ├── train_cfg.py                      # Step 3：CFG 策略训练
           └── run_cfg_rl.sh                     # Step 3 启动脚本

   rlinf/
   ├── models/embodiment/value_model/steam/     # 评论器、集成、配置、合并
   │   ├── modeling_steam.py / modeling_critic.py
   │   ├── ensemble_modeling_critic.py          # worst-of-N + coerce_to_ensemble
   │   └── checkpoint_merge.py
   ├── data/datasets/steam/                     # pair_dataset.py、mixture.py、binning.py
   └── data/process/                            # 共享、模型无关（RECAP + STEAM）
       ├── advantage.py                         # 分位数阈值 + 布尔标签
       ├── distributed.py                       # 分片推理辅助
       └── mixture_config.py                    # meta/mixture_config.yaml tag I/O

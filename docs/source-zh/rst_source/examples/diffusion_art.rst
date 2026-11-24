ART 扩散模型微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   使用 DiffusionART 训练的 SD3 与 Wan2.2 生成模型。

使用 DiffusionART 微调 SD3 或 Wan2.2，通过 OCR 奖励改善文字渲染。ART 将同一 prompt 下各候选的奖励与干净 latent 终点聚合为共享方向，再向生成过程中所经状态上的冻结速度目标训练。完成 :doc:`共用准备步骤 <video_models>` 后，选择图像或视频配置，并按下文启动训练、检查结果。

机器人 Pi0 策略请看 :ref:`LIBERO Object 与 Spatial 配方 <diffusion-art-libero>`。该路径用 anchor 残差训练完整轨迹，按 reset 组与决策序号或近邻状态分组。

概览
----

两个配置均使用 LoRA、FSDP2 和确定性 ODE 生成。每个 prompt 的候选分成两个不重叠的更新窗口，共用第一次更新前冻结的目标。若预测已超出偏移阈值，且更新会让偏移继续增大，修正 mask 就会屏蔽该更新。算法推导与参考实现见 `DiffusionART 论文 <https://arxiv.org/abs/2610.12572>`_、`官方代码 <https://github.com/vanzll/DiffusionART>`_ 和 `项目页 <https://vanzll.github.io/diffusionart-project/>`_。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 算法
      :text-align: center

      DiffusionART

   .. grid-item-card:: 输出
      :text-align: center

      图像 / 视频

   .. grid-item-card:: 奖励
      :text-align: center

      OCR / 视频 OCR

| **操作流程：** 选择配置 → 启动训练 → 检查 OCR reward 和生成样本。
| **前置条件：** :doc:`共用安装、模型与数据集准备 <video_models>`。记录中的 Wan 实验使用四张 NVIDIA A800 GPU。

任务
~~~~

按需要生成的内容选择配置。

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - 模型
     - 配置
     - 任务
   * - SD3.5 Medium
     - ``sd3_art_ocr``
     - 图像 OCR
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_art_video_ocr``
     - 文生视频 OCR

观测与动作
~~~~~~~~~~

环境对每个完整候选打分；视频奖励先取帧均值，再用于构造 prompt 组的 ART 引导。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 数据
     - 含义
   * - 观测
     - 含有引号内目标文字的文本 prompt。
   * - 动作
     - 生成的图像或视频。
   * - 奖励
     - 每个候选一个 OCR 相似度标量，范围为 ``[0, 1]``。

安装与资源准备
--------------

按共用步骤完成 :ref:`安装 <diffusion-model-setup>`、:ref:`下载模型 <diffusion-model-checkpoints>` 和 :ref:`下载 OCR 数据集 <diffusion-ocr-dataset>`。两个 ART 配置均设置 ``env.*.reward.use_gpu: False``，使用 CPU OCR 安装即可。

运行训练
--------

启动前，打开 ``examples/diffusion/config/sd3_art_ocr.yaml`` 或 ``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml``。在所选 YAML 中，将 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path`` 设为下载目录，并根据实际硬件和下文的 ART 约束检查 ``cluster.num_nodes``、``cluster.component_placement`` 及 batch 设置。

检查 YAML 后，在仓库根目录传入对应配置名启动训练：

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

启动脚本生成候选、计算奖励并更新 LoRA 参数，日志写入 ``logs/<timestamp>-<config>/``。

配置 ART 更新
-------------

调整 rollout 数量或 placement 时，确保每个 actor shard 都收到完整 prompt 组。``algorithm.art_updates_per_rollout`` 只能为 ``1`` 或 ``2``，且必须整除 ``algorithm.group_size``；``env.train.group_size`` 也需与该组大小一致。

.. warning::

   ``actor.global_batch_size`` 应等于 ``env.train.total_num_envs * env.train.rollout_epoch`` 除以 ``algorithm.art_updates_per_rollout``。例如，96 个环境、12 个 rollout epoch、两次更新，对应 ``actor.global_batch_size=576``。每个 rank 的更新窗口还需能被 ``actor.micro_batch_size`` 整除。环境数量应保证分到 env 和 actor 各 rank 后，prompt 组仍然完整。

ART 要求 ``actor.training_backend=fsdp``、``rollout.pipeline_stage_num=1``、``algorithm.update_epoch=1``、``runner.weight_sync_interval=1``。保留 ``env.train.max_steps_per_rollout_epoch=actor.model.num_action_chunks``，使每个 rollout epoch 生成一个完整候选。SD3 使用 ``actor.model.sd3.train_sampler=ode`` 和 ``actor.model.sd3.noise_level=0.0``；Wan 记录 ODE 轨迹。

.. list-table:: 主要更新参数
   :header-rows: 1
   :widths: 40 60

   * - 参数
     - 作用
   * - ``algorithm.art_timestep_fraction``
     - 用于训练的去噪状态比例；配置中 SD3 为 ``0.5``，Wan 为 ``0.25``。
   * - ``algorithm.art_guidance_gain``
     - 调整奖励引导幅度；两个配置均为 ``1.0``。
   * - ``algorithm.art_corrective_mask_enabled``
     - 启用偏移修正 mask；两个配置均为 ``True``。
   * - ``algorithm.art_mask_threshold``
     - mask 的偏移阈值；两个配置均为 ``1e-5``，配合 ``art_mask_metric=sde_kl`` 使用。

.. _diffusion-art-results:

预期实验结果
------------

Wan 关注 ``env/video_ocr``，SD3 关注 ``env/avg``，同时检查生成样本。``actor/active_group_frac`` 表示提供有效引导的 prompt 组比例；``actor/skipped_zero_guidance`` 标记所有 rank 都没有可用引导的轮次。日志与 TensorBoard 设置见 :doc:`训练指标 </rst_source/reference/metrics>`。OCR reward 不衡量视觉质量或时序一致性，ART 与 NFT 的 loss 也不能直接比较。

2026 年 11 月 11 日启动的这批实验使用四张 NVIDIA A800 80 GB GPU、Wan2.2 TI2V 5B、LoRA（rank 为 ``32``，alpha 为 ``64``）、FSDP2、BF16 和 CPU OCR。第一张表统一比较生成 **40,320 条轨迹** 时的结果，对应 ART 第 35 轮、NFT 第 70 轮。每次运行先取该预算下最后 5760 条轨迹的 ``env/video_ocr`` 均值；NFT 再报告两次运行的均值和样本标准差，ART 只有一次运行。

.. list-table:: 生成 40,320 条轨迹时的 Wan 视频 OCR 结果
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - 方法
     - 运行次数
     - 每轮轨迹数
     - 学习率
     - 训练 Reward
     - 耗时（h）
   * - ART
     - 1
     - 1152
     - 1e-4
     - 0.8584
     - 10.01
   * - NFT
     - 2
     - 576
     - 4e-4
     - 0.7565 ± 0.0544
     - 11.29 ± 0.52

在相同采样预算下，ART 的 reward 比 NFT 均值高 13.5%（绝对增加 ``0.1019``），训练耗时减少 11.3%。耗时取自训练循环日志，不含模型初始化。按同样的 5760 条轨迹窗口计算，ART 在生成 21,888 条轨迹时首次达到 ``0.8``，耗时 5.48 h；两次 NFT 运行在 40,320 条轨迹内均未达到这一窗口均值阈值。

与 ART 使用相同每轮轨迹数和学习率的 NFT 运行在第 29 轮后停止。因此，下表统一比较生成 **33,408 条轨迹** 时的结果，仍取最后 5760 条轨迹的 reward 均值，两者各运行一次。

.. list-table:: 每轮轨迹数和学习率对齐的结果
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - 方法
     - 每轮轨迹数
     - 学习率
     - 训练 Reward
     - 耗时（h）
   * - ART
     - 1152
     - 1e-4
     - 0.8446
     - 8.31
   * - NFT
     - 1152
     - 1e-4
     - 0.1076
     - 8.23

每轮轨迹数和学习率对齐后，ART 的 reward 高出 ``0.7370``，相对提高 685.2%，耗时基本相当。这里比较的是前 29 轮的表现，不代表最终收敛结果。这些是训练奖励，不是独立测试集分数；第一张表中的每轮 576 条轨迹 NFT 配置还改变了学习率和奖励归约方式，不能据此单独归因于算法。

复现时，ART 选择 ``wan22_ti2v_5b_art_video_ocr``，NFT 选择 ``wan22_ti2v_5b_nft_video_ocr``，模型和数据路径按上文准备。两者均设置 ``cluster.component_placement={"actor,env,rollout": "0,1,2,3"}``、``env.train.total_num_envs=96``、``algorithm.group_size=8`` 和 ``runner.val_check_interval=-1``。将 ``env.train.reward.use_gpu`` 与 ``env.eval.reward.use_gpu`` 设为 ``False``。在 ``actor.model.wan22_ti2v_5b`` 下设置 ``max_generation_batch_size=2``、``compile_transformer_forward=True`` 和 ``offload_auxiliary_modules=True``；保留配置中的 512 × 512 分辨率、5 帧、10 个去噪步骤和 ``1.0`` 的 guidance scale。

ART 使用 ``env.train.rollout_epoch=12``、``actor.global_batch_size=576``、``actor.micro_batch_size=2``、``actor.optim.lr=1e-4``、``runner.save_interval=10``、``runner.max_epochs=70`` 和 ``runner.max_steps=70``。每轮 576 条轨迹的 NFT 使用 ``env.train.rollout_epoch=6``、``actor.global_batch_size=288``、``actor.micro_batch_size=1``、``actor.optim.lr=4e-4``、``runner.save_interval=-1``、``runner.max_epochs=70`` 和 ``runner.max_steps=70``。两者的 ``env.*.reward.reward_type`` 分别设为 ``video_level`` 和 ``frame_level``。

每轮 1152 条轨迹的 NFT 选择 NFT 配置，并采用上面 ART 的覆盖参数，包括 ``video_level`` 奖励，但将 ``runner.max_epochs=200`` 和 ``runner.max_steps=200``；该运行在第 29 轮后停止。两组 NFT 均保留 ``algorithm.nft_tau=[1.0,0.01,0,70]``。

这批实验均使用 ``actor.seed=42`` 和环境 seed ``42``，但这些设置没有固定 LoRA 初始化和采样涉及的全部全局随机数状态。比较结果时应重复运行，并记录代码版本、依赖和随机种子。

.. _diffusion-art-libero:

LIBERO Object 与 Spatial
-------------------------

Pi0 在完整 LIBERO 轨迹上训练，支持 natural_anchor 和 cluster 两种分组。两者都先从动作中减去状态相关的 anchor，再计算动作与回报的协方差。Anchor 用共享的正负参考噪声各做一次确定性去噪，取平均来估计该状态的平均动作；实际采样动作不变。

.. list-table:: LIBERO ART 配方
   :header-rows: 1

   * - 任务套件
     - Natural Anchor
     - State Clusters
   * - Object
     - ``libero_object_0_art_openpi``
     - ``libero_object_0_art_cluster_openpi``
   * - Spatial
     - ``libero_spatial_0_art_openpi``
     - ``libero_spatial_0_art_cluster_openpi``

配方默认训练 task 0，使用 128 个训练环境、4 个去噪步、5e-7 学习率和 FP32 master 参数，共训练 50 轮。每条 episode 最多执行 240 个环境步，每次决策执行 5 个动作。两次更新各用 3,072 行决策数据，轨迹互不重叠；终止后的行不参与训练。Guidance、restoration 和 trust-region mask 都只覆盖已执行的动作坐标。

选择分组方式
~~~~~~~~~~~~

**Natural anchor**\ （``algorithm.art_grouping=index``）比较同一 reset 组、同一决策序号的 8 条轨迹。它们从相同的模拟器初始化出发，独立采样动作。后续决策训练时，组内状态已经分叉；减去 anchor 能减小不同状态下平均动作的差异。

**State clusters**\ （``algorithm.art_grouping=cluster``）跨 actor rank 对存活状态聚类。每簇属于同一任务和接触阶段，包含 4–8 个成员，每条轨迹最多贡献一个决策。第一个决策保留精确的 reset 分组，后续未入簇的行不参与训练。特征包括夹爪、物体和目标的相对位置、完整姿态、夹爪开度及剩余时间。``algorithm.art_cluster_radius=1.0`` 表示归一化特征空间中的半径。

运行训练
~~~~~~~~

按 :doc:`../start/installation` 安装 OpenPi/LIBERO 环境，使用 4 张可见 GPU。下载包含归一化统计的 SFT 权重：

.. code-block:: bash

   hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
     --local-dir /path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

在 ``examples/embodiment/config/libero_object_0_art_openpi.yaml`` 中，将 ``actor.model.model_path`` 设置为下载目录。四个配方都继承这个路径，rollout 自动使用 actor 的权重。设置运行环境：

.. code-block:: bash

   export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

用 natural-anchor 分组训练 Object task 0：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_object_0_art_openpi LIBERO

复现 Spatial task 6 的 cluster 结果时，在 ``examples/embodiment/config/libero_spatial_0_art_cluster_openpi.yaml`` 中加入以下覆盖项：

.. code-block:: yaml

   env:
     train:
       task_id_filter: [6]
     eval:
       task_id_filter: [6]

启动选定的配方：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_0_art_cluster_openpi LIBERO

其他单任务需要同时修改这两处过滤器；切换分组方式时，选择表中对应的配方。复现 Spatial 全部 10 个任务的联合训练结果时，将两个过滤器设为 ``null``，并将下列配置合入所选 Spatial 配方已有的 YAML 段落：

.. code-block:: yaml

   runner:
     max_steps: 200
     max_epochs: 200
   env:
     train:
       task_id_filter: null
     eval:
       task_id_filter: null
       max_steps_per_rollout_epoch: 2640

Natural-anchor 使用 ``bash examples/embodiment/run_embodiment.sh libero_spatial_0_art_openpi LIBERO`` 启动，cluster 则选择 ``libero_spatial_0_art_cluster_openpi``。每 10 轮评估一次，日志保存在 ``logs/`` 下。

实验结果
~~~~~~~~

实验使用 4 张 GPU，actor 的随机种子为 42，训练和评估环境的随机种子均为 0。成功率指标为 ``eval/success_once``。每个 checkpoint 的评估轨迹数为：单任务 50 条，Spatial 联合训练 500 条。SFT 列为所选实验初始 policy 在第 0 步的评估结果。

下列单任务实验均训练 50 轮。同一任务、同一分组方式有多次完整实验时，选择第 50 轮成功率最高的一次，并列出这次实验对应的 SFT 基线。

.. list-table:: Natural anchor 单任务成功率
   :header-rows: 1

   * - 任务
     - SFT
     - ART，第 50 轮
     - 提升（百分点）
   * - Object 0
     - 32%
     - 100%
     - +68
   * - Spatial 0
     - 90%
     - 94%
     - +4
   * - Spatial 1
     - 84%
     - 100%
     - +16
   * - Spatial 2
     - 100%
     - 100%
     - +0
   * - Spatial 3
     - 74%
     - 94%
     - +20
   * - Spatial 4
     - 66%
     - 88%
     - +22
   * - Spatial 5
     - 2%
     - 0%
     - -2
   * - Spatial 6
     - 62%
     - 100%
     - +38
   * - Spatial 7
     - 50%
     - 86%
     - +36
   * - Spatial 8
     - 68%
     - 88%
     - +20
   * - Spatial 9
     - 20%
     - 68%
     - +48

.. list-table:: State clusters 单任务成功率
   :header-rows: 1

   * - 任务
     - SFT
     - ART，第 50 轮
     - 提升（百分点）
   * - Spatial 6
     - 56%
     - 96%
     - +40
   * - Spatial 7
     - 50%
     - 78%
     - +28
   * - Spatial 8
     - 68%
     - 86%
     - +18
   * - Spatial 9
     - 18%
     - 38%
     - +20

每次联合训练都用一个 policy 学习 Spatial 的全部 10 个任务，共训练 200 轮。下表列出最终成功率、训练期间最高成功率及其对应轮次。

.. list-table:: Spatial 全部 10 个任务联合训练成功率
   :header-rows: 1

   * - 分组方式
     - SFT
     - ART，第 200 轮
     - 提升（百分点）
     - ART 最高成功率
     - 最高成功率对应轮次
   * - Natural anchor
     - 61.4%
     - 74.0%
     - +12.6
     - 75.0%
     - 190
   * - State clusters
     - 61.4%
     - 72.6%
     - +11.2
     - 77.6%
     - 170

在 TensorBoard 中查看 ``eval/success_once``。聚类的 ``actor/cluster_row_frac`` 表示入簇覆盖率，``actor/cluster_active_frac`` 表示产生非零 guidance 的簇占比。训练指标见 :doc:`../reference/metrics`，独立评估流程见 :doc:`../evaluations/index`。

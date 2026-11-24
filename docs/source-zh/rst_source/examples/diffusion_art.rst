ART 扩散模型微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   使用 DiffusionART 训练的 SD3 与 Wan2.2 生成模型。

使用 DiffusionART 微调 SD3 或 Wan2.2，通过 OCR 奖励改善图像和视频中的文字渲染。完成 :doc:`共用准备步骤 <video_models>` 后，选择对应配置启动训练。

Pi0 机器人 policy 的训练见 :ref:`LIBERO Object 和 Spatial 配方 <diffusion-art-libero>`。

概览
----

两个配置均使用 LoRA、FSDP2 和确定性 ODE 采样，奖励来自生成文字与提示词中目标文字的 OCR 相似度。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      DiffusionART

   .. grid-item-card:: 模型
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      OCR 提示词

   .. grid-item-card:: 训练
      :text-align: center

      LoRA · FSDP2 · BF16

任务
~~~~

选择图像生成或视频生成配置。

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

每个生成结果对应一个 OCR 分数，视频奖励取采样帧的分数均值。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 数据
     - 含义
   * - 观测
     - 文本提示词，目标文字用引号标出。
   * - 动作
     - 生成的图像或视频。
   * - 奖励
     - OCR 相似度分数，范围为 ``[0, 1]``。

ART 如何训练
------------

ART 根据同一提示词下各候选的奖励和最终潜变量计算共享更新方向。训练使用生成轨迹中的去噪状态，两个更新窗口共用第一次更新前冻结的速度目标。

算法推导与参考实现见 `DiffusionART 论文 <https://arxiv.org/abs/2610.12572>`_、`官方代码 <https://github.com/vanzll/DiffusionART>`_ 和 `项目页 <https://vanzll.github.io/diffusionart-project/>`_。

安装
----

按共用步骤完成 :ref:`安装 <diffusion-model-setup>`、:ref:`下载模型 <diffusion-model-checkpoints>` 和 :ref:`下载 OCR 数据集 <diffusion-ocr-dataset>`。两个 ART 配置均使用 CPU OCR。

运行
----

打开 ``examples/diffusion/config/sd3_art_ocr.yaml`` 或 ``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml``，将 ``actor.model.model_path``、``env.train.dataset.path`` 和 ``env.eval.dataset.path`` 设为模型与数据目录。GPU 分配由 ``cluster.component_placement`` 设置。

在仓库根目录运行对应配置：

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

训练日志写入 ``logs/<timestamp>-<config>/``。图像 OCR 分数记录为 ``env/avg``，视频 OCR 分数记录为 ``env/video_ocr``。

配置 ART 更新
~~~~~~~~~~~~~

每轮样本数为 ``env.train.total_num_envs * env.train.rollout_epoch``。默认 ``algorithm.art_updates_per_rollout=2``，每轮分两次更新，``actor.global_batch_size`` 取每轮样本数的一半。例如，96 个环境、12 个 rollout epoch 对应每轮 1152 个样本，全局 batch size 为 576。

主要更新参数如下：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 参数
     - 作用
   * - ``algorithm.art_timestep_fraction``
     - 用于训练的去噪状态比例；SD3 为 ``0.5``，Wan 为 ``0.25``。
   * - ``algorithm.art_guidance_gain``
     - 奖励引导幅度，默认为 ``1.0``。
   * - ``algorithm.art_corrective_mask_enabled``
     - 启用偏移修正 mask，默认为 ``True``。
   * - ``algorithm.art_mask_threshold``
     - 偏移修正阈值，默认为 ``1e-5``，使用 ``sde_kl`` 度量。

.. _diffusion-art-results:

预期实验结果
------------

实验使用 4 张 NVIDIA A800 80 GB GPU、Wan2.2 TI2V 5B、LoRA（rank 为 ``32``，alpha 为 ``64``）、FSDP2、BF16 和 CPU OCR。生成设置为 512 × 512 分辨率、5 帧、10 个去噪步骤，guidance scale 为 ``1.0``。

ART 使用 ``wan22_ti2v_5b_art_video_ocr`` 配置，NFT 使用 ``wan22_ti2v_5b_nft_video_ocr`` 配置。训练奖励取最后 5760 条轨迹的 ``env/video_ocr`` 均值。多次运行的结果用均值 ± 样本标准差表示。

第一组实验中，ART 使用 ``video_level`` 奖励，NFT 使用 ``frame_level`` 奖励。

.. list-table:: 生成 40,320 条轨迹时的 Wan 视频 OCR 结果
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - 方法
     - 运行次数
     - 每轮轨迹数
     - 学习率
     - 训练奖励
     - 训练耗时（h）
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

在这组配置下，生成 40,320 条轨迹后，ART 的训练奖励比 NFT 均值高 **13.5%**，训练耗时少 **11.3%**。耗时取训练循环的计时结果。

第二组实验中，两种方法的每轮轨迹数和学习率相同，均使用 ``video_level`` 奖励，各运行一次。表中比较两次运行的前 29 轮。

.. list-table:: 生成 33,408 条轨迹时的 Wan 视频 OCR 结果
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - 方法
     - 每轮轨迹数
     - 学习率
     - 训练奖励
     - 训练耗时（h）
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

复现实验时，在对应配置中设置模型与数据目录，并使用以下共用设置：

.. code-block:: yaml

   cluster:
     component_placement:
       actor,env,rollout: "0-3"
   algorithm:
     group_size: 8
   actor:
     seed: 42
     model:
       wan22_ti2v_5b:
         max_generation_batch_size: 2
         compile_transformer_forward: True
         offload_auxiliary_modules: True
   env:
     train:
       total_num_envs: 96
       seed: 42
       reward:
         use_gpu: False
     eval:
       seed: 42
       reward:
         use_gpu: False
   runner:
     val_check_interval: -1

各组实验的 batch 和训练步数设置如下。两组 NFT 均使用 ``algorithm.nft_tau=[1.0,0.01,0,70]``。

.. list-table:: 分组训练配置
   :header-rows: 1
   :widths: 46 18 18 18

   * - 参数
     - ART，1152 条/轮
     - NFT，576 条/轮
     - NFT，1152 条/轮
   * - ``env.train.rollout_epoch``
     - 12
     - 6
     - 12
   * - ``actor.global_batch_size``
     - 576
     - 288
     - 576
   * - ``actor.micro_batch_size``
     - 2
     - 1
     - 2
   * - ``actor.optim.lr``
     - 1e-4
     - 4e-4
     - 1e-4
   * - ``env.train.reward.reward_type``
     - ``video_level``
     - ``frame_level``
     - ``video_level``
   * - ``runner.max_epochs`` / ``runner.max_steps``
     - 70
     - 70
     - 200
   * - ``runner.save_interval``
     - 10
     - -1
     - 10

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

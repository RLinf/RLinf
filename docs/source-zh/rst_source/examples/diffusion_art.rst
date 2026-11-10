ART 扩散模型微调
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   使用 DiffusionART 训练的 SD3 与 Wan2.2 生成模型。

使用 DiffusionART 微调 SD3 或 Wan2.2，通过 OCR 奖励改善文字渲染。ART 将同一 prompt 下各候选的奖励与干净 latent 终点聚合为共享方向，再向生成过程中所经状态上的冻结速度目标训练。完成 :doc:`共用准备步骤 <video_models>` 后，选择图像或视频配置，并按下文启动训练、检查结果。

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

记录中的 Wan 对比使用四张 NVIDIA A800 GPU、LoRA、FSDP2 和 CPU 视频 OCR。下表取每种配置的一次运行，报告按 5760 条生成轨迹平滑后，训练 reward 首次达到 ``0.8`` 的时间与采样量。表中的采样预算与当前用于启动训练的配置不同。

.. list-table:: Wan 视频 OCR 运行记录
   :header-rows: 1
   :widths: 18 22 18 20 22

   * - 方法
     - 每轮轨迹数
     - 学习率
     - 记录轮数
     - 平滑 Reward 首次达到 0.8
   * - ART
     - 1152
     - 1e-4
     - 148
     - 5.43 h / 21888 条轨迹
   * - NFT
     - 1152
     - 1e-4
     - 111
     - 29.50 h 内未达到
   * - NFT
     - 576
     - 4e-4
     - 70
     - 7.83 h / 32832 条轨迹

这些是单次运行的训练奖励，不是独立测试集成功率。两个 NFT 运行同时改变了学习率和采样预算，记录长度也不同，不能把最终 reward 当作等预算结果比较。

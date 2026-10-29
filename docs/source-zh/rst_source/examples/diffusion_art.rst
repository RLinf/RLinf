ART 扩散模型微调
================

本页介绍 DiffusionART 在 SD3.5 Medium 图像文字生成和 Wan2.2 TI2V 5B 视频文字生成中的微调，内容涵盖算法与公式、OCR 数据、训练配置、ART/NFT 奖励与耗时实测结果，以及训练指标定义，最后的 LIBERO 一节介绍 Pi0 动作块的固定前缀和 snapshot 状态采样配方。

概览
----

两个配置均使用 LoRA、FSDP2 和确定性 ODE 生成。每个 prompt 的候选分成两个不重叠的更新窗口，共用第一次更新前冻结的目标。若预测已超出偏移阈值，且更新会让偏移继续增大，修正 mask 就会屏蔽该更新。算法推导与参考实现见 `DiffusionART 论文 <https://arxiv.org/abs/2610.12572>`_、`官方代码 <https://github.com/vanzll/DiffusionART>`_ 和 `项目页 <https://vanzll.github.io/diffusionart-project/>`_。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      SD3.5 Medium / Wan2.2 TI2V 5B / Pi0

   .. grid-item-card:: 算法
      :text-align: center

      图像 OCR / 视频 OCR / LIBERO

   .. grid-item-card:: 输出
      :text-align: center

      图像、视频：LoRA + FSDP2；Pi0：action expert + FSDP1

任务
~~~~

图像、视频配置位于 ``examples/diffusion/config/``，使用 Diffusers checkpoint 和 OCR 奖励；LIBERO 配置位于 ``examples/embodiment/config/``，使用 Pi0 和环境回报。

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
     - OCR prompt；``visual.video_ocr`` 计算视频级平均分。
   * - 机器人操作
     - ``libero_object_0_art_openpi`` （``examples/embodiment/config/``）
     - LIBERO Object task 0；环境 episode 回报。

观测与动作
~~~~~~~~~~

图像、视频任务的每个 episode 生成一张图像或一段视频。ART 为每个候选保留一个标量奖励，再比较同一 prompt 下的候选。:ref:`LIBERO 动作配方 <diffusion-art-libero>` 则比较从同一机器人状态采样的动作块，用续跑回报评价候选。

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

在这些记录中，ART 达到 0.8 的累计时间比 NFT-576 少 30.6%，轨迹数少 33.3%。两组 NFT 同时改变学习率和采样量，本次对比无法拆分各自的贡献。记录长度也不同，不能把最后一个奖励值当作相同预算下的终点比较。这些结果来自单次运行的训练奖励，不能解释成独立测试集成功率，也不代表复现了论文的视频 benchmark。

耗时与诊断
~~~~~~~~~~

执行成本统一统计累计轨迹区间 ``5760 < cumulative trajectories <= 40320``，每组覆盖 34560 条轨迹。对应 ART、NFT-1152 的第 6 至 35 轮，以及 NFT-576 的第 11 至 70 轮。

.. list-table:: 共同区间内的平均耗时
   :header-rows: 1
   :widths: 34 22 22 22

   * - 指标
     - ART
     - NFT，1152，1e-4
     - NFT，576，4e-4
   * - 每轮耗时（s）
     - 1026.24
     - 971.35
     - 495.00
   * - 每轮 actor 训练（s）
     - 105.23
     - 61.56
     - 63.37
   * - 每千条轨迹耗时（s）
     - 890.83
     - 843.18
     - 859.37
   * - 每分钟轨迹数
     - 67.35
     - 71.16
     - 69.82

在每轮 1152 条轨迹的两组中，ART 在该区间内每轮 actor 训练多用约 44 s，整轮多用约 55 s。Wan ART 配方为每条轨迹选择两个去噪状态，预计算对应旧速度，并在两个状态上反向传播；NFT 配方每条轨迹随机选择一个训练状态。额外的 forward 与 backward 是 actor 耗时差异的可能来源，现有 timer 尚未单独量出这部分成本。三组记录中，生成仍占最大的阶段耗时。

``algorithm.art_timestep_fraction=0.1`` 对应训练时刻成本的消融配置。当前生成步数为 10，该值会把每条轨迹的训练状态从两个减为一个。其余设置一致、计时排除初始化的对照可反映这一调整的影响。奖励随累计时间的变化与吞吐量反映不同侧面，因为减少训练状态也会改变学习预算。

本页的指标定义说明更新诊断与奖励、耗时的关系。ART 与 NFT 的 loss 定义不同，仅凭日志中的梯度大小不能判断哪种算法更稳定。论文的图像、视频与 pairwise 实验，以及当前实现的边界见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。

.. _diffusion-training-metrics:

扩散模型训练指标
----------------

这些指标描述 ART 和 NFT 的图像、视频生成训练。任务奖励反映文字生成效果，更新诊断记录模型变化，计时记录执行成本。终端表格会缩短 namespace：Training/Actor 下的 ``actor/art_loss`` 对应日志中的 ``train/actor/art_loss``，Environment 下的 ``video_ocr`` 对应 ``env/video_ocr``。

任务奖励与 rollout 信号
~~~~~~~~~~~~~~~~~~~~~~~

例如 prompt 要求生成 "OPEN"，OCR 会读取生成文字，再与目标比较。评分器去除空格、忽略大小写，按归一化编辑距离评分：

.. math::

   R_{\mathrm{OCR}}=1-\frac{\min(d(\hat y,y),|y|)}{|y|}.

其中 :math:`y` 是目标文字，:math:`\hat y` 是识别文字，:math:`d` 统计插入、删除和替换次数。子串匹配也会得到满分。视频 OCR 对每帧评分，``reward_type=video_level`` 使用所有帧的平均分，``frame_level`` 保留逐帧分数供后续分组。奖励 0.8 表示文字相似度，不代表 80% 的视频文字完全正确。

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 终端指标
     - 含义
   * - ``avg`` / ``video_ocr`` / ``ocr``
     - 奖励 backend 的评分，``reward.key`` 选择训练奖励。统一的 backend 字段支持跨运行对比；帧和 chunk 归约不同，日志聚合结果也可能不同。
   * - ``return``
     - episode 奖励之和。在单步、标量奖励的 ART 配方中，它等于该 episode 的评分。
   * - ``episode_len``
     - 每个 episode 的环境步数。一段视频可以只对应一个 episode，内部仍包含多个去噪步骤。
   * - ``num_trajectories``
     - 当前采集轮次完成的候选数，不是累计计数器。
   * - ``rewards``
     - rollout 侧对训练奖励 tensor 的聚合。
   * - ``advantages_mean`` / ``advantages_min`` / ``advantages_max``
     - NFT 分组比较的相对 advantage 统计。归一化后均值接近零是预期行为。ART 直接从奖励计算引导。
   * - ``loss_mask_fraction``
     - rollout 中标记为可参与训练的项所占比例，与 ART 修正 mask、NFT 裁剪判断含义不同。

ART 引导与更新诊断
~~~~~~~~~~~~~~~~~~

ART 先从完整 prompt 组估计方向，再在选中的去噪状态上拟合冻结速度目标。公式与速度符号约定见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。其 loss 为 :math:`\mathcal L=w\operatorname{MSE}(v_\theta,v_{\mathrm{old}}-H_c)`。

.. list-table:: Training/Actor 下的终端字段
   :header-rows: 1
   :widths: 36 64

   * - 指标
     - 含义
   * - ``actor/art_reward_mean``
     - 经过 ``art_reward_reduction`` 后，各候选标量奖励的平均值。
   * - ``actor/art_reward_std``
     - 每个 actor shard 内候选奖励的总体标准差，再跨 shard 平均。它不等于同一 prompt 内的奖励差异。
   * - ``actor/active_group_frac``
     - 引导非零的 prompt 组占比。组内奖励相同或汇总方向为零时，该组无有效引导。
   * - ``actor/covariance_rms``
     - 有效组的原始奖励与终点协方差 RMS 的平均值，尺度受奖励与 latent 坐标影响。
   * - ``actor/guidance_rms``
     - 每个候选的方向乘幅度后计算 RMS，再对候选平均，包含无效候选。单位归一化时通常接近有效组比例。
   * - ``actor/skipped_zero_guidance``
     - 所有 actor shard 均无引导、跳过优化时为一；至少一个 shard 有信号时为零。
   * - ``actor/art_loss``
     - 加权并经过修正 mask 后的训练 loss，平均时包含 loss 为零的项。
   * - ``actor/unmasked_loss``
     - 修正 mask 生效前的加权 loss，仅对有效项平均，分母与 ``actor/art_loss`` 不同。
   * - ``actor/old_deviate``
     - 训练项上速度 MSE 的平均值，即 :math:`D^{-1}\|v_\theta-v_{\mathrm{old}}\|_2^2`。
   * - ``actor/movement_score``
     - 偏移代理量的均值。``sde_kl`` 使用参考 SDE 噪声与步长；``fixed_mse`` 使用速度 MSE。
   * - ``actor/outward_frac``
     - 有效项中，下降方向与相对于旧速度的偏移内积为正的比例。
   * - ``actor/mask_frac``
     - 有效项中，同时因偏移过大和向外移动条件而被屏蔽的比例。
   * - ``actor/total_loss``
     - ART worker 中等于 ``actor/art_loss``。

这些比例与 loss 汇总各 micro-batch、各训练时刻的统计，再跨 worker 平均。``outward_frac`` 包含任意偏移大小的向外方向，``mask_frac`` 还要求越过阈值。屏蔽更多训练项也会让 masked loss 变小，奖励、有效组比例和屏蔽比例共同构成解释该值的依据。

NFT 能量与裁剪
~~~~~~~~~~~~~~

NFT 根据当前速度变化 :math:`\Delta v=v_\theta-v_{\mathrm{old}}` 构造两个分支：

.. math::

   v_\pm=v_{\mathrm{old}}\pm\beta\Delta v,\qquad
   E_\pm=\operatorname{reduce}\left[
      w_\pm\bigl(T(v_\pm)-x_{\mathrm{target}}\bigr)^2\right].

变换 :math:`T` 将速度转换到配置指定的预测空间，例如 ``nft_target_space=x0``。能量衡量预测误差，``nft_weight_mode`` 和 ``nft_energy_reduction`` 决定它的尺度。正负指两个分支，能量均值通常包含所有候选。采用 ``nft_loss_form=mse`` 时，转换后的 advantage 是 :math:`r\in[0,1]` 的权重，裁剪前的逐项 loss 为：

.. math::

   \ell_{\mathrm{NFT}}=\frac{A_{\max}}{\beta}
   \left[rE_+ +(1-r)E_-\right],
   \qquad A_{\max}=\text{advantage clipping bound}.

这里的裁剪上限为 ``adv_clip_max``。启用裁剪后，实现取裁剪前后逐项 loss 的较大值。它与 ART 目标 MSE 的定义和尺度不同。

.. list-table:: Training/Actor 下的终端字段
   :header-rows: 1
   :widths: 36 64

   * - 指标
     - 含义
   * - ``actor/E_pos_mean`` / ``actor/E_neg_mean``
     - 所有候选的正分支与负分支平均能量。
   * - ``actor/E_pos_mean_pos_only``
     - 转换后 advantage 满足 :math:`r>0.5` 的有效项上的正分支能量。
   * - ``actor/E_neg_mean_neg_only``
     - 满足 :math:`r<0.5` 的有效项上的负分支能量。
   * - ``actor/delta_E_mean``
     - :math:`E_+-E_-` 的均值，相反方向的贡献可能在聚合时相互抵消。
   * - ``actor/delta_v_norm``
     - 按配置的 gather、reduction 轴计算速度变化范数，与 ART 的平均平方偏移不同。
   * - ``actor/clip_frac``
     - 归约后的速度项中，裁剪系数小于一的比例。计数单位由 ``nft_gather_type`` 决定。
   * - ``actor/clip_loss_frac``
     - 有效 loss 项中，裁剪后的 loss 大于原 loss、最终被选中的比例。
   * - ``actor/nft_tau``
     - schedule 插值后的 rollout/reference EMA 当前 actor 混合系数，越小越保留历史参数。
   * - ``actor/nft_loss``
     - 梯度累积缩放前的 NFT 目标值，包含所配置的裁剪行为。
   * - ``actor/total_loss``
     - NFT worker 中除以梯度累积因子后的 loss。使用 SFT co-training 时，还包含 SFT 项。
   * - ``actor/grad_norm`` / ``actor/lr``
     - optimizer 报告的梯度范数与当前学习率，ART 也记录这两个指标。

能量均值、loss 和梯度可用于诊断单次训练，但它们的尺度受权重、目标空间与梯度累积影响。相同时间或采样预算下的任务奖励构成算法表现对比的依据。

耗时与进度
~~~~~~~~~~

runner 的整轮计时与互不重叠的阶段计时反映端到端性能，内部 timer 反映各阶段内的成本。重叠计时相加会重复统计部分执行时间。

.. list-table:: Time 下的终端字段
   :header-rows: 1
   :widths: 43 57

   * - 指标
     - 含义
   * - ``step``
     - runner 当前轮次的耗时。
   * - ``generate_rollouts``
     - 采集阶段，包含生成、奖励反馈和轨迹交付。
   * - ``actor_training`` / ``actor/run_training``
     - runner 和 worker 视角的 actor 训练计时，包含 ART/NFT 输入准备。
   * - ``sync_weights``
     - runner 权重同步阶段。
   * - ``actor/actor/recv_traj``
     - actor 等待、接收轨迹的时间，与 rollout 采集重叠。
   * - ``actor/actor/compute_adv`` / ``cal_adv_and_returns``
     - worker 和 runner 视角的 advantage、return 处理时间；ART 在这个阶段报告 rollout 统计。
   * - ``rollout/predict``
     - 模型生成调用，包含 wrapper 内的生成与解码。
   * - ``env/env_interact_step`` / ``env/env/bootstrap_step``
     - 包含奖励评分的环境交互，以及 bootstrap 工作。
   * - ``env/interact`` / ``env/run_interact_once``
     - 环境侧交互区间，包含通信和等待生成结果。
   * - ``rollout/generate_one_epoch`` / ``rollout/rollout/generate``
     - ``generate_rollouts`` 内部的嵌套 rollout 循环计时。
   * - ``actor/actor/sync_model_to_rollout`` / ``rollout/sync_model_from_actor``
     - actor 与 rollout 两侧的权重传输计时，可能重叠。

表头 Global Step 统计采集与训练轮次，一轮 ART 可以包含两次 optimizer update。Elapsed 是累计墙钟时间，ETA 是预估剩余时间，表头 Step Time 是累计墙钟时间除以已完成轮数得到的平均每轮耗时。Time 区的 ``step`` 记录单轮成本，每千条轨迹耗时与每分钟轨迹数反映不同采样量下的执行成本。相同累计采样区间的对比见 :ref:`训练结果 <diffusion-art-results>`。

.. _diffusion-art-libero:

LIBERO Object Task 0
--------------------

本节介绍 Pi0 在 LIBERO Object task 0 上如何从同一机器人状态比较候选动作块。两个配方只在状态选取方式上不同：``libero_object_0_art_openpi`` 共享固定前缀，在决策索引 8（从 0 计数，即第 9 次决策）分叉；``libero_object_0_art_states_openpi`` 则恢复从采集轨迹中抽取的 snapshot。两者都不会因为后续决策序号相同，就把独立到达的不同状态放进同一比较组。

两个配方都加载 ``RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``。Rollout 由 Hugging Face backend 的 ``MultiStepRolloutWorker`` 执行，训练由 FSDP1 NO_SHARD 的 ``EmbodiedARTFSDPPolicy`` 执行。视觉语言骨干保持冻结，更新 action expert 及相关可训练层，不使用 LoRA。输入包含双相机图像、机器人状态和任务指令；Pi0 生成完整动作 tensor，每次决策向环境执行 5 个 7 维动作。

固定前缀与 Snapshot 采样
~~~~~~~~~~~~~~~~~~~~~~~~~

ART 要求同组候选具有相同的条件状态。仅 task ID 相同不够：被训练动作块开始之前，物体位置、机器人状态、观测和指令都要对齐。两种采集方式分别建立这一条件：

.. list-table:: Object Task 0 采集模式
   :header-rows: 1
   :widths: 18 41 41

   * - 模式
     - 状态构造
     - 训练位置
   * - branch8
     - 同组从相同 reset state 开始，在决策 0 到 7 执行组长的动作块。
     - 决策索引 8，即执行 40 个环境步之后。候选动作与后续续跑各自独立采样。
   * - ``states``
     - 先用冻结的当前 policy 采集轨迹，在决策边界保存尚未结束的组长 snapshot；按任务匹配、时间分层抽样，并将每个选中状态恢复到同组 K 个环境。
     - 恢复后的第一个决策，不一定是整条任务的初始决策。之后各自独立续跑。

branch8 设置 ``algorithm.art_branch_chunk_index=8``、``art_state_sampling=False``。Rollout 在共享前缀和分叉点检查模拟器状态、机器人观测、任务指令及两路图像。如果检查失败，或同组成员在分叉前已终止，该组训练贡献会被屏蔽，避免比较不同状态。

``states`` 设置 ``algorithm.art_state_sampling=True``、``art_branch_chunk_index=null``。Snapshot 包含物理状态、控制器和夹爪状态、观测缓存及模拟器进程的 RNG 状态。恢复时还保留已用步数和前一步奖励，不会重新获得完整 episode 时间预算。状态库在每次 policy 同步后重新采集，恢复后清空。该模式会多一次采集 rollout，并占用 snapshot 存储，不是零开销的固定分叉替代方案。

两种模式下，K=8 个候选通过不同的 policy 采样噪声生成不同动作块。要求一致的是被训练动作块之前的状态，而不是后续整条轨迹。回报累加分叉或恢复之后直到终止或原始 240 步上限的环境奖励；同组回报完全相同时，引导为零。每个候选只训练选中的一个动作块，后续续跑用于提供蒙特卡洛评价。

动作引导与更新
~~~~~~~~~~~~~~

固定条件状态后，在组内有效执行坐标上计算回报与动作的协方差，再做 RMS 归一化得到引导 H。Pi0 与 SD3/Wan 的速度符号约定一致：在缓存的反向去噪状态上，冻结目标是 ``v_old - H``，不是从终点重新前向加噪。模型仍处理完整动作 tensor，但 guidance、restoration 和 trust-region mask 只覆盖有效执行坐标。

配方默认 ``algorithm.art_reference_source=actor``：两次更新开始前在 actor 上重算一次 v_old，之后始终冻结这一参考。这样避免把跨 backend 回放差异直接当成相对参考的位移。选择 ``rollout`` 可以使用缓存的 rollout 速度进行对照。FP32 master 参数先累积小幅优化器更新，再复制回低精度模型权重。这些设置不代表已经验证成功率，需要用真实 checkpoint 和模拟器确认效果。

先用全部 K 个候选计算引导，再把每组候选分入两个互不重叠的更新窗口，同一候选轨迹不会同时出现在两个窗口。128 个环境、K=8、每个环境训练一个动作块、每轮两次更新，对应 16 个组和每次更新 ``actor.global_batch_size=64``。``art_timestep_fraction=1.0`` 训练全部 4 个缓存去噪步；两种模式统一使用学习率 5e-7，默认训练 100 轮。

运行 Branch8 与 States
~~~~~~~~~~~~~~~~~~~~~~~

按 :doc:`Pi0 <embodied/pi0>` 安装 OpenPi/LIBERO 环境；已有环境可跳过安装：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero \
     --venv .venv-openpi-libero
   source .venv-openpi-libero/bin/activate

在仓库根目录执行以下设置，将 MODEL_PATH 指向已有 SFT checkpoint；命令不会下载模型权重。两个配方的训练和评估都限定为 Object task 0：

.. code-block:: bash

   set -o pipefail
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
   export HYDRA_FULL_ERROR=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
   export MODEL_PATH=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

先在 4 张可见 GPU 上运行固定前缀配方。日志目录、实验名和终端日志文件名均以时间戳开头。Bash 入口依次接受配置名、机器人平台和 Hydra overrides；LOG_DIR、MEGA_LOG_FILE 指定输出路径：

.. code-block:: bash

   RUN_NAME="$(date +%Y%m%d_%H%M%S)_object0_branch8_T4_lr5e7_seed42"
   LOG_DIR="$PWD/logs/$RUN_NAME"
   LOG_DIR="$LOG_DIR" MEGA_LOG_FILE="$LOG_DIR/$RUN_NAME.log" \
     bash examples/embodiment/run_embodiment.sh libero_object_0_art_openpi LIBERO \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     env.train.total_num_envs=128 env.eval.total_num_envs=48 \
     actor.global_batch_size=64 actor.micro_batch_size=2 \
     actor.seed=42 actor.optim.lr=5e-7 \
     actor.optim.use_fp32_master_params=true \
     algorithm.art_reference_source=actor \
     actor.model.num_steps=4 algorithm.art_timestep_fraction=1.0 \
     runner.max_steps=100 runner.max_epochs=100 \
     runner.eval_at_start=true runner.val_check_interval=10 \
     runner.save_interval=-1 \
     runner.logger.experiment_name="$RUN_NAME"

Snapshot 采样使用 states 配方，从相同 SFT checkpoint 单独启动。两条命令在同一组 GPU 上顺序执行：

.. code-block:: bash

   RUN_NAME="$(date +%Y%m%d_%H%M%S)_object0_states_T4_lr5e7_seed42"
   LOG_DIR="$PWD/logs/$RUN_NAME"
   LOG_DIR="$LOG_DIR" MEGA_LOG_FILE="$LOG_DIR/$RUN_NAME.log" \
     bash examples/embodiment/run_embodiment.sh libero_object_0_art_states_openpi LIBERO \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     env.train.total_num_envs=128 env.eval.total_num_envs=48 \
     actor.global_batch_size=64 actor.micro_batch_size=2 \
     actor.seed=42 actor.optim.lr=5e-7 \
     actor.optim.use_fp32_master_params=true \
     algorithm.art_reference_source=actor \
     actor.model.num_steps=4 algorithm.art_timestep_fraction=1.0 \
     runner.max_steps=100 runner.max_epochs=100 \
     runner.eval_at_start=true runner.val_check_interval=10 \
     runner.save_interval=-1 \
     runner.logger.experiment_name="$RUN_NAME"

两组实验都在训练前及每 10 轮训练后评估原始完整任务，评估不使用训练时的分叉或 snapshot 恢复。实际评估条数看 ``eval/num_trajectories``，48 个并行环境槽位不等于 48 个独立 reset state。以上命令关闭 checkpoint 保存；需要保留模型时，将 ``runner.save_interval`` 设为正数。

结果与验证
~~~~~~~~~~

比较两种模式时，以完整任务的评估成功率为准。训练中的续跑成功率以采样状态为条件，不能直接视为完整任务成功率。通过 ``actor/active_group_frac``、``actor/branch_valid_group_frac``、``actor/branch_state_error_max`` 和 ``actor/branch_image_error_max`` 检查有效信号与同状态分组。``actor/mask_frac_global`` 和 ``actor/outward_frac_global`` 先汇总有效样本计数再计算比例；``actor/u1/``、``actor/u2/`` 分别记录两次更新。

可选的 ``algorithm.art_replay_diagnostics=true`` 会增加回放前向。``actor/replay_mse`` 比较更新前 actor 与 rollout 速度；``actor/update_u1_exec_rms``、``actor/update_u2_exec_rms`` 则相对于冻结的 actor 参考计算位移。它们不是成功率指标，默认配方关闭这类诊断。

以下 CPU 回归测试覆盖候选分组、回报截取、冻结目标及 snapshot 恢复：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py \
     tests/unit_tests/test_libero_snapshot.py

这些测试不能替代 GPU 训练及真实 MuJoCo/checkpoint 验证。

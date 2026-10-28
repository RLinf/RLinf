ART 扩散模型微调
================

本页介绍 DiffusionART 在 SD3.5 Medium 图像文字生成和 Wan2.2 TI2V 5B 视频文字生成中的微调，内容涵盖算法与公式、OCR 数据、训练配置、ART/NFT 奖励与耗时实测结果，以及训练指标定义，并给出 Pi0 在 LIBERO 多步动作轨迹上的实验性 ART 配方。

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
     - OCR prompt；``visual.video_ocr`` 计算视频级平均分。
   * - 机器人操作
     - ``libero_object_0_art_openpi`` （``examples/embodiment/config/``）
     - LIBERO Object task 0；环境 episode 回报。

观测与动作
~~~~~~~~~~

图像、视频任务的每个 episode 生成一张图像或一段视频。ART 为每个候选保留一个标量奖励，再比较同一 prompt 下的候选。:ref:`LIBERO 动作配方 <diffusion-art-libero>` 则对完整机器人轨迹分组，按决策位置对齐动作块。

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

.. _diffusion-art-libero:

LIBERO 动作微调
---------------

``libero_object_0_art_openpi`` 使用与 ``examples/embodiment/config/libero_object_0_nft_openpi.yaml`` 相同的 SFT checkpoint，在 LIBERO Object task 0 上微调 Pi0。配方沿用环境交互与 rollout 流程，使用完整 episode 回报、rollout 缓存速度，以及每轮两次 ART 更新。

.. warning::

   这是一种实验性的轨迹级适配。同组轨迹共享任务与模拟器初始状态，执行不同动作之后，观测会逐渐分化。因此，后续决策的聚合条件与论文的同条件图像、视频设置存在差异。成功率、指导强度和漂移阈值需要通过 LIBERO 实验验证。``art_first_chunk_only=True`` 仅训练初始决策，可作为同初始观测对照。

轨迹回报与动作指导
~~~~~~~~~~~~~~~~~~

一个机器人 episode 包含多次决策，每次决策分别通过去噪生成一个动作块。配方先累加终止之前及终止动作上的奖励，再把完整 episode 回报分配给有效决策，让终局成功反馈参与早期动作训练。

对于组 :math:`g`、第 :math:`h` 次决策和实际执行的动作坐标 :math:`j`，有效轨迹集合记为 :math:`\mathcal V_{g,h,j}`。协方差仅在该集合内计算：

.. math::

   C_{g,h,j}=\frac{1}{n_{g,h,j}-1}
   \sum_{i\in\mathcal V_{g,h,j}}
   (R_i-\bar R_{g,h,j})(a_{i,h,j}-\bar a_{g,h,j}).

其中 :math:`R_i` 是完整 episode 回报，:math:`a_{i,h,j}` 是模型内部的归一化动作坐标，:math:`n_{g,h,j}=|\mathcal V_{g,h,j}|`。有效轨迹少于两条的坐标，其指导信号为零。协方差在有效动作坐标上归一化，再分配给各轨迹；回报完全相同的组产生零指导信号。

Transformer 仍接收完整动作生成长度和模型维度；协方差、目标损失和漂移代理只使用实际执行的动作与真实环境维度。同一个动作块中，终止之后的动作也会被排除。损失归一化统计所有 actor rank 上的有效动作块。

两次更新始终使用 rollout 时缓存的速度。每个初始状态组分别向两个窗口分配一半轨迹，同一条轨迹的全部动作块归入同一窗口。组指导在划分窗口前计算。Pi0 与 SD3/Wan 实现使用相同的速度符号，目标为 :math:`v_{\mathrm{old}}-H_{g,h}`，现有 corrective mask 在有效动作坐标上计算。

配置与启动
~~~~~~~~~~

OpenPi 环境和 checkpoint 设置沿用 :doc:`Pi0 <embodied/pi0>`。LIBERO 安装选项准备该配方使用的模拟器素材：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero \
      --venv .venv-openpi-libero
   source .venv-openpi-libero/bin/activate

这会创建独立的 OpenPi 环境。Pi0 SFT checkpoint 已位于本地时，以下命令从仓库根目录启动实验：

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1

   python examples/embodiment/train_embodied_agent.py \
      --config-path "$EMBODIED_PATH/config" \
      --config-name libero_object_0_art_openpi \
      actor.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
      rollout.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

默认每轮采集 64 条 episode，每条最多 240 个环境步，每次决策执行 5 个动作。终止掩码生效前，每轮共有 :math:`64\times(240/5)=3072` 个动作块，分成两次不重叠更新，因此 ``actor.global_batch_size=1536``。通用关系为 ``total_num_envs * rollout_epoch * decisions / art_updates_per_rollout``。设置 ``algorithm.art_first_chunk_only=True`` 时，决策数为一，该配方对应的 batch 覆盖值为 ``actor.global_batch_size=32``。

模型使用 ``openpi.rl_mode=art``、``is_nft=False``，以及零转移噪声的 ``noise_method=flow_ode``。训练要求 ``auto_reset=False``、``ignore_terminations=False`` 和固定的分组初始状态。四步采样器结合 ``art_timestep_fraction=0.25``，为每个动作块选择一个实际 rollout 去噪状态。完整配置位于 ``examples/embodiment/config/libero_object_0_art_openpi.yaml``。

固定位置分叉采样
~~~~~~~~~~~~~~~~

共享动作前缀将同条件比较扩展到任务中途。``algorithm.art_branch_chunk_index`` 为所有组指定一个从零开始的决策索引。到达该位置之前，同组环境执行首个成员采样的动作块；从分叉处开始，各成员使用冻结的 rollout policy 独立采样并继续完成任务。Actor 只训练分叉动作块，同时读取该位置的观测、去噪状态和缓存旧速度。回报累加范围为分叉处至终止：

.. math::

   R_i^{(b)}=\sum_{t=t_b}^{T_i}r_{i,t}.

分叉模式在现有启动命令上附加以下覆盖值：

.. code-block:: bash

   algorithm.art_first_chunk_only=False \
   algorithm.art_branch_chunk_index=8 \
   actor.global_batch_size=32

每个动作块执行 5 个动作时，索引 8 对应先执行 40 个环境步再分叉；索引 0 用于初始决策对照。每条 episode 提供一个训练动作块，因此 64 条 episode、两次互斥更新对应全局 batch 32。``art_first_chunk_only`` 与非空分叉索引互斥。分叉模式要求同步批次，且每个 environment、rollout 和 actor rank 均包含完整组。评估保持标准的完整任务 rollout。

Rollout 在前缀和分叉处检查同组任务、机器人观测、可获取的模拟器状态及两路相机图像。状态最大差异默认容差为 ``art_branch_state_atol=1e-5``，各相机平均绝对像素差默认容差为 ``art_branch_image_atol=1.0``。任一前缀决策不满足一致性要求，或任一组员在分叉前终止，该组整体退出训练，批次形状保持不变。检查范围为可获取的状态和观测，不包含控制器快照恢复。

验证与指标
~~~~~~~~~~

张量与配置测试覆盖终止掩码、分组对齐、冻结目标和缓存状态回放。模型测试使用小型 Gemma/SigLIP 与构造的观测，可以在本地直接运行：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py

启动实验附加 ``runner.max_steps=2``、``runner.max_epochs=2``、``runner.val_check_interval=1`` 和 ``runner.save_interval=1``，会先执行两轮训练、评估与 checkpoint 保存。GPU LIBERO 训练和 checkpoint 恢复需要真实模型与模拟器素材。

``actor/art_reward_mean`` 和 ``actor/art_reward_std`` 描述 episode 回报；``actor/valid_chunk_frac`` 和 ``actor/valid_action_frac`` 描述终止掩码与可选奖励过滤后保留的比例。现有 ``active_group_frac``、``mask_frac`` 和 ``movement_score`` 反映指导信号与漂移。ART/NFT 对照使用相同 SFT 初始化、初始状态和评估状态，同时记录环境交互量、optimizer 更新次数与耗时；两种损失的数值尺度对应不同目标。

分叉实验额外记录 ``actor/branch_chunk_index``、``actor/branch_valid_group_frac``、``actor/branch_state_error_max`` 和 ``actor/branch_image_error_max``。误差汇总前缀检查结果，有效组比例还排除分叉前终止的组。先运行索引 0 和索引 8 的短实验，分别检查初始决策行为与中途分叉，再增加训练预算。这些实验需要 GPU LIBERO 验证；单元测试覆盖采样、缓存对齐、回报截取和无效组屏蔽。

完整输出速度诊断通过 ``algorithm.art_replay_diagnostics=true`` 开启。
``actor/replay_{region}_rms`` 记录更新前的回放误差；
``actor/drift_u1_{region}_rms`` 和 ``actor/drift_u2_{region}_rms`` 分别记录两次 optimizer 更新后相对同一份冻结 rollout 速度的漂移。
``region`` 为 ``exec`` 时统计有效执行坐标，``future`` 统计后续 action token 的环境动作维度，``padding`` 统计其余模型维度，``unexec`` 统计全部未执行坐标，``full`` 统计完整输出。
完全无效的行不参与统计；平方误差与坐标数先跨 microbatch、rank 求和，再计算 RMS。
探针使用固定缓存状态和已选取的去噪时间步，不重新生成动作链。训练 loss 和 trust-region mask 仍只覆盖执行坐标。
完整速度缓存增加 rollout 存储与传输量；开启诊断会在更新前及每次更新后增加 eval 模式前向计算。
同一开关还会记录 ``actor/art_output_restore_ratio`` 和 ``actor/art_output_restore_cosine``，比较 loss 对执行坐标速度输出的梯度中 restoration 与 guidance 两部分，统计已计入 loss 缩放、梯度累积和跨 rank 平均。

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

排查 LIBERO 更新与状态采样
--------------------------

Long task 0 使用 ``libero_10_0_art_diagnostics_openpi``，Object task 0 使用
``libero_object_0_art_diagnostics_openpi``。两者均选择 128 个训练环境、K=8、
全部 4 步去噪、actor 端参考速度、FP32 master AdamW，并默认运行 5 轮。
原有配方仍使用 rollout 参考速度和仅组长轨迹的状态采集。

先区分回放误差与参数更新。设置 ``algorithm.art_reference_source=actor``，
在两次 ART 更新前只重算一次冻结参考。开启
``algorithm.art_replay_diagnostics=True`` 后，对比
``actor/replay_mse`` （更新前 actor 与 rollout 的差异）、
``actor/reference_pre_mse`` （actor 与所选参考的差异），以及
``actor/update_u1_exec_rms`` / ``actor/update_u2_exec_rms`` （相对于 actor
参考的更新位移）。``actor/drift_u1_*`` 和 ``actor/drift_u2_*`` 继续使用
rollout 参考；``update_u*`` 指标只在选择 actor 参考时输出。

短程 FSDP1 NO_SHARD 排查时开启 ``algorithm.art_parameter_diagnostics=True``。
``actor/param_u1_bf16_changed_frac`` 和 ``actor/param_u2_bf16_update_rms``
记录实际模型权重变化，``actor/param_u2_master_update_rms`` 记录 FP32 主权重
的累积更新。诊断在 CPU 保留可训练参数副本，位移均相对于第一次更新前计算。
设置 ``algorithm.art_replay_batch_sizes=[1,2,8]``，可在固定数据子集上比较
不同 actor 推理 batch 大小的回放差异。

使用 ``actor/mask_frac_global`` 和 ``actor/outward_frac_global`` 查看按
active 样本计数加权的比例。``actor/u1/`` 和 ``actor/u2/`` 下的同名指标
区分两次更新；``movement_score_active`` 使用相同的 active 样本分母。
原有 ``mask_frac`` 和 ``outward_frac`` 保留 microbatch 平均口径，用于对照旧日志。

随后设置 ``algorithm.art_probe_only=True`` 冻结 SFT 策略。此模式跳过反向传播、
optimizer、scheduler，以及首次之后的权重同步。通过
``algorithm.art_state_sampler`` 选择以下状态池：

.. list-table:: 状态采样消融
   :header-rows: 1
   :widths: 25 75

   * - 取值
     - 采集与选取方式
   * - ``legacy``
     - 保存所有仍有效的组长状态，按任务匹配并进行时间分层采样。
   * - ``uniform``
     - 从所有轨迹建立有容量上限的 reservoir，再进行时间分层采样。
   * - ``success_mixture``
     - 使用有界状态池，成功来源状态约占 ``art_success_state_fraction``
       （默认 0.7）；某类来源为空时使用可用的另一类来源。
   * - ``early_failure``
     - 使用有界状态池，保留成功来源的全部状态，仅保留失败来源在
       ``art_failure_max_step_fraction`` 时间比例之前的状态（默认 0.5）。

有界状态池的 ``algorithm.art_states_per_trajectory`` 默认为 4。
先进行 reservoir 接纳决策，再捕获 snapshot。采集结束后标注来源轨迹结局，
恢复时保留原始剩余预算。``<log_path>/art_probe/`` 下的 JSONL 按组记录回报、
来源结局、来源轨迹和 reset 标识、采样时间及剩余步数。探针比较采样分布，保持奖励不变。

激活 OpenPi/LIBERO 环境后运行实验矩阵：

.. code-block:: bash

   python examples/embodiment/run_art_ablation.py diagnose \
     --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Long-SFT --gpus 0,1,2,3
   python examples/embodiment/run_art_ablation.py probe \
     --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Long-SFT --gpus 0,1,2,3
   python examples/embodiment/run_art_ablation.py summary

``diagnose`` 运行 4 组各 5 轮的实验：rollout/actor 参考与关闭/开启 FP32 master
交叉对比。``probe`` 为 4 种采样方式各运行 10 轮冻结策略采集。
``summary`` 按初始 reset ID 划分筛选集与留出集，分别报告信号产出率。
同时查看来源构成、全失败和全成功比例，不要只按 active 比例选方案；汇总不会自动选择赢家。

选定采样方式后，运行 ``train --sampler uniform`` （将 ``uniform`` 换成选定方案），
比较两组各 30 轮训练：采用 ``legacy`` 采样的数值修正基线，以及相同基线加选定采样。
``confirm --sampler uniform`` 使用 seed 42 和 43，将这两组各重跑 50 轮。
``object --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``
运行 25 轮 Object task 0 基线。每个阶段的运行名、控制台日志和完整命令清单均以时间戳开头。
添加 ``--dry-run`` 可先检查命令，``--override key=value`` 可追加 Hydra 覆盖项。

训练阶段在 step 0 及每 10 轮评估一次（``object`` 为每 5 轮）。
``runner.eval_repeats`` 重复完整的可用 reset 集合，``runner.eval_seed``
为各实验配对策略采样噪声，同时保持训练 RNG 不变。当任务包含 50 个 reset 状态时，
重复 3 次得到 150 条轨迹；这是重复初始状态，不是 150 个不同场景。确认阶段重复 5 次。
``eval`` 在 step 0 重复评估 4 次，训练轮数为 0；使用 ``num_trajectories``
核对实际完成的评估预算。

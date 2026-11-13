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

LIBERO Object Task 0
--------------------

本节介绍 Pi0 在 LIBERO Object task 0 上如何从同一机器人状态比较候选动作块。两个配方只在状态选取方式上不同：``libero_object_0_art_openpi`` 共享固定前缀，在决策索引 8（从 0 计数，即第 9 次决策）分叉；``libero_object_0_art_states_openpi`` 则恢复从采集轨迹中抽取的 snapshot。两者都不会因为后续决策序号相同，就把独立到达的不同状态放进同一比较组。

两个配方都加载 ``RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``。Rollout 使用 Hugging Face backend，训练使用 FSDP1 NO_SHARD。视觉语言骨干保持冻结，更新 action expert 及相关可训练层，不使用 LoRA。输入包含双相机图像、机器人状态和任务指令；Pi0 生成完整动作 tensor，每次决策向环境执行 5 个 7 维动作。

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

比较两种模式时，以完整任务的评估成功率为准。训练中的续跑成功率以采样状态为条件，不能直接视为完整任务成功率。通过 ``actor/active_group_frac``、``actor/branch_valid_group_frac``、``actor/branch_state_error_max`` 和 ``actor/branch_image_error_max`` 检查有效信号与同状态分组。``actor/mask_frac_global`` 和 ``actor/outward_frac_global`` 先汇总有效样本计数再计算比例；``actor/u1/``、``actor/u2/`` 分别记录两次更新。``actor/mask_frac`` 和 ``actor/outward_frac`` 也按有效样本数聚合 micro-batch 和 actor rank，不再直接平均各局部比例。

可选的 ``algorithm.art_replay_diagnostics=true`` 会增加回放前向。``actor/replay_mse`` 比较更新前 actor 与 rollout 速度；``actor/update_u1_exec_rms``、``actor/update_u2_exec_rms`` 则相对于冻结的 actor 参考计算位移。它们不是成功率指标，默认配方关闭这类诊断。

以下 CPU 回归测试覆盖候选分组、回报截取、冻结目标及 snapshot 恢复：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py \
     tests/unit_tests/test_libero_snapshot.py

这些测试不能替代 GPU 训练及真实 MuJoCo/checkpoint 验证。

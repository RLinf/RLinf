OpenWAM 监督微调
=================

本配方通过 RLinf 的 Ray 管理 FSDP runner，在原生 LIBERO 数据集上微调 OpenWAM。训练会从 OpenWAM checkpoint 目录读取模型配置，复用原生 dataloader 和视频、动作联合 loss，并使用 Full-shard FSDP 更新未冻结模块。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenWAM Wan2.2

   .. grid-item-card:: 方法
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: 数据
      :text-align: center

      原生 LIBERO reader

   .. grid-item-card:: 硬件
      :text-align: center

      默认 8 张 GPU 与 FSDP2

OpenWAM checkpoint 会提供模型和 dataloader 设置。将 ``data.train_data_paths`` 指向数据集根目录（也可以是多个根目录的列表，各数据集用同一套 dataloader 设置读取后按样本数比例拼接）；loader 会读取 ``actor.model.model_path`` 下的 ``config.yaml``，并保留原生的帧数、动作和归一化约定。

安装
----

安装 OpenWAM 环境和 RLinf：

.. code:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

运行
----

在 ``examples/sft/config/model/openwam.yaml`` 和 ``examples/sft/config/libero_sft_openwam.yaml`` 中设置 checkpoint 与数据集路径。配方默认使用 8 张 GPU（``0-7``），为 OpenWAM 的可训练 DiT/action 参数和优化器状态留出更多显存余量。其他 OpenWAM SFT 配方也继承这一默认卡数。

第四轮审查在 H200 上使用 LIBERO 数据、``micro_batch_size: 1`` 和 ``global_batch_size: 8`` 测得：4 卡时每卡峰值已分配显存为 72.3 GiB、保留显存为 113.3 GiB；8 卡时分别为 55.5 GiB 和 83.7 GiB。默认改为 8 卡是依据这组 H200 数据，并不代表已经验证能在 80 GiB 显卡上运行；实际显存需求仍需结合 checkpoint 和硬件确认。

数据读取器由 checkpoint 的 ``config.yaml`` 决定，因此配方要把 checkpoint 和同类型的数据配对。``examples/sft/config/`` 下每种 OpenWAM 读取器各有一份配方，都继承 ``libero_sft_openwam.yaml``，只改路径和实验名：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 配方
     - 数据
   * - ``libero_sft_openwam``
     - LIBERO（LeRobot v3，EEF10 动作）
   * - ``agibotworld_sft_openwam``
     - AgibotWorld
   * - ``interndata_a1_sft_openwam``
     - InterData A1
   * - ``mixture_sft_openwam``
     - OpenWAM reader 混合数据
   * - ``muka_franka_sft_openwam``
     - Muka Franka
   * - ``oxe_droid_sft_openwam``
     - OXE DROID
   * - ``robocoin_sft_openwam``
     - RoboCoin
   * - ``robotwin_sft_openwam``
     - RoboTwin 2.0（aloha-agilex，20 维双臂 EEF）
   * - ``robodojo_sft_openwam``
     - RoboDojo 真机数据
   * - ``ebench_sft_openwam``
     - EBench
   * - ``robocasa365_sft_openwam``
     - RoboCasa365
   * - ``robocasa_gr1_sft_openwam``
     - RoboCasa GR1 人形
   * - ``vlabench_sft_openwam``
     - VLABench

对 ``mixture`` checkpoint，``data.train_data_paths`` 只填一个根目录：checkpoint 的 mixture 配置中每个启用的数据源 ``datasets.<name>`` 都从 ``<root>/<name>`` 读取，单个数据源可用 ``data.openwam.datasets.<name>.dataset_dir`` 指向别处。各数据源各自保留归一化统计量，因此 mixture 训练不会在 checkpoint 旁保存 ``normalization_stats.npy``，导出时使用源 checkpoint 中的文件。

基础配方默认启用 ``fsdp_config.gradient_checkpointing: true``，并将它转发给 OpenWAM 自己的分块 checkpointing（``use_gradient_checkpointing``）。同时使用 ``global_batch_size: 8`` 和 ``micro_batch_size: 1``，在不增加每个 rank 激活显存的情况下提高有效 batch。

启动由 Ray 管理的 FSDP runner：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

修改 GPU 数量时，同时修改 ``cluster.component_placement.actor``，并确保 ``actor.global_batch_size`` 能被 actor world size 整除。

预设以 fp32 加载权重（``precision: fp32``），优化器持有 fp32 主权重，FSDP 用 bf16 计算（``mixed_precision.param_dtype``）；若主权重是 bf16，``lr: 1e-6`` 下几乎所有更新都会被舍入掉。由于 OpenWAM 的联合去噪驱动会在块的 forward 之外直接读取块权重，policy 使用一个根 FSDP2 单元。``reshard_after_forward`` 只作用于 wrap policy 指定的冻结 ``ResidualBlock`` 子单元；可训练的根参数在联合 forward 和 backward 期间仍需驻留。配方仍默认启用 gradient checkpointing，但第四轮审查的 4 卡开关对比显示，在 ``micro_batch_size: 1`` 下它未降低峰值显存。模型预设同时保持 ``load_to_device: false``：每个 rank 先在 CPU 上构建模型，FSDP 在包装时把各自的分片搬到 GPU。评测配方则用 ``load_to_device: true`` 直接加载到 GPU。

验证与断点续训
--------------

设置 ``data.val_data_paths``\ （一个或多个数据集根目录，用同一套 dataloader 设置读取）和 ``runner.val_check_interval`` 后，会在验证集上平均 OpenWAM 的原生 loss，记录为 ``eval/loss``、``eval/loss_video`` 和 ``eval/loss_action``。``actor.eval_batch_size`` 是每个 rank 的验证 batch，``actor.eval_max_batches`` 可以限制大数据集上每个 rank 跑的验证 batch 数。LeRobot 风格的读取器按 split 选取 episode：验证默认读 ``val`` split，如果验证集是一个只有 train split 的独立数据集，请设置 ``data.openwam_val_split: train``\ （验证集为空时会在启动阶段直接报错）。多卡验证遇到短 shard 时，会重复一个样本仅用于对齐各 rank 的 FSDP forward 次数；补齐样本不会计入最终 loss。

checkpoint 会把 dataloader、sampler（含 shuffle 的 epoch）和随机数状态与模型权重一起保存，因此 ``runner.resume_dir=<log_path>/<experiment_name>/checkpoints/global_step_<N>`` 会从下一个未见过的 batch 继续，而不是重头开始这一轮数据。OpenWAM 配方设置了 ``runner.strict_resume: true``，缺少 ``data.pt`` 或 ``rng.pt`` 的旧 checkpoint 会直接报错；只有明确接受重新开始数据流时才应取消该设置。

查看结果
--------

在 TensorBoard 中观察 ``train/loss``、``train/loss_video`` 和 ``train/loss_action``。RLinf 会将 FSDP 模型和 optimizer shards 写入 ``runner.logger.log_path/<experiment_name>/checkpoints/global_step_<N>/actor``。

导出部署 checkpoint
-------------------

FSDP worker 会把完整的 ``OpenWAMPolicy`` state dict 保存在 ``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``\ （只有分片的 ``dcp_checkpoint`` 目录时，导出脚本会自动合并）。用下面的命令把它重建为自包含的 OpenWAM checkpoint 目录：

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
       --rlinf-checkpoint ../results/libero_sft_openwam/checkpoints/global_step_1000 \
       --source-checkpoint /path/to/openwam-libero-sft-30000 \
       --output /path/to/openwam-libero-sft-rlinf-step1000 --link-assets --verify cuda

导出脚本去掉 ``architecture.`` 前缀、丢弃 ``vlm_backbone.*``\ （OpenWAM 以目录形式保存 VLM）、校验键集合与源 checkpoint 一致，并写出 ``checkpoint_step_<N>.safetensors``；config、tokenizer 和归一化文件从 ``--source-checkpoint`` 复制（``--link-assets`` 时为软链）。``--verify`` 用 ``openwam.deploy.load_from_checkpoint_dir`` 重新加载核对。导出目录既可以交给 OpenWAM 自身的工具，也可以作为下文评测配方的 ``rollout.model.model_path``。

评估
----

LIBERO 评估配方与本配方使用同一套 checkpoint 约定。``evaluations/libero/`` 提供 ``libero_{spatial,object,goal,10}_openwam_eval.yaml``\ （见 :doc:`../../evaluations/guides/libero`）；每条 episode 会记录 ``[libero eval] task_id=.., trial_id=.., success=..``，可以按任务拆分成功率。RoboTwin checkpoint 使用 ``evaluations/robotwin/robotwin_<task>_openwam_eval.yaml``，覆盖全部 50 个任务（见 :doc:`../../evaluations/guides/robotwin`）。

设置 ``MUJOCO_GL=egl`` 和 ``PYOPENGL_PLATFORM=egl`` 后，可以先运行 smoke 配方（1 个环境、30 步，即三次 10 步生成）。配方把 env worker 与 rollout worker 分到不同 GPU，避免 EGL 渲染和 OpenWAM 推理争用同一张卡：

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path ../tests/e2e_tests/evaluations --config-name libero_spatial_openwam_eval

自行缩短配方时，``env.eval.max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks``\ （默认 ``openwam.inference_horizon`` 下为 10）整除。

配方默认面向用 native delta EEF10 动作训练的 checkpoint（``env.eval.openwam_action_representation: native_delta_eef10``）。如果 checkpoint 按标准 LIBERO 数据训练、输出绝对 EEF10 目标位姿，请改为 ``absolute_eef10``：RLinf 会在每个环境 step 根据当前实际位姿计算目标差值，再发送 7D OSC 动作。该值必须与被评估 checkpoint 的数据 metadata 一致。


其他视频编码器
--------------

部分旧 OpenWAM checkpoint 使用 ``vjepa2_1``、``flux_vae`` 或 ``wan_vae`` 这些旧名称，部署时 RLinf 会自动转换。如果编码器权重不在 checkpoint 内，可设置 ``rollout.model.encoder_model_path`` 指向本机编码器目录。RLinf 会使用临时配置加载，不会修改 checkpoint。

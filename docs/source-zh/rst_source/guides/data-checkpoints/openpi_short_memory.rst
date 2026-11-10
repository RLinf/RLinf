OpenPI Short Memory 评测与训练
==============================

本页介绍如何在 LIBERO-Mem 上评测带 short memory 的 π₀.₅ checkpoint，以及如何通过 FSDP1 微调模型。先确定历史帧配置，再准备 checkpoint 和归一化统计量，最后启动评测或 SFT。下列配置使用 RLinf 原生 ``openpi`` 模型与 LIBERO transforms。Short memory 支持 ``openpi.task: eval`` 和 ``sft``；RL replay 尚未携带相机历史。

配置视觉历史
------------

评测与训练都从当前 episode 中选取相机帧，按时间从旧到新排列。``num_frames`` 包含当前帧，可以自由配置，示例中的八帧只是默认值。``stride`` 在评测中表示环境 control step 数，在 SFT 中表示数据帧数。在 20 Hz 的评测环境中，八帧、stride 70 对应 24.5 秒的历史跨度。

评测通过 ``env.eval.short_memory`` 配置历史，SFT 使用 ``data.short_memory``，并从 LeRobot metadata 读取采样率：

.. code-block:: yaml

   # Evaluation: examples/embodiment/config/env/libero_mem.yaml
   short_memory:
     num_frames: 8
     stride: 70
     fps: 20.0

   # SFT: examples/sft/config/libero_mem_sft_openpi_pi05.yaml
   data:
     short_memory:
       num_frames: 8
       stride: 70

评测的 ``fps`` 应与环境控制频率一致。评测根据 control step 差计算相对时间，SFT 则使用所选帧在数据集中的 timestamp。只有数据按相同频率保留了每个 control step，两处 stride 才表示相同的物理时间间隔。如果转换过程删除 idle frames 并重新编号 timestamp，训练时间轴就会被压缩；读取这些 timestamp 无法恢复被删除的时间。比较训练与评测的历史跨度前，应先核对转换流程。下文的公开数据集声明了 20 Hz，但仅凭 metadata 不能确认它保留了所有 control steps。Episode 开始时缺少的历史帧填零，并将对应 mask 设为 false；环境 reset 时清空该环境的历史。

只有相机输入带历史。State 和 prompt 始终属于当前帧，训练动作目标从当前帧开始。视觉编码器默认在从零编号的 ``[3, 7, 11, 15]`` 层融合历史，在第 15 层之后只保留当前帧的 tokens。可通过 ``openpi.short_memory_temporal_layers`` 和 ``openpi.short_memory_drop_history_layer`` 调整这些位置。这一过程复用视觉权重，不增加可学习参数，也不增加语言模型接收的视觉 token 数。``short_memory`` 默认为 ``false``，此时 OpenPI 使用原有图像路径。

准备环境与权重
--------------

仿真评测需要 :doc:`LIBERO </rst_source/examples/embodied/libero>` 页面介绍的 OpenPI 环境与 LIBERO-Mem fork。SFT 直接读取 LeRobot demonstrations，不启动模拟器。训练环境应安装与数据集匹配的 LeRobot 和视频 decoder 依赖。使用 TorchCodec 解码时，需要兼容的 PyTorch/TorchCodec 版本及 FFmpeg 动态库；图像存储和视频存储共用同一 short-memory loader。

以下命令使用上述安装步骤创建的独立环境 ``.venv-libero-mem``。

SFT 可以从公开的 `RLinf-Pi05-LIBERO-SFT 权重 <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__ 开始：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-SFT \
     --local-dir checkpoints/RLinf-Pi05-LIBERO-SFT

该命令下载下方训练配置使用的初始权重。LIBERO-Mem 的归一化统计量需要按下文步骤从训练数据计算。

评测时，准备微调后的 checkpoint 及其 OpenPI ``norm_stats.json``。配置中的 ``/path/to/...`` 都需要替换成实际路径。``model_path`` 可以指向 safetensors checkpoint 目录，或包含 ``actor/model_state_dict/full_weights.pt`` 的 RLinf checkpoint，也可以指向后者的 ``actor`` 或 ``model_state_dict`` 目录。直接传入 safetensors 或 ``.pt`` 权重文件也受支持。模型形状与 transforms 由 YAML 配置决定，因此 LIBERO transforms、action horizon 和归一化统计量应与权重一致。单独加载权重会开始新的训练；恢复完整训练状态需要 ``runner.resume_dir``。

在 LIBERO-Mem 上评测
--------------------

准备好模拟器与权重后，激活环境，在仓库根目录运行专用评测配置：

.. code-block:: bash

   source .venv-libero-mem/bin/activate
   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=/path/to/short_memory_checkpoint \
     rollout.model.openpi_data.norm_stats_path=/path/to/libero-mem/norm_stats.json

该配置使用 ``env_type: libero``、``task_suite_name: libero_mem``、20 个并行环境和 700-step horizon。模型每次预测 10 个动作，执行其中五个，使用五步去噪。历史按每个 control step 记录，也包含 action chunk 内的中间步骤。启用历史时保持 ``env.eval.skip_intermediate_renders=False``。

不使用历史进行评测时，同时设置 ``rollout.model.openpi.short_memory=False`` 和 ``env.eval.short_memory=null``。

如需调整帧数与间隔，可在命令后添加 ``env.eval.short_memory.num_frames=3 env.eval.short_memory.stride=35``。比较评测结果时，应采用 checkpoint 训练时的历史配置。输出说明见 :doc:`LIBERO 评测 </rst_source/evaluations/guides/libero>`，LIBERO-Mem 成功条件见 LIBERO 页面。

历史帧会增加视觉编码器的计算量，虽然语言模型接收的视觉 token 数保持不变。选择历史长度时，应在实际 batch size 与硬件上测量开销。

准备训练数据
------------

将 `CollisionCode/libero_mem_lerobot_v2.1 <https://huggingface.co/datasets/CollisionCode/libero_mem_lerobot_v2.1>`__ 下载到 LeRobot 缓存目录。其 metadata 记录了 961 个 episodes、321,405 帧、10 个任务和 20 Hz 采样率，所有 episodes 均属于训练集：

.. code-block:: bash

   export HF_LEROBOT_HOME="$PWD/datasets"
   hf download CollisionCode/libero_mem_lerobot_v2.1 --repo-type dataset \
     --local-dir "$HF_LEROBOT_HOME/CollisionCode/libero_mem_lerobot_v2.1"

后续命令保持 ``HF_LEROBOT_HOME`` 不变。SFT 配置选择 ``pi05_libero_mem``，直接读取公开数据集中的字段，并将其映射到与评测一致的 LIBERO policy 输入：

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - 数据集字段
     - Policy 输入
     - 内容
   * - ``observation.images.image``
     - ``observation/image``
     - 第三人称 RGB 视频。
   * - ``observation.images.wrist_image``
     - ``observation/wrist_image``
     - 腕部 RGB 视频。
   * - ``observation.state``
     - ``observation/state``
     - 八维末端执行器与夹爪状态。
   * - ``action``
     - ``actions``
     - 七维增量位姿与夹爪动作。

Task prompt 来自 LeRobot task metadata，loader 使用 episode 边界与 timestamp 采样历史。如果已有数据已经转换为 ``image``、``wrist_image``、``state`` 和 ``actions``，计算归一化统计量和运行 SFT 时都应使用 ``pi05_libero``，而不是 ``pi05_libero_mem``。复现已有实验时，保持原数据与配置不变。

使用与 SFT 相同的数据配置计算归一化统计量：

.. code-block:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_libero_mem \
     --repo-id CollisionCode/libero_mem_lerobot_v2.1

该命令将 ``norm_stats.json`` 写入 ``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/``。训练和评测时显式指定这个文件。其他数据集的归一化流程见 :doc:`OpenPI SFT </rst_source/examples/embodied/sft_openpi>`。

通过 FSDP 微调
--------------

准备好数据集与统计量后，将 ``data.train_data_paths`` 设为本地数据根目录或 LeRobot repository ID。历史相机查询限制在同一 episode 内，当前 state、prompt 和向前的 action window 沿用 OpenPI transforms。每个 sample 只执行一次 prompt tokenization。

训练仅使用当前帧的对照模型时，保留 ``data.short_memory``，并设置 ``actor.model.openpi.short_memory=False``。这样会使用相同的可恢复 loader 和 sampler，只返回当前相机帧。未配置 ``data.short_memory`` 且模型关闭 short memory 时，沿用原有 loader。

直接运行 SFT 入口，使 Hydra overrides 传入配置：

.. code-block:: bash

   source .venv-libero-mem/bin/activate
   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-name libero_mem_sft_openpi_pi05 \
     data.train_data_paths=CollisionCode/libero_mem_lerobot_v2.1 \
     actor.model.model_path="$PWD/checkpoints/RLinf-Pi05-LIBERO-SFT" \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json" \
     runner.logger.log_path="$PWD/logs/libero_mem_short_memory"

示例使用现有 flow-matching loss 训练完整模型，更新 12,750 步，每个 rank 的 microbatch 为四，全局 batch 为 128。根据可用 GPU 调整 batch 与更新预算。历史配置项为 ``data.short_memory.num_frames`` 和 ``data.short_memory.stride``；添加 ``+data.episodes='[0,1]'`` 可以按数据集 episode ID 限定小规模验证范围。

模型保留 FP32 参数用于 optimizer 更新，backbone activations 使用 BF16。``openpi_fsdp`` preset 将 action/time projections、RMSNorm 和 vision embedding 模块的计算保留为 FP32，gradient checkpointing 使用 non-reentrant 实现。Short memory 不增加独立 optimizer 或可训练的 memory 参数。

恢复训练
--------

Short-memory loader 使用 ``StatefulDataLoader`` 与 distributed sampler。SFT checkpoint 会在训练状态之外，将各 rank 的 loader state 和 sampler epoch 保存为 ``actor/data.pt``，RNG state 保存为 ``actor/rng.pt``。继续训练时保留这些文件。

在 SFT 命令后添加 ``+runner.resume_dir=/path/to/checkpoints/global_step_2125``，恢复完整 checkpoint。保持数据集、历史配置、worker 数量、world size 和训练配置与保存时一致。Checkpoint 布局与通用恢复方式见 :doc:`恢复训练 <../resume>`。

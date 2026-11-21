DreamZero 监督微调和 Franka 真机部署
========================================

.. figure:: https://dreamzero0.github.io/images/project_overview.png
   :align: center
   :width: 90%

   DreamZero：由视频生成世界模型微调得到的 VLA 策略。

在 RLinf 中运行 DreamZero 监督微调（SFT）：准备模型与 LeRobot 数据，启动训练，执行仿真评测，并将训练后的策略部署到 Franka 真机。

概览
----------------------------------------

将基于 WAN 的 DreamZero 世界模型微调成操作策略，在 LeRobot 数据上训练，在仿真中评测，并部署到 Franka。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      WAN2.1 · WAN2.2

   .. grid-item-card:: 方法
      :text-align: center

      SFT · Mixture SFT

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO · DROID · Franka PnP

   .. grid-item-card:: 硬件
      :text-align: center

      1+ 节点 · NVIDIA CUDA · :ref:`华为昇腾 CANN <sft-dreamzero-hardware>` （SFT）

| **你将完成：** 安装 → 准备模型和 LeRobot 数据 → 生成 ``metadata.json`` → 运行 ``run_vla_sft.sh`` → 在仿真或 Franka 上评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · `DreamZero 仓库 <https://github.com/RLinf/dreamzero>`_（``DREAMZERO_PATH``）· 一个 LeRobot 数据集。

**当前支持**

- **数据集：** LIBERO（``libero_sim``）、OXE DROID（``oxe_droid``）、Franka pick-and-place（``franka_pnp``）；支持跨 embodiment 的 **mixture SFT** （见 ``libero_franka_mix_sft_dreamzero_5b.yaml``）。
- **骨干网络：** WAN2.1（如 DreamZero-DROID 14B）、WAN2.2（如 Wan2.2-TI2V-5B 冷启动）。

.. _sft-dreamzero-hardware:

一键上手
----------------------------------------

.. include:: _setup_common.rst

``examples/sft/config/`` 下的 SFT 参考配置：

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 配方
     - 配置文件
     - 骨干网络 / Checkpoint
   * - LIBERO 14B
     - ``libero_sft_dreamzero_14b.yaml``
     - WAN2.1（`DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_）
   * - LIBERO 5B
     - ``libero_sft_dreamzero_5b.yaml``
     - WAN2.2 冷启动（`RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_）
   * - DROID 14B
     - ``droid_sft_dreamzero_14b.yaml``
     - WAN2.1（`DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_ / `DreamZero-AgiBot <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>`_）
   * - LIBERO + Franka 混合
     - ``libero_franka_mix_sft_dreamzero_5b.yaml``
     - WAN2.2 混合 SFT（``dreamzero_5b``）

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. 安装并克隆 DreamZero**\ （``examples/sft/run_vla_sft.sh`` 读取 ``DREAMZERO_PATH`` 导入外部包）：

      .. tab-set::

         .. tab-item:: 仅 SFT 训练

            .. code-block:: bash

               git clone https://github.com/RLinf/RLinf.git && cd RLinf
               bash requirements/install.sh embodied --model dreamzero && source .venv/bin/activate
               git clone https://github.com/RLinf/dreamzero.git ../dreamzero
               export DREAMZERO_PATH="$(realpath ../dreamzero)"

         .. tab-item:: SFT + LIBERO 评测

            .. code-block:: bash

               git clone https://github.com/RLinf/RLinf.git && cd RLinf
               bash requirements/install.sh embodied --model dreamzero --env libero && source .venv/bin/activate
               git clone https://github.com/RLinf/dreamzero.git ../dreamzero
               export DREAMZERO_PATH="$(realpath ../dreamzero)"

      **2. 下载模型权重**\ （从已发布 checkpoint 继续训练，或使用 WAN2.2 组件冷启动）：

      .. tab-set::

         .. tab-item:: 从 Checkpoint 继续（14B）

            .. code-block:: bash

               hf download GEAR-Dreams/DreamZero-DROID --local-dir ./DreamZero-DROID

            在 ``droid_sft_dreamzero_14b.yaml`` 中设置 ``actor.model.model_path``\ （AgiBot 数据换为 ``./DreamZero-AgiBot``）：

            .. code-block:: yaml

               actor:
                 model:
                   model_path: ./DreamZero-DROID
                   tokenizer_path: google/umt5-xxl
                   embodiment_tag: oxe_droid

         .. tab-item:: 从头训练（WAN2.2 5B 冷启动）

            下载 `Wan-AI/Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_、`Wan2.1 CLIP <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P>`_ 权重和 `google/umt5-xxl <https://huggingface.co/google/umt5-xxl>`_：

            .. code-block:: bash

               hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
               hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
               hf download google/umt5-xxl --local-dir ./umt5-xxl

            在 ``libero_sft_dreamzero_5b.yaml`` 中设置 ``model_path: null`` 和各 ``*_pretrained_path``：

            .. code-block:: yaml

               actor:
                 model:
                   model_path: null
                   tokenizer_path: ./umt5-xxl
                   diffusion_model_pretrained_path: ./Wan2.2-TI2V-5B
                   image_encoder_pretrained_path: ./Wan2.1-CLIP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
                   text_encoder_pretrained_path: ./Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
                   vae_pretrained_path: ./Wan2.2-TI2V-5B/Wan2.2_VAE.pth
                   metadata_json_path: ./libero_metadata.json
                   embodiment_tag: libero_sim

      **3. 下载 LeRobot 数据集并生成 ``metadata.json``**\ （随后在配置中填写 ``data.train_data_paths`` 和 ``actor.model.metadata_json_path``）：

      .. tab-set::

         .. tab-item:: LIBERO

            .. code-block:: bash

               hf download physical-intelligence/libero --repo-type dataset --local-dir ./libero
               python toolkits/lerobot/generate_dreamzero_metadata.py \
                 --preset libero_sim \
                 --dataset-root ./libero \
                 --output-metadata ./libero_metadata.json

         .. tab-item:: DROID

            .. code-block:: bash

               hf download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./droid
               python toolkits/lerobot/generate_dreamzero_metadata.py \
                 --preset oxe_droid \
                 --dataset-root ./droid \
                 --output-metadata ./droid_metadata.json \
                 --merge

         .. tab-item:: Franka PnP

            .. code-block:: bash

               hf download RLinf/dreamzero-franka-pnp --repo-type dataset --local-dir ./franka_pnp
               python toolkits/lerobot/generate_dreamzero_metadata.py \
                 --preset franka_pnp \
                 --dataset-root ./franka_pnp \
                 --output-metadata ./franka_pnp_metadata.json

      **4. 启动训练**\ （在仓库根目录执行；日志写入 ``logs/<timestamp>-<config_name>/run_embodiment.log``，断点续训设置 ``runner.resume_dir``）：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b
         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b
         bash examples/sft/run_vla_sft.sh droid_sft_dreamzero_14b
         bash examples/sft/run_vla_sft.sh libero_franka_mix_sft_dreamzero_5b

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      华为昇腾 CANN 支持 DreamZero SFT，沿用相同的 checkpoint、数据集和配置；CI 在两张昇腾 910B NPU 上以 PyTorch 与 ``torch-npu`` 2.6.0 运行 WAN2.2 5B SFT。

      使用昇腾 LIBERO 容器，或已安装 CANN 与 NPU 驱动的宿主机：

      .. include:: _ascend_libero.rst

      克隆 RLinf，安装昇腾 DreamZero 环境并克隆 DreamZero（国内下载可添加 ``--use-mirror``；安装器会安装匹配的 ``torch-npu`` 包并跳过 CUDA flash-attention）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform ascend embodied --model dreamzero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero
         export DREAMZERO_PATH="$(realpath ../dreamzero)"

      按 NVIDIA CUDA 标签页准备模型、数据和 ``metadata.json``。在昇腾上，如果 ``torchcodec`` 的 wheel 无法加载，安装器会将其移除，因此在 SFT 配置中设置 ``data.video_backend: decord``。在 64 GB NPU 上调小 ``actor.micro_batch_size``\ （默认按 80 GB GPU 配置），并保证 ``actor.global_batch_size`` 能被 ``micro_batch_size`` 与 NPU 数量之积整除。在仓库根目录启动 SFT：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

      actor 或 rollout worker 位于 NPU 上时，RLinf 会在构建 DreamZero 前导入 ``torch_npu.contrib.transfer_to_npu``，让 DreamZero 源码中的 CUDA 调用在 NPU 上执行，同时关闭 TorchDynamo，使所有 ``torch.compile`` 以 eager 模式运行。这两项改动作用于整个 worker 进程：该进程中其他调用 ``torch.cuda``、``torch.compile`` 或 ``torch.jit.script`` 的代码也会受到同样的影响。

.. dropdown:: 进阶配置（可选）

   配置文件由 Hydra 管理（入口脚本 ``examples/sft/train_vla_sft.py``），已内置采样、时间对齐、混合精度和 FSDP2 默认项。

   **数据相关配置**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 字段
        - 含义与作用
      * - ``train_data_paths``
        - 单数据集：LeRobot 根路径或 HF ``repo_id``。**混合训练**：YAML 列表，每项含 ``dataset_path``\ （或路径列表）、``weight``、``embodiment_tag``、``metadata_json_path`` 等。可选 ``distribute_weights: true`` 在单条 spec 含多路径时按 episode 长度分配权重。
      * - ``lazy_load``
        - 是否懒加载 mp4 视频。``multi_anchor`` 采样模式下必须设为 ``True``\ （否则无法按锚点随机取帧）。
      * - ``sampling_mode``
        - ``multi_anchor``\ （默认，推荐）：在同一语言片段内按多个时间锚点采样，宏观时间块数由 ``max_chunk_size`` 决定；``fixed_window`` 为连续固定窗口。
      * - ``video_backend``
        - LeRobot 视频解码后端：``pyav``、``torchcodec`` 或 ``decord``。推荐使用 ``torchcodec``；在昇腾等无法加载 ``torchcodec`` 的环境中使用 ``decord``。
      * - ``video_tolerance_s``
        - 视频时间戳与目标帧时间的容差（秒）。
      * - ``parquet_cache_size``
        - Parquet episode 缓存上限（episode 数），影响内存与 IO。
      * - ``num_workers`` / ``prefetch_factor``
        - DataLoader 并行与预取，影响数据吞吐。

   **时间对齐要点（数据采样 vs 模型块）**

   - 宏观时间块数来自 ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size``\ （常见为 4；官方 Groot DROID 配方可为 5）。
   - ``actor.model.action_horizon`` 是 DreamTransform / WAN 每个块内的动作步数（LIBERO 常用 16，DROID 常用 24），不是数据集宏观步长。
   - ``multi_anchor`` 下，数据集侧动作序列长度约为 ``action_horizon * max_chunk_size``\ （如 LIBERO 64、DROID 96）。
   - 视频时间维在预设里配置 ``action_head_cfg.config.num_frames``\ （DreamZero 默认 33，对应 ``8 * max_chunk_size + 1``）；未设置时自动推导。

   **模型与训练相关配置**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 字段
        - 含义与作用
      * - ``model_type``
        - 固定为 ``dreamzero``。
      * - ``model_path``
        - 完整 checkpoint 目录；非 ``null`` 时从 ``config.json`` 读架构并加载权重。``null`` 时使用 YAML / 预设 + 各 ``*_pretrained_path`` 冷启动。
      * - ``tokenizer_path``
        - UMT5 分词器路径（训练与 collate 均需）。
      * - ``diffusion_model_pretrained_path``
        - 因果 DiT 预训练权重；冷启动必填。
      * - ``image_encoder_pretrained_path``
        - WAN 图像编码器；WAN2.2 需指向 WAN2.1 CLIP 权重。
      * - ``text_encoder_pretrained_path``
        - T5 文本编码器权重。
      * - ``vae_pretrained_path``
        - VAE 权重；WAN2.2 使用 38 通道 WAN2.2 VAE。
      * - ``metadata_json_path``
        - 数据集 ``metadata.json``；未设置则回退到 ``model_path/experiment_cfg/metadata.json``。
      * - ``embodiment_tag``
        - 选择数据变换与 collate 模板：``libero_sim``、``oxe_droid``、``franka_pnp``。混合训练时各子项在 ``train_data_paths`` 中单独指定，``actor.model.embodiment_tag`` 仍须设为主数据来源。
      * - ``action_horizon`` / ``state_horizon`` / ``num_action_per_block``
        - 每个 WAN 时间块内的动作步数（LIBERO 16，DROID 24）、每个样本状态行数（通常为 1）和 DiT ``num_action_per_block``。
      * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
        - DreamTransform 填充与文本序列上限。
      * - ``target_video_height`` / ``target_video_width``
        - WAN 策略头在 **多视角拼接后** 的目标分辨率（5B 预设如 176×320；LIBERO 常用 160×320）。仅作用于模型内部 resize，**不要** 用于单视角 resize。
      * - ``droid_view_height`` / ``droid_view_width``
        - 可选的 DROID 各视角 resize 覆盖。
      * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
        - 是否使用相对动作及作用维度；DROID 常对 ``joint_position`` 等开启 ``relative_action: True``。

   - ``precision``：推荐保持 ``precision: fp32`` 配合 ``actor.fsdp_config.mixed_precision``\ （``param_dtype``、``reduce_dtype``、``buffer_dtype: bf16``），使优化器状态和主权重保持 FP32，前向/反向在 BF16 计算。
   - ``is_lora``：是否 LoRA 微调（DreamZero SFT 示例多为全参 ``False``）。
   - ``actor.micro_batch_size`` / ``actor.global_batch_size``、``actor.optim.*``、``actor.fsdp_config``：批大小、学习率调度和 FSDP2 分片/重计算设置。

   **配置示例对照**

   .. code:: yaml

      # 数据（单数据集）
      data:
        train_data_paths: ./libero
        lazy_load: True
        sampling_mode: multi_anchor
        video_backend: torchcodec
        num_workers: 8

      # 数据（混合，见 libero_franka_mix_sft_dreamzero_5b.yaml）
      data:
        train_data_paths:
          - dataset_path: ./libero
            weight: 4
            embodiment_tag: libero_sim
            metadata_json_path: ./libero_metadata.json
          - dataset_path: ./franka_pnp
            weight: 1
            embodiment_tag: franka_pnp
            metadata_json_path: ./franka_pnp_metadata.json

      # 模型（从 checkpoint 继续）
      actor:
        model:
          model_path: ./DreamZero-DROID
          tokenizer_path: ./umt5-xxl
          embodiment_tag: oxe_droid
          action_horizon: 24
          metadata_json_path: ./droid_metadata.json


独立评测
----------------------------------------

独立的仿真或真机评测由统一的 Evaluation 章节负责。本 SFT 页面只保留 DreamZero
特有的衔接点。

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 目标
     - 从这里开始
     - DreamZero 专属字段
   * - LIBERO 仿真
     - :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>` 和 ``evaluations/libero/libero_spatial_dreamzero_eval.yaml``
     - 将 ``runner.ckpt_path`` 指向 ``full_weights.pt``；保持 ``actor.model.metadata_json_path`` 与 ``actor.model.embodiment_tag: libero_sim`` 和 SFT 一致。
   * - Franka 部署 / 评测
     - :doc:`真机评测指南 <../../evaluations/guides/realworld>` 和 ``evaluations/realworld/realworld_pnp_eval_dreamzero.yaml``
     - 设置完整 DreamZero checkpoint 目录、``embodiment_tag: franka_pnp``、机器人 IP、相机序列号和任务位姿字段。

命令格式、Hydra 覆盖、日志与结果文件见 :doc:`Evaluation CLI 参考 <../../evaluations/reference/cli>`
和 :doc:`Evaluation 结果参考 <../../evaluations/reference/results>`。如果 SFT checkpoint 仍是
``.distcp`` 分片格式，请先按 :doc:`checkpoint 转换指南 <../../guides/convertor>` 转换。

.. note::

   DreamZero rollout 评测要求 ``max_steps_per_rollout_epoch`` 能被
   ``actor.model.num_action_chunks`` 整除。

**预训练 checkpoint 评测结果**

`RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_ 在 LIBERO Spatial 上的评测结果（``num_trajectory=512``）：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 训练步数
     - success_once
   * - 3000
     - 7.81%
   * - 6000
     - 66.41%
   * - 9000
     - 89.06%
   * - 12000
     - 88.48%
   * - 15000
     - 66.60%
   * - 18000
     - 96.68%
   * - 21000
     - 90.43%

可视化与结果
----------------------------------------

1. 查看 ``run_embodiment.log``：``time/step`` 是否稳定；``train/loss``、``train/action_loss``、``train/dynamics_loss`` 是否合理。

2. TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. 开跑后尽早检查：

   - ``images`` / ``state`` / ``action`` 的 shape、dtype、数值范围
   - ``state_mask`` / ``action_mask`` / ``text_attention_mask`` 有效比例
   - WAN2.2 时确认输入分辨率与 ``frame_seqlen`` 与 ``config.json`` 或预设一致


扩展 DreamZero 到新的 ``embodiment_tag``
-------------------------------------------

当要在 **新的机器人 或 新 LeRobot 数据集** 上训练 DreamZero SFT 时，需要新增一个 ``embodiment_tag``，并在 RLinf 中注册对应的数据变换与元数据生成逻辑。建议以现有实现为模板对照修改：

- ``rlinf/data/datasets/dreamzero/data_transforms/libero_sim.py`` （双视角、简单 state/action 列）
- ``rlinf/data/datasets/dreamzero/data_transforms/franka_pnp.py`` （双视角，继承 ``libero_sim``，自定义 ``num_frames`` 等）
- ``rlinf/data/datasets/dreamzero/data_transforms/oxe_droid.py`` （三视角， ``meta/modality.json`` 切片）

整体数据流：

.. code:: text

   LeRobot 数据集
        → DreamZeroLeRobotDataset（按 transform 链里的 keys 读 parquet/mp4）
        → ComposedModalityTransform + DreamTransform（归一化、多视角拼接、tokenize）
        → DreamZeroCollator → 训练

步骤 1：实现 embodiment 变换模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``rlinf/data/datasets/dreamzero/data_transforms/`` 下新建 ``your_tag.py``，实现 ``DreamZeroEmbodimentTransform`` 协议（见 ``base.py``），至少包含：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 成员 / 方法
     - 说明
   * - ``TAG``
     - 字符串标识，与配置里 ``actor.model.embodiment_tag``、``metadata.json`` 顶层键完全一致。
   * - ``DEFAULT_TAG_MAPPING``
     - ``{TAG: <int>}``，映射到 WAN 动作头里的 **embodiment projector ID** 。继续微调已有 DreamZero 权重时，ID 须出现在 checkpoint ``config.json`` 的 ``action_loss_embodiment_ids`` 中（ 如 5B 预设含 17、21、26）； **全新 ID** 需接受 projector 随机初始化或改模型配置。
   * - ``DEFAULT_ACTION_HORIZON``
     - 该 embodiment 默认每块动作步数（LIBERO 16、DROID 24），与 ``actor.model.action_horizon`` 一致。
   * - ``get_modality_config()``
     - 返回 ``video`` / ``state`` / ``action`` / ``language`` 的 ``ModalityConfig`` （ ``delta_indices``、 ``modality_keys``）。 ``language`` 的 key 必须在数据集中存在（任务文本列）。视频/动作 ``delta_indices`` 需与 Groot 配方一致（现实现多为 video ``range(25)``、action ``range(24)``），否则 ``multi_anchor`` 时间对齐会错。
   * - ``get_transform(...)``
     - 组装 ``Video*`` → ``StateAction*`` → ``ConcatTransform`` → ``DreamTransform`` 链；``DreamTransform`` 使用 RLinf 子类（``dream_transform.py``），会从 registry 调用多视角拼接。
   * - ``format_training_prompt(instruction)``
     - 为多视角布局生成 T5 文本前缀（须与 Groot 训练模板语义一致）。
   * - ``concat_multiview_video(images)``
     - 将 ``(v, t, c, h, w)`` 拼成 ``(1, t, c, H, W)``；布局须与 ``format_training_prompt`` 描述一致。
   * - ``ROLLOUT_OBS_LAYOUT``
     - ``RolloutObsLayout`` 实例：将 RLinf rollout 的 ``main_images`` / ``wrist_images`` / ``states`` / ``task_descriptions`` 映射到上述 ``modality_keys``。推理时由 ``convert_rollout_env_obs(embodiment_tag, env_obs)`` 调用（见 ``data_transforms/__init__.py``）。

``modality_keys`` 命名约定（与 ``DreamZeroLeRobotDataset`` 解析逻辑挂钩）：

- 视频：``video.short_name`` （如 ``video.image``），短名通过 ``meta/modality.json`` 的 ``original_key`` 或 ``info.json`` 的 ``observation.images.*`` / 裸列名解析到真实特征列。
- 状态/动作：``state.name``、``action.name``；有 ``meta/modality.json`` 时用 ``start``/``end`` 切片；否则回退到 ``observation.state`` / ``action`` 整列或启发式切片（见 ``lerobot_dataset.py`` 中 ``_build_component_sources``）。
- 训练 YAML 里的 ``video.*`` / ``state.*`` / ``action.*`` 必须与 transform 里 ``ConcatTransform`` 的 ``*_concat_order`` 一致。

步骤 2：注册到 RLinf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 在 ``rlinf/data/datasets/dreamzero/data_transforms/embodiment_tag.py`` 的 ``EmbodimentTag`` 枚举中增加成员（值等于 ``TAG`` 字符串）。
2. 编辑 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``：

   - ``from ...your_tag import YourEmbodimentDataTransform``
   - 在 ``_EMBODIMENT_REGISTRY`` 中加入 ``YourEmbodimentDataTransform.TAG: YourEmbodimentDataTransform``

无需手写 Groot patch：``get_model()`` 会将 ``groot.vla.data.schema.embodiment_tags.EmbodimentTag`` 替换为上述 RLinf 枚举。

未注册时，``build_dreamzero_composed_transform`` 会报错并列出已有 tag。

步骤 3：生成 ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为新数据集计算归一化统计，输出键名必须等于 ``TAG``：

方式 A（推荐）：在 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 的 ``PRESETS`` 中增加一项（字段参考 ``libero_sim`` / ``oxe_droid``：``state_key``、``action_key``、``video_keys``、``use_modality_json``），然后：

.. code:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset YOUR_TAG \
     --dataset-root /path/to/lerobot_dataset \
     --output-metadata /path/to/metadata.json

方式 B：不改脚本，用手动参数（``--embodiment-tag``、``--state-key``、``--action-key``、``--video-keys``、``--use-modality-json``）。

在训练配置中设置 ``actor.model.metadata_json_path`` （或放到 ``model_path/experiment_cfg/metadata.json``）。

步骤 4：编写 / 调整训练配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

复制 ``libero_sft_dreamzero_14b.yaml``、``libero_sft_dreamzero_5b.yaml`` 或 ``droid_sft_dreamzero_14b.yaml``，更新数据路径、``embodiment_tag``、``metadata_json_path`` 和 ``action_horizon``\ （冷启动 WAN 时，还需在 ``examples/sft/config/model/dreamzero_5b.yaml`` 或 ``dreamzero_14b.yaml`` 的 ``action_head_cfg.config.action_loss_embodiment_ids`` 中加入新 ID）：

.. dropdown:: 进阶配置（可选）

   .. code:: yaml

      data:
        train_data_paths: /path/to/your_lerobot
        lazy_load: True              # multi_anchor 必须为 True（mp4 数据）
        sampling_mode: multi_anchor

      actor:
        model:
          embodiment_tag: "YOUR_TAG"
          metadata_json_path: /path/to/metadata.json
          action_horizon: 16         # 与 DEFAULT_ACTION_HORIZON 一致
          # 从 checkpoint 继续时核对 action_loss_embodiment_ids 是否包含你的 projector ID
          target_video_height: 160
          target_video_width: 320
          relative_action: False
          relative_action_keys: []

步骤 5：验证（短跑 + 数据检查）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 单独跑 metadata 脚本，确认 ``metadata.json`` 中对应 tag 条目的 ``statistics`` / ``modalities`` 维度与 parquet 一致。
2. 用 50–200 step 启动 SFT，检查日志无 ``Could not map transform video keys``、``embodiment_tag not found in metadata`` 等错误。
3. 在 TensorBoard / 日志中确认 ``train/action_loss`` 有限；检查 batch 内 ``images`` 拼接形状、``embodiment_id`` 与 ``DEFAULT_TAG_MAPPING`` 一致。

**易错细节 checklist**

- ``embodiment_tag`` 字符串在四处置一致：``embodiment_tag.py`` 枚举成员值、``TAG``、配置 / ``train_data_paths`` 子项、``metadata.json`` 顶层键。
- ``multi_anchor`` + mp4 数据：必须将 ``data.lazy_load`` 设为 ``True``。
- ``action_horizon`` × ``max_chunk_size`` 决定数据集动作长度；勿只改其一。
- 多视角拼接顺序与 prompt 文案不一致会导致训练信号错乱。
- 继续微调官方权重时，随意改 ``DEFAULT_TAG_MAPPING`` 的整数 ID 会导致 projector 对不上。
- 视频 resize：单视角 ``VideoResize`` 写在各 embodiment 的 ``data_transforms`` 代码中（如 ``libero_sim``、``franka_pnp`` 均为 256×256）；``target_video_height/width`` 仅用于 WAN 在多视角拼接 **之后** 的模型内 resize，二者勿混用。 **混合数据集训练** 须保证各子数据集经 ``DreamTransform`` 拼接后输出相同 ``images`` 空间形状（H×W），否则 collate 无法组 batch；若各 embodiment 拼接布局不同（如 ``oxe_droid`` 为 2×2 网格）或单视角默认尺寸不一致，请在对应 transform 模块中手动对齐 ``VideoResize`` 参数。
- 推理 / 评测：``examples/embodiment/config/`` 下的 DreamZero 评测配置 中同样需要正确的 ``embodiment_tag``。

若仅推理、不改 RLinf 代码，且 Groot/DreamZero 上游已支持该 tag，有时只需准备 ``metadata.json`` 与评测配置；**SFT 新数据** 则须完成上述枚举成员、registry 注册与 transform 实现（``get_model`` 会自动 patch Groot ``EmbodimentTag``）。


常见问题
----------------------------------------

1. **找不到权重（No safetensors weights）**

   - 检查 ``model_path`` 下是否存在 ``model.safetensors`` 或分片索引
   - 冷启动时确认各 ``*_pretrained_path`` 可访问且与架构匹配

2. **WAN2.2 维度不匹配**

   - 核对有效配置（``model_path/config.json`` 或 ``dreamzero_5b`` 预设）中 ``diffusion_model_cfg`` 是否为 ti2v、``in_dim/out_dim=48``、``vae_cfg`` 为 ``WanVideoVAE38``
   - 图像编码器须使用 WAN2.1 CLIP 路径

3. **metadata.json 找不到**

   - 运行 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 并设置 ``metadata_json_path``
   - 确认 JSON 内包含与 ``embodiment_tag`` 同名的键

4. **action_loss 异常偏高**

   - 检查归一化统计是否与当前数据集一致
   - 检查 ``relative_action`` 与数据是否冲突
   - 核对 ``action_horizon``、``max_chunk_size`` 与 ``sampling_mode`` 是否匹配

5. **DROID 视频尺寸错误**

   - 勿将 ``target_video_height/width`` 用于 data transform 的单视角 resize；DROID 视角尺寸在 ``oxe_droid`` transform 代码中调整

6. **multi_anchor 报错要求 lazy_load**

   - 设置 ``data.lazy_load: True``

7. ``AttributeError: GR1_UNIFIED_SEGMENTATION`` 或未知 ``EmbodimentTag``

   - 数据 transform 链须使用 ``dream_transform.DreamTransform`` (RLinf 子类)，勿直接实例化 Groot 基类
   - 新 tag 须在 ``embodiment_tag.py`` 与 ``_EMBODIMENT_REGISTRY`` 注册；训练经 ``get_model()`` 加载模型时会 patch Groot 枚举


实践建议
----------------------------------------

- 追求稳定收敛时，优先从已发布的 DreamZero 权重继续 SFT（设置 ``model_path``）。
- 全量适配 WAN2.2 可冷启动，但需更大数据与更长训练；改配置后先用 50–200 step 试跑验证 shape 与 loss。
- 每次更换数据集或 ``embodiment_tag``，务必重新生成或更新 ``metadata.json``。
- LIBERO 与 DROID 的 ``action_horizon``、 ``embodiment_tag``、多视角拼接逻辑不同，不要混用配置模板。


训练加速
----------------------------------------

RLinf 团队对 DreamZero 的训练管线进行了深度的系统级重构与加速。相比 DreamZero 官方提供的基线训练脚本，RLinf **实现了近 4 倍的训练吞吐加速**，同时保持甚至优化了收敛效果。


端到端性能实测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下所有测试均在 Droid 数据集（单样本含左、右、腕部三个视角，视频规格 33 frames × 480 × 640）上，使用 8×H100 GPU 完成。

**DreamZero-14B**

在 14B 大模型上，由于显存压力巨大，官方基线通常被迫采用 DeepSpeed ZeRO-offload 方案，这导致了严重的计算/通信浪费与 CPU 换入换出开销。我们通过工程优化，以 FSDP2 full_shard 替代 DeepSpeed ZeRO-offload 方案，并进一步结合了计算图优化（算子融合与 CUDA Graph）。

.. list-table::
   :header-rows: 1
   :widths: 38 22 22 18

   * - 实验配置
     - 迭代耗时 (Step Time)
     - 训练吞吐 (Samples/sec/GPU)
     - 性能收益 (vs. 基线)
   * - DeepSpeed ZeRO2 + Offload（官方版本）
     - 18.0 s
     - 0.055
     - 基线
   * - FSDP2 Base（原生支持）
     - 9.0 s
     - 0.111
     - +100%（2.0x）
   * - **RLinf 深度优化版**
     - **6.7 s**
     - **0.150**
     - **+170%（2.7x）**

14B 模型使用 MBS=1 和 GBS=8 进行测试。RLinf 相比原生 DeepSpeed 方案实现了 **2.7 倍**\ 的加速；即便相比于未经优化的 FSDP2，吞吐量也进一步提升了 **35%**。

**DreamZero-5B**

对于 5B 中等规模模型，RLinf 的优势在于能够通过高效率的重计算逻辑稳定开启更大的 Microbatch Size，并配合计算图调优，彻底释放 GPU 算力。

.. list-table::
   :header-rows: 1
   :widths: 38 22 22 18

   * - 实验配置
     - 迭代耗时 (Step Time)
     - 训练吞吐 (Samples/sec/GPU)
     - 性能收益 (vs. 基线)
   * - DeepSpeed ZeRO2 + Offload（官方版本，mbs=32 × 8 GPU）
     - 30.0 s
     - 1.10
     - 基线
   * - FSDP2 Base（mbs=1 × 8 GPU）
     - 1.8 s
     - 0.56
     - -49%（受限于小 MBS 算子效率低、CPU 开销显著、FSDP2 通信无法掩盖）
   * - **RLinf 深度优化版（mbs=32 + Recompute × 8 GPU）**
     - **7.2 s**
     - **4.44**
     - **+300%（4.0x）**

5B 模型使用 GBS=256 测试。FSDP2 Base 版本由于 PyTorch 的一些限制不能开大 MBS，导致吞吐受限；RLinf 解决了这些问题并取得了显著的吞吐增长。训练吞吐从官方代码的 1.1 samples/sec/gpu 飙升至 4.44 samples/sec/gpu，实现了约 4 倍的训练加速。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0acctime.jpg
   :align: center
   :width: 45%

   DreamZero 5B 与 14B 模型的加速效果对比

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0accthpt.jpg
   :align: center
   :width: 45%

   DreamZero 5B 与 14B 模型的吞吐提升对比

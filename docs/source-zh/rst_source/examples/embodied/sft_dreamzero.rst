DreamZero 监督微调
==================

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

**支持范围**

- **数据集：** LIBERO（``libero_sim``）、OXE DROID（``oxe_droid``）、Franka pick-and-place（``franka_pnp``）；支持跨 embodiment 的 **mixture SFT** （见 ``libero_franka_mix_sft_dreamzero_5b.yaml``）。
- **骨干网络：** WAN2.1（如 DreamZero-DROID 14B）、WAN2.2（如 Wan2.2-TI2V-5B 冷启动）。

.. _sft-dreamzero-hardware:

安装与运行
----------

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

      **1. 安装环境**

      安装 RLinf、DreamZero 与 LIBERO 评测依赖（仅跑 SFT 时可省略 ``--env libero``），在同级目录克隆 DreamZero 并导出 ``DREAMZERO_PATH``\ （``examples/sft/run_vla_sft.sh`` 据此导入外部包）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dreamzero --env libero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero
         export DREAMZERO_PATH="$(realpath ../dreamzero)"

      **2. 下载模型与数据**

      下载 DreamZero-DROID 14B 基础权重与 LIBERO LeRobot 数据集，并生成 ``libero_metadata.json``：

      .. code-block:: bash

         hf download GEAR-Dreams/DreamZero-DROID --local-dir ./DreamZero-DROID
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./libero
         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset libero_sim \
           --dataset-root ./libero \
           --output-metadata ./libero_metadata.json

      .. dropdown:: 其他数据集与冷启动骨干权重（DROID、Franka PnP、WAN2.2 5B）

         **数据集与 metadata 预设**\ （``hf download`` 需加 ``--repo-type dataset``，随后运行 ``toolkits/lerobot/generate_dreamzero_metadata.py``）：

         .. list-table::
            :header-rows: 1
            :widths: 16 36 18 30

            * - 数据集
              - Hugging Face 仓库
              - Metadata 预设
              - 附加参数 / 输出路径
            * - LIBERO
              - ``physical-intelligence/libero``
              - ``libero_sim``
              - ``--output-metadata ./libero_metadata.json``
            * - DROID
              - ``GEAR-Dreams/DreamZero-DROID-Data``
              - ``oxe_droid``
              - ``--output-metadata ./droid_metadata.json --merge``
            * - Franka PnP
              - ``RLinf/dreamzero-franka-pnp``
              - ``franka_pnp``
              - ``--output-metadata ./franka_pnp_metadata.json``

         **从头训练（WAN2.2 5B 冷启动）** —— 下载 `Wan-AI/Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_、`Wan2.1 CLIP <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P>`_ 权重和 `google/umt5-xxl <https://huggingface.co/google/umt5-xxl>`_：

         .. code-block:: bash

            hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
            hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
            hf download google/umt5-xxl --local-dir ./umt5-xxl

         在 ``examples/sft/config/libero_sft_dreamzero_5b.yaml`` 中设置 ``model_path: null`` 与各 ``*_pretrained_path``：

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

      **3. 启动训练**

      在 ``examples/sft/config/libero_sft_dreamzero_14b.yaml`` 中填入数据集、模型权重与 metadata 路径：

      .. code-block:: yaml

         data:
           train_data_paths: ./libero
         actor:
           model:
             model_path: ./DreamZero-DROID
             metadata_json_path: ./libero_metadata.json

      在仓库根目录启动 SFT（日志写入 ``logs/<timestamp>-<config_name>/run_embodiment.log``，断点续训设置 ``runner.resume_dir``）：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

      其他配方替换配置后缀即可：``libero_sft_dreamzero_5b``、``droid_sft_dreamzero_14b`` 或 ``libero_franka_mix_sft_dreamzero_5b``。

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      华为昇腾 CANN 支持 DreamZero SFT，沿用相同的 checkpoint、数据集和配置；CI 在两张昇腾 910B NPU 上以 PyTorch 与 ``torch-npu`` 2.6.0 运行 WAN2.2 5B SFT。

      **1. 安装环境**

      使用昇腾 LIBERO 容器，或已安装 CANN 与 NPU 驱动的宿主机：

      .. include:: _ascend_libero.rst

      克隆 RLinf，安装昇腾 DreamZero 环境并克隆 DreamZero（安装器会安装匹配的 ``torch-npu`` 包并跳过 CUDA flash-attention）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform ascend embodied --model dreamzero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero
         export DREAMZERO_PATH="$(realpath ../dreamzero)"

      **2. 下载模型与数据**

      按 **NVIDIA CUDA** 标签页下载模型权重、LeRobot 数据集并生成 ``metadata.json``\ （WAN2.2 5B 冷启动见该标签页内的冷启动折叠面板）。

      **3. 启动训练**

      在昇腾上，如果 ``torchcodec`` 的 wheel 无法加载，安装器会将其移除，因此在 SFT 配置中设置 ``data.video_backend: decord``。在 64 GB NPU 上调小 ``actor.micro_batch_size``\ （默认按 80 GB GPU 配置），并保证 ``actor.global_batch_size`` 能被 ``micro_batch_size`` 与 NPU 数量之积整除。在仓库根目录启动 SFT：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

      actor 或 rollout worker 运行在 NPU 上时，RLinf 会自动将 DreamZero 外部包中的 CUDA 调用重定向到 NPU，并在该 worker 进程内以 eager 模式运行编译算子。

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
   - ``actor.model.action_horizon`` 是 WAN 每个时间块内的动作步数（LIBERO 常用 16，DROID 常用 24），不是数据集宏观步长。
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
        - 数据变换填充与文本序列上限。
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

   .. code-block:: yaml

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

.. dropdown:: 实践建议

   - 追求稳定收敛时，优先从已发布的 DreamZero 权重继续 SFT（设置 ``model_path``）。
   - 全量适配 WAN2.2 可冷启动，但需更大数据与更长训练；改配置后先用 50–200 step 试跑验证 shape 与 loss。
   - 每次更换数据集或 ``embodiment_tag``，务必重新生成或更新 ``metadata.json``。
   - LIBERO 与 DROID 的 ``action_horizon``、``embodiment_tag``、多视角拼接逻辑不同，不要混用配置模板。

.. dropdown:: 独立评测与 Franka 真机部署

   独立的仿真或真机评测由统一的 Evaluation 章节负责，这里只列出 DreamZero 特有的衔接点：

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

   命令格式、Hydra 覆盖、日志与结果文件见 :doc:`Evaluation CLI 参考 <../../evaluations/reference/cli>` 和 :doc:`Evaluation 结果参考 <../../evaluations/reference/results>`。如果 SFT checkpoint 仍是 ``.distcp`` 分片格式，请先按 :doc:`checkpoint 转换指南 <../../guides/convertor>` 转换。

   .. note::

      DreamZero rollout 评测要求 ``max_steps_per_rollout_epoch`` 能被 ``actor.model.num_action_chunks`` 整除。

.. dropdown:: 扩展 DreamZero 到新的具身标签

   在新的机器人或 LeRobot 数据集上训练 DreamZero SFT 时，按以下步骤新增并注册 ``embodiment_tag``：

   1. **实现并注册具身数据变换**：参考内置的 ``libero_sim``\ （双视角）、``franka_pnp``\ （双视角，自定义 ``num_frames``）或 ``oxe_droid``\ （三视角，读取 ``meta/modality.json`` 切片）实现新的变换模块。在其中定义标签字符串、WAN 动作头 projector ID（续训官方权重时须包含在 checkpoint ``config.json`` 的 ``action_loss_embodiment_ids`` 中，5B 预设含 17、21、26）、默认 ``action_horizon``、模态键名（``video.<short_name>``、``state.<name>``、``action.<name>``、``language``）、与 T5 提示词布局一致的多视角拼接逻辑及推理观测映射，然后在 RLinf 的 DreamZero 具身枚举与注册表中登记该标签。
   2. **生成 ``metadata.json``**：在 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 中新增预设（或直接传入 ``--embodiment-tag``、``--state-key``、``--action-key``、``--video-keys``、``--use-modality-json``）：

      .. code-block:: bash

         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset YOUR_TAG \
           --dataset-root ./your_lerobot \
           --output-metadata ./your_metadata.json

   3. **编写训练配置**：复制 ``libero_sft_dreamzero_14b.yaml``、``libero_sft_dreamzero_5b.yaml`` 或 ``droid_sft_dreamzero_14b.yaml``，更新数据路径、``embodiment_tag``、``metadata_json_path`` 和 ``action_horizon``\ （冷启动 WAN 时，还需在 ``examples/sft/config/model/dreamzero_5b.yaml`` 或 ``dreamzero_14b.yaml`` 的 ``action_head_cfg.config.action_loss_embodiment_ids`` 中加入新 ID）：

      .. code-block:: yaml

         data:
           train_data_paths: ./your_lerobot
           lazy_load: True              # multi_anchor 必须为 True（mp4 数据）
           sampling_mode: multi_anchor

         actor:
           model:
             embodiment_tag: "YOUR_TAG"
             metadata_json_path: ./your_metadata.json
             action_horizon: 16
             target_video_height: 160
             target_video_width: 320
             relative_action: False
             relative_action_keys: []

   4. **短跑验证（50–200 step）**：确认 ``metadata.json`` 中对应 tag 的统计维度与 parquet 一致，检查 ``train/action_loss`` 数值有限，并核对 batch 内拼接后的 ``images`` 形状与 projector ID。

   **易错细节**

   - ``embodiment_tag`` 字符串须在四处完全一致：具身枚举值、变换模块标签、YAML 配置 / ``train_data_paths`` 子项、``metadata.json`` 顶层键。
   - ``sampling_mode: multi_anchor`` 搭配 mp4 数据时，必须设置 ``data.lazy_load: True``。
   - 数据集动作长度由 ``action_horizon × max_chunk_size`` 决定，调整时需保持同步。
   - 多视角拼接顺序必须与提示词文案一致；混合数据集训练时，各子数据集拼接后的 ``images`` 空间形状（H×W）也必须一致。
   - 续训官方权重时不要随意改动 projector 整数 ID。
   - 推理或评测时，在 ``examples/embodiment/config/`` 或 ``evaluations/`` 下的 DreamZero 配置中同步设置正确的 ``embodiment_tag``。

可视化与结果
----------------------------------------

1. 查看 ``run_embodiment.log``：``time/step`` 是否稳定；``train/loss``、``train/action_loss``、``train/dynamics_loss`` 是否合理。

2. 启动 TensorBoard：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

3. 开跑后尽早检查：

   - ``images`` / ``state`` / ``action`` 的 shape、dtype、数值范围
   - ``state_mask`` / ``action_mask`` / ``text_attention_mask`` 有效比例
   - WAN2.2 时确认输入分辨率与 ``frame_seqlen`` 与 ``config.json`` 或预设一致

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

.. dropdown:: 训练加速实测（8×H100）

   RLinf 以 FSDP2 ``full_shard`` 替代 DeepSpeed ZeRO-offload，并结合算子融合、CUDA Graph 与重计算调优。在 **Droid 数据集**\ （左、右、腕部三视角，33 frames × 480 × 640）的 8×H100 实测中，相比官方基线最高取得 **约 4 倍** 训练吞吐加速。

   **DreamZero-14B**\ （MBS=1、GBS=8 —— 相比 DeepSpeed 基线提速 **2.7 倍**，相比原生 FSDP2 吞吐再提升 **35%**）：

   .. list-table::
      :header-rows: 1
      :widths: 38 22 22 18

      * - 实验配置
        - 迭代耗时（Step Time）
        - 训练吞吐（Samples/sec/GPU）
        - 性能收益（vs. 基线）
      * - DeepSpeed ZeRO2 + Offload（官方版本）
        - 18.0 s
        - 0.055
        - 基线
      * - FSDP2 Base（原生支持）
        - 9.0 s
        - 0.111
        - +100%（2.0×）
      * - **RLinf 深度优化版**
        - **6.7 s**
        - **0.150**
        - **+170%（2.7×）**

   **DreamZero-5B**\ （GBS=256 —— 吞吐从 1.10 提升至 4.44 samples/sec/GPU，实现 **4.0 倍** 加速）：

   .. list-table::
      :header-rows: 1
      :widths: 38 22 22 18

      * - 实验配置
        - 迭代耗时（Step Time）
        - 训练吞吐（Samples/sec/GPU）
        - 性能收益（vs. 基线）
      * - DeepSpeed ZeRO2 + Offload（官方版本，mbs=32 × 8 GPU）
        - 30.0 s
        - 1.10
        - 基线
      * - FSDP2 Base（mbs=1 × 8 GPU）
        - 1.8 s
        - 0.56
        - -49%（受限于小 MBS 与通信开销）
      * - **RLinf 深度优化版（mbs=32 + Recompute × 8 GPU）**
        - **7.2 s**
        - **4.44**
        - **+300%（4.0×）**

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0acctime.jpg
      :align: center
      :width: 45%

      DreamZero 5B 与 14B 模型的加速效果对比

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0accthpt.jpg
      :align: center
      :width: 45%

      DreamZero 5B 与 14B 模型的吞吐提升对比

常见问题
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 排查与修复
   * - 找不到权重（``No safetensors weights``）
     - 检查 ``model_path`` 下是否存在 ``model.safetensors`` 或分片索引；冷启动时确认各 ``*_pretrained_path`` 可访问且与架构匹配。
   * - WAN2.2 维度不匹配
     - 核对 ``model_path/config.json`` 或 ``dreamzero_5b`` 预设：``diffusion_model_cfg`` 须为 ``ti2v`` 且 ``in_dim/out_dim=48``，VAE 使用 38 通道 WAN2.2 VAE，图像编码器使用 WAN2.1 CLIP 权重。
   * - ``metadata.json`` 找不到
     - 运行 ``toolkits/lerobot/generate_dreamzero_metadata.py`` 并设置 ``actor.model.metadata_json_path``，确认 JSON 内包含与 ``embodiment_tag`` 同名的顶层键。
   * - ``action_loss`` 异常偏高
     - 检查归一化统计是否匹配当前数据集、``relative_action`` 是否与数据一致，并核对 ``action_horizon``、``max_chunk_size`` 与 ``sampling_mode``。
   * - DROID 视频尺寸错误
     - ``target_video_height`` / ``target_video_width`` 仅用于拼接后的模型内 resize；单视角尺寸请通过 ``droid_view_height`` / ``droid_view_width`` 或 ``oxe_droid`` 变换调整。
   * - ``multi_anchor`` 报错要求 ``lazy_load``
     - 设置 ``data.lazy_load: True``。
   * - ``AttributeError: GR1_UNIFIED_SEGMENTATION`` 或未知具身标签
     - 在 RLinf 的 DreamZero 具身枚举与变换注册表中登记自定义标签，并确保使用 RLinf 的 DreamZero 数据变换链（加载模型时会自动为上游枚举打补丁）。

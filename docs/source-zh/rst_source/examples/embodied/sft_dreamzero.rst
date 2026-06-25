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

      1+ 节点 · GPU

| **你将完成：** 安装 → 准备模型和 LeRobot 数据 → 生成 ``metadata.json`` → 运行 ``run_vla_sft.sh`` → 在仿真或 Franka 上评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · `DreamZero 仓库 <https://github.com/RLinf/dreamzero>`_（``DREAMZERO_PATH``）· 一个 LeRobot 数据集。

**当前支持**

- **数据集：** LIBERO（``libero_sim``）、OXE DROID（``oxe_droid``）、Franka pick-and-place（``franka_pnp``）；支持跨 embodiment 的 **mixture SFT** （见 ``libero_franka_mix_sft_dreamzero_5b.yaml``）。
- **骨干网络：** WAN2.1（如 DreamZero-DROID 14B）、WAN2.2（如 Wan2.2-TI2V-5B 冷启动）。

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：仅 SFT 环境** — 安装 DreamZero，不安装仿真器依赖：

.. code-block:: bash

   # 国内用户可以添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model dreamzero
   source .venv/bin/activate

**选项 2：SFT + LIBERO 评测** — 额外安装 LIBERO 仿真依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model dreamzero --env libero
   source .venv/bin/activate

单独克隆 DreamZero 仓库，并在 SFT 或评测前设置 ``DREAMZERO_PATH``：

.. code-block:: bash

   git clone https://github.com/RLinf/dreamzero.git
   export DREAMZERO_PATH=/path/to/dreamzero

**这些命令会：**

1. 通过 ``requirements/install.sh`` 创建 DreamZero 专用 uv 虚拟环境。
2. 默认只安装离线 SFT 依赖；如果需要仿真评测，则额外安装 LIBERO。
3. 通过 ``DREAMZERO_PATH`` 让外部 DreamZero 包可导入；``examples/sft/run_vla_sft.sh`` 也会将其加入 ``PYTHONPATH``。

模型准备
----------------------------------------

从 checkpoint 继续训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

设置 ``actor.model.model_path`` 为已下载的权重目录；架构与权重从该目录加载。可选 checkpoint：

- DreamZero 14B（DROID / AgiBot）： `DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_、 `DreamZero-AgiBot <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>`_ — 参考 ``droid_sft_dreamzero_14b.yaml``
- RLinf 5B（LIBERO SFT）： `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_ — 参考 ``libero_sft_dreamzero_5b.yaml`` 并将 ``model_path`` 指向该目录

下载示例：

.. code:: bash

   pip install -U huggingface_hub
   huggingface-cli download GEAR-Dreams/DreamZero-DROID --local-dir ./DreamZero-DROID

YAML 示例（DROID + 官方 14B，见 ``droid_sft_dreamzero_14b.yaml``）：

.. code:: yaml

   defaults:
     - model/dreamzero_14b@actor.model

   actor:
     model:
       model_path: ./DreamZero-DROID
       tokenizer_path: google/umt5-xxl
       embodiment_tag: oxe_droid

AgiBot 数据将 ``model_path`` 换为 ``./DreamZero-AgiBot`` 即可。

从头训练（WAN2.2 组件冷启动）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

设置 ``model_path: null``，并填写各 ``*_pretrained_path``。需从 Hugging Face 下载：

- `Wan-AI/Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_ （DiT、T5、VAE）
- `Wan2.1 CLIP <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P>`_  （ ``models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`` 不在 5B 包内）
- `google/umt5-xxl <https://huggingface.co/google/umt5-xxl>`_

下载示例：

.. code:: bash

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
   huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
     models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
   huggingface-cli download google/umt5-xxl --local-dir ./umt5-xxl

YAML 示例（LIBERO 冷启动，见 ``libero_sft_dreamzero_5b.yaml``）：

.. code:: yaml

   defaults:
     - model/dreamzero_5b@actor.model

   actor:
     model:
       model_path: null
       tokenizer_path: google/umt5-xxl
       diffusion_model_pretrained_path: Wan-AI/Wan2.2-TI2V-5B
       image_encoder_pretrained_path: Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
       text_encoder_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
       vae_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: libero_sim


数据准备
----------------------------------------

训练数据需为 LeRobot v2/v3 布局（含 ``meta/``、``data/`` 等）。通过 ``data.train_data_paths`` 指定本地目录或 Hugging Face 数据集 ID。

数据集下载
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前支持：

- LIBERO： `physical-intelligence/libero <https://huggingface.co/datasets/physical-intelligence/libero>`_ — ``embodiment_tag: libero_sim``，配置见 ``libero_sft_dreamzero_14b.yaml`` / ``libero_sft_dreamzero_5b.yaml``
- DROID： `GEAR-Dreams/DreamZero-DROID-Data <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>`_ — ``embodiment_tag: oxe_droid``，配置见 ``droid_sft_dreamzero_14b.yaml``
- Franka PnP：`RLinf/dreamzero-franka-pnp <https://huggingface.co/datasets/RLinf/dreamzero-franka-pnp>`_ — ``embodiment_tag: franka_pnp``，变换实现见 ``data_transforms/franka_pnp.py``（继承 ``libero_sim`` 双视角布局）
- 混合训练：``libero_franka_mix_sft_dreamzero_5b.yaml`` 中 ``data.train_data_paths`` 为列表，每项可指定不同的 ``dataset_path`` / ``embodiment_tag`` / ``metadata_json_path`` / ``weight``

下载示例：

.. code:: bash

   pip install -U huggingface_hub
   # LIBERO
   huggingface-cli download physical-intelligence/libero --repo-type dataset --local-dir ./libero
   # DROID
   huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./DreamZero-DROID-Data
   # Franka PnP 真机数据
   huggingface-cli download RLinf/dreamzero-franka-pnp --repo-type dataset --local-dir ./franka_pnp

生成 metadata.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在新数据集或冷启动（无 ``experiment_cfg/metadata.json``）时，必须先为对应 ``embodiment_tag`` 生成归一化统计：

.. code:: bash

   # LIBERO
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

   # DROID（多数据集可 --merge）
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset oxe_droid \
     --dataset-root /path/to/droid \
     --output-metadata /path/to/metadata.json \
     --merge

   # Franka PnP
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset franka_pnp \
     --dataset-root /path/to/franka_pnp \
     --output-metadata /path/to/franka_pnp_metadata.json

然后在配置中设置 ``actor.model.metadata_json_path`` （ 或放到 ``model_path/experiment_cfg/metadata.json`` ） 。


配置参考
----------------------------------------

配置文件由 Hydra 管理，入口脚本为 ``examples/sft/train_vla_sft.py``。下面按 **数据相关** 与 **模型及训练超参相关** 分别说明含义与作用。

数据相关配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``train_data_paths``
     - 单数据集：LeRobot 根路径或 HF ``repo_id``。**混合训练**：YAML 列表，每项含 ``dataset_path``（或路径列表）、``weight``、``embodiment_tag``、``metadata_json_path`` 等；由 ``build_dreamzero_mixture_dataset_from_spec`` 按权重采样。可选 ``distribute_weights: true`` 在单条 spec 含多路径时按 episode 长度分配权重。
   * - ``lazy_load``
     - 是否懒加载 mp4 视频。 ``multi_anchor`` 采样模式下必须将 ``lazy_load`` 设为 ``True`` （否则无法按锚点随机取帧）。
   * - ``sampling_mode``
     - ``multi_anchor`` （默认，推荐）：在同一语言片段内按多个时间锚点采样；宏观时间块数由 ``max_chunk_size`` 决定。``fixed_window`` 为连续固定窗口。
   * - ``video_backend``
     - LeRobot 视频解码后端：``pyav`` 或 ``torchcodec``，影响懒加载 mp4 的速度与兼容性，推荐使用 ``torchcodec``。
   * - ``video_tolerance_s``
     - 视频时间戳与目标帧时间的容差（秒）。
   * - ``parquet_cache_size``
     - Parquet episode 缓存上限（episode 数），影响内存与 IO。
   * - ``num_workers`` / ``prefetch_factor``
     - DataLoader 并行与预取，影响数据吞吐。

**时间对齐要点（数据采样 vs 模型块）**

- 宏观时间块数来自 ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size`` （常见为 4；官方 Groot DROID 配方可为 5）。
- ``actor.model.action_horizon`` 是 DreamTransform / WAN 每个块内的动作步数（LIBERO 常用 16，DROID 常用 24），不是数据集宏观步长。
- ``multi_anchor`` 下，数据集侧动作序列长度约为 ``action_horizon * max_chunk_size`` （如 LIBERO 64、DROID 96）。
- 视频时间维在预设里配置 ``action_head_cfg.config.num_frames`` （DreamZero 默认 33，对应 ``8 * max_chunk_size + 1``）；未设置时自动推导。

模型与训练相关配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**标识与权重路径**

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
     - 因果 DiT（扩散骨干）预训练权重；冷启动必填。
   * - ``image_encoder_pretrained_path``
     - WAN 图像编码器；WAN2.2 需指向 WAN2.1 CLIP 权重。
   * - ``text_encoder_pretrained_path``
     - T5 文本编码器权重。
   * - ``vae_pretrained_path``
     - VAE 权重；WAN2.2 对应 ``WanVideoVAE38``。
   * - ``metadata_json_path``
     - 数据集 ``metadata.json``；未设置则回退到 ``model_path/experiment_cfg/metadata.json``。
   * - ``embodiment_tag``
     - 选择数据变换与 collate 模板：``libero_sim``、``oxe_droid``、``franka_pnp``（定义于 ``data_transforms/embodiment_tag.py``）。单数据集训练时须与数据一致；混合训练时各子项在 ``train_data_paths`` 里单独指定， ``actor.model.embodiment_tag`` 仍须设置（通常与主数据来源一致，供 ``get_model`` 加载 policy 侧 metadata）。

**时序与动作形状（需与数据、WAN 容量一致）**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``action_horizon``
     - 每个 WAN 时间块内的动作步数（LIBERO 16，DROID 24）。
   * - ``state_horizon``
     - 每个样本的状态行数（通常为 1，每个宏观锚点一个状态）。
   * - ``num_action_per_block``
     - 与 ``action_head_cfg`` 中 DiT 的 ``num_action_per_block`` 对齐（常等于 ``action_horizon``）。
   * - ``action_head_cfg...diffusion_model_cfg.max_chunk_size``
     - 多锚点宏观时间块数 / Causal DiT 容量；与 ``data.sampling_mode: multi_anchor`` 强相关。视频帧数 ``num_frames`` 由 ``8 * max_chunk_size + 1`` 推导。
   * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
     - DreamTransform 填充与文本序列上限。

**视频尺寸与 DROID 特有项**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义与作用
   * - ``target_video_height`` / ``target_video_width``
     - WAN 策略头在 **多视角拼接后** 的目标分辨率（5B 预设如 176×320；Libero 常用 160×320）。仅作用于模型内部 resize， **不要** 用于 data transform 的单视角 resize。
   * - ``droid_view_height`` / ``droid_view_width``
     - （可选）DROID 各视角 resize 覆盖。
   * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
     - 是否使用相对动作及作用维度；DROID 常对 ``joint_position`` 等开启 ``relative_action: True``。

**其它模型训练项**

- ``precision``：Actor / Optimizer 侧的主精度设置（ ``fp32`` / ``bf16``）。推荐 ``precision: fp32``，并配合 ``actor.fsdp_config.mixed_precision`` 做混合精度训练：优化器状态与主参数保持 FP32（数值更稳），前向/反向的实际矩阵运算由 FSDP 在 ``mixed_precision`` 中降为 BF16（省显存、提速）。示例：

  .. code:: yaml

     actor:
       model:
         precision: fp32
       fsdp_config:
         mixed_precision:
           param_dtype: bf16
           reduce_dtype: bf16
           buffer_dtype: bf16

  若将 ``precision`` 设为 ``bf16``，优化器也会以较低精度维护状态，一般不如上述组合稳定。启用 FSDP CPU offload 时，通常保持 ``precision: fp32``。
- ``is_lora``：是否 LoRA 微调（DreamZero SFT 示例多为全参 ``False``）。
- ``actor.micro_batch_size`` / ``actor.global_batch_size``：每 GPU 微批与全局有效 batch（需能被 GPU 数整除关系约束）。
- ``actor.optim.*``：学习率、warmup、cosine 等。
- ``actor.fsdp_config``：FSDP2 分片、梯度检查点；``mixed_precision`` 控制计算/通信 dtype（与 ``actor.model.precision`` 配合，见上）。

**配置示例对照**

.. code:: yaml

   # ---------- 数据（单数据集）----------
   data:
     train_data_paths: /path/to/libero
     lazy_load: True
     sampling_mode: multi_anchor
     video_backend: torchcodec
     num_workers: 8

   # ---------- 数据（混合，见 libero_franka_mix_sft_dreamzero_5b.yaml）----------
   data:
     train_data_paths:
       - dataset_path: /path/to/libero
         weight: 4
         embodiment_tag: libero_sim
         metadata_json_path: /path/to/libero_metadata.json
       - dataset_path: /path/to/franka_pnp
         weight: 1
         embodiment_tag: franka_pnp
         metadata_json_path: /path/to/franka_metadata.json

   # ---------- 模型（从 checkpoint 继续）----------
   actor:
     model:
       model_path: /path/to/DreamZero-DROID
       tokenizer_path: /path/to/umt5-xxl
       embodiment_tag: oxe_droid
       action_horizon: 24
       metadata_json_path: /path/to/metadata.json   # 若无 experiment_cfg/metadata.json

运行
----------------------------------------

在仓库根目录执行：

.. code:: bash

   # LIBERO + WAN2.1（checkpoint，dreamzero_14b 预设）
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

   # LIBERO + WAN2.2（冷启动，dreamzero_5b 预设）
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

   # DROID + WAN2.1（dreamzero_14b 预设，model_path 指向 DreamZero-DROID）
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero_14b

   # LIBERO + Franka 混合（WAN2.2，见 libero_franka_mix_sft_dreamzero_5b.yaml）
   bash examples/sft/run_vla_sft.sh libero_franka_mix_sft_dreamzero_5b

脚本等价于：

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name CONFIG_NAME \
     runner.logger.log_path=LOG_DIR

日志目录：

- 仓库根目录下 ``logs/时间戳-config_name/run_embodiment.log``

断点续训可设置 ``runner.resume_dir`` 指向 checkpoint 目录。


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

复制 ``libero_sft_dreamzero_14b.yaml``、``libero_sft_dreamzero_5b.yaml`` 或 ``droid_sft_dreamzero_14b.yaml``，至少修改：

.. code:: yaml

   data:
     train_data_paths: /path/to/your_lerobot
     lazy_load: True              # multi_anchor 必须为 True（mp4 数据）
     sampling_mode: multi_anchor

   actor:
     model:
       embodiment_tag: "YOUR_TAG"
       metadata_json_path: /path/to/metadata.json
       action_horizon: 16  # 与 DEFAULT_ACTION_HORIZON 一致
       # 从 checkpoint 继续时核对 action_loss_embodiment_ids 是否包含你的 projector ID
       target_video_height: ...
       target_video_width: ...
       relative_action: ...
       relative_action_keys: [...]

若冷启动 WAN，在 ``examples/sft/config/model/dreamzero_5b.yaml`` （ 或 14b）的 ``action_head_cfg.config.action_loss_embodiment_ids`` 中加入新 ID。

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

以下逐一拆解背后的三项核心优化技术。


算子与计算图优化：Torch Compile + CUDA Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 层面的算子与调度开销往往是限制 GPU 峰值性能的"隐形杀手"。RLinf 深度融合了 ``torch.compile`` 和 CUDA Graph 技术：

- **Torch Compile**：通过底层编译优化，对算子进行深度融合（Kernel Fusion），包括 WanRMSNorm、adaLN-zero 等 Diffusion 架构中的低效算子。
- **CUDA Graph**：将计算图固化，消除 GPU kernel launch 的 CPU 调度瓶颈。DreamZero 训练中，CausalWanSelfAttention 部分的 kernel launch 较为密集，CUDA Graph 可对其有效优化。

通过该项优化，DreamZero 5B 和 14B 模型在不改变 mbs=1（每 GPU 微批大小，下同）的配置下，分别获得 50%（从 1.8s/step 降到 1.2s/step）和 34%（从 9s/step 降到 6.7s/step）的训练加速。


FSDP2 并行优化：解锁 mbs 与 Recompute 的全方位调优
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

支持任意 Microbatch Size、FSDP2（Fully Sharded Data Parallel 2）并行策略的参数调优以及 Recompute（激活重计算），是业界训练大模型时必不可少的性能调优手段。然而，在 DreamZero 官方的 baseline 中存在明显的工程局限——默认使用 DeepSpeed 的 zero2 offload 并行方法、image encoder 不拼 batch 逐样本执行等，大大限制了性能调优空间。

RLinf 从底层夯实了工程底座，交付了一套健壮且高度可配的调优矩阵：

**稳定适配 FSDP2**
  FSDP2 是 PyTorch 官方团队推出的最新 ZeRO 实现，也是 RLinf 面向中等规模大模型的默认并行方案。此前 DreamZero 官方代码使用 DeepSpeed 方案存在局限性：由于 ZeRO3 与 VAE 模块中 causal conv 的上下文维护机制存在兼容性冲突，开发者往往被迫回退至性能较低的 ZeRO2 offload 模式；此外 DeepSpeed 在反向传播阶段的 post backward hook 产生了较高的 CPU 侧开销，制约了整体训练吞吐。通过向 FSDP2 训练后端的迁移，RLinf 彻底解决了上述架构冲突与性能瓶颈。用户可以根据显存配置需求，在不同的分片策略间灵活切换，确保训练过程的高效与稳定。

**灵活的 Microbatch 设置**
  在 FSDP2 支持 DreamZero 模型训练的初始版本中，Microbatch Size（mbs）、Recompute 与 FSDP2 的策略组合往往会触发复杂的底层计算图冲突，且 image encoder 不拼 batch 会吞掉一部分开大 mbs 的加速收益。RLinf 通过工程上的努力，彻底解决了 mbs > 1 时与上述特性共存的不兼容问题，并使 image encoder 能够高效地拼 batch 执行。这一改进使得用户可以不约束地配置任意 mbs，从而根据硬件资源的显存水位与计算吞吐需求，进行精细化的参数调优。例如，对 DreamZero 5B 模型训练，不开启 Recompute 的情况下，mbs 从 1 开到 2，单步耗时几乎不变（1.2s/step → 1.3s/step），吞吐增加 85%。

**Recompute 机制与加速算子的深度协同**
  针对 PyTorch 原生框架在复杂并行策略下的兼容性局限，RLinf 通过深度的底层工程优化，实现了 Recompute（激活重计算）与 CUDA Graph、FSDP2 的稳定解耦与协同。这一改进将 Recompute 转化为一个高可靠、可量化的性能调优维度。在显存受限的硬件环境下，系统能够以微小的计算耗时为代价，换取显著的显存空间释放，从而支持更大的 mbs，大幅提升整体训练吞吐。在 DreamZero 5B 训练中，不开启 Recompute 时单卡 mbs 只能开到 2，最佳速度约 1.2s/step（即 1.7 samples/sec/gpu）；开启 Recompute 后单卡 mbs 可开到 32，获得 7.2s/step（即 4.4 samples/sec/gpu），**同等算力下吞吐提升 158%**。

通过以上 FSDP2、mbs、Recompute 的全局参数调优，在 DreamZero 5B 模型训练上，在第一项算子优化的基础上（即 1.2 samples/sec/gpu）将训练性能进一步提升了 **266%**，达到 4.4 samples/sec/gpu。


视频数据处理管线：突破 I/O 吞吐瓶颈
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

随着计算密度的显著提升，数据加载效率逐渐成为制约整体训练吞吐的新瓶颈。DreamZero 训练中，视频数据的解码与预处理过程极其消耗 CPU 资源——多视角场景下需同时处理左视角、右视角、腕部视角三个视频流。传统的 PyAV 方案解码性能难以支撑高频吞吐需求，而单纯增加 ``num_workers`` 往往治标不治本（过多数据读取进程会抢占 CPU 资源，反而拖慢 GPU kernel 下发节奏）。

为在"解码速度"与"系统资源开销"之间寻找最优解，RLinf 团队对主流视频处理库进行了深度 Benchmark：

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - 解码方案
     - 单视频平均解码耗时
     - 系统资源占用特点
   * - PyAV
     - 816 ms
     - 较高（资源消耗多但速度最慢）
   * - Decord
     - 435 ms
     - 高（瞬时负载波动大）
   * - Torchcodec
     - 446 ms
     - 低（资源消耗曲线平稳）

虽然 Decord 在纯解码速度上略胜一筹，但 **Torchcodec** 在保持同梯队性能的同时，表现出更优的 CPU 占用稳定性。这使得 RLinf 能够预留出足够计算余量给训练主线程，并支持开启更多的 ``num_workers`` 来并发处理数据。

相比原生 PyAV 方案，单个视频的解码时间缩短了近 **400ms**。在 DreamZero 多视角（左视角、右视角、腕部视角三个视频）训练场景下，视频解码时间累计节省约 **1.2s**，为后续进一步压榨 GPU 计算潜力提供了充足的数据"弹药"。

要启用 Torchcodec 后端，在配置中设置：

.. code:: yaml

   data:
     video_backend: torchcodec


端到端性能实测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下所有测试均在 Droid 数据集（单样本含左、右、腕部三个视角，视频规格 33 frames × 480 × 640）上，使用 8×H100 GPU 完成。

**DreamZero-14B**

在 14B 大模型上，由于显存压力巨大，官方基线通常被迫采用 DeepSpeed ZeRO-offload 方案，这导致了严重的计算/通信浪费与 CPU 换入换出开销。

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

14B 模型使用 MBS=1 和 GBS=8 进行测试。RLinf 相比原生 DeepSpeed 方案实现了 **2.7 倍**的加速；即便相比于未经优化的 FSDP2，吞吐量也进一步提升了 **35%**。

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
   :width: 90%

   DreamZero 5B 与 14B 模型的加速效果对比。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0accthpt.jpg
   :align: center
   :width: 90%

   DreamZero 5B 与 14B 模型的吞吐提升对比。

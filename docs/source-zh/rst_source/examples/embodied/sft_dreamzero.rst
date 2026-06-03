DreamZero 监督微调
====================================

本文档介绍如何在 RLinf 中运行 DreamZero 监督微调（SFT），覆盖从 **模型与数据准备**、 **配置填写** 到 **启动训练**、 **评测** 与 **排错** 的完整流程。

当前支持：

- **数据集**: LIBERO (``libero_sim``)、OXE DROID (``oxe_droid``)、Franka 抓取放置 (``franka_pnp``)；支持多 embodiment **混合训练** (见 ``libero_franka_mix_sft_dreamzero_5b.yaml``)
- **骨干网络**：WAN2.1（如 DreamZero-DROID 14B）、WAN2.2（如 Wan2.2-TI2V-5B 冷启动）


环境准备
----------------

1. 克隆 RLinf 仓库并进入根目录：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 使用 ``requirements/install.sh`` 创建并安装 **DreamZero 专用 uv 虚拟环境** ：

.. code:: bash

   # 仅做 SFT（LeRobot 离线数据，不跑仿真）— 推荐
   bash requirements/install.sh embodied --model dreamzero

   # 若后续还要在 LIBERO 仿真里评测，可一并安装 libero 环境
   bash requirements/install.sh embodied --model dreamzero --env libero

说明：

- 国内网络可加 ``--use-mirror`` 加速 PyPI / Python / GitHub 下载。
- 自定义 venv 目录： ``--venv <dir>``；无 root 且系统依赖已就绪： ``--no-root``。

安装完成后激活环境：

.. code:: bash

   source .venv/bin/activate

3. 单独克隆 `DreamZero 代码库 <https://github.com/RLinf/dreamzero>`_，并设置 ``DREAMZERO_PATH`` 指向其 Python 包根目录：

.. code:: bash

   git clone https://github.com/RLinf/dreamzero.git
   export DREAMZERO_PATH=/path/to/dreamzero

模型准备
----------------

从 checkpoint 继续训练
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
----------------

训练数据需为 LeRobot v2/v3 布局（含 ``meta/``、``data/`` 等）。通过 ``data.train_data_paths`` 指定本地目录或 Hugging Face 数据集 ID。

数据集下载
~~~~~~~~~~~~~~~~

当前支持：

- LIBERO： `physical-intelligence/libero <https://huggingface.co/datasets/physical-intelligence/libero>`_ — ``embodiment_tag: libero_sim``，配置见 ``libero_sft_dreamzero_14b.yaml`` / ``libero_sft_dreamzero_5b.yaml``
- DROID： `GEAR-Dreams/DreamZero-DROID-Data <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>`_ — ``embodiment_tag: oxe_droid``，配置见 ``droid_sft_dreamzero_14b.yaml``
- Franka PnP：自定义 LeRobot 数据集 — ``embodiment_tag: franka_pnp``，变换实现见 ``data_transforms/franka_pnp.py``（继承 ``libero_sim`` 双视角布局）
- 混合训练：``libero_franka_mix_sft_dreamzero_5b.yaml`` 中 ``data.train_data_paths`` 为列表，每项可指定不同的 ``dataset_path`` / ``embodiment_tag`` / ``metadata_json_path`` / ``weight``

下载示例：

.. code:: bash

   pip install -U huggingface_hub
   # LIBERO
   huggingface-cli download physical-intelligence/libero --repo-type dataset --local-dir ./libero
   # DROID
   huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./DreamZero-DROID-Data

生成 metadata.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

然后在配置中设置 ``actor.model.metadata_json_path`` （ 或放到 ``model_path/experiment_cfg/metadata.json`` ） 。


配置说明
---------------

配置文件由 Hydra 管理，入口脚本为 ``examples/sft/train_vla_sft.py``。下面按 **数据相关** 与 **模型及训练超参相关** 分别说明含义与作用。

数据相关配置
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - WAN 策略头目标分辨率（5B 预设如 176×320；可在 YAML 覆盖）。避免在 transform 代码里写死尺寸，以兼容 WAN2.1 / WAN2.2。
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

启动训练
-------------

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


评测
--------

SFT 完成后，可在数据集对应具身环境中评测策略。下文以 **LIBERO** 仿真环境为例说明完整流程（任务套件为 LIBERO Spatial）；对应示例配置为 ``examples/embodiment/config/libero_spatial_eval_dreamzero.yaml``。其它支持 ``env.eval`` 的仿真环境亦可按相同方式编写配置并调用 ``eval_embodiment.sh``。

**前置条件**

1. 安装时需包含 LIBERO 仿真环境（见上文 **环境准备** 中的 ``--env libero``）。
2. 已设置 ``DREAMZERO_PATH`` 指向 DreamZero 代码库根目录（``eval_embodiment.sh`` 会将其加入 ``PYTHONPATH``）。
3. 已准备与训练一致的 ``metadata.json``（``actor.model.metadata_json_path``）。

**配置评测 YAML**

复制或编辑 ``examples/embodiment/config/libero_spatial_eval_dreamzero.yaml``，至少修改以下字段：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 字段
     - 说明
   * - ``runner.ckpt_path``
     - 待评测的 SFT 权重（``.pt``）。训练保存路径一般为 ``{log_path}/{experiment_name}/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。若仅有 ``.distcp`` 格式，请先按 :doc:`Checkpoint 转换 <../../tutorials/usage/convertor>` 转为 ``.pt``。
   * - ``actor.model.*_pretrained_path`` / ``tokenizer_path``
     - 与 SFT 冷启动配置一致（``model_path: null`` 时从各预训练路径构建骨干，再由 ``ckpt_path`` 覆盖可训练权重）。
   * - ``actor.model.metadata_json_path``
     - LIBERO 归一化统计（``embodiment_tag: libero_sim`` 时与 SFT 使用同一份 ``metadata.json``）。
   * - ``actor.model.embodiment_tag``
     - 须为 ``libero_sim``，与 LIBERO 数据及 rollout 观测变换一致。
   * - ``actor.model.action_horizon`` / ``num_action_chunks``
     - 与 SFT 一致（LIBERO 常用 16）。
   * - ``algorithm.eval_rollout_epoch``
     - 评测轮数；每轮在相同种子下跑完测试集，最终指标为多轮平均。
   * - ``env.eval.total_num_envs`` / ``auto_reset`` / ``max_steps_per_rollout_epoch``
     - 并行环境数与是否通过 ``auto_reset`` 覆盖更大测试集；详见 :doc:`评估教程 <../../start/vla-eval>`。
   * - ``env.eval.video_cfg.save_video``
     - 设为 ``True`` 可在 ``{log_path}/video/eval`` 下保存评测视频。

配置片段示例：

.. code:: yaml

   runner:
     only_eval: True
     ckpt_path: /path/to/logs/libero_sft_dreamzero/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt

   actor:
     model:
       model_path: null
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: libero_sim
       action_horizon: 16
       num_action_chunks: 16

   env:
     eval:
       total_num_envs: 64
       auto_reset: True
       ignore_terminations: True
       max_episode_steps: 480
       max_steps_per_rollout_epoch: 480

**启动评测**

在仓库根目录、已激活 DreamZero 环境且 ``DREAMZERO_PATH`` 已设置时执行：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh libero_spatial_eval_dreamzero

脚本会调用 ``eval_embodied_agent.py``，将日志写入 ``logs/<时间戳>-libero_spatial_eval_dreamzero/eval_embodiment.log``，并在终端输出 ``eval/success_once``、 ``eval/return`` 等指标。更多通用评测参数说明见 :doc:`评估教程 <../../start/vla-eval>`。

可选：若需将 SFT 的 ``full_weights.pt`` 转为 Hugging Face ``safetensors`` 目录（便于外部推理或发布），可使用 ``fsdp_dreamzero_convertor`` 配置运行 ``convert_pt_to_hf``（见 ``rlinf/utils/ckpt_convertor/fsdp_convertor/config/fsdp_dreamzero_convertor.yaml``）。在 LIBERO 等仿真环境中评测时，只需将 ``runner.ckpt_path`` 指向 ``.pt`` 权重文件即可。

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

监控与 sanity check
-------------------------------

1. 查看 ``run_embodiment.log``：``time/step`` 是否稳定；``train/loss``、``train/action_loss``、``train/dynamics_loss`` 是否合理。

2. TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. 开跑后尽早检查：

   - ``images`` / ``state`` / ``action`` 的 shape、dtype、数值范围
   - ``state_mask`` / ``action_mask`` / ``text_attention_mask`` 有效比例
   - WAN2.2 时确认输入分辨率与 ``frame_seqlen`` 与 ``config.json`` 或预设一致


扩展：新增 ``embodiment_tag``
------------------------------------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 在 ``rlinf/data/datasets/dreamzero/data_transforms/embodiment_tag.py`` 的 ``EmbodimentTag`` 枚举中增加成员（值等于 ``TAG`` 字符串）。
2. 编辑 ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``：

   - ``from ...your_tag import YourEmbodimentDataTransform``
   - 在 ``_EMBODIMENT_REGISTRY`` 中加入 ``YourEmbodimentDataTransform.TAG: YourEmbodimentDataTransform``

无需手写 Groot patch：``get_model()`` 会将 ``groot.vla.data.schema.embodiment_tags.EmbodimentTag`` 替换为上述 RLinf 枚举。

未注册时，``build_dreamzero_composed_transform`` 会报错并列出已有 tag。

步骤 3：生成 ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
- 视频 resize：优先在 transform 链或 ``target_video_height/width`` 配置，避免写死尺寸导致 WAN2.1/2.2 不兼容。
- 推理 / 评测：``examples/embodiment/config/*_dreamzero.yaml`` 中同样需要正确的 ``embodiment_tag``。

若仅推理、不改 RLinf 代码，且 Groot/DreamZero 上游已支持该 tag，有时只需准备 ``metadata.json`` 与评测配置；**SFT 新数据** 则须完成上述枚举成员、registry 注册与 transform 实现（``get_model`` 会自动 patch Groot ``EmbodimentTag``）。


常见问题
--------------

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

   - 勿在代码中写死分辨率；使用 ``target_video_height/width`` 或 ``droid_view_*`` 配置项

6. **multi_anchor 报错要求 lazy_load**

   - 设置 ``data.lazy_load: True``

7. ``AttributeError: GR1_UNIFIED_SEGMENTATION`` 或未知 ``EmbodimentTag``

   - 数据 transform 链须使用 ``dream_transform.DreamTransform`` (RLinf 子类)，勿直接实例化 Groot 基类
   - 新 tag 须在 ``embodiment_tag.py`` 与 ``_EMBODIMENT_REGISTRY`` 注册；训练经 ``get_model()`` 加载模型时会 patch Groot 枚举


实践建议
------------------

- 追求稳定收敛时，优先从已发布的 DreamZero 权重继续 SFT（设置 ``model_path``）。
- 全量适配 WAN2.2 可冷启动，但需更大数据与更长训练；改配置后先用 50–200 step 试跑验证 shape 与 loss。
- 每次更换数据集或 ``embodiment_tag``，务必重新生成或更新 ``metadata.json``。
- LIBERO 与 DROID 的 ``action_horizon``、 ``embodiment_tag``、多视角拼接逻辑不同，不要混用配置模板。

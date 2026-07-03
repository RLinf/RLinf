具身智能配置
=============

本节介绍具身智能 RL 训练专用的配置参数（机器人操作、模拟器、VLA 模型）。
这些参数扩展了 :doc:`basic_config` 中的共享配置。

defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/maniskill_put_on_plate_in_scene_25_main@env.train
    - env/maniskill_ood_template@env.eval
    - model/openvla_oft@actor.model
    - training_backend/fsdp@actor.fsdp_config
    - weight_syncer/patch_syncer@weight_syncer

``defaults``：Hydra 配置继承。选择要组合到本次运行中的环境、模型、训练后端与权重同步器预设。
``<group>@<target>`` 语法把某个预设挂载到配置节点上（例如
``model/openvla_oft@actor.model`` 填充 ``actor.model``）。

hydra
~~~~~~~~~~~~~~~

.. code:: yaml

  hydra:
    searchpath:
      - file://${oc.env:EMBODIED_PATH}/config/

``hydra.searchpath``：配置文件的额外搜索路径。``EMBODIED_PATH`` 指向具身示例的配置目录。

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    only_eval: False
    overlap_env_bootstrap: False
    enable_decoupled_mode: False

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``runner.only_eval``
     - 仅运行评估，不进行训练。
   * - ``runner.overlap_env_bootstrap``
     - 让环境引导（重置）与 Actor 训练重叠，以隐藏重置延迟。当环境重置较慢时尤为有用。
       仅当 ``env.enable_offload`` 为 False 时生效；当环境与 Actor 共用同一加速器时，
       开启此选项可能增加显存压力。
   * - ``runner.enable_decoupled_mode``
     - 将 Env Worker 与 Rollout Worker 解耦。开启后，Env Worker 不再绑定到固定的
       Rollout Worker rank：Env Worker 将观测推送到共享 Channel，空闲的 Rollout
       Worker 动态拉取批次。当 Env 步进耗时波动较大（长尾延迟）或 Env Worker 多于
       Rollout Worker 时有帮助。要求 ``env_world_size >= rollout_world_size``，且与
       ``enable_p2p`` 不兼容。详见 :doc:`env_decoupled_mode`。

诸如 ``runner.task_type``\ （设为 ``embodied``）、``runner.max_epochs``、
``runner.save_interval``、``runner.val_check_interval`` 与 ``runner.resume_dir``
等共享键在 :doc:`basic_config` 中说明。

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    normalize_advantages: True
    kl_penalty: kl

    reward_type: chunk_level
    logprob_type: token_level
    entropy_type: token_level

    adv_type: gae
    loss_type: actor_critic
    loss_agg_func: "token-mean"

    bootstrap_type: always
    kl_beta: 0.0
    entropy_bonus: 0.0
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    clip_ratio_c: 3.0
    value_clip: 0.2
    huber_delta: 10.0

    gamma: 0.99
    gae_lambda: 0.95

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``algorithm.normalize_advantages``
     - 在批次内对优势进行归一化。
   * - ``algorithm.kl_penalty``
     - 如何估计 KL 散度（``kl`` 或 ``kl_penalty``）。
   * - ``algorithm.reward_type``
     - 奖励聚合粒度（``chunk_level`` 或 ``action_level``）。默认同时驱动
       ``actor.model.value_type``。
   * - ``algorithm.logprob_type``
     - 对数概率计算粒度（例如 ``token_level``）。
   * - ``algorithm.entropy_type``
     - 熵计算粒度（例如 ``token_level``）。
   * - ``algorithm.adv_type``
     - 优势估计器（例如 PPO 使用 ``gae``）。
   * - ``algorithm.loss_type``
     - 策略损失类型。``actor_critic`` 启用带价值头的 PPO（要求
       ``actor.model.add_value_head: True``）。
   * - ``algorithm.loss_agg_func``
     - token 损失的聚合方式（例如 ``token-mean``）。
   * - ``algorithm.bootstrap_type``
     - 在 episode 边界如何对 Q 值进行自举：``standard`` 仅在截断时自举；``always`` 在
       截断或终止时都自举。
   * - ``algorithm.kl_beta``
     - 加入奖励的 KL 惩罚权重。
   * - ``algorithm.entropy_bonus``
     - 熵奖励系数。
   * - ``algorithm.clip_ratio_low`` / ``clip_ratio_high``
     - 重要性比值的 PPO 上/下裁剪界（非对称裁剪）。
   * - ``algorithm.clip_ratio_c``
     - 悲观 PPO 界的 dual-clip 常数。
   * - ``algorithm.value_clip``
     - 价值函数裁剪阈值（PPO 价值损失）。
   * - ``algorithm.huber_delta``
     - 价值损失使用的 Huber 损失 delta。
   * - ``algorithm.gamma``
     - 折扣因子。
   * - ``algorithm.gae_lambda``
     - GAE 平滑因子。

``algorithm.group_size``\ （每个 prompt 的回复数；GRPO 始终 > 1）为共享键，在
:doc:`basic_config` 中说明。

env
~~~~~~~~~~~~~~~

.. code:: yaml

  env:
    group_name: "EnvGroup"
    enable_offload: True

    train:
      rollout_epoch: 1
      total_num_envs: 128
      auto_reset: True
      ignore_terminations: False
      use_fixed_reset_state_ids: False
      max_episode_steps: 80
      max_steps_per_rollout_epoch: 160

    eval:
      rollout_epoch: 1
      total_num_envs: 16
      auto_reset: True
      ignore_terminations: True
      use_fixed_reset_state_ids: True
      max_episode_steps: 80
      max_steps_per_rollout_epoch: 80

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``env.group_name``
     - 环境 Worker 组的逻辑名称。
   * - ``env.enable_offload``
     - 卸载环境以降低内存占用。
   * - ``env.train.rollout_epoch`` / ``env.eval.rollout_epoch``
     - 每个训练/评估步的 rollout 轮数。评估指标会在相同种子的多次遍历上取平均。
   * - ``env.train.total_num_envs`` / ``env.eval.total_num_envs``
     - 训练/评估的并行环境总数。
   * - ``env.train.auto_reset`` / ``env.eval.auto_reset``
     - episode 终止时自动重置环境。
   * - ``env.train.ignore_terminations`` / ``env.eval.ignore_terminations``
     - 忽略 episode 终止；开启后 episode 仅在 ``max_episode_steps`` 处结束。
   * - ``env.train.use_fixed_reset_state_ids`` / ``env.eval.use_fixed_reset_state_ids``
     - 使用固定的重置状态 ID（False 表示随机化）。GRPO 始终为 True；PPO 默认 False。
   * - ``env.train.max_episode_steps`` / ``env.eval.max_episode_steps``
     - 每个 episode 的最大步数（截断范围）。
   * - ``env.train.max_steps_per_rollout_epoch`` / ``env.eval.max_steps_per_rollout_epoch``
     - 每个 rollout 轮采集的最大环境步数。必须能被 ``actor.model.num_action_chunks``
       整除。

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    sampling_params:
      do_sample: True
      temperature_train: 1.0
      temperature_eval: 0.6
      top_k: 0
      top_p: 1.0
      repetition_penalty: 1.0

    group_name: "RolloutGroup"
    backend: "huggingface"
    enable_offload: True
    pipeline_stage_num: 1
    rollout_queue_size: 0

    model:
      model_path: "/path/to/hf_model"
      precision: ${actor.model.precision}

**sampling_params（自回归 VLA 策略）：** 连续策略（MLP、CNN、OpenPI、GR00T 等）不使用
``rollout.sampling_params``。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``rollout.sampling_params.do_sample``
     - 为 False 时使用确定性解码。
   * - ``rollout.sampling_params.temperature_train`` / ``temperature_eval``
     - 训练与评估时的采样温度。
   * - ``rollout.sampling_params.top_k`` / ``top_p``
     - top-k 与核采样参数。
   * - ``rollout.sampling_params.repetition_penalty``
     - 对重复 token 的惩罚。
   * - ``rollout.group_name``
     - Rollout Worker 组的逻辑名称。
   * - ``rollout.backend``
     - 模型后端（例如 ``huggingface``）。
   * - ``rollout.enable_offload``
     - 卸载 rollout 模型以降低显存占用。
   * - ``rollout.pipeline_stage_num``
     - rollout 的流水线阶段数。
   * - ``rollout.rollout_queue_size``
     - 仅当 ``runner.enable_decoupled_mode`` 为 True 时使用。限制单个 Rollout Worker
       一次聚合多少个 Env 分片。``0`` 使用默认值
       ``ceil(env_world_size // rollout_world_size)``；更小的值减少等待时间，更大的值
       提升推理批利用率。详见 :doc:`env_decoupled_mode`。
   * - ``rollout.model.model_path``
     - rollout 使用的模型检查点路径（可与 Actor 相同）。
   * - ``rollout.model.precision``
     - rollout 的推理精度。

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    group_name: "ActorGroup"
    training_backend: "fsdp"
    micro_batch_size: 40
    global_batch_size: 640
    seed: 1234
    enable_offload: True

    model:
      model_path: "/path/to/huggingface_model"
      model_type: "openvla_oft"
      implement_version: "rlinf"
      action_dim: 7
      num_action_chunks: 8
      use_proprio: False
      use_film: False
      unnorm_key: bridge_orig
      value_type: ${algorithm.reward_type}
      add_value_head: True
      center_crop: True
      do_sample: False
      max_prompt_length: 30

      precision: "bf16"
      vocab_size: 32000
      hidden_size: 4096
      policy_setup: "widowx_bridge"
      image_size: [224, 224]
      is_lora: True
      lora_rank: 32
      lora_path: /path/to/models/oft-sft/lora_004000/
      num_images_in_input: 1
      attn_implementation: "flash_attention_2"
      low_cpu_mem_usage: True
      trust_remote_code: True

    optim:
      lr: 1.0e-4
      value_lr: 3.0e-3
      adam_beta1: 0.9
      adam_beta2: 0.999
      adam_eps: 1.0e-08
      weight_decay: 0.01
      clip_grad: 10.0
      critic_warmup_steps: 0

**顶层**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``actor.group_name``
     - Actor Worker 组的逻辑名称。
   * - ``actor.training_backend``
     - 训练后端（具身 VLA 训练使用 ``fsdp``）。
   * - ``actor.micro_batch_size``
     - 每个 GPU 的微批大小。
   * - ``actor.global_batch_size``
     - 所有 GPU 上的全局批大小。
   * - ``actor.seed``
     - 用于可复现性的全局随机种子。
   * - ``actor.enable_offload``
     - 卸载 Actor 模型以降低内存占用。

**模型配置**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``actor.model.model_type``
     - 模型架构名称（例如 ``openvla_oft``）。
   * - ``actor.model.model_path``
     - HuggingFace 模型路径。
   * - ``actor.model.implement_version``
     - OpenVLA-OFT 的实现变体：``rlinf``\ （RLinf 发布的 LIBERO / ManiSkill 模型）或
       ``official``\ （RoboTwin / 上游 OpenVLA-OFT，支持 ``use_film`` 与 ``use_proprio``）。
   * - ``actor.model.action_dim``
     - 动作空间维度。
   * - ``actor.model.num_action_chunks``
     - 每个序列的动作块数量。
   * - ``actor.model.use_proprio``
     - 是否向模型输入本体感受状态。
   * - ``actor.model.use_film``
     - 使用 FiLM 语言条件（仅 ``official`` OpenVLA-OFT）。
   * - ``actor.model.unnorm_key``
     - 动作反归一化统计量的键。
   * - ``actor.model.value_type``
     - 价值头粒度；继承自 ``algorithm.reward_type``。
   * - ``actor.model.add_value_head``
     - 添加价值头。当 ``algorithm.loss_type`` 为 ``actor_critic`` 时必须为 True。
   * - ``actor.model.center_crop``
     - 是否对输入图像进行中心裁剪。
   * - ``actor.model.do_sample``
     - 动作生成时是否采样（而非贪心）。
   * - ``actor.model.max_prompt_length``
     - 输入 VLA 主干的最大 prompt 长度（token）。
   * - ``actor.model.precision``
     - 数值精度（``bf16``、``fp16``、``fp32``）。
   * - ``actor.model.vocab_size``
     - 词表大小。
   * - ``actor.model.hidden_size``
     - 隐藏层维度。
   * - ``actor.model.policy_setup``
     - 策略/本体设定（例如 ``widowx_bridge``）。
   * - ``actor.model.image_size``
     - 输入图像尺寸 ``[height, width]``。
   * - ``actor.model.is_lora``
     - 是否使用 LoRA 微调。
   * - ``actor.model.lora_rank``
     - LoRA 低秩适配的秩。
   * - ``actor.model.lora_path``
     - LoRA 权重路径。
   * - ``actor.model.num_images_in_input``
     - 模型输入中的图像数量。
   * - ``actor.model.attn_implementation``
     - 注意力实现（例如 ``flash_attention_2``）。
   * - ``actor.model.low_cpu_mem_usage``
     - 使用低 CPU 内存初始化。
   * - ``actor.model.trust_remote_code``
     - 加载模型时信任远程代码。

.. note::

   OpenVLA（基础版）架构在 ``actor.model`` 下还额外提供 ``add_bias_linear`` 与
   ``add_qkv_bias``；OpenVLA-OFT 不使用它们。

**优化器配置**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``actor.optim.lr``
     - 策略网络的学习率。
   * - ``actor.optim.value_lr``
     - 价值头的学习率。
   * - ``actor.optim.adam_beta1`` / ``adam_beta2`` / ``adam_eps``
     - Adam 优化器超参数。
   * - ``actor.optim.weight_decay``
     - L2 权重衰减。
   * - ``actor.optim.clip_grad``
     - 梯度裁剪范数。
   * - ``actor.optim.critic_warmup_steps``
     - 在更新策略前，仅训练价值头的步数（0 表示禁用预热）。

分词器（``actor.tokenizer.*``）与 FSDP（``actor.fsdp_config.*``）子节遵循
:doc:`agentic_config` 与 :doc:`basic_config` 中的共享 schema；FSDP 预设从
``training_backend/fsdp`` 挂载。

环境专用配置
------------------------------------

以下参数位于环境预设中（通过 ``defaults`` 挂载），以 LIBERO-10 为例。

**环境类型**

.. code:: yaml

  env_type: libero
  task_suite_name: libero_10

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``env_type``
     - 模拟器类型（``libero`` 表示 LIBERO 基准）。
   * - ``task_suite_name``
     - 任务套件（``libero_10`` 表示 10 任务基准）。

**奖励配置**

.. code:: yaml

  use_rel_reward: true
  reward_coef: 5.0

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``use_rel_reward``
     - 使用相对奖励（当前步与上一步奖励之差）。
   * - ``reward_coef``
     - 用于缩放奖励的系数。

**随机化与分组**

.. code:: yaml

  seed: 0
  group_size: 1
  use_fixed_reset_state_ids: True

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``seed``
     - 环境初始化的随机种子。
   * - ``group_size``
     - 每组的环境数量；继承自 ``algorithm.group_size``。
   * - ``use_fixed_reset_state_ids``
     - 使用固定的重置状态 ID（False 表示随机化）。GRPO 始终为 True；PPO 默认 False。

**视频录制**

.. code:: yaml

  video_cfg:
    save_video: true
    info_on_video: true
    video_base_dir: ${runner.logger.log_path}/video/train

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``video_cfg.save_video``
     - 启用视频录制。
   * - ``video_cfg.info_on_video``
     - 在视频上叠加训练信息。
   * - ``video_cfg.video_base_dir``
     - 视频保存目录。

**相机配置**

.. code:: yaml

  init_params:
    camera_heights: 256
    camera_widths: 256

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 参数
     - 说明
   * - ``init_params.camera_heights``
     - 相机图像高度（像素）。
   * - ``init_params.camera_widths``
     - 相机图像宽度（像素）。

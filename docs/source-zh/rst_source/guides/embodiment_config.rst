具身智能配置
=============

本节介绍具身智能 RL 训练专用的配置参数（机器人操作、模拟器、VLA 模型）。
这些参数扩展了 :doc:`basic_config` 中的共享配置。

.. contents::
   :depth: 1
   :local:

defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/manikill_put_carrot_on_plate_in_scene@env.train
    - env/manikill_put_carrot_on_plate_in_scene@env.eval

``defaults``：Hydra 配置继承。指定训练与评估加载的环境配置。

hydra
~~~~~~~~~~~~~~~

.. code:: yaml

  hydra:
    searchpath:
      - file://${oc.env:REPO_PATH}/config/

``hydra.searchpath``：额外的配置文件搜索路径。

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    only_eval: False
    max_prompt_length: 30
    overlap_env_bootstrap: False

``runner.only_eval``：只运行评估，不进行训练。

``runner.max_prompt_length``：最大提示长度（token 数）。

``runner.overlap_env_bootstrap``：
将环境的 bootstrap（重置/reset）过程与 Actor 训练过程重叠，以隐藏重置延迟。
当环境重置较慢时，此功能非常有用。
**注意：** 仅在 ``env.train.enable_offload`` 为 False 时生效。
如果环境和 Actor 共享同一个加速器，开启此功能可能会增加重合期间的 GPU 显存压力。

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    normalize_advantages: True
    kl_penalty: kl

    reward_type: chunk_level
    logprob_type: token_level
    entropy_type: token_level

``algorithm.normalize_advantages``：是否对优势值归一化处理。

``algorithm.reward_type``：奖励聚合层级（chunk_level、action_level）。

``algorithm.logprob_type``：对数概率的计算层级。

``algorithm.entropy_type``：熵的计算层级。

env
~~~~~~~~~~~~~~~

.. code:: yaml

  env:
    group_name: "EnvGroup"
    enable_offload: True

    train:
      rollout_epoch: 1
      total_num_envs: null
      auto_reset: False
      ignore_terminations: False
      use_fixed_reset_state_ids: True
      max_episode_steps: 10

    eval:
      rollout_epoch: 1
      total_num_envs: null
      auto_reset: False
      ignore_terminations: False
      use_fixed_reset_state_ids: True
      max_episode_steps: 10

``env.group_name``：环境 worker 组的逻辑名称。

``env.enable_offload``：启用环境侧的下放以降低内存占用。

``env.train.rollout_epoch``：每个训练步骤前的 rollout 轮数。

``env.train.total_num_envs``：训练用的并行环境总数。

``env.train.auto_reset``：训练时当 episode 终止时自动重置环境。

``env.train.ignore_terminations``：训练时忽略 episode 终止（如果启用，episode 仅在达到 ``max_episode_steps`` 时结束）。

``env.train.use_fixed_reset_state_ids``：使用固定的 reset 状态 ID（false 则随机化）。GRPO 始终为 True，PPO 默认为 False。

``env.train.max_episode_steps``：训练时每个 episode 的最大步数。

``env.eval.rollout_epoch``：评估 rollout 轮数；多轮在相同种子下跑完测试集后取平均。

``env.eval.total_num_envs``：评估用的并行环境总数。

``env.eval.auto_reset``：评估时当 episode 终止时自动重置环境。

``env.eval.ignore_terminations``：评估时忽略 episode 终止（如果启用，episode 仅在达到 ``max_episode_steps`` 时结束）。

``env.eval.use_fixed_reset_state_ids``：使用固定的 reset 状态 ID（false 则随机化）。GRPO 始终为 True，PPO 默认为 False。

``env.eval.max_episode_steps``：评估时每个 episode 的最大步数。

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
      max_new_tokens: 7

    group_name: "RolloutGroup"
    backend: "huggingface"
    enable_offload: True
    pipeline_stage_num: 2

    model:
      model_path: "/path/to/hf_model"
      precision: ${actor.model.precision}

**sampling_params（VLA 等自回归策略）：**

``rollout.sampling_params.do_sample``：False 时使用贪心解码。

``rollout.sampling_params.temperature_train`` / ``temperature_eval``：训练与评估时的采样温度。

``rollout.sampling_params.top_k`` / ``top_p``：top-k 与 nucleus 采样参数。

``rollout.sampling_params.repetition_penalty``：重复惩罚系数。

``rollout.sampling_params.max_new_tokens``：单步最大生成 token 数（动作维度）。

连续策略（MLP、CNN、OpenPI、GR00T 等）不使用 ``rollout.sampling_params``。

``rollout.group_name``：rollout worker 组的逻辑名称。

``rollout.backend``：模型后端（huggingface、vllm）。

``rollout.enable_offload``：启用 rollout 模型下放以降低显存占用。

``rollout.pipeline_stage_num``：rollout 的流水线阶段数。

``rollout.model.model_path``：rollout 使用的模型权重路径（可与 actor 共享）。

``rollout.model.precision``：rollout 推理精度。

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    group_name: "ActorGroup"
    training_backend: "fsdp"
    micro_batch_size: 8
    global_batch_size: 160
    enable_offload: True

    model:
      model_path: "/path/to/hf_model"
      model_type: "openvla_oft"
      action_dim: 7
      num_action_chunks: 8
      use_proprio: False
      unnorm_key: bridge_orig
      value_type: ${algorithm.reward_type}
      val_micro_batch_size: 8
      center_crop: True
      do_sample: False

      precision: "bf16"
      add_bias_linear: False
      add_qkv_bias: True
      vocab_size: 32000
      hidden_size: 4096
      policy_setup: "widowx_bridge"
      image_size: [224, 224]
      is_lora: True
      lora_rank: 32
      lora_path: /storage/models/oft-sft/lora_004000
      num_images_in_input: 1
      attn_implementation: "flash_attention_2"
      low_cpu_mem_usage: True
      trust_remote_code: True

    tokenizer:
      tokenizer_type: "HuggingFaceTokenizer"
      tokenizer_model: "/storage/download_models/Openvla-oft-SFT-libero10-trajall/"
      extra_vocab_size: 421
      use_fast: False
      trust_remote_code: True
      padding_side: "right"

    optim:
      lr: 1.0e-4
      value_lr: 3.0e-3
      adam_beta1: 0.9
      adam_beta2: 0.999
      adam_eps: 1.0e-05
      clip_grad: 10.0

``actor.group_name``：actor worker 组的逻辑名称。

``actor.training_backend``：训练后端（分布式 FSDP）。

``actor.micro_batch_size``：每张 GPU 的微批大小。

``actor.global_batch_size``：全局批大小（跨所有 GPU）。

``actor.enable_offload``：启用模型下放以降低内存占用。

**模型配置：**

``actor.model.model_type``：模型类型（openvla_oft）。

``actor.model.action_dim``：动作空间维度。

``actor.model.num_action_chunks``：每条序列的动作块数量。

``actor.model.use_proprio``：是否使用本体感知信息。

``actor.model.unnorm_key``：动作反归一化的键。

``actor.model.value_type``：价值函数类型（继承自 algorithm.reward_type）。

``actor.model.val_micro_batch_size``：价值函数计算的微批大小。

``actor.model.center_crop``：是否对输入图像做中心裁剪。

``actor.model.do_sample``：推理时是否采样。

``actor.model.precision``：数值精度（bf16/fp16/fp32）。

``actor.model.add_bias_linear / add_qkv_bias``：线性/QKV 是否加 bias。

``actor.model.vocab_size / hidden_size``：词表大小与隐藏维度。

``actor.model.policy_setup``：策略配置（widowx_bridge）。

``actor.model.image_size``：输入图像尺寸 [H, W]。

``actor.model.is_lora / lora_rank / lora_path``：是否使用 LoRA、秩与权重路径。

``actor.model.megatron_checkpoint``：模型 checkpoint 路径。

``actor.model.num_images_in_input``：输入的图像数量。

``actor.model.attn_implementation``：注意力实现（flash_attention_2）。

``actor.model.low_cpu_mem_usage``：低内存初始化。

``actor.model.trust_remote_code``：加载模型时信任远程代码。

**分词器配置：**

``actor.tokenizer.tokenizer_type``：分词器类型（HuggingFaceTokenizer）。

``actor.tokenizer.tokenizer_model``：分词器模型路径。

``actor.tokenizer.extra_vocab_size``：额外词表大小。

``actor.tokenizer.use_fast``：是否使用 fast 版本。

``actor.tokenizer.trust_remote_code``：信任远程代码。

``actor.tokenizer.padding_side``：填充方向（left/right）。

**优化器配置：**

``actor.optim.lr``：策略网络学习率。

``actor.optim.value_lr``：价值网络学习率。

``actor.optim.adam_beta1/beta2/eps``：Adam 超参数。

``actor.optim.clip_grad``：梯度裁剪阈值。

环境配置参考
----------------------

以下示例以 Libero-10 为例说明环境关键参数。

路径为

**环境类型**

.. code:: yaml

  env_type: libero
  task_suite_name: libero_10

``env_type``：模拟器类型（libero 表示 Libero 基准）。

``task_suite_name``：任务集合（libero_10 表示 10 个任务的基准）。

**Episode 配置**

.. code:: yaml

  auto_reset: False
  ignore_terminations: False
  max_episode_steps: 512

``auto_reset``：episode 结束时是否自动重置（在 ``env.train`` / ``env.eval`` 中配置）。

``ignore_terminations``：训练时是否忽略终止（在 ``env.train`` 中配置）。

``max_episode_steps``：每个 episode 的最大步数（复杂 Libero 任务通常取 512）。

**奖励配置**

.. code:: yaml

  use_rel_reward: true
  reward_coef: 5.0

``use_rel_reward``：使用相对奖励（当前步与前一状态的差值）。

``reward_coef``：奖励缩放系数（如 5.0 强化奖励信号）。

**随机化与分组**

.. code:: yaml

  seed: 0
  group_size: 1
  use_fixed_reset_state_ids: True

``seed``：环境初始化随机种子（0 便于复现）。

``group_size``：每个分组的环境数（继承自 algorithm.group_size）。

``use_fixed_reset_state_ids``：是否使用固定 reset 状态（GRPO 为 True，PPO 默认 False）。

**环境规模**

.. code:: yaml

  total_num_envs: null

``total_num_envs``：总并行环境数用于训练或评估。

**视频记录**

.. code:: yaml

  video_cfg:
    save_video: true
    info_on_video: true
    video_base_dir: ${runner.logger.log_path}/video/train

``video_cfg.save_video``：训练时保存视频。

``video_cfg.info_on_video``：在视频上叠加训练信息。

``video_cfg.video_base_dir``：视频保存目录。

**相机配置**

.. code:: yaml

  init_params:
    camera_heights: 256
    camera_widths: 256

``init_params.camera_heights``：相机图像高度（像素）。

``init_params.camera_widths``：相机图像宽度（像素）。

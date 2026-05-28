Embodiment Configuration
=========================

This section covers configuration parameters specific to embodied RL training
(robot manipulation, simulators, VLA models). These extend the shared
configuration described in :doc:`gpu_cluster_config`.

.. contents::
   :depth: 1
   :local:

defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/manikill_put_carrot_on_plate_in_scene@env.train
    - env/manikill_put_carrot_on_plate_in_scene@env.eval

``defaults``: Hydra configuration inheritance. Specifies which environment configurations to load for training and evaluation.

hydra
~~~~~~~~~~~~~~~

.. code:: yaml

  hydra:
    searchpath:
      - file://${oc.env:REPO_PATH}/config/

``hydra.searchpath``: Additional search paths for configuration files.


runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    only_eval: False
    max_prompt_length: 30
    overlap_env_bootstrap: False

``runner.only_eval``: Run evaluation only without training.

``runner.max_prompt_length``: Maximum prompt length in tokens.

``runner.overlap_env_bootstrap``:
Overlap environment bootstrap (reset) with actor training to hide reset latency.
This is particularly useful when environment reset is slow.
**Note:** This is only effective when ``env.train.enable_offload`` is False.
Enabling this may increase GPU memory pressure if the environment and actor share the same accelerator.

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    normalize_advantages: True
    kl_penalty: kl
    rollout_epoch: 1

    reward_type: chunk_level
    logprob_type: token_level
    entropy_type: token_level


    length_params:
      max_new_token: null
      max_length: 1024
      min_length: 1

``algorithm.normalize_advantages``: Normalize advantages across the batch.

``algorithm.rollout_epoch``: Number of rollout epochs per training step.

``algorithm.reward_type``: Reward aggregation level (chunk_level, action_level).

``algorithm.logprob_type``: Log probability computation level.

``algorithm.entropy_type``: Entropy computation level.

**length_params:**

``algorithm.length_params.max_new_token``: Maximum new tokens to generate.

``algorithm.length_params.max_length``: Maximum total sequence length.

``algorithm.length_params.min_length``: Minimum sequence length.

env
~~~~~~~~~~~~~~~

.. code:: yaml

  env:
    group_name: "EnvGroup"
    channel:
      name: "env_buffer_list"
      queue_name: "obs_buffer"
      queue_size: 0
    enable_offload: True

    train:
      total_num_envs: null
      auto_reset: False
      ignore_terminations: False
      use_fixed_reset_state_ids: True
      max_episode_steps: 10

    eval:
      total_num_envs: null
      auto_reset: False
      ignore_terminations: False
      use_fixed_reset_state_ids: True
      max_episode_steps: 10

``env.group_name``: Logical name for environment worker group.

``env.channel.name``: Shared memory channel name for inter-process communication.

``env.channel.queue_name``: Queue name for observation buffer.

``env.channel.queue_size``: Queue size (0 for unlimited).

``env.enable_offload``: Enable environment offloading to reduce memory usage.

``env.train.total_num_envs``: Total number of parallel environments for training.

``env.train.auto_reset``: Automatically reset environments when episodes terminate.

``env.train.ignore_terminations``: Ignore episode terminations during training (if enabled, episode only ends when it reaches the ``max_episode_steps``).

``env.train.use_fixed_reset_state_ids``: Use fixed reset state IDs (false for randomization). Always True for GRPO, default be False for PPO.

``env.train.max_episode_steps``: Maximum number of steps per episode for training.

``env.eval.total_num_envs``: Total number of parallel environments for evaluation.

``env.eval.auto_reset``: Automatically reset environments when episodes terminate for evaluation.

``env.eval.ignore_terminations``: Ignore episode terminations during evaluation (if enabled, episode only ends when it reaches the ``max_episode_steps`` for evaluation).

``env.eval.use_fixed_reset_state_ids``: Use fixed reset state IDs (false for randomization). Always True for GRPO, default be False for PPO.

``env.eval.max_episode_steps``: Maximum number of steps per episode for evaluation.

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    channel:
      name: ${env.channel.name}
      queue_name: "action_buffer"
      queue_size: 0
    mode: "collocate"
    backend: "huggingface"
    enforce_eager: True
    enable_offload: True
    pipeline_stage_num: 2


``rollout.channel.name``: Shared memory channel (inherits from env).

``rollout.channel.queue_name``: Queue name for action buffer.

``rollout.channel.queue_size``: Queue size.

``rollout.mode``: Rollout mode (collocate for shared GPU).

``rollout.backend``: Model backend (huggingface, vllm).

``rollout.pipeline_stage_num``: Number of pipeline stages for rollout.

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    channel:
      name: ${env.channel.name}
      queue_name: "replay_buffer"
      queue_size: 0
    training_backend: "fsdp"
    micro_batch_size: 8
    global_batch_size: 160
    enable_offload: True

    model:
      model_path: "/path/to/huggingface_model"
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


``actor.channel.name``: Shared memory channel (inherits from env).

``actor.channel.queue_name``: Queue name for replay buffer.

``actor.training_backend``: Training backend (fsdp for distributed training).

``actor.micro_batch_size``: Micro-batch size per GPU.

``actor.global_batch_size``: Global batch size across all GPUs.

``actor.enable_offload``: Enable model offloading to reduce memory usage.

**Model Configuration:**

``actor.model.model_type``: Model architecture name (openvla_oft).

``actor.model.model_path``: Path to huggingface model.

``actor.model.action_dim``: Action space dimensionality.

``actor.model.num_action_chunks``: Number of action chunks per sequence.

``actor.model.use_proprio``: Whether to use proprioceptive information.

``actor.model.unnorm_key``: Key for action normalization.

``actor.model.value_type``: Value function type (inherits from algorithm.reward_type).

``actor.model.val_micro_batch_size``: Micro-batch size for value function computation.

``actor.model.center_crop``: Whether to center crop input images.

``actor.model.do_sample``: Whether to use sampling during inference.

``actor.model.precision``: Numerical precision (bf16, fp16, fp32).

``actor.model.add_bias_linear``: Add bias to linear layers.

``actor.model.add_qkv_bias``: Add bias to QKV projections.

``actor.model.vocab_size``: Vocabulary size.

``actor.model.hidden_size``: Hidden dimension size.

``actor.model.policy_setup``: Policy configuration (widowx_bridge).

``actor.model.image_size``: Input image dimensions [height, width].

``actor.model.is_lora``: Whether to use LoRA fine-tuning.

``actor.model.lora_rank``: LoRA rank for low-rank adaptation.

``actor.model.lora_path``: Path to LoRA weights.

``actor.model.num_images_in_input``: Number of images in model input.

``actor.model.attn_implementation``: Attention implementation (flash_attention_2).

``actor.model.low_cpu_mem_usage``: Use low CPU memory initialization.

``actor.model.trust_remote_code``: Trust remote code in model loading.

**Tokenizer Configuration:**

``actor.tokenizer.tokenizer_type``: Tokenizer type (HuggingFaceTokenizer).

``actor.tokenizer.tokenizer_model``: Path to tokenizer model.

``actor.tokenizer.extra_vocab_size``: Additional vocabulary size.

``actor.tokenizer.use_fast``: Use fast tokenizer implementation.

``actor.tokenizer.trust_remote_code``: Trust remote code in tokenizer.

``actor.tokenizer.padding_side``: Padding side (left or right).

**Optimizer Configuration:**

``actor.optim.lr``: Learning rate for policy network.

``actor.optim.value_lr``: Learning rate for value function.

``actor.optim.adam_beta1/beta2``: Adam optimizer beta parameters.

``actor.optim.adam_eps``: Adam optimizer epsilon.

``actor.optim.clip_grad``: Gradient clipping norm.



Environment-Specific Configuration
------------------------------------

The following configuration describes the key parameters of the environment, using Libero-10 as an example.

The path is

**Environment Type**

.. code:: yaml

  env_type: libero
  task_suite_name: libero_10

``env_type``: Specifies the simulator type (libero for Libero benchmark).

``task_suite_name``: Specifies the task suite (libero_10 for 10-task benchmark).

**Episode Configuration**

.. code:: yaml

  auto_reset: ${algorithm.auto_reset}
  ignore_terminations: ${algorithm.ignore_terminations}
  max_episode_steps: 512

``auto_reset``: Automatically reset environment when episode terminates (inherits from algorithm config).

``ignore_terminations``: Ignore episode terminations during training (inherits from algorithm config).

``max_episode_steps``: Maximum number of steps per episode (512 for complex Libero tasks).

**Reward Configuration**

.. code:: yaml

  use_rel_reward: true
  reward_coef: 5.0

``use_rel_reward``: Use relative rewards (difference between current and previous step rewards).

``reward_coef``: Reward coefficient for scaling rewards (5.0 for amplified reward signals).

**Randomization and Groups**

.. code:: yaml

  seed: 0
  group_size: 1
  use_fixed_reset_state_ids: True

``seed``: Random seed for environment initialization (0 for reproducibility).

``group_size``: Number of environments per group (inherits from algorithm.group_size).

``use_fixed_reset_state_ids``: Use fixed reset state IDs (false for randomization). Always True for GRPO, default be False for PPO.

**Environment Scaling**

.. code:: yaml

  total_num_envs: null

``total_num_envs``: Total number of parallel environments for training or evaluation.

**Video Recording**

.. code:: yaml

  video_cfg:
    save_video: true
    info_on_video: true
    video_base_dir: ${runner.logger.log_path}/video/train

``video_cfg.save_video``: Enable video recording during training.

``video_cfg.info_on_video``: Overlay training information on videos.

``video_cfg.video_base_dir``: Directory to save training videos.

**Camera Configuration**

.. code:: yaml

  init_params:
    camera_heights: 256
    camera_widths: 256

``init_params.camera_heights``: Camera image height in pixels (256).

``init_params.camera_widths``: Camera image width in pixels (256).

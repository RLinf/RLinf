Embodiment Configuration
=========================

This section covers configuration parameters specific to embodied RL training
(robot manipulation, simulators, VLA models). These extend the shared
configuration described in :doc:`basic_config`.

defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/maniskill_put_on_plate_in_scene_25_main@env.train
    - env/maniskill_ood_template@env.eval
    - model/openvla_oft@actor.model
    - training_backend/fsdp@actor.fsdp_config
    - weight_syncer/patch_syncer@weight_syncer

``defaults``: Hydra configuration inheritance. Selects the environment, model,
training-backend, and weight-syncer presets to compose into the run. The
``<group>@<target>`` syntax mounts a preset onto a config node (e.g.
``model/openvla_oft@actor.model`` fills ``actor.model``).

hydra
~~~~~~~~~~~~~~~

.. code:: yaml

  hydra:
    searchpath:
      - file://${oc.env:EMBODIED_PATH}/config/

``hydra.searchpath``: Additional search paths for configuration files. ``EMBODIED_PATH``
points at the embodiment example config directory.

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

   * - Parameter
     - Description
   * - ``runner.only_eval``
     - Run evaluation only, without training.
   * - ``runner.overlap_env_bootstrap``
     - Overlap environment bootstrap (reset) with actor training to hide reset
       latency. Useful when environment reset is slow. Only effective when
       ``env.enable_offload`` is False; enabling it may increase GPU memory
       pressure when the environment and actor share the same accelerator.
   * - ``runner.enable_decoupled_mode``
     - Decouple Env Workers from Rollout Workers. When enabled, an Env Worker is
       no longer bound to a fixed Rollout Worker rank: Env Workers push
       observations to a shared Channel and idle Rollout Workers fetch batches
       dynamically. Helps when Env step time varies (long-tail latency) or there
       are more Env Workers than Rollout Workers. Requires
       ``env_world_size >= rollout_world_size`` and is not compatible with
       ``enable_p2p``. See :doc:`env_decoupled_mode`.

Shared keys such as ``runner.task_type`` (set to ``embodied``),
``runner.max_epochs``, ``runner.save_interval``, ``runner.val_check_interval``,
and ``runner.resume_dir`` are documented in :doc:`basic_config`.

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

   * - Parameter
     - Description
   * - ``algorithm.normalize_advantages``
     - Normalize advantages across the batch.
   * - ``algorithm.kl_penalty``
     - How to estimate the KL divergence (``kl`` or ``kl_penalty``).
   * - ``algorithm.reward_type``
     - Reward aggregation level (``chunk_level`` or ``action_level``). Also drives
       ``actor.model.value_type`` by default.
   * - ``algorithm.logprob_type``
     - Log-probability computation level (e.g., ``token_level``).
   * - ``algorithm.entropy_type``
     - Entropy computation level (e.g., ``token_level``).
   * - ``algorithm.adv_type``
     - Advantage estimator (e.g., ``gae`` for PPO).
   * - ``algorithm.loss_type``
     - Policy loss type. ``actor_critic`` enables PPO with a value head (requires
       ``actor.model.add_value_head: True``).
   * - ``algorithm.loss_agg_func``
     - How to aggregate token losses (e.g., ``token-mean``).
   * - ``algorithm.bootstrap_type``
     - How to bootstrap Q-values at episode boundaries: ``standard`` bootstraps
       only on truncation; ``always`` bootstraps on truncation or termination.
   * - ``algorithm.kl_beta``
     - Weight of the KL penalty added to rewards.
   * - ``algorithm.entropy_bonus``
     - Entropy reward coefficient.
   * - ``algorithm.clip_ratio_low`` / ``clip_ratio_high``
     - Lower/upper PPO clipping bounds for the importance ratio (asymmetric clip).
   * - ``algorithm.clip_ratio_c``
     - Dual-clip constant for the pessimistic PPO bound.
   * - ``algorithm.value_clip``
     - Value-function clipping threshold (PPO value loss).
   * - ``algorithm.huber_delta``
     - Huber-loss delta used by the value loss.
   * - ``algorithm.gamma``
     - Discount factor.
   * - ``algorithm.gae_lambda``
     - GAE smoothing factor.

``algorithm.group_size`` (responses per prompt; always > 1 for GRPO) is shared and
documented in :doc:`basic_config`.

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

   * - Parameter
     - Description
   * - ``env.group_name``
     - Logical name for the environment worker group.
   * - ``env.enable_offload``
     - Offload the environment to reduce memory usage.
   * - ``env.train.rollout_epoch`` / ``env.eval.rollout_epoch``
     - Number of rollout epochs per training/evaluation step. Evaluation metrics
       are averaged over passes with the same seeds.
   * - ``env.train.total_num_envs`` / ``env.eval.total_num_envs``
     - Total number of parallel environments for training/evaluation.
   * - ``env.train.auto_reset`` / ``env.eval.auto_reset``
     - Automatically reset environments when episodes terminate.
   * - ``env.train.ignore_terminations`` / ``env.eval.ignore_terminations``
     - Ignore episode terminations; when enabled, an episode only ends at
       ``max_episode_steps``.
   * - ``env.train.use_fixed_reset_state_ids`` / ``env.eval.use_fixed_reset_state_ids``
     - Use fixed reset state IDs (False randomizes). Always True for GRPO; default
       False for PPO.
   * - ``env.train.max_episode_steps`` / ``env.eval.max_episode_steps``
     - Maximum number of steps per episode (truncation horizon).
   * - ``env.train.max_steps_per_rollout_epoch`` / ``env.eval.max_steps_per_rollout_epoch``
     - Maximum environment steps collected per rollout epoch. Must be divisible by
       ``actor.model.num_action_chunks``.

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

**sampling_params (autoregressive VLA policies):** continuous policies (MLP, CNN,
OpenPI, GR00T, etc.) do not use ``rollout.sampling_params``.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``rollout.sampling_params.do_sample``
     - Deterministic decoding if False.
   * - ``rollout.sampling_params.temperature_train`` / ``temperature_eval``
     - Sampling temperature for training and evaluation.
   * - ``rollout.sampling_params.top_k`` / ``top_p``
     - Top-k and nucleus sampling parameters.
   * - ``rollout.sampling_params.repetition_penalty``
     - Penalize repeated tokens.
   * - ``rollout.group_name``
     - Logical name for the rollout worker group.
   * - ``rollout.backend``
     - Model backend (e.g., ``huggingface``).
   * - ``rollout.enable_offload``
     - Offload the rollout model to reduce GPU memory usage.
   * - ``rollout.pipeline_stage_num``
     - Number of pipeline stages for rollout.
   * - ``rollout.rollout_queue_size``
     - Only used when ``runner.enable_decoupled_mode`` is True. Caps how many Env
       shards a single Rollout Worker aggregates at once. ``0`` uses the default
       ``ceil(env_world_size // rollout_world_size)``; a smaller value reduces
       waiting time, a larger value improves inference batch utilization. See
       :doc:`env_decoupled_mode`.
   * - ``rollout.model.model_path``
     - Model checkpoint path used by rollout (may match the actor).
   * - ``rollout.model.precision``
     - Inference precision for rollout.

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

**Top-level**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``actor.group_name``
     - Logical name for the actor worker group.
   * - ``actor.training_backend``
     - Training backend (``fsdp`` for embodied VLA training).
   * - ``actor.micro_batch_size``
     - Micro-batch size per GPU.
   * - ``actor.global_batch_size``
     - Global batch size across all GPUs.
   * - ``actor.seed``
     - Global seed for reproducibility.
   * - ``actor.enable_offload``
     - Offload the actor model to reduce memory usage.

**Model Configuration**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``actor.model.model_type``
     - Model architecture name (e.g., ``openvla_oft``).
   * - ``actor.model.model_path``
     - Path to the HuggingFace model.
   * - ``actor.model.implement_version``
     - Implementation variant for OpenVLA-OFT: ``rlinf`` (RLinf-released LIBERO /
       ManiSkill models) or ``official`` (RoboTwin / upstream OpenVLA-OFT, which
       supports ``use_film`` and ``use_proprio``).
   * - ``actor.model.action_dim``
     - Action-space dimensionality.
   * - ``actor.model.num_action_chunks``
     - Number of action chunks per sequence.
   * - ``actor.model.use_proprio``
     - Whether to feed proprioceptive state to the model.
   * - ``actor.model.use_film``
     - Use FiLM language conditioning (``official`` OpenVLA-OFT only).
   * - ``actor.model.unnorm_key``
     - Key for action de-normalization statistics.
   * - ``actor.model.value_type``
     - Value-head granularity; inherits from ``algorithm.reward_type``.
   * - ``actor.model.add_value_head``
     - Attach a value head. Must be True when ``algorithm.loss_type`` is
       ``actor_critic``.
   * - ``actor.model.center_crop``
     - Whether to center-crop input images.
   * - ``actor.model.do_sample``
     - Whether to sample (vs. greedy) during action generation.
   * - ``actor.model.max_prompt_length``
     - Maximum prompt length in tokens fed to the VLA backbone.
   * - ``actor.model.precision``
     - Numerical precision (``bf16``, ``fp16``, ``fp32``).
   * - ``actor.model.vocab_size``
     - Vocabulary size.
   * - ``actor.model.hidden_size``
     - Hidden dimension size.
   * - ``actor.model.policy_setup``
     - Policy/embodiment setup (e.g., ``widowx_bridge``).
   * - ``actor.model.image_size``
     - Input image dimensions ``[height, width]``.
   * - ``actor.model.is_lora``
     - Whether to use LoRA fine-tuning.
   * - ``actor.model.lora_rank``
     - LoRA rank for low-rank adaptation.
   * - ``actor.model.lora_path``
     - Path to LoRA weights.
   * - ``actor.model.num_images_in_input``
     - Number of images in the model input.
   * - ``actor.model.attn_implementation``
     - Attention implementation (e.g., ``flash_attention_2``).
   * - ``actor.model.low_cpu_mem_usage``
     - Use low-CPU-memory initialization.
   * - ``actor.model.trust_remote_code``
     - Trust remote code during model loading.

.. note::

   The OpenVLA (base) architecture additionally exposes ``add_bias_linear`` and
   ``add_qkv_bias`` under ``actor.model``; OpenVLA-OFT does not use them.

**Optimizer Configuration**

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``actor.optim.lr``
     - Learning rate for the policy network.
   * - ``actor.optim.value_lr``
     - Learning rate for the value head.
   * - ``actor.optim.adam_beta1`` / ``adam_beta2`` / ``adam_eps``
     - Adam optimizer hyper-parameters.
   * - ``actor.optim.weight_decay``
     - L2 weight decay.
   * - ``actor.optim.clip_grad``
     - Gradient-clipping norm.
   * - ``actor.optim.critic_warmup_steps``
     - Number of steps to train only the value head before updating the policy
       (0 disables warm-up).

The tokenizer (``actor.tokenizer.*``) and FSDP (``actor.fsdp_config.*``) sub-sections
follow the shared schemas in :doc:`agentic_config` and :doc:`basic_config`; the FSDP
preset is mounted from ``training_backend/fsdp``.

Environment-Specific Configuration
------------------------------------

The following parameters live in the environment preset (mounted via ``defaults``),
using LIBERO-10 as an example.

**Environment Type**

.. code:: yaml

  env_type: libero
  task_suite_name: libero_10

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``env_type``
     - Simulator type (``libero`` for the LIBERO benchmark).
   * - ``task_suite_name``
     - Task suite (``libero_10`` for the 10-task benchmark).

**Reward Configuration**

.. code:: yaml

  use_rel_reward: true
  reward_coef: 5.0

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``use_rel_reward``
     - Use relative rewards (difference between current and previous step rewards).
   * - ``reward_coef``
     - Reward coefficient for scaling rewards.

**Randomization and Groups**

.. code:: yaml

  seed: 0
  group_size: 1
  use_fixed_reset_state_ids: True

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``seed``
     - Random seed for environment initialization.
   * - ``group_size``
     - Number of environments per group; inherits from ``algorithm.group_size``.
   * - ``use_fixed_reset_state_ids``
     - Use fixed reset state IDs (False randomizes). Always True for GRPO; default
       False for PPO.

**Video Recording**

.. code:: yaml

  video_cfg:
    save_video: true
    info_on_video: true
    video_base_dir: ${runner.logger.log_path}/video/train

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``video_cfg.save_video``
     - Enable video recording.
   * - ``video_cfg.info_on_video``
     - Overlay training information on videos.
   * - ``video_cfg.video_base_dir``
     - Directory to save videos.

**Camera Configuration**

.. code:: yaml

  init_params:
    camera_heights: 256
    camera_widths: 256

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``init_params.camera_heights``
     - Camera image height in pixels.
   * - ``init_params.camera_widths``
     - Camera image width in pixels.

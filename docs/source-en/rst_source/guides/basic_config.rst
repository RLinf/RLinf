Basic Configuration
===================

Below is a complete reference for the core configuration parameters shared across all RLinf workloads.
Every important key in the YAML is documented so that you can confidently adapt the file to your own cluster, model, or research ideas.
Parameters are grouped exactly by their top-level key.

This section covers the fundamental GPU and cluster configuration that applies to both embodied and agentic training.
For task-specific configuration, see :doc:`embodiment_config` and :doc:`agentic_config`.

hydra
~~~~~~

.. code:: yaml

  hydra:
    run:
      dir: .
    output_subdir: null

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``hydra.run.dir``
     - Working directory for Hydra runs.
   * - ``hydra.output_subdir``
     - Output subdirectory (``null`` disables subdirectory creation).

cluster
~~~~~~~~~~~~~~~

.. code:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      actor,inference,rollout: all

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``cluster.num_nodes``
     - Physical nodes to use for training.
   * - ``cluster.component_placement``
     - The placement strategy for each component. Each line is a
       ``component_names: resource_ranks`` mapping: the key is one or more
       component names (e.g. ``rollout`` or ``rollout,inference,actor``) and the
       value is the hardware (e.g. GPU) ranks allocated to them.

The ranks value accepts the following forms:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Value
     - Meaning
   * - ``all``
     - Use all accelerators in the cluster.
   * - ``3``
     - A single integer: use accelerator 3.
   * - ``0,2,3``
     - A comma-separated list: use accelerators 0, 2, and 3.
   * - ``0-3``
     - A hyphenated range: use accelerators 0, 1, 2, and 3.
   * - ``0-3,5,14``
     - A combination: use accelerators 0, 1, 2, 3, 5 (node 0), and 14
       (accelerator 6 on node 1).

For more advanced usage of component placement (e.g., heterogeneous cluster with different GPU models, robotic hardware, or CPU-only nodes) and customization in code, see :doc:`../concepts/placement`.

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    task_type: math
    logger:
      log_path: ${runner.output_dir}/${runner.experiment_name}
      project_name: rlinf
      experiment_name: ${runner.experiment_name}
      logger_backends: ["tensorboard"] # wandb, swanlab

    max_epochs: 5
    max_steps: -1

    val_check_interval: 1
    save_interval: 50

    seq_length: 2048

    resume_dir: null
    experiment_name: grpo-1.5b
    output_dir: ../results

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``runner.task_type``
     - Task type identifier (``math`` or ``embodied``).
   * - ``runner.logger.log_path``
     - Base directory for log files.
   * - ``runner.logger.project_name``
     - Project name for experiment tracking.
   * - ``runner.logger.experiment_name``
     - Specific experiment name.
   * - ``runner.logger.logger_backends``
     - List of logging backends (``tensorboard``, ``wandb``, ``swanlab``). See
       :doc:`logger`.
   * - ``runner.max_epochs``
     - Maximum number of training epochs.
   * - ``runner.max_steps``
     - Maximum training steps. If set to ``-1``, it is derived automatically from
       ``runner.max_epochs``.
   * - ``runner.val_check_interval``
     - How often to launch a validation rollout (``-1`` to disable).
   * - ``runner.save_interval``
     - Checkpoint frequency in trainer steps.
   * - ``runner.seq_length``
     - Total sequence length (prompt + generated response) fed into models.

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    group_size: 2

    logprob_forward_micro_batch_size: 1

    val_rollout_batch_size_per_gpu: 4

    loss_type: ppo
    loss_agg_func: "token-mean"
    kl_beta: 0.0
    kl_penalty_type: low_var_kl
    ratio_clip_eps: 0.2
    entropy_bonus: 0.0
    calculate_entropy: False
    clip_ratio_c: null

    adv_type: grpo
    normalize_advantages: True
    early_stop_imp_ratio: 5.0
    use_valid_token_scale: False

    sampling_params:
      do_sample: True
      temperature: 1.0
      top_k: 1000000
      top_p: 1.0
      repetition_penalty: 1.0

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``algorithm.group_size``
     - Responses per prompt (set > 1 to enable group baselines).
   * - ``algorithm.logprob_forward_micro_batch_size``
     - Micro-batch size for log-prob forward passes.
   * - ``algorithm.val_rollout_batch_size_per_gpu``
     - Validation rollout micro-batch per GPU.
   * - ``algorithm.loss_type``
     - Policy loss type (e.g., ``ppo``).
   * - ``algorithm.loss_agg_func``
     - How to aggregate token losses (e.g., ``token-mean``).
   * - ``algorithm.kl_beta``
     - Weight of the KL penalty added to rewards.
   * - ``algorithm.kl_penalty_type``
     - KL shaping variant (e.g., ``low_var_kl``).
   * - ``algorithm.ratio_clip_eps``
     - PPO clipping epsilon for importance ratios.
   * - ``algorithm.entropy_bonus``
     - Entropy reward coefficient.
   * - ``algorithm.calculate_entropy``
     - Whether to compute/persist entropy terms.
   * - ``algorithm.clip_ratio_c``
     - Dual-clip constant for the pessimistic PPO bound (``null`` disables it).
   * - ``algorithm.adv_type``
     - Advantage estimator type (e.g., ``grpo``).
   * - ``algorithm.normalize_advantages``
     - Normalize advantages across the batch.
   * - ``algorithm.early_stop_imp_ratio``
     - Stop an update early if ratios exceed this threshold.
   * - ``algorithm.use_valid_token_scale``
     - Scale losses/advantages by valid-token masks.
   * - ``algorithm.sampling_params.do_sample``
     - Deterministic decoding if False.
   * - ``algorithm.sampling_params.temperature``
     - Softmax temperature during sampling.
   * - ``algorithm.sampling_params.top_k``
     - Top-k cutoff (use a very large value to disable).
   * - ``algorithm.sampling_params.top_p``
     - Nucleus sampling threshold.
   * - ``algorithm.sampling_params.repetition_penalty``
     - Penalize repeated tokens.

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    group_name: "RolloutGroup"

    gpu_memory_utilization: 0.55

    model:
      model_path: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
      model_type: qwen2.5

    recompute_logprobs: True

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``rollout.group_name``
     - Logical name for rollout/inference workers.
   * - ``rollout.gpu_memory_utilization``
     - Target GPU memory utilization fraction.
   * - ``rollout.model.model_path``
     - Path to the HF model used by the generation backend.
   * - ``rollout.model.model_type``
     - Internal architecture tag used by the backend (e.g., ``qwen2.5``).
   * - ``rollout.recompute_logprobs``
     - Recompute log-probs for sampled sequences.

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    group_name: "ActorGroup"

    model:
      megatron_checkpoint: null

    seed: 1234

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``actor.group_name``
     - Logical name for the training (actor) workers.
   * - ``actor.model.megatron_checkpoint``
     - Path to a Megatron model checkpoint to load before training.
   * - ``actor.seed``
     - Global seed for reproducibility.

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    use_reward_model: false

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``reward.use_reward_model``
     - Whether to use a reward model.

critic
~~~~~~~~~~~~~~~

.. code:: yaml

  critic:
    use_critic_model: false

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Parameter
     - Description
   * - ``critic.use_critic_model``
     - Whether to use a critic model.

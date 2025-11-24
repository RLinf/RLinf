YAML Configuration
=====================


Below is a complete reference for the configuration file used in the RLinf
Every important key in the YAML is documented below so that you can confidently adapt the file to your own cluster, model, or research ideas.  
Parameters are grouped exactly by their top-level key.

For clarity, this section includes the following three main parts: 
**Basic Configuration**, **MATH-specific Configuration**, and **Embody-specific Configuration**.
Therefore, users can find the corresponding configuration information according to their own needs.

.. contents::
   :depth: 1
   :local:

Basic Configuration
---------------------

hydra
~~~~~~

.. code:: yaml

  hydra:
    run:
      dir: .
    output_subdir: null 

``hydra.run.dir``: Working directory for Hydra runs.

``hydra.output_subdir``: Output subdirectory (null disables subdirectory creation).


cluster
~~~~~~~~~~~~~~~

.. code:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      actor,inference,rollout: all


``cluster.num_nodes``: Physical nodes to use for training.

``cluster.component_placement``: 
The *placement strategy* for each component.
Each line of component placement config is a dictionary of ``component_names: resource_ranks``.
In this simple example of running on GPU nodes, the meaning is:

- The key is the names of components, e.g., ``rollout``, or ``rollout,inference,actor``
- The value is the hardware (e.g., GPU) ranks allocated to the components, which can be:
   - "all": use all accelerators in the cluster
   - A single integer, e.g., "3": use accelerator 3
   - A list of integers separated by comma, e.g., "0,2,3": use accelerator 0, 2, and 3
   - A range of integers separated by hyphen, e.g., "0-3": use accelerator 0, 1, 2, and 3
   - A combination of the above two, e.g., "0-3,5,14": use accelerator 0, 1, 2, 3, 5 (on node 0), and 14 (i.e., accelerator 6 on node 1)

For more advanced usage of component placement (e.g., heterogeneous cluster with different GPU models, robotic hardware, or CPU-only nodes) and customization in code, see :doc:`./placement`.

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

``runner.task_type``: Task type identifier, math or embodied.

**logger:**

``runner.logger.log_path``: Base directory for log files.

``runner.logger.project_name``: Project name for experiment tracking.

``runner.logger.experiment_name``: Specific experiment name.

``runner.logger.logger_backends``: List of logging backends (tensorboard, wandb, swanlab).

See more details about logger backends in :doc:`../advance/logger`.

``runner.max_epochs``: Maximum number of training epochs.

``runner.max_steps``: Maximum training steps. If set to -1, this defaults to set automatially based on the ``runner.max_epochs``.

``runner.val_check_interval``: How often to launch a validation rollout (-1 to disable).

``runner.save_interval``: Checkpoint frequency in trainer steps.

``runner.seq_length``: Total sequence length (prompt + generated response) fed into models.


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
      use_greedy: False
      temperature: 1.0
      top_k: 1000000
      top_p: 1.0
      repetition_penalty: 1.0


``algorithm.group_size``: Responses per prompt (set > 1 to enable group baselines).

``algorithm.logprob_forward_micro_batch_size``: Micro-batch size for log-prob forward passes.

``algorithm.val_rollout_batch_size_per_gpu``: Validation rollout micro-batch per GPU.

``algorithm.loss_type``: Policy loss type (e.g., ppo).

``algorithm.loss_agg_func``: How to aggregate token losses (e.g., token-mean).

``algorithm.kl_beta``: Weight of KL penalty added to rewards.

``algorithm.kl_penalty_type``: KL shaping variant (e.g., low_var_kl).

``algorithm.ratio_clip_eps``: PPO clipping epsilon for importance ratios.

``algorithm.entropy_bonus``: Entropy reward coefficient.

``algorithm.calculate_entropy``: Whether to compute/persist entropy terms.

``algorithm.adv_type``: Advantage estimator type (e.g., grpo).

``algorithm.normalize_advantages``: Normalize advantages across the batch.

``algorithm.early_stop_imp_ratio``: Stop an update early if ratios exceed this threshold.

``algorithm.use_valid_token_scale``: Scale losses/advantages by valid-token masks.

**sampling_params:**

``algorithm.sampling_params.use_greedy``: Deterministic decoding if True.

``algorithm.sampling_params.temperature``: Softmax temperature during sampling.

``algorithm.sampling_params.top_k``: Top-k cutoff (use a very large value to disable).

``algorithm.sampling_params.top_p``: Nucleus sampling threshold.

``algorithm.sampling_params.repetition_penalty``: Penalize repeated tokens.



rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    group_name: "RolloutGroup"

    gpu_memory_utilization: 0.55

    model_dir: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
    model_arch: qwen2.5

    recompute_logprobs: True

``rollout.gpu_memory_utilization``: Target GPU memory utilization fraction.

``rollout.group_name``: Logical name for rollout/inference workers.

``rollout.model_dir``: Path to the HF model used by the generation backend.

``rollout.model_arch``: Internal architecture tag used by the backend (e.g., qwen2.5).

``rollout.recompute_logprobs``: Recompute log-probs for sampled sequences.



actor
~~~~~~~~~~~~~~~

.. code:: yaml


  actor:
    group_name: "ActorGroup"

    checkpoint_load_path: null

    seed: 1234


**Top-level**

``actor.group_name``: Logical name for the training (actor) workers.

``actor.checkpoint_load_path``: Path to a checkpoint to load before training.

``actor.seed``: Global seed for reproducibility.

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    use_reward_model: false

``reward.use_reward_model``: Whether to use a reward model.

critic
~~~~~~~~~~~~~~~

.. code:: yaml

  critic:
    use_critic_model: false


``critic.use_critic_model``: Whether to use a critic model.



MATH-specific Configuration
----------------------------

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

``runner.enable_dynamic_batch_size``: Whether to user dynamic batch size when training by Megatron.

``runner.max_tokens_per_mbs``: Upper limit of tokens in a Megatron microbatch when dynamic batching is enabled.


algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:

    n_minibatches: 4
    training_batch_size_per_gpu: 1 
    rollout_batch_size_per_gpu: null 

    sampling_params:
      max_new_tokens: ${subtract:${runner.seq_length}, ${data.max_prompt_length}}
      min_new_tokens: 1

``algorithm.n_minibatches``: Number of gradient update per batch.

``algorithm.training_batch_size_per_gpu``: Micro-batch size on each actor GPU.

``algorithm.rollout_batch_size_per_gpu``: Inference micro-batch per GPU; null divides the global rollout batch evenly.


**sampling_params:**


``algorithm.sampling_params.max_new_tokens``: Max generated tokens; computed from runner.seq_length and data.max_prompt_length.

``algorithm.sampling_params.min_new_tokens``: Minimum generated tokens.



rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    enforce_eager: False         # if False, rollout engine will capture cuda graph, which will take more time to initialize.
    distributed_executor_backend: mp   # ray or mp
    disable_log_stats: False
    detokenize: False            # Whether to detokenize the output. During RL we actually don't need to detokenize it. Can be set to True for debugging.
    padding: null               # will be tokenizer.pad_token_id if null. it is used to filter megatron's padding for rollout engine
    eos: null                   # will be tokenizer.eos_token_id if null.

    attention_backend: triton

    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    
    validate_weight: False # whether to send all weights at first for weight comparison.
    validate_save_dir: null # the directory to save the weights for comparison. If validate_weight is True, this will be used to save the weights for comparison.
    print_outputs: False         # whether to print the outputs (token ids, texts, etc.) of rollout engine.

    sglang_decode_log_interval: 500000 # the interval for SGLang to log the decode time and other stats.
    max_running_requests: 64 # the maximum number of running requests in the rollout engine.
    cuda_graph_max_bs: 128 # the maximum batch size for cuda graph. If the batch size is larger than this, cuda graph will not be used.

    use_torch_compile: False # enable torch_compile in SGLang for rollout.
    torch_compile_max_bs: 128 # the maximum batch size for torch compile. If the batch size is larger than this, torch compile will not be used.



``rollout.enforce_eager``: If True, disable CUDA graph capture to shorten warm-up.

``rollout.distributed_executor_backend``: Backend for launching rollout workers (mp or ray).

``rollout.disable_log_stats``: Suppress periodic backend stats logging.

``rollout.detokenize``: Detokenize outputs for debugging (RL usually uses token ids only).

``rollout.padding``: Pad token id override; null uses tokenizer.pad id.

``rollout.eos``: EOS token id override; null uses tokenizer.eos id.

``rollout.attention_backend``: Attention kernel backend (e.g., triton). 

``rollout.tensor_parallel_size``: TP degree inside the generation backend.

``rollout.pipeline_parallel_size``: PP degree inside the generation backend.

See more details about the parallelism in :doc:`../advance/5D`.

``rollout.validate_weight``: Send full weights once for cross-check/validation.

``rollout.validate_save_dir``: Directory to store weights for comparison when validation is enabled.

``rollout.print_outputs``: Print token ids/texts from the engine for debugging.

``rollout.sglang_decode_log_interval``: Interval for SGLang to log decode stats.

``rollout.max_running_requests``: Max concurrent decode requests.

``rollout.cuda_graph_max_bs``: Max batch size eligible for CUDA graph.

``rollout.use_torch_compile``: Enable torch.compile inside SGLang.

``rollout.torch_compile_max_bs``: Max batch size eligible for torch.compile.



data
~~~~~~~~~~~~~~~

.. code:: yaml

  data:
    type: math
    max_prompt_length: 1024
    rollout_batch_size: 64
    val_rollout_batch_size: null
    num_workers: 2
    prompt_key: prompt
    shuffle: True
    validation_shuffle: True
    seed: 1234
    train_data_paths: ["../../data/boba/AReaL-boba-106k.jsonl"]
    val_data_paths: ["../../data/boba/AReaL-boba-106k.jsonl"]

``data.type``: Dataset/task family (e.g., math).

``data.max_prompt_length``: Maximum tokens allowed for prompts.

``data.rollout_batch_size``: Global rollout batch size across engines.

``data.val_rollout_batch_size``: Global validation rollout batch size; null falls back to data.rollout_batch_size.

``data.num_workers``: Data loader workers per actor rank.

``data.prompt_key``: JSONL key that stores the prompt text.

``data.shuffle``: Shuffle training data each epoch.

``data.validation_shuffle``: Shuffle validation data (usually keep True for on-policy eval variety).

``data.seed``: RNG seed for loaders and sampling.

``data.train_data_paths``: List of training JSONL file paths.

``data.val_data_paths``: List of validation JSONL file paths.

actor
~~~~~~~~~~~~~~~

.. code:: yaml


  actor:
    training_backend: megatron
    mcore_gpt: True
    spec_name: decoder_gpt

    offload_optimizer: True
    offload_weight: True
    offload_grad: True

    enable_dp_load_balance: False

    calculate_flops: False

    model:
      precision: fp16
      add_bias_linear: False

      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1

      activation: swiglu
      sequence_parallel: True
      # recompute_method: block
      # recompute_granularity: selective

      recompute_method: block
      recompute_granularity: full
      recompute_num_layers: 20

      seq_length: ${runner.seq_length}
      encoder_seq_length: ${runner.seq_length}

      normalization: rmsnorm

      position_embedding_type: rope

      apply_rope_fusion: True
      bias_dropout_fusion: False
      persist_layer_norm: False
      bias_activation_fusion: False
      attention_softmax_in_fp32: True
      batch_p2p_comm: False
      variable_seq_lengths: True
      gradient_accumulation_fusion: False
      moe_token_dispatcher_type: alltoall
      use_cpu_initialization: False

    optim:
      optimizer: adam
      bf16: False
      fp16: True
      lr: 2e-05
      adam_beta1: 0.9
      adam_beta2: 0.95
      adam_eps: 1.0e-05
      min_lr: 2.0e-6
      weight_decay: 0.05
      use_distributed_optimizer: True
      overlap_grad_reduce: True
      overlap_param_gather: True
      optimizer_enable_pin: false
      overlap_param_gather_with_optimizer_step: False
      clip_grad: 1.0
      loss_scale_window: 5

    lr_sched:
      lr_warmup_fraction: 0.01
      lr_warmup_init: 0.0
      lr_warmup_iters: 0
      max_lr: 2.0e-5
      min_lr: 0.0
      lr_decay_style: constant
      lr_decay_iters: 10

    tokenizer:
      tokenizer_model: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
      use_fast: False
      trust_remote_code: True
      padding_side: 'right'

    megatron:
      ddp_bucket_size: null
      distributed_backend: nccl # Support 'nccl' and 'gloo'
      distributed_timeout_minutes: 30
      ckpt_format: torch
      use_dist_ckpt: False
      tp_comm_bootstrap_backend: nccl
      tp_comm_overlap_cfg: null 
      use_hf_ckpt: True # if true, will transfer hf model to generate megatron checkpoint and use it for training.
      
      ckpt: # config for ckpt convertor
        model: DeepSeek-R1-Distill-Qwen-1.5B
        model_type: null # will be set by hf model's config if null
        hf_model_path: ${rollout.model_dir} # path to the hf model
        save_path: ${runner.output_dir}/${runner.experiment_name}/actor/megatron_ckpt_from_hf
        use_gpu_num : 0
        use_gpu_index: null # 
        process_num: 16 # number of processes to use for checkpointing
        tensor_model_parallel_size: ${actor.model.tensor_model_parallel_size}
        pipeline_model_parallel_size: ${actor.model.pipeline_model_parallel_size}

    fsdp_config:

      strategy: "fsdp"

      sharding_strategy: "no_shard"

      cpu_offload: False
      offload_pin_memory: False
      reshard_after_forward: True

      enable_gradient_accumulation: True
      forward_prefetch: False
      limit_all_gathers: False
      backward_prefetch: null
      use_orig_params: False
      use_liger_kernel: False

      fsdp_size: -1

      mixed_precision:
        param_dtype: ${actor.model.precision}
        reduce_dtype: ${actor.model.precision}
        buffer_dtype: ${actor.model.precision}

      amp:
        enabled: False
        precision: "bf16"
        use_grad_scaler: False

**Top-level**


``actor.training_backend``: Training backend (megatron).

``actor.mcore_gpt``: Use Megatron-Core GPT stack. 

``actor.spec_name``: Model spec/preset name (e.g., decoder-only GPT). 

``actor.offload_optimizer``: Offload optimizer state to CPU to reduce GPU memory.

``actor.offload_weight``: Offload model weights to CPU when possible (ZeRO-style). 

``actor.offload_grad``: Offload gradients to CPU to reduce GPU memory.

``actor.enable_dp_load_balance``: Enable data-parallel load balancing. 

``actor.calculate_flops``: Compute and log FLOPs for profiling.


**Model sub-section**

``actor.model.precision``: Numerical precision for training (e.g., fp16).

``actor.model.add_bias_linear``: Add bias terms to linear layers.

``actor.model.tensor_model_parallel_size``: TP degree for actor.

``actor.model.pipeline_model_parallel_size``: PP degree for actor.

``actor.model.activation``: Activation function (e.g., swiglu).

``actor.model.sequence_parallel``: Enable sequence parallelism (requires TP).

``actor.model.recompute_method``: Activation recompute strategy (e.g., block).

``actor.model.recompute_granularity``: Recompute scope (e.g., full or selective).

``actor.model.recompute_num_layers``: Number of layers to checkpoint/recompute.

``actor.model.seq_length``: Decoder context length for training.

``actor.model.encoder_seq_length``: Encoder length (for encoder-decoder; mirrors seq_length here).

``actor.model.normalization``: Norm layer type (e.g., rmsnorm).

``actor.model.position_embedding_type``: Positional embedding type (e.g., rope).

``actor.model.apply_rope_fusion``: Use fused RoPE kernels if available.

``actor.model.bias_dropout_fusion``: Fuse bias + dropout kernels. 

``actor.model.persist_layer_norm``: Persist LN params in higher precision. 

``actor.model.bias_activation_fusion``: Fuse bias + activation kernels. 

``actor.model.attention_softmax_in_fp32``: Compute attention softmax in FP32 for stability.

``actor.model.batch_p2p_comm``: Batch P2P communications across layers. 

``actor.model.variable_seq_lengths``: Allow variable sequence lengths per micro-batch.

``actor.model.gradient_accumulation_fusion``: Fused gradient accumulation. 

``actor.model.moe_token_dispatcher_type``: MoE token dispatcher (e.g., alltoall).

``actor.model.use_cpu_initialization``: Initialize weights on CPU to reduce GPU spikes.

**Optimizer**

``actor.optim.optimizer``: Optimizer choice (adam).

``actor.optim.bf16 / actor.optim.fp16``: Mixed precision flags.

``actor.optim.lr``: Base learning rate.

``actor.optim.adam_beta1 / adam_beta2 / adam_eps``: Adam hyper-parameters.

``actor.optim.min_lr``: Minimum LR (for schedulers that decay below base LR).

``actor.optim.weight_decay``: L2 weight decay.

``actor.optim.use_distributed_optimizer``: Use Megatron distributed optimizer.

``actor.optim.overlap_grad_reduce``: Overlap gradient reduction with backward pass.

``actor.optim.overlap_param_gather``: Overlap parameter all-gather with forward pass.

``actor.optim.optimizer_enable_pin``: Pin optimizer memory. 

``actor.optim.overlap_param_gather_with_optimizer_step``: Overlap param gather with step. 

``actor.optim.clip_grad``: Global gradient clipping norm.

``actor.optim.loss_scale_window``: Dynamic loss scale window for FP16. 

**LR schedule**

``actor.lr_sched.lr_warmup_fraction``: Warm-up as a fraction of total iters.

``actor.lr_sched.lr_warmup_init``: Initial LR value during warm-up.

``actor.lr_sched.lr_warmup_iters``: Warm-up iterations (overrides fraction when > 0).

``actor.lr_sched.max_lr / min_lr``: LR bounds for schedulers.

``actor.lr_sched.lr_decay_style``: Decay policy (e.g., constant).

``actor.lr_sched.lr_decay_iters``: Total decay iterations.

**Tokenizer**

``actor.tokenizer.tokenizer_model``: Path/name of the tokenizer.

``actor.tokenizer.use_fast``: Use HF fast tokenizer.

``actor.tokenizer.trust_remote_code``: Allow custom tokenizer code.

``actor.tokenizer.padding_side``: left or right padding.

**Megatron integration**

``actor.megatron.ddp_bucket_size``: DDP gradient bucket size. 

``actor.megatron.distributed_backend``: Distributed backend (nccl or gloo).

``actor.megatron.distributed_timeout_minutes``: Backend communication timeout.

``actor.megatron.ckpt_format``: Checkpoint format (e.g., torch).

``actor.megatron.use_dist_ckpt``: Use distributed checkpointing (sharded). 

``actor.megatron.tp_comm_bootstrap_backend``: Backend used for TP bootstrap (e.g., nccl).

``actor.megatron.tp_comm_overlap_cfg``: YAML path for TP comm/compute overlap. 

``actor.megatron.use_hf_ckpt``: Convert/load from a HuggingFace checkpoint for training.

**Megatron checkpoint converter**

``actor.megatron.ckpt.model``: Model name for the converter metadata.

``actor.megatron.ckpt.model_type``: Model type; inferred from HF config when null.

``actor.megatron.ckpt.hf_model_path``: Source HF model path.

``actor.megatron.ckpt.save_path``: Target directory to write Megatron checkpoints.

``actor.megatron.ckpt.use_gpu_num``: Number of GPUs to use for conversion. 

``actor.megatron.ckpt.use_gpu_index``: Specific GPU index to use. 

``actor.megatron.ckpt.process_num``: CPU processes for conversion work.

``actor.megatron.ckpt.tensor_model_parallel_size``: TP degree for converted checkpoints.

``actor.megatron.ckpt.pipeline_model_parallel_size``: PP degree for converted checkpoints.

**FSDP Integration:**

``actor.fsdp_config.strategy``: Determines the FSDP strategy used, supporting fsdp and fsdp2 (case-insensitive).

``actor.fsdp_config.sharding_strategy``: FSDP/FSDP2 parameter, indicating the sharding strategy used by FSDP, supporting full_shard, shard_grad_op, hybrid_shard, and no_shard.

``actor.fsdp_config.cpu_offload``: FSDP2 parameter, determines whether FSDP2 places parameters on the CPU side, transmitting them to the GPU side only when necessary.

``actor.fsdp_config.offload_pin_memory``: FSDP2 parameter, only effective when the cpu_offload option is True. If true, the CPU-side memory is pinned memory to improve transmission efficiency.

``actor.fsdp_config.reshard_after_forward``: FSDP2 parameter, indicates whether to reslice parameters after forward propagation to save GPU memory.

``actor.fsdp_config.enable_gradient_accumulation``: FSDP/FSDP2 parameter, indicates whether to enable gradient accumulation. If true, communication and gradient updates are only performed after the last micro-batch. Enabling this increases GPU memory usage but speeds up training.

``actor.fsdp_config.forward_prefetch``: FSDP parameter, indicates whether to prefetch the next all-gather operation during forward propagation. Enabling this increases GPU memory usage; it is recommended to enable it when GPU memory is sufficient to overlap communication and computation, thereby improving performance.

``actor.fsdp_config.limit_all_gathers``: FSDP parameter, indicates whether to limit the number of concurrent all-gather operations. It is recommended to enable this when CPU or memory is a bottleneck.

``actor.fsdp_config.backward_prefetch``: FSDP parameter, indicating the prefetch strategy during backpropagation (null/'pre'/'post'). If 'pre', the next all-gather operation is prefetched during gradient computation, resulting in more aggressive overlap and higher throughput. If 'post', the next all-gather operation is prefetched after the current gradient computation is complete, which is more conservative than 'pre'.

``actor.fsdp_config.use_orig_params``: FSDP parameter, indicating whether to use the module's original parameters, exposing the original parameters (nn.Module.named_parameters) instead of the flattened parameters of FSDP. This improves compatibility but introduces additional communication overhead and reduces performance.

``actor.fsdp_config.use_liger_kernel``: FSDP/FSDP2 parameter, determines whether to use liger_kernel (currently only supported for some models, including qwen2.5 and qwen2.5-vl). Enabling it can reduce GPU memory usage and improve training speed.

``actor.fsdp_config.fsdp_size``: FSDP2 parameter. If not -1, FSDP2 will group slices according to the size specified by this parameter.

``actor.fsdp_config.mixed_precision.param_dtype``: FSDP/FSDP2 parameter, specifying the parameter type.

``actor.fsdp_config.mixed_precision.reduce_dtype``: FSDP/FSDP2 parameter, specifying the data type used during reduction.

``actor.fsdp_config.mixed_precision.buffer_dtype``: FSDP parameter, specifying the data type used for the buffer.

``actor.fsdp_config.amp.enabled``: FSDP/FSDP2 parameter, indicating whether automatic mixed-precision training is enabled.

``actor.fsdp_config.amp.precision``: FSDP/FSDP2 parameter, indicating the numerical precision used by AMP.

``actor.fsdp_config.amp.use_grad_scaler``: FSDP/FSDP2 parameter, indicating whether the gradient scaler is enabled.

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0


``reward.reward_type``: Which reward type to use for the training.

``reward.reward_scale``: when the answer is correct, it receives ``reward_scale``; when it is incorrect, it receives ``-reward_scale``.


Embody-specific Configuration
-------------------------------


defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/train: PutCarrotOnPlateInScene
    - env/eval: PutCarrotOnPlateInScene

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

``runner.only_eval``: Run evaluation only without training.

``runner.max_prompt_length``: Maximum prompt length in tokens.

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    auto_reset: True
    ignore_terminations: True
    use_fixed_reset_state_ids: False
    normalize_advantages: True
    kl_penalty: kl

    n_chunk_steps: 10
    n_eval_chunk_steps: 10
    num_group_envs: 32
    rollout_epoch: 1

    reward_type: chunk_level
    logprob_type: token_level
    entropy_type: token_level


    length_params:
      max_new_token: null
      max_length: 1024
      min_length: 1

``algorithm.auto_reset``: Automatically reset environments when episodes terminate.

``algorithm.ignore_terminations``: Ignore episode terminations during training (if enabled, episode only ends when it reaches the ``max_episode_steps``).

``algorithm.use_fixed_reset_state_ids``: Use fixed reset state IDs (false for randomization). Always True for GRPO, default be False for PPO.

``algorithm.normalize_advantages``: Normalize advantages across the batch.

``algorithm.n_chunk_steps``: Number of chunks (i.e., times the model is called to predict action chunks) within one rollout epoch.

``algorithm.n_eval_chunk_steps``: Number of chunks in evaluation.

``algorithm.num_group_envs``: Number of environment groups.

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

``env.group_name``: Logical name for environment worker group.

``env.channel.name``: Shared memory channel name for inter-process communication.

``env.channel.queue_name``: Queue name for observation buffer.

``env.channel.queue_size``: Queue size (0 for unlimited).

``env.enable_offload``: Enable environment offloading to reduce memory usage.

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

``rollout.pipeline_stage_num``: Number of pipeline stages for model parallelism.

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
      model_name: "openvla_oft"
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
      ckpt_path: null
      num_images_in_input: 1
      use_wrist_image: False
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

``actor.model.model_name``: Model architecture name (openvla_oft).

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

``actor.model.ckpt_path``: Path to model checkpoint.

``actor.model.num_images_in_input``: Number of images in model input.

``actor.model.use_wrist_image``: Whether to use wrist image in model input.

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



Env-based 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following configuration describes the key parameters of the environment, using Libero-10 as an example.

The path is 

**Environment Type**

.. code:: yaml

  simulator_type: libero
  task_suite_name: libero_10

``simulator_type``: Specifies the simulator type (libero for Libero benchmark).

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
  num_task: ${algorithm.num_group_envs}
  num_group: ${algorithm.num_group_envs}
  group_size: ${algorithm.group_size}
  use_fixed_reset_state_ids: ${algorithm.use_fixed_reset_state_ids}

``seed``: Random seed for environment initialization (0 for reproducibility).

``num_task``: Number of tasks to use (inherits from algorithm.num_group_envs).

``num_group``: Number of environment groups (inherits from algorithm.num_group_envs).

``group_size``: Number of environments per group (inherits from algorithm.group_size).

``use_fixed_reset_state_ids``: Use fixed reset state IDs (false for randomization). Always True for GRPO, default be False for PPO (inherits from algorithm.use_fixed_reset_state_ids).

**Input Configuration**

.. code:: yaml

  use_wrist_image: False

``use_wrist_image``: If set to True, wrist images will be added in model inputs.

**Environment Scaling**

.. code:: yaml

  num_envs: ${multiply:${algorithm.group_size}, ${algorithm.num_group_envs}}

``num_envs``: Total number of environments (calculated as group_size × num_group_envs).

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
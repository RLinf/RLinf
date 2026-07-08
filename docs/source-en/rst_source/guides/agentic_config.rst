Agentic RL Configuration
=========================

This section covers configuration parameters specific to agentic and reasoning RL training
(math reasoning, coding agents, multi-agent systems). These extend the shared
configuration described in :doc:`basic_config`.

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``runner.enable_dynamic_batch_size``
     - Whether to use dynamic batch size when training with Megatron.
   * - ``runner.max_tokens_per_mbs``
     - Upper limit of tokens in a Megatron micro-batch when dynamic batching is
       enabled.

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    n_minibatches: 4
    training_batch_size_per_gpu: 1
    rollout_batch_size_per_gpu: null

    recompute_logprobs: True
    shuffle_rollout: False

    clip_ratio_low: null
    clip_ratio_high: null

    sampling_params:
      max_new_tokens: ${subtract:${runner.seq_length}, ${data.max_prompt_length}}
      min_new_tokens: 1

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``algorithm.n_minibatches``
     - Number of gradient updates per batch.
   * - ``algorithm.training_batch_size_per_gpu``
     - Micro-batch size on each actor GPU.
   * - ``algorithm.rollout_batch_size_per_gpu``
     - Inference micro-batch per GPU; ``null`` divides the global rollout batch
       evenly across inference instances.
   * - ``algorithm.recompute_logprobs``
     - Recompute log-probs in the training engine instead of trusting the rollout
       engine's values.
   * - ``algorithm.shuffle_rollout``
     - Shuffle rollout samples before the optimization step.
   * - ``algorithm.clip_ratio_low`` / ``clip_ratio_high``
     - Asymmetric PPO clip bounds; ``null`` falls back to ``ratio_clip_eps``
       (see :doc:`basic_config`).
   * - ``algorithm.sampling_params.max_new_tokens``
     - Max generated tokens; computed from ``runner.seq_length`` and
       ``data.max_prompt_length``.
   * - ``algorithm.sampling_params.min_new_tokens``
     - Minimum generated tokens.

Shared loss/advantage keys (``loss_type``, ``adv_type``, ``kl_beta``,
``ratio_clip_eps``, ``group_size``, ``sampling_params.temperature``, …) are
documented in :doc:`basic_config`.

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    rollout_backend: sglang       # [sglang, vllm]

    enforce_eager: False
    distributed_executor_backend: mp   # ray or mp
    disable_log_stats: False
    detokenize: False
    padding: null                 # tokenizer.pad_token_id if null
    eos: null                     # tokenizer.eos_token_id if null

    tensor_parallel_size: 1
    pipeline_parallel_size: 1

    return_logprobs: ${not:${algorithm.recompute_logprobs}}

    validate_weight: False
    validate_save_dir: null
    print_outputs: False

    max_running_requests: 64
    cuda_graph_max_bs: 128

    sglang:
      attention_backend: triton   # [flashinfer, triton]
      decode_log_interval: 500000
      use_torch_compile: False
      torch_compile_max_bs: 128

    vllm:
      attention_backend: FLASH_ATTN  # [FLASH_ATTN, XFORMERS]
      enable_chunked_prefill: True
      enable_prefix_caching: True
      enable_flash_infer_sampler: True
      max_num_batched_tokens: null

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``rollout.rollout_backend``
     - Generation backend to use (``sglang`` or ``vllm``). Selects which of the
       ``sglang`` / ``vllm`` sub-blocks applies.
   * - ``rollout.enforce_eager``
     - If True, disable CUDA-graph capture to shorten warm-up.
   * - ``rollout.distributed_executor_backend``
     - Backend for launching rollout workers (``mp`` or ``ray``).
   * - ``rollout.disable_log_stats``
     - Suppress periodic backend stats logging.
   * - ``rollout.detokenize``
     - Detokenize outputs for debugging (RL usually uses token ids only).
   * - ``rollout.padding``
     - Pad token id override; ``null`` uses ``tokenizer.pad_token_id``.
   * - ``rollout.eos``
     - EOS token id override; ``null`` uses ``tokenizer.eos_token_id``.
   * - ``rollout.tensor_parallel_size``
     - TP degree inside the generation backend. See :doc:`5D`.
   * - ``rollout.pipeline_parallel_size``
     - PP degree inside the generation backend. See :doc:`5D`.
   * - ``rollout.return_logprobs``
     - Whether the engine returns log-probs; defaults to the negation of
       ``algorithm.recompute_logprobs``.
   * - ``rollout.validate_weight``
     - Send full weights once for cross-check/validation.
   * - ``rollout.validate_save_dir``
     - Directory to store weights for comparison when validation is enabled.
   * - ``rollout.print_outputs``
     - Print token ids/texts from the engine for debugging.
   * - ``rollout.max_running_requests``
     - Max concurrent decode requests.
   * - ``rollout.cuda_graph_max_bs``
     - Max batch size eligible for CUDA graph.

**SGLang backend (``rollout.sglang``):**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``rollout.sglang.attention_backend``
     - Attention kernel backend (``flashinfer`` or ``triton``).
   * - ``rollout.sglang.decode_log_interval``
     - Interval (in steps) for SGLang to log decode stats.
   * - ``rollout.sglang.use_torch_compile``
     - Enable ``torch.compile`` inside SGLang.
   * - ``rollout.sglang.torch_compile_max_bs``
     - Max batch size eligible for ``torch.compile``.

**vLLM backend (``rollout.vllm``):**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``rollout.vllm.attention_backend``
     - Attention backend (``FLASH_ATTN`` or ``XFORMERS``).
   * - ``rollout.vllm.enable_chunked_prefill``
     - Enable chunked prefill.
   * - ``rollout.vllm.enable_prefix_caching``
     - Enable prefix caching.
   * - ``rollout.vllm.enable_flash_infer_sampler``
     - Use FlashInfer for sampling.
   * - ``rollout.vllm.max_num_batched_tokens``
     - Maximum tokens batched together; ``null`` uses the vLLM default.

``rollout.group_name``, ``rollout.gpu_memory_utilization``, and ``rollout.model.*``
are shared keys documented in :doc:`basic_config`.

data
~~~~~~~~~~~~~~~

.. code:: yaml

  data:
    type: math
    dataset_name: boba
    max_prompt_length: 1024
    filter_prompt_by_length: True
    rollout_batch_size: 512
    val_rollout_batch_size: null
    num_workers: 2
    prompt_key: prompt
    answer_key: solutions
    apply_chat_template: False
    shuffle: True
    validation_shuffle: True
    seed: 1234
    train_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]
    val_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``data.type``
     - Dataset/task family (e.g., ``math``).
   * - ``data.dataset_name``
     - Dataset identifier used to select preprocessing/formatting.
   * - ``data.max_prompt_length``
     - Maximum tokens allowed for prompts.
   * - ``data.filter_prompt_by_length``
     - Drop prompts longer than ``max_prompt_length`` instead of truncating.
   * - ``data.rollout_batch_size``
     - Global rollout batch size across engines.
   * - ``data.val_rollout_batch_size``
     - Global validation rollout batch size; ``null`` falls back to
       ``data.rollout_batch_size``.
   * - ``data.num_workers``
     - Data-loader workers per actor rank.
   * - ``data.prompt_key``
     - JSONL key that stores the prompt text.
   * - ``data.answer_key``
     - JSONL key that stores the reference answer/solution.
   * - ``data.apply_chat_template``
     - Wrap prompts with the tokenizer chat template before tokenizing.
   * - ``data.shuffle``
     - Shuffle training data each epoch.
   * - ``data.validation_shuffle``
     - Shuffle validation data.
   * - ``data.seed``
     - RNG seed for loaders and sampling.
   * - ``data.train_data_paths``
     - List of training JSONL file paths.
   * - ``data.val_data_paths``
     - List of validation JSONL file paths.

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
    seed: 1234

    model:
      precision: fp16
      add_bias_linear: False

      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1

      activation: swiglu
      sequence_parallel: True

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
      overlap_grad_reduce: False
      overlap_param_gather: False
      optimizer_enable_pin: false
      overlap_param_gather_with_optimizer_step: False
      clip_grad: 0.8
      loss_scale: 65536

    lr_sched:
      lr_warmup_fraction: 0.01
      lr_warmup_init: 0.0
      lr_warmup_iters: 0
      max_lr: 2.0e-5
      min_lr: 0.0
      lr_decay_style: constant
      lr_decay_iters: 10

    tokenizer:
      tokenizer_model: /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B/
      use_fast: False
      trust_remote_code: True
      padding_side: 'right'

    megatron:
      ddp_bucket_size: null
      distributed_backend: nccl # 'nccl' or 'gloo'
      distributed_timeout_minutes: 30
      ckpt_format: torch
      use_dist_ckpt: False
      tp_comm_bootstrap_backend: nccl
      tp_comm_overlap_cfg: null
      use_hf_ckpt: True
      use_profiler: False

      ckpt_convertor: # config for the checkpoint convertor
        model: DeepSeek-R1-Distill-Qwen-1.5B
        hf_model_path: ${rollout.model.model_path}
        save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor
        use_gpu_num: 0
        use_gpu_index: null
        process_num: 16
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

      mixed_precision:
        param_dtype: ${actor.model.precision}
        reduce_dtype: ${actor.model.precision}
        buffer_dtype: ${actor.model.precision}

      amp_autocast:
        enabled: False
        precision: "bf16"

      grad_scaler:
        enabled: False

**Top-level**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.training_backend``
     - Training backend (``megatron`` or ``fsdp``).
   * - ``actor.mcore_gpt``
     - Use the Megatron-Core GPT stack.
   * - ``actor.spec_name``
     - Model spec/preset name (e.g., decoder-only GPT).
   * - ``actor.offload_optimizer``
     - Offload optimizer state to CPU to reduce GPU memory.
   * - ``actor.offload_weight``
     - Offload model weights to CPU when possible.
   * - ``actor.offload_grad``
     - Offload gradients to CPU to reduce GPU memory.
   * - ``actor.enable_dp_load_balance``
     - Enable data-parallel load balancing.
   * - ``actor.calculate_flops``
     - Compute and log FLOPs for profiling.
   * - ``actor.seed``
     - Global seed for reproducibility.

**Model sub-section**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.model.precision``
     - Numerical precision for training (e.g., ``fp16``).
   * - ``actor.model.add_bias_linear``
     - Add bias terms to linear layers.
   * - ``actor.model.tensor_model_parallel_size``
     - TP degree for the actor.
   * - ``actor.model.pipeline_model_parallel_size``
     - PP degree for the actor.
   * - ``actor.model.activation``
     - Activation function (e.g., ``swiglu``).
   * - ``actor.model.sequence_parallel``
     - Enable sequence parallelism (requires TP).
   * - ``actor.model.recompute_method``
     - Activation-recompute strategy (e.g., ``block``).
   * - ``actor.model.recompute_granularity``
     - Recompute scope (``full`` or ``selective``).
   * - ``actor.model.recompute_num_layers``
     - Number of layers to checkpoint/recompute.
   * - ``actor.model.seq_length``
     - Decoder context length for training.
   * - ``actor.model.encoder_seq_length``
     - Encoder length (mirrors ``seq_length`` for decoder-only models).
   * - ``actor.model.normalization``
     - Norm-layer type (e.g., ``rmsnorm``).
   * - ``actor.model.position_embedding_type``
     - Positional-embedding type (e.g., ``rope``).
   * - ``actor.model.apply_rope_fusion``
     - Use fused RoPE kernels if available.
   * - ``actor.model.bias_dropout_fusion``
     - Fuse bias + dropout kernels.
   * - ``actor.model.persist_layer_norm``
     - Persist LayerNorm params in higher precision.
   * - ``actor.model.bias_activation_fusion``
     - Fuse bias + activation kernels.
   * - ``actor.model.attention_softmax_in_fp32``
     - Compute attention softmax in FP32 for stability.
   * - ``actor.model.batch_p2p_comm``
     - Batch P2P communications across layers.
   * - ``actor.model.variable_seq_lengths``
     - Allow variable sequence lengths per micro-batch.
   * - ``actor.model.gradient_accumulation_fusion``
     - Fused gradient accumulation.
   * - ``actor.model.moe_token_dispatcher_type``
     - MoE token dispatcher (e.g., ``alltoall``).
   * - ``actor.model.use_cpu_initialization``
     - Initialize weights on CPU to reduce GPU spikes.

**Optimizer**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.optim.optimizer``
     - Optimizer choice (``adam``).
   * - ``actor.optim.bf16`` / ``actor.optim.fp16``
     - Mixed-precision flags.
   * - ``actor.optim.lr``
     - Base learning rate.
   * - ``actor.optim.adam_beta1`` / ``adam_beta2`` / ``adam_eps``
     - Adam hyper-parameters.
   * - ``actor.optim.min_lr``
     - Minimum LR (for schedulers that decay below the base LR).
   * - ``actor.optim.weight_decay``
     - L2 weight decay.
   * - ``actor.optim.use_distributed_optimizer``
     - Use the Megatron distributed optimizer.
   * - ``actor.optim.overlap_grad_reduce``
     - Overlap gradient reduction with the backward pass.
   * - ``actor.optim.overlap_param_gather``
     - Overlap parameter all-gather with the forward pass.
   * - ``actor.optim.optimizer_enable_pin``
     - Pin optimizer memory.
   * - ``actor.optim.overlap_param_gather_with_optimizer_step``
     - Overlap param gather with the optimizer step.
   * - ``actor.optim.clip_grad``
     - Global gradient-clipping norm.
   * - ``actor.optim.loss_scale``
     - Static FP16 loss scale (use ``loss_scale_window`` instead for dynamic
       scaling).

**LR schedule**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.lr_sched.lr_warmup_fraction``
     - Warm-up as a fraction of total iterations.
   * - ``actor.lr_sched.lr_warmup_init``
     - Initial LR value during warm-up.
   * - ``actor.lr_sched.lr_warmup_iters``
     - Warm-up iterations (overrides the fraction when > 0).
   * - ``actor.lr_sched.max_lr`` / ``min_lr``
     - LR bounds for the scheduler.
   * - ``actor.lr_sched.lr_decay_style``
     - Decay policy (e.g., ``constant``).
   * - ``actor.lr_sched.lr_decay_iters``
     - Total decay iterations.

**Tokenizer**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.tokenizer.tokenizer_model``
     - Path/name of the tokenizer.
   * - ``actor.tokenizer.use_fast``
     - Use the HuggingFace fast tokenizer.
   * - ``actor.tokenizer.trust_remote_code``
     - Allow custom tokenizer code.
   * - ``actor.tokenizer.padding_side``
     - ``left`` or ``right`` padding.

**Megatron integration**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.megatron.ddp_bucket_size``
     - DDP gradient bucket size.
   * - ``actor.megatron.distributed_backend``
     - Distributed backend (``nccl`` or ``gloo``).
   * - ``actor.megatron.distributed_timeout_minutes``
     - Backend communication timeout.
   * - ``actor.megatron.ckpt_format``
     - Checkpoint format (e.g., ``torch``).
   * - ``actor.megatron.use_dist_ckpt``
     - Use distributed (sharded) checkpointing.
   * - ``actor.megatron.tp_comm_bootstrap_backend``
     - Backend used for TP bootstrap (e.g., ``nccl``).
   * - ``actor.megatron.tp_comm_overlap_cfg``
     - YAML path for TP comm/compute overlap.
   * - ``actor.megatron.use_hf_ckpt``
     - Convert/load from a HuggingFace checkpoint for training.
   * - ``actor.megatron.use_profiler``
     - Enable the Torch profiler during training (affects performance).

**Megatron checkpoint converter (``actor.megatron.ckpt_convertor``)**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.megatron.ckpt_convertor.model``
     - Model name for the converter metadata.
   * - ``actor.megatron.ckpt_convertor.hf_model_path``
     - Source HF model path.
   * - ``actor.megatron.ckpt_convertor.save_path``
     - Target directory for the converted Megatron checkpoint.
   * - ``actor.megatron.ckpt_convertor.use_gpu_num``
     - Number of GPUs to use for conversion.
   * - ``actor.megatron.ckpt_convertor.use_gpu_index``
     - Specific GPU index to use.
   * - ``actor.megatron.ckpt_convertor.process_num``
     - CPU processes for conversion work.
   * - ``actor.megatron.ckpt_convertor.tensor_model_parallel_size``
     - TP degree for the converted checkpoint.
   * - ``actor.megatron.ckpt_convertor.pipeline_model_parallel_size``
     - PP degree for the converted checkpoint.

**FSDP integration (``actor.fsdp_config``)**

Used when ``actor.training_backend`` is ``fsdp``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``actor.fsdp_config.strategy``
     - FSDP strategy: ``fsdp`` or ``fsdp2`` (case-insensitive).
   * - ``actor.fsdp_config.sharding_strategy``
     - Sharding strategy: ``full_shard``, ``shard_grad_op``, ``hybrid_shard``, or
       ``no_shard``.
   * - ``actor.fsdp_config.cpu_offload``
     - FSDP2: keep parameters on CPU, moving them to GPU only when needed.
   * - ``actor.fsdp_config.offload_pin_memory``
     - FSDP2: use pinned CPU memory (only when ``cpu_offload`` is True) for faster
       transfers.
   * - ``actor.fsdp_config.reshard_after_forward``
     - FSDP2: re-shard parameters after the forward pass to save GPU memory.
   * - ``actor.fsdp_config.enable_gradient_accumulation``
     - Communicate/update only after the last micro-batch. Speeds up training at
       the cost of GPU memory.
   * - ``actor.fsdp_config.forward_prefetch``
     - FSDP: prefetch the next all-gather during the forward pass (more memory,
       better overlap).
   * - ``actor.fsdp_config.limit_all_gathers``
     - FSDP: limit concurrent all-gathers (recommended when CPU/memory bound).
   * - ``actor.fsdp_config.backward_prefetch``
     - FSDP: prefetch strategy in the backward pass (``null`` / ``pre`` / ``post``).
   * - ``actor.fsdp_config.use_orig_params``
     - FSDP: expose original (unflattened) parameters; better compatibility, more
       communication overhead.
   * - ``actor.fsdp_config.use_liger_kernel``
     - Use Liger kernels (currently Qwen2.5 / Qwen2.5-VL) to cut memory and speed
       up training.
   * - ``actor.fsdp_config.mixed_precision.param_dtype``
     - Parameter dtype.
   * - ``actor.fsdp_config.mixed_precision.reduce_dtype``
     - Reduction dtype.
   * - ``actor.fsdp_config.mixed_precision.buffer_dtype``
     - Buffer dtype.
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - Enable automatic mixed-precision (AMP) training.
   * - ``actor.fsdp_config.amp_autocast.precision``
     - Numerical precision used by AMP.
   * - ``actor.fsdp_config.grad_scaler.enabled``
     - Enable the gradient scaler.

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Parameter
     - Description
   * - ``reward.reward_type``
     - Which reward type to use for training (e.g., ``math``).
   * - ``reward.reward_scale``
     - A correct answer receives ``reward_scale``; an incorrect one receives
       ``-reward_scale``.

Agentic RL Configuration
=========================

This section covers configuration parameters specific to agentic and reasoning RL training
(math reasoning, coding agents, multi-agent systems). These extend the shared
configuration described in :doc:`gpu_cluster_config`.

.. contents::
   :depth: 1
   :local:

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

``runner.enable_dynamic_batch_size``: Whether to use dynamic batch size when training by Megatron.

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
        hf_model_path: ${rollout.model.model_path} # path to the hf model
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

``actor.fsdp_config.mixed_precision.param_dtype``: FSDP/FSDP2 parameter, specifying the parameter type.

``actor.fsdp_config.mixed_precision.reduce_dtype``: FSDP/FSDP2 parameter, specifying the data type used during reduction.

``actor.fsdp_config.mixed_precision.buffer_dtype``: FSDP parameter, specifying the data type used for the buffer.

``actor.fsdp_config.amp_autocast.enabled``: FSDP/FSDP2 parameter, indicating whether automatic mixed-precision training is enabled.

``actor.fsdp_config.amp_autocast.precision``: FSDP/FSDP2 parameter, indicating the numerical precision used by AMP.

``actor.fsdp_config.grad_scaler.enabled``: FSDP/FSDP2 parameter, indicating whether the gradient scaler is enabled.

``actor.fsdp_config.grad_scaler.init_scale``: FSDP/FSDP2 parameter, indicating the initial scale factor used by the gradient scaler to prevent numerical underflow.

``actor.fsdp_config.grad_scaler.growth_interval``: FSDP/FSDP2 parameter, indicating the number of consecutive steps without gradient overflows required before the scale factor is increased.

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0


``reward.reward_type``: Which reward type to use for the training.

``reward.reward_scale``: when the answer is correct, it receives ``reward_scale``; when it is incorrect, it receives ``-reward_scale``.

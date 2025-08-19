Basic YAML Configuration
================================


Below is a complete reference for the configuration file used in the `GRPO + Megatron-LM + SGLang` reinforcement learning pipeline. 
Every important key in the YAML is documented below so that you can confidently adapt the file to your own cluster, model, or research ideas.  
Parameters are grouped exactly by their top-level key.



cluster
----------------

.. code:: yaml

  cluster:
    num_nodes: 1
    num_gpus_per_node: 1
    component_placement:
      actor,inference,rollout: all


``cluster.num_nodes``: Physical nodes to use for training.

``cluster.num_gpus_per_node``: GPUs per node that the placement strategy should assume are free. 

``cluster.component_placement``: 
The *placement strategy* for each processes.
Each line of component placement config looks like: ``actor,inference: 0-4``, 
which means both the actor and inference groups occupy GPU 0 to 4
Alternatively, "all" can be used to specify all GPUs

runner
-------------

.. code:: yaml

  runner:
    task_type: math
    logger:
      path: ${runner.output_dir}/${runner.experiment_name}/tensorboard
      tensorboard:
        enable: True
        queue_size: 10
      wandb:
        enable: False
        project_name: infini-rl
        experiment_name: ${runner.experiment_name}

    max_epochs: 5
    max_steps: -1

    val_check_interval: 1
    save_interval: 50

    seq_length: 2048

    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

    resume_dir: null
    experiment_name: grpo-1.5b
    output_dir: ../results


``runner.logger``: The logging parameter including tensorboard and wandb

``runner.max_epochs``: Maximum number of *epochs* to run. Epoch here = walking once through the dataset

``runner.max_steps``: Maximum number of training steps before stopping early. If set to -1, this defaults to set automatially based on the ``runner.max_epochs``.

``runner.val_check_interval``: How often to launch a validation rollout.

``runner.save_interval``: Checkpoint frequency in trainer steps.

``runner.seq_length``: Total sequence length (prompt + generated response) fed into Megatron during RL updates.

``runner.enable_dynamic_batch_size``: Whether to user dynamic batch size when training by Megatron.

``runner.max_tokens_per_mbs``: Upper limit of tokens in a Megatron microbatch when dynamic batching is enabled.


algorithm
---------

.. code:: yaml

  algorithm:
    group_size: 2

    n_minibatches: 4
    training_batch_size_per_gpu: 1 
    rollout_batch_size_per_gpu: null 

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
      max_new_tokens: ${subtract:${runner.seq_length}, ${data.max_prompt_length}}
      min_new_tokens: 1


``algorithm.group_size``: Responses per prompt (set > 1 to enable group baselines).

``algorithm.n_minibatches``: Number of gradient update per batch.

``algorithm.training_batch_size_per_gpu``: Micro-batch size on each actor GPU.

``algorithm.rollout_batch_size_per_gpu``: Inference micro-batch per GPU; null divides the global rollout batch evenly.

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

``algorithm.sampling_params.max_new_tokens``: Max generated tokens; computed from runner.seq_length and data.max_prompt_length.

``algorithm.sampling_params.min_new_tokens``: Minimum generated tokens.


inference
---------

.. code:: yaml

  inference:
    group_name: "InferenceGroup"
    load_from_actor: True
    model:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      sequence_parallel: False

``inference.group_name``: Logical name for the inference worker group.

``inference.load_from_actor``: Initialize inference weights from the actor.

``inference.model.tensor_model_parallel_size``: TP degree inside the inference engine.

``inference.model.pipeline_model_parallel_size``: PP degree inside the inference engine.

``inference.model.sequence_parallel``: Enable Megatron sequence parallelism for inference

See more details about the parallelism in :doc:`../advance/5D`.

rollout
-------

.. code:: yaml

  rollout:
    group_name: "RolloutGroup"

    gpu_memory_utilization: 0.55

    model_dir: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
    model_arch: qwen2.5
    enforce_eager: False         # if False, vllm will capture cuda graph, which will take more time to initialize.
    distributed_executor_backend: mp   # ray or mp
    disable_log_stats: False
    detokenize: False            # Whether to detokenize the output. During RL we actually don't need to detokenize it. Can be set to True for debugging.
    padding: null               # will be tokenizer.pad_token_id if null. it is used to filter megatron's padding for vllm rollout
    eos: null                   # will be tokenizer.eos_token_id if null.

    attention_backend: triton
    recompute_logprobs: True

    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    
    validate_weight: False # whether to send all weights at first for weight comparison.
    validate_save_dir: null # the directory to save the weights for comparison. If validate_weight is True, this will be used to save the weights for comparison.
    print_outputs: False         # whether to print the outputs (token ids, texts, etc.) of inference engine.

    sglang_decode_log_interval: 500000 # the interval for SGLang to log the decode time and other stats.
    max_running_requests: 64 # the maximum number of running requests in the inference engine.
    cuda_graph_max_bs: 128 # the maximum batch size for cuda graph. If the batch size is larger than this, cuda graph will not be used.

    use_torch_compile: False # enable torch_compile in SGLang for rollout.
    torch_compile_max_bs: 128 # the maximum batch size for torch compile. If the batch size is larger than this, torch compile will not be used.

``rollout.group_name``: Logical name for rollout/inference workers.

``rollout.gpu_memory_utilization``: Target VRAM fraction per rollout worker.

``rollout.model_dir``: Path to the HF model used by the generation backend.

``rollout.model_arch``: Internal architecture tag used by the backend (e.g., qwen2.5).

``rollout.enforce_eager``: If True, disable CUDA graph capture to shorten warm-up.

``rollout.distributed_executor_backend``: Backend for launching rollout workers (mp or ray).

``rollout.disable_log_stats``: Suppress periodic backend stats logging.

``rollout.detokenize``: Detokenize outputs for debugging (RL usually uses token ids only).

``rollout.padding``: Pad token id override; null uses tokenizer.pad id.

``rollout.eos``: EOS token id override; null uses tokenizer.eos id.

``rollout.attention_backend``: Attention kernel backend (e.g., triton). 

``rollout.recompute_logprobs``: Recompute log-probs for sampled sequences.

``rollout.tensor_parallel_size``: TP degree inside the generation backend.

``rollout.pipeline_parallel_size``: PP degree inside the generation backend.

``rollout.validate_weight``: Send full weights once for cross-check/validation.

``rollout.validate_save_dir``: Directory to store weights for comparison when validation is enabled.

``rollout.print_outputs``: Print token ids/texts from the engine for debugging.

``rollout.sglang_decode_log_interval``: Interval for SGLang to log decode stats.

``rollout.max_running_requests``: Max concurrent decode requests.

``rollout.cuda_graph_max_bs``: Max batch size eligible for CUDA graph.

``rollout.use_torch_compile``: Enable torch.compile inside SGLang.

``rollout.torch_compile_max_bs``: Max batch size eligible for torch.compile.



data
----

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
-----

.. code:: yaml


  actor:
    group_name: "ActorGroup"
    training_backend: megatron
    mcore_gpt: True
    spec_name: decoder_gpt

    checkpoint_load_path: null

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
      tp_comm_overlap_cfg: null #/mnt/public/megatron-infinigence-rl/examples/megatron_tp_comm_overlap_cfg.yaml
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

**Top-level**

``actor.group_name``: Logical name for the training (actor) workers.

``actor.training_backend``: Training backend (megatron).

``actor.mcore_gpt``: Use Megatron-Core GPT stack. TODO: exact scope.

``actor.spec_name``: Model spec/preset name (e.g., decoder-only GPT). TODO: preset mapping.

``actor.checkpoint_load_path``: Path to a checkpoint to load before training.

``actor.offload_optimizer``: Offload optimizer state to CPU to reduce GPU memory.

``actor.offload_weight``: Offload model weights to CPU when possible (ZeRO-style). 

``actor.offload_grad``: Offload gradients to CPU to reduce GPU memory.

``actor.enable_dp_load_balance``: Enable data-parallel load balancing. 

``actor.calculate_flops``: Compute and log FLOPs for profiling.

``actor.seed``: Global seed for reproducibility.

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

``actor.model.bias_dropout_fusion``: Fuse bias + dropout kernels. TODO: kernel availability.

``actor.model.persist_layer_norm``: Persist LN params in higher precision. TODO: exact semantics.

``actor.model.bias_activation_fusion``: Fuse bias + activation kernels. TODO: kernel availability.

``actor.model.attention_softmax_in_fp32``: Compute attention softmax in FP32 for stability.

``actor.model.batch_p2p_comm``: Batch P2P communications across layers. TODO: behavior.

``actor.model.variable_seq_lengths``: Allow variable sequence lengths per micro-batch.

``actor.model.gradient_accumulation_fusion``: Fused gradient accumulation. TODO: support matrix.

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

``actor.optim.optimizer_enable_pin``: Pin optimizer memory. TODO: effects and scope.

``actor.optim.overlap_param_gather_with_optimizer_step``: Overlap param gather with step. TODO.

``actor.optim.clip_grad``: Global gradient clipping norm.

``actor.optim.loss_scale_window``: Dynamic loss scale window for FP16. TODO: exact algorithm.

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

``actor.megatron.ddp_bucket_size``: DDP gradient bucket size. TODO: units.

``actor.megatron.distributed_backend``: Distributed backend (nccl or gloo).

``actor.megatron.distributed_timeout_minutes``: Backend communication timeout.

``actor.megatron.ckpt_format``: Checkpoint format (e.g., torch).

``actor.megatron.use_dist_ckpt``: Use distributed checkpointing (sharded). TODO: behavior.

``actor.megatron.tp_comm_bootstrap_backend``: Backend used for TP bootstrap (e.g., nccl).

``actor.megatron.tp_comm_overlap_cfg``: YAML path for TP comm/compute overlap. TODO: schema.

``actor.megatron.use_hf_ckpt``: Convert/load from a HuggingFace checkpoint for training.

**Megatron checkpoint converter**

``actor.megatron.ckpt.model``: Model name for the converter metadata.

``actor.megatron.ckpt.model_type``: Model type; inferred from HF config when null.

``actor.megatron.ckpt.hf_model_path``: Source HF model path.

``actor.megatron.ckpt.save_path``: Target directory to write Megatron checkpoints.

``actor.megatron.ckpt.use_gpu_num``: Number of GPUs to use for conversion. TODO: behavior.

``actor.megatron.ckpt.use_gpu_index``: Specific GPU index to use. TODO:

``actor.megatron.ckpt.process_num``: CPU processes for conversion work.

``actor.megatron.ckpt.tensor_model_parallel_size``: TP degree for converted checkpoints.

``actor.megatron.ckpt.pipeline_model_parallel_size``: PP degree for converted checkpoints.


reward
------

.. code:: yaml

  reward:
    use_reward_model: false
    reward_type: math
    reward_scale: 5.0

``reward.use_reward_model``: Whether to use a reward model.

``reward.reward_type``: Which reward type to use for the training.

``reward.reward_scale``: when the answer is correct, it receives ``reward_scale``; when it is incorrect, it receives ``-reward_scale``.

critic
------

.. code:: yaml

  critic:
    use_critic_model: false


``critic.use_critic_model``: Whether to use a critic model.
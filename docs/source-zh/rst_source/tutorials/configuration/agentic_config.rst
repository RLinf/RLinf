智能体强化学习配置
=====================

本节介绍智能体和推理 RL 训练专用的配置参数（数学推理、代码智能体、多智能体系统）。
这些参数扩展了 :doc:`gpu_cluster_config` 中的共享配置。

.. contents::
   :depth: 1
   :local:

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

``runner.enable_dynamic_batch_size``：使用 Megatron 训练时是否启用动态批大小。

``runner.max_tokens_per_mbs``：启用动态批时每个微批的 token 上限。


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

``algorithm.n_minibatches``：每个 batch 的梯度更新次数。

``algorithm.training_batch_size_per_gpu``：每张 actor GPU 的训练微批大小。

``algorithm.rollout_batch_size_per_gpu``：每 GPU 的推理微批大小；为 null 时按全局大小平均分配。


**sampling_params：**

``algorithm.sampling_params.max_new_tokens``：最大生成长度（由 runner.seq_length 与 data.max_prompt_length 计算）。

``algorithm.sampling_params.min_new_tokens``：最小生成长度。

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    enforce_eager: False         # 若为 False，rollout 引擎将使用 CUDA graph，初始化更久但运行更快
    distributed_executor_backend: mp   # 可选 ray 或 mp
    disable_log_stats: False
    detokenize: False            # 是否反词元化输出；RL 训练通常只需 token id。调试可设 True
    padding: null               # 为空则使用 tokenizer.pad_token_id；用于过滤 megatron 的 padding
    eos: null                   # 为空则使用 tokenizer.eos_token_id

    attention_backend: triton

    tensor_parallel_size: 1
    pipeline_parallel_size: 1

    validate_weight: False # 是否在开始时发送全部权重进行一致性校验
    validate_save_dir: null # 若启用校验，保存用于比对的权重目录
    print_outputs: False         # 是否打印 rollout 引擎的输出（token id/文本等）

    sglang_decode_log_interval: 500000 # SGLang 打印解码耗时与统计信息的间隔
    max_running_requests: 64 # rollout 引擎内最大并发请求数
    cuda_graph_max_bs: 128 # 使用 CUDA graph 的最大 batch size；超过则不使用

    use_torch_compile: False # 在 SGLang 中为 rollout 启用 torch.compile
    torch_compile_max_bs: 128 # 启用 torch.compile 的最大 batch size；超过则不使用

``rollout.enforce_eager``：True 时禁用 CUDA graph，加快预热启动。

``rollout.distributed_executor_backend``：rollout worker 的启动后端（mp 或 ray）。

``rollout.disable_log_stats``：是否关闭后端周期性统计日志。

``rollout.detokenize``：是否将输出 detokenize（调试用）。

``rollout.padding``：pad token id 重载；null 则用 tokenizer 的 pad id。

``rollout.eos``：EOS token id 重载；null 则用 tokenizer 的 eos id。

``rollout.attention_backend``：注意力算子后端（如 triton）。

``rollout.tensor_parallel_size``：生成后端的张量并行度（TP）。

``rollout.pipeline_parallel_size``：生成后端的流水并行度（PP）。

并行化细节见 :doc:`../advance/5D`。

``rollout.validate_weight``：是否发送完整权重进行校验。

``rollout.validate_save_dir``：启用校验时的权重保存目录。

``rollout.print_outputs``：是否打印调试输出。

``rollout.sglang_decode_log_interval``：SGLang 解码统计的间隔。

``rollout.max_running_requests``：最大并发解码请求数。

``rollout.cuda_graph_max_bs``：可使用 CUDA graph 的最大批大小。

``rollout.use_torch_compile``：启用 torch.compile。

``rollout.torch_compile_max_bs``：可使用 torch.compile 的最大批大小。

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

``data.type``：数据集/任务类型（如 math）。

``data.max_prompt_length``：提示的最大 token 数。

``data.rollout_batch_size``：全局 rollout 批大小。

``data.val_rollout_batch_size``：全局验证批大小；为 null 则回退到 ``data.rollout_batch_size``。

``data.num_workers``：每个 actor rank 的数据加载进程数。

``data.prompt_key``：JSONL 中提示文本的键名。

``data.shuffle``：训练数据是否每 epoch 乱序。

``data.validation_shuffle``：验证数据是否乱序（on-policy 评估通常建议 True）。

``data.seed``：数据加载与采样用的随机种子。

``data.train_data_paths``：训练 JSONL 文件列表。

``data.val_data_paths``：验证 JSONL 文件列表。

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
      distributed_backend: nccl # 支持 'nccl' 与 'gloo'
      distributed_timeout_minutes: 30
      ckpt_format: torch
      use_dist_ckpt: False
      tp_comm_bootstrap_backend: nccl
      tp_comm_overlap_cfg: null
      use_hf_ckpt: True # 为 True 时将 HF 模型转为 Megatron checkpoint 并用于训练

      ckpt: # checkpoint 转换器配置
        model: DeepSeek-R1-Distill-Qwen-1.5B
        hf_model_path: ${rollout.model.model_path} # HF 模型所在路径
        save_path: ${runner.output_dir}/${runner.experiment_name}/actor/megatron_ckpt_from_hf
        use_gpu_num : 0
        use_gpu_index: null #
        process_num: 16 # 转换使用的进程数
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

**顶层：**

``actor.training_backend``：训练后端（megatron）。

``actor.mcore_gpt``：是否使用 Megatron-Core GPT 栈。

``actor.spec_name``：模型规格/预设（如 decoder_gpt）。

``actor.offload_optimizer/weight/grad``：将优化器/权重/梯度尽可能下放到 CPU 以节省显存。

``actor.enable_dp_load_balance``：是否启用数据并行负载均衡。

``actor.calculate_flops``：是否计算并记录 FLOPs（分析用）。

**Model 子项：**

``actor.model.precision``：训练数值精度（fp16 等）。

``actor.model.add_bias_linear``：线性层是否带 bias。

``actor.model.tensor_model_parallel_size``：actor 端 TP 并行度。

``actor.model.pipeline_model_parallel_size``：actor 端 PP 并行度。

``actor.model.activation``：激活函数（如 swiglu）。

``actor.model.sequence_parallel``：启用序列并行（需配合 TP）。

``actor.model.recompute_method/granularity/num_layers``：重计算策略/粒度/层数。

``actor.model.seq_length / encoder_seq_length``：训练时解码/编码序列长度。

``actor.model.normalization``：归一化层类型（rmsnorm）。

``actor.model.position_embedding_type``：位置编码类型（rope）。

``actor.model.apply_rope_fusion``：是否使用融合的 RoPE 内核。

``actor.model.*fusion``：若干算子融合开关。

``actor.model.attention_softmax_in_fp32``：注意力 softmax 用 FP32 保稳。

``actor.model.batch_p2p_comm``：跨层批量 P2P 通信。

``actor.model.variable_seq_lengths``：允许不同微批序列长度。

``actor.model.gradient_accumulation_fusion``：梯度累积融合。

``actor.model.moe_token_dispatcher_type``：MoE token 分发方式（如 alltoall）。

``actor.model.use_cpu_initialization``：在 CPU 上初始化权重以降低 GPU 峰值。

**优化器：**

``actor.optim.optimizer``：优化器选择（如 adam）。

``actor.optim.bf16 / actor.optim.fp16``：混合精度训练相关开关。

``actor.optim.lr``：基础学习率（Base learning rate）。

``actor.optim.adam_beta1 / adam_beta2 / adam_eps``：Adam 优化器的超参数。

``actor.optim.min_lr``：最小学习率（适用于 LR 衰减低于基准 LR 的情况）。

``actor.optim.weight_decay``：L2 正则化权重衰减。

``actor.optim.use_distributed_optimizer``：是否使用 Megatron 分布式优化器。

``actor.optim.overlap_grad_reduce``：是否在反向传播时与梯度归约操作重叠执行。

``actor.optim.overlap_param_gather``：是否在前向传播时与参数 all-gather 重叠执行。

``actor.optim.optimizer_enable_pin``：是否固定优化器的内存位置。

``actor.optim.overlap_param_gather_with_optimizer_step``：是否在执行优化器 step 时与参数 all-gather 重叠。

``actor.optim.clip_grad``：全局梯度裁剪范数（Gradient clipping norm）。

``actor.optim.loss_scale_window``：FP16 的动态 loss scaling 窗口。

**学习率调度：**

``actor.lr_sched.lr_warmup_fraction``：学习率预热阶段占总迭代的比例。

``actor.lr_sched.lr_warmup_init``：预热初始学习率值。

``actor.lr_sched.lr_warmup_iters``：学习率预热的迭代次数（>0 时覆盖上面比例设置）。

``actor.lr_sched.max_lr / min_lr``：学习率调度的上限 / 下限。

``actor.lr_sched.lr_decay_style``：学习率衰减策略（如 constant）。

``actor.lr_sched.lr_decay_iters``：学习率衰减持续的总迭代次数。

**分词器：**

``actor.tokenizer.tokenizer_model``：分词器路径/名称。

``actor.tokenizer.use_fast``：是否使用 fast tokenizer。

``actor.tokenizer.trust_remote_code``：允许自定义分词器代码。

``actor.tokenizer.padding_side``：填充方向（left/right）。

**Megatron 集成：**

``actor.megatron.*``：分布式后端、超时、checkpoint 格式、HF checkpoint 转换等设置。

**Megatron checkpoint 转换器：**

``actor.megatron.ckpt.model``：转换器元信息中的模型名称。

``actor.megatron.ckpt.hf_model_path``：源 HF 模型路径。

``actor.megatron.ckpt.save_path``：转换后 Megatron checkpoint 保存目录。

``actor.megatron.ckpt.use_gpu_num``：转换使用的 GPU 数量。

``actor.megatron.ckpt.use_gpu_index``：指定使用的 GPU 索引。

``actor.megatron.ckpt.process_num``：转换过程使用的 CPU 进程数。

``actor.megatron.ckpt.tensor_model_parallel_size``：转换后 checkpoint 的张量并行度（TP）。

``actor.megatron.ckpt.pipeline_model_parallel_size``：转换后 checkpoint 的流水线并行度（PP）。

**FSDP 集成：**

``actor.fsdp_config.strategy``: 决定所使用FSDP 策略，支持fsdp, fsdp2（不区分大小写）

``actor.fsdp_config.sharding_strategy``: FSDP/FSDP2参数,表示FSDP所使用的切片策略,支持full_shard, shard_grad_op, hybrid_shard, no_shard

``actor.fsdp_config.cpu_offload``: FSDP2参数，决定FSDP2是否将参数放置于CPU侧，需要时在传输到GPU侧

``actor.fsdp_config.offload_pin_memory``: FSDP2参数，仅当cpu_offload选项为True时有效，如果为真则此时CPU侧内存为pinned memory以提高传输效率

``actor.fsdp_config.reshard_after_forward``: FSDP2参数，表示是否在前向传播后重新切片参数以节省显存

``actor.fsdp_config.enable_gradient_accumulation``: FSDP/FSDP2参数，表示是否启用梯度累积，如果为真则仅在最后一个micro batch结束后再进行通信并更新梯度，开启会增加一定显存占用，但会加快训练

``actor.fsdp_config.forward_prefetch``: FSDP1参数，表示是否在前向传播时预取下一个 all-gather 操作。开启时会增加显存占用，建议当显存足够时可以开启以重叠通信与计算，从而提升性能

``actor.fsdp_config.limit_all_gathers``: FSDP1参数，表示是否限制并发 all-gather 操作的数量，建议当CPU或内存成为瓶颈时开启。

``actor.fsdp_config.backward_prefetch``: FSDP1参数，表示后向传播时的预取策略（null/'pre'/'post'）， 如果为 'pre'，则在计算梯度时预取下一个 all-gather 操作，这样重叠更激进，吞吐更高；如果为 'post'，则在当前梯度计算完成后预取下一个 all-gather 操作，相较于 'pre' 更保守一些。

``actor.fsdp_config.use_orig_params``: FSDP1参数，表示是否使用模块的原始参数，让模块暴露原始参数（nn.Module.named_parameters），而非 FSDP 的扁平参数。可以提高兼容性，但是会引入额外的通信开销降低性能。

``actor.fsdp_config.use_liger_kernel``: FSDP/FSDP2参数，是否使用 liger_kernel（目前仅支持部分模型，包括：qwen2.5，qwen2.5-vl），开启则可以降低显存占用并提升训练速度。

``actor.fsdp_config.mixed_precision.param_dtype``: FSDP/FSDP2参数，指定参数类型

``actor.fsdp_config.mixed_precision.reduce_dtype``: FSDP/FSDP2参数，指定规约时使用的数据类型

``actor.fsdp_config.mixed_precision.buffer_dtype``: FSDP1参数，指定缓冲区使用的数据类型

``actor.fsdp_config.amp_autocast.enabled``: FSDP/FSDP2参数，表示是否启用自动混合精度训练

``actor.fsdp_config.amp_autocast.precision``: FSDP/FSDP2参数，表示AMP使用的数值精度

``actor.fsdp_config.grad_scaler.enabled``: FSDP/FSDP2参数，表示是否启用梯度缩放器

``actor.fsdp_config.grad_scaler.init_scale``: FSDP/FSDP2 参数，表示梯度缩放器的初始缩放因子，用于在训练初期放大梯度以防止数值下溢（Underflow）。

``actor.fsdp_config.grad_scaler.growth_interval``: FSDP/FSDP2 参数，表示在不发生梯度溢出的情况下，缩放因子增加所需的连续迭代步数。

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0

``reward.reward_type``：训练所使用的奖励类型。

``reward.reward_scale``：答对奖励为 ``reward_scale``，答错为 ``-reward_scale``。

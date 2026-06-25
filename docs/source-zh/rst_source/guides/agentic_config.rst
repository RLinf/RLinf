智能体强化学习配置
=====================

本节介绍智能体和推理 RL 训练专用的配置参数（数学推理、代码智能体、多智能体系统）。
这些参数扩展了 :doc:`basic_config` 中的共享配置。

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    enable_dynamic_batch_size: False
    max_tokens_per_mbs: 2048

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``runner.enable_dynamic_batch_size``
     - 使用 Megatron 训练时是否启用动态批大小。
   * - ``runner.max_tokens_per_mbs``
     - 启用动态批时，单个 Megatron 微批中 token 数的上限。

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

   * - 参数
     - 说明
   * - ``algorithm.n_minibatches``
     - 每个批次的梯度更新次数。
   * - ``algorithm.training_batch_size_per_gpu``
     - 每个 Actor GPU 的微批大小。
   * - ``algorithm.rollout_batch_size_per_gpu``
     - 每个 GPU 的推理微批；``null`` 表示将全局 rollout 批均匀分配到各推理实例。
   * - ``algorithm.recompute_logprobs``
     - 在训练引擎中重新计算对数概率，而非信任 rollout 引擎的取值。
   * - ``algorithm.shuffle_rollout``
     - 在优化步之前对 rollout 样本进行打乱。
   * - ``algorithm.clip_ratio_low`` / ``clip_ratio_high``
     - 非对称 PPO 裁剪界；``null`` 时回退到 ``ratio_clip_eps``\ （见 :doc:`basic_config`）。
   * - ``algorithm.sampling_params.max_new_tokens``
     - 最大生成 token 数；由 ``runner.seq_length`` 与 ``data.max_prompt_length`` 计算得到。
   * - ``algorithm.sampling_params.min_new_tokens``
     - 最小生成 token 数。

共享的损失/优势相关键（``loss_type``、``adv_type``、``kl_beta``、
``ratio_clip_eps``、``group_size``、``sampling_params.temperature`` 等）在
:doc:`basic_config` 中说明。

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    rollout_backend: sglang       # [sglang, vllm]

    enforce_eager: False
    distributed_executor_backend: mp   # ray 或 mp
    disable_log_stats: False
    detokenize: False
    padding: null                 # 为 null 时使用 tokenizer.pad_token_id
    eos: null                     # 为 null 时使用 tokenizer.eos_token_id

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

   * - 参数
     - 说明
   * - ``rollout.rollout_backend``
     - 使用的生成后端（``sglang`` 或 ``vllm``）。决定 ``sglang`` / ``vllm`` 哪个子块生效。
   * - ``rollout.enforce_eager``
     - 为 True 时禁用 CUDA 图捕获以缩短预热时间。
   * - ``rollout.distributed_executor_backend``
     - 启动 rollout Worker 的后端（``mp`` 或 ``ray``）。
   * - ``rollout.disable_log_stats``
     - 抑制后端的周期性统计日志。
   * - ``rollout.detokenize``
     - 为调试反 token 化输出（RL 通常仅使用 token id）。
   * - ``rollout.padding``
     - pad token id 覆盖；``null`` 时使用 ``tokenizer.pad_token_id``。
   * - ``rollout.eos``
     - EOS token id 覆盖；``null`` 时使用 ``tokenizer.eos_token_id``。
   * - ``rollout.tensor_parallel_size``
     - 生成后端内部的 TP 并行度。见 :doc:`5D`。
   * - ``rollout.pipeline_parallel_size``
     - 生成后端内部的 PP 并行度。见 :doc:`5D`。
   * - ``rollout.return_logprobs``
     - 引擎是否返回对数概率；默认为 ``algorithm.recompute_logprobs`` 的取反。
   * - ``rollout.validate_weight``
     - 首次发送全部权重以进行交叉校验/验证。
   * - ``rollout.validate_save_dir``
     - 启用验证时用于保存对比权重的目录。
   * - ``rollout.print_outputs``
     - 打印引擎的 token id/文本以供调试。
   * - ``rollout.max_running_requests``
     - 最大并发解码请求数。
   * - ``rollout.cuda_graph_max_bs``
     - 可使用 CUDA 图的最大批大小。

**SGLang 后端（``rollout.sglang``）：**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``rollout.sglang.attention_backend``
     - 注意力内核后端（``flashinfer`` 或 ``triton``）。
   * - ``rollout.sglang.decode_log_interval``
     - SGLang 记录解码统计的间隔（步）。
   * - ``rollout.sglang.use_torch_compile``
     - 在 SGLang 内启用 ``torch.compile``。
   * - ``rollout.sglang.torch_compile_max_bs``
     - 可使用 ``torch.compile`` 的最大批大小。

**vLLM 后端（``rollout.vllm``）：**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``rollout.vllm.attention_backend``
     - 注意力后端（``FLASH_ATTN`` 或 ``XFORMERS``）。
   * - ``rollout.vllm.enable_chunked_prefill``
     - 启用分块预填充。
   * - ``rollout.vllm.enable_prefix_caching``
     - 启用前缀缓存。
   * - ``rollout.vllm.enable_flash_infer_sampler``
     - 使用 FlashInfer 进行采样。
   * - ``rollout.vllm.max_num_batched_tokens``
     - 一起批处理的最大 token 数；``null`` 时使用 vLLM 默认值。

``rollout.group_name``、``rollout.gpu_memory_utilization`` 与 ``rollout.model.*``
为共享键，在 :doc:`basic_config` 中说明。

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

   * - 参数
     - 说明
   * - ``data.type``
     - 数据集/任务族（例如 ``math``）。
   * - ``data.dataset_name``
     - 用于选择预处理/格式化的数据集标识。
   * - ``data.max_prompt_length``
     - prompt 允许的最大 token 数。
   * - ``data.filter_prompt_by_length``
     - 丢弃长度超过 ``max_prompt_length`` 的 prompt，而非截断。
   * - ``data.rollout_batch_size``
     - 跨引擎的全局 rollout 批大小。
   * - ``data.val_rollout_batch_size``
     - 全局验证 rollout 批大小；``null`` 时回退到 ``data.rollout_batch_size``。
   * - ``data.num_workers``
     - 每个 Actor rank 的数据加载 worker 数。
   * - ``data.prompt_key``
     - 存储 prompt 文本的 JSONL 键。
   * - ``data.answer_key``
     - 存储参考答案/解答的 JSONL 键。
   * - ``data.apply_chat_template``
     - 在 token 化前用分词器的对话模板包裹 prompt。
   * - ``data.shuffle``
     - 每个 epoch 打乱训练数据。
   * - ``data.validation_shuffle``
     - 打乱验证数据。
   * - ``data.seed``
     - 加载器与采样的随机种子。
   * - ``data.train_data_paths``
     - 训练 JSONL 文件路径列表。
   * - ``data.val_data_paths``
     - 验证 JSONL 文件路径列表。

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
      distributed_backend: nccl # 'nccl' 或 'gloo'
      distributed_timeout_minutes: 30
      ckpt_format: torch
      use_dist_ckpt: False
      tp_comm_bootstrap_backend: nccl
      tp_comm_overlap_cfg: null
      use_hf_ckpt: True
      use_profiler: False

      ckpt_convertor: # 检查点转换器的配置
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

**顶层**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.training_backend``
     - 训练后端（``megatron`` 或 ``fsdp``）。
   * - ``actor.mcore_gpt``
     - 使用 Megatron-Core GPT 栈。
   * - ``actor.spec_name``
     - 模型 spec/预设名称（例如仅解码器 GPT）。
   * - ``actor.offload_optimizer``
     - 将优化器状态卸载到 CPU 以降低显存。
   * - ``actor.offload_weight``
     - 在可能时将模型权重卸载到 CPU。
   * - ``actor.offload_grad``
     - 将梯度卸载到 CPU 以降低显存。
   * - ``actor.enable_dp_load_balance``
     - 启用数据并行负载均衡。
   * - ``actor.calculate_flops``
     - 计算并记录 FLOPs 以供性能分析。
   * - ``actor.seed``
     - 用于可复现性的全局随机种子。

**模型子节**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.model.precision``
     - 训练的数值精度（例如 ``fp16``）。
   * - ``actor.model.add_bias_linear``
     - 为线性层添加偏置项。
   * - ``actor.model.tensor_model_parallel_size``
     - Actor 的 TP 并行度。
   * - ``actor.model.pipeline_model_parallel_size``
     - Actor 的 PP 并行度。
   * - ``actor.model.activation``
     - 激活函数（例如 ``swiglu``）。
   * - ``actor.model.sequence_parallel``
     - 启用序列并行（需要 TP）。
   * - ``actor.model.recompute_method``
     - 激活重计算策略（例如 ``block``）。
   * - ``actor.model.recompute_granularity``
     - 重计算范围（``full`` 或 ``selective``）。
   * - ``actor.model.recompute_num_layers``
     - 进行检查点/重计算的层数。
   * - ``actor.model.seq_length``
     - 训练的解码器上下文长度。
   * - ``actor.model.encoder_seq_length``
     - 编码器长度（仅解码器模型下与 ``seq_length`` 一致）。
   * - ``actor.model.normalization``
     - 归一化层类型（例如 ``rmsnorm``）。
   * - ``actor.model.position_embedding_type``
     - 位置编码类型（例如 ``rope``）。
   * - ``actor.model.apply_rope_fusion``
     - 在可用时使用融合的 RoPE 内核。
   * - ``actor.model.bias_dropout_fusion``
     - 融合 bias + dropout 内核。
   * - ``actor.model.persist_layer_norm``
     - 以更高精度持久化 LayerNorm 参数。
   * - ``actor.model.bias_activation_fusion``
     - 融合 bias + 激活内核。
   * - ``actor.model.attention_softmax_in_fp32``
     - 以 FP32 计算注意力 softmax 以提升稳定性。
   * - ``actor.model.batch_p2p_comm``
     - 跨层批处理 P2P 通信。
   * - ``actor.model.variable_seq_lengths``
     - 允许每个微批使用可变序列长度。
   * - ``actor.model.gradient_accumulation_fusion``
     - 融合的梯度累积。
   * - ``actor.model.moe_token_dispatcher_type``
     - MoE token 分发器（例如 ``alltoall``）。
   * - ``actor.model.use_cpu_initialization``
     - 在 CPU 上初始化权重以降低显存峰值。

**优化器**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.optim.optimizer``
     - 优化器选择（``adam``）。
   * - ``actor.optim.bf16`` / ``actor.optim.fp16``
     - 混合精度标志。
   * - ``actor.optim.lr``
     - 基础学习率。
   * - ``actor.optim.adam_beta1`` / ``adam_beta2`` / ``adam_eps``
     - Adam 超参数。
   * - ``actor.optim.min_lr``
     - 最小学习率（用于会衰减到基础学习率以下的调度器）。
   * - ``actor.optim.weight_decay``
     - L2 权重衰减。
   * - ``actor.optim.use_distributed_optimizer``
     - 使用 Megatron 分布式优化器。
   * - ``actor.optim.overlap_grad_reduce``
     - 让梯度规约与反向传播重叠。
   * - ``actor.optim.overlap_param_gather``
     - 让参数 all-gather 与前向传播重叠。
   * - ``actor.optim.optimizer_enable_pin``
     - 固定（pin）优化器内存。
   * - ``actor.optim.overlap_param_gather_with_optimizer_step``
     - 让参数 gather 与优化器步重叠。
   * - ``actor.optim.clip_grad``
     - 全局梯度裁剪范数。
   * - ``actor.optim.loss_scale``
     - 静态 FP16 损失缩放（动态缩放请改用 ``loss_scale_window``）。

**学习率调度**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.lr_sched.lr_warmup_fraction``
     - 预热占总迭代数的比例。
   * - ``actor.lr_sched.lr_warmup_init``
     - 预热期间的初始学习率。
   * - ``actor.lr_sched.lr_warmup_iters``
     - 预热迭代数（> 0 时覆盖比例设置）。
   * - ``actor.lr_sched.max_lr`` / ``min_lr``
     - 调度器的学习率上下界。
   * - ``actor.lr_sched.lr_decay_style``
     - 衰减策略（例如 ``constant``）。
   * - ``actor.lr_sched.lr_decay_iters``
     - 总衰减迭代数。

**分词器**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.tokenizer.tokenizer_model``
     - 分词器的路径/名称。
   * - ``actor.tokenizer.use_fast``
     - 使用 HuggingFace 快速分词器。
   * - ``actor.tokenizer.trust_remote_code``
     - 允许自定义分词器代码。
   * - ``actor.tokenizer.padding_side``
     - ``left`` 或 ``right`` 填充。

**Megatron 集成**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.megatron.ddp_bucket_size``
     - DDP 梯度分桶大小。
   * - ``actor.megatron.distributed_backend``
     - 分布式后端（``nccl`` 或 ``gloo``）。
   * - ``actor.megatron.distributed_timeout_minutes``
     - 后端通信超时。
   * - ``actor.megatron.ckpt_format``
     - 检查点格式（例如 ``torch``）。
   * - ``actor.megatron.use_dist_ckpt``
     - 使用分布式（分片）检查点。
   * - ``actor.megatron.tp_comm_bootstrap_backend``
     - 用于 TP 引导的后端（例如 ``nccl``）。
   * - ``actor.megatron.tp_comm_overlap_cfg``
     - TP 通信/计算重叠的 YAML 路径。
   * - ``actor.megatron.use_hf_ckpt``
     - 从 HuggingFace 检查点转换/加载以进行训练。
   * - ``actor.megatron.use_profiler``
     - 训练期间启用 Torch profiler（会影响性能）。

**Megatron 检查点转换器（``actor.megatron.ckpt_convertor``）**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.megatron.ckpt_convertor.model``
     - 转换器元数据使用的模型名称。
   * - ``actor.megatron.ckpt_convertor.hf_model_path``
     - 源 HF 模型路径。
   * - ``actor.megatron.ckpt_convertor.save_path``
     - 转换后 Megatron 检查点的目标目录。
   * - ``actor.megatron.ckpt_convertor.use_gpu_num``
     - 用于转换的 GPU 数量。
   * - ``actor.megatron.ckpt_convertor.use_gpu_index``
     - 使用的具体 GPU 索引。
   * - ``actor.megatron.ckpt_convertor.process_num``
     - 用于转换的 CPU 进程数。
   * - ``actor.megatron.ckpt_convertor.tensor_model_parallel_size``
     - 转换后检查点的 TP 并行度。
   * - ``actor.megatron.ckpt_convertor.pipeline_model_parallel_size``
     - 转换后检查点的 PP 并行度。

**FSDP 集成（``actor.fsdp_config``）**

当 ``actor.training_backend`` 为 ``fsdp`` 时使用。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``actor.fsdp_config.strategy``
     - FSDP 策略：``fsdp`` 或 ``fsdp2``\ （不区分大小写）。
   * - ``actor.fsdp_config.sharding_strategy``
     - 分片策略：``full_shard``、``shard_grad_op``、``hybrid_shard`` 或 ``no_shard``。
   * - ``actor.fsdp_config.cpu_offload``
     - FSDP2：参数保留在 CPU，仅在需要时移到 GPU。
   * - ``actor.fsdp_config.offload_pin_memory``
     - FSDP2：使用固定（pinned）CPU 内存（仅当 ``cpu_offload`` 为 True）以加快传输。
   * - ``actor.fsdp_config.reshard_after_forward``
     - FSDP2：前向后重新分片参数以节省显存。
   * - ``actor.fsdp_config.enable_gradient_accumulation``
     - 仅在最后一个微批后进行通信/更新。以显存换训练速度。
   * - ``actor.fsdp_config.forward_prefetch``
     - FSDP：在前向期间预取下一次 all-gather（更多显存，更好重叠）。
   * - ``actor.fsdp_config.limit_all_gathers``
     - FSDP：限制并发 all-gather（CPU/内存受限时建议开启）。
   * - ``actor.fsdp_config.backward_prefetch``
     - FSDP：反向中的预取策略（``null`` / ``pre`` / ``post``）。
   * - ``actor.fsdp_config.use_orig_params``
     - FSDP：暴露原始（未展平）参数；兼容性更好，但通信开销更大。
   * - ``actor.fsdp_config.use_liger_kernel``
     - 使用 Liger 内核（目前支持 Qwen2.5 / Qwen2.5-VL）以降低显存并加速训练。
   * - ``actor.fsdp_config.mixed_precision.param_dtype``
     - 参数数据类型。
   * - ``actor.fsdp_config.mixed_precision.reduce_dtype``
     - 规约数据类型。
   * - ``actor.fsdp_config.mixed_precision.buffer_dtype``
     - 缓冲区数据类型。
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - 启用自动混合精度（AMP）训练。
   * - ``actor.fsdp_config.amp_autocast.precision``
     - AMP 使用的数值精度。
   * - ``actor.fsdp_config.grad_scaler.enabled``
     - 启用梯度缩放器。

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 参数
     - 说明
   * - ``reward.reward_type``
     - 训练使用的奖励类型（例如 ``math``）。
   * - ``reward.reward_scale``
     - 答案正确时获得 ``reward_scale``；错误时获得 ``-reward_scale``。

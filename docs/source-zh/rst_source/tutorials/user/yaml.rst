YAML 配置
=====================

下面给出 RLinf 使用的配置文件的完整参考说明。  
我们对 YAML 中每个重要键做了说明，便于你针对自己的集群、模型或研究需求进行调整。  
各参数按顶层键分组组织。

为便于查阅，本节分为三部分：**基础配置**、**MATH 专用配置** 与 **具身智能（Embodied）专用配置**。  
你可根据实际需求查找对应的小节。

.. contents::
   :depth: 1
   :local:

基础配置
---------------------

hydra
~~~~~~

.. code:: yaml

  hydra:
    run:
      dir: .
    output_subdir: null 

``hydra.run.dir``：Hydra 运行的工作目录。

``hydra.output_subdir``：输出子目录（设为 null 则不创建子目录）。


cluster
~~~~~~~~~~~~~~~

.. code:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      actor,inference,rollout: all

``cluster.num_nodes``：用于训练的物理节点数量。

``cluster.component_placement``：  
各组件（进程）的 *放置策略*。  

在上面运行于GPU节点的简单示例中：

- 键 (key) 是组件的名称，例如 ``rollout``，或 ``rollout,inference,actor``
- 值 (value) 是分配给这些组件的全局 GPU Rank，可以是：
   - "all"：使用集群中的所有 GPU
   - 单个整数，例如 "3"：使用 GPU 3
   - 逗号分隔的整数列表，例如 "0,2,3"：使用 GPU 0、2 和 3
   - 连字符分隔的整数范围，例如 "0-3"：使用 GPU 0、1、2 和 3
   - 上述两种方式的组合，例如 "0-3,5,7"：使用 GPU 0、1、2、3、5 和 7

而对于更高级的组件放置用法（例如，异构集群中使用不同型号的 GPU、机器人硬件或仅 CPU 节点）以及代码中的自定义，请参见 :doc:`./placement`。

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

``runner.task_type``：任务类型标识（math 或 embodied）。

**logger：**

``runner.logger.log_path``：日志输出的根目录。  

``runner.logger.project_name``：实验跟踪的项目名。  

``runner.logger.experiment_name``：实验名称。  

``runner.logger.logger_backends``：日志后端（tensorboard、wandb、swanlab）。

关于日志后端详见 :doc:`../advance/logger`。

``runner.max_epochs``：最大训练 epoch 数。  

``runner.max_steps``：最大全局步数；为 -1 时，依据 ``runner.max_epochs`` 自动确定。  

``runner.val_check_interval``：验证 rollout 的触发频率（-1 关闭）。  

``runner.save_interval``：保存 checkpoint 的步数间隔。  

``runner.seq_length``：输入到模型的总序列长度（提示 + 生成）。

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

``algorithm.group_size``：每个提示采样的响应个数（>1 时启用组基线）。  

``algorithm.logprob_forward_micro_batch_size``：log-prob 前向的微批大小。

``algorithm.val_rollout_batch_size_per_gpu``：验证阶段每 GPU 的 rollout 微批大小。

``algorithm.loss_type``：策略损失类型（如 ppo）。  

``algorithm.loss_agg_func``：token 损失的聚合方式（如 token-mean）。  

``algorithm.kl_beta``：加入到奖励中的 KL 权重。  

``algorithm.kl_penalty_type``：KL 形态（如 low_var_kl）。  

``algorithm.ratio_clip_eps``：PPO 比率裁剪阈值。  

``algorithm.entropy_bonus``：熵奖励系数。  

``algorithm.calculate_entropy``：是否计算/记录熵项。  

``algorithm.adv_type``：优势函数估计类型（如 grpo）。  

``algorithm.normalize_advantages``：是否对优势进行归一化。  

``algorithm.early_stop_imp_ratio``：当重要性比超出阈值时提前终止本次更新。 

``algorithm.use_valid_token_scale``：是否按有效 token 掩码缩放损失/优势。

**sampling_params：**

``algorithm.sampling_params.use_greedy``：True 时使用贪心解码。
 
``algorithm.sampling_params.temperature``：采样温度。  

``algorithm.sampling_params.top_k``：top-k 截断（设很大值等于禁用）。  

``algorithm.sampling_params.top_p``：nucleus 采样阈值。  

``algorithm.sampling_params.repetition_penalty``：重复惩罚系数。

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    group_name: "RolloutGroup"

    gpu_memory_utilization: 0.55

    model_dir: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
    model_arch: qwen2.5

    recompute_logprobs: True

``rollout.gpu_memory_utilization``：目标 GPU 显存占用比例。  

``rollout.group_name``：rollout / inference worker 的逻辑分组名。  

``rollout.model_dir``：生成后端所用 HF 模型路径。  

``rollout.model_arch``：后端内部使用的模型架构标记（如 qwen2.5）。  

``rollout.recompute_logprobs``：是否为采样序列重新计算对数概率。

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    group_name: "ActorGroup"

    checkpoint_load_path: null

    seed: 1234

**顶层：**

``actor.group_name``：训练（actor）worker 的逻辑分组名。  

``actor.checkpoint_load_path``：训练前加载的 checkpoint 路径。 

``actor.seed``：全局随机种子，便于复现。

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    use_reward_model: false

``reward.use_reward_model``：是否使用奖励模型。

critic
~~~~~~~~~~~~~~~

.. code:: yaml

  critic:
    use_critic_model: false

``critic.use_critic_model``：是否使用价值网络（critic）。

MATH 专用配置
----------------------------

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
        model_type: null # 若为 null，将由 HF 配置推断
        hf_model_path: ${rollout.model_dir} # HF 模型所在路径
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

      fsdp_size: -1

      mixed_precision:
        param_dtype: ${actor.model.precision}
        reduce_dtype: ${actor.model.precision}
        buffer_dtype: ${actor.model.precision}

      amp:
        enabled: False
        precision: "bf16"
        use_grad_scaler: False

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

``actor.megatron.ckpt.model_type``：模型类型；为 null 时会从 HF 配置中推断。

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

``actor.fsdp_config.fsdp_size``: FSDP2参数，如果不为-1，则FSDP2会按照该参数指定的大小进行分组切片

``actor.fsdp_config.mixed_precision.param_dtype``: FSDP/FSDP2参数，指定参数类型

``actor.fsdp_config.mixed_precision.reduce_dtype``: FSDP/FSDP2参数，指定规约时使用的数据类型

``actor.fsdp_config.mixed_precision.buffer_dtype``: FSDP1参数，指定缓冲区使用的数据类型

``actor.fsdp_config.amp.enabled``: FSDP/FSDP2参数，表示是否启用自动混合精度训练

``actor.fsdp_config.amp.precision``: FSDP/FSDP2参数，表示AMP使用的数值精度

``actor.fsdp_config.amp.use_grad_scaler``: FSDP/FSDP2参数，表示是否启用梯度缩放器

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    reward_type: math
    reward_scale: 5.0

``reward.reward_type``：训练所使用的奖励类型。  

``reward.reward_scale``：答对奖励为 ``reward_scale``，答错为 ``-reward_scale``。

具身智能（Embodied）专用配置
-------------------------------

defaults
~~~~~~~~~~~~~~~

.. code:: yaml

  defaults:
    - env/train: PutCarrotOnPlateInScene
    - env/eval: PutCarrotOnPlateInScene

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

``runner.only_eval``：只运行评估，不进行训练。  

``runner.max_prompt_length``：最大提示长度（token 数）。

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

``algorithm.auto_reset``：是否在 episode 结束时自动重置环境。

``algorithm.ignore_terminations``：训练时是否忽略 episode 的终止信号（若开启，episode 仅在达到最大步数时结束）。

``algorithm.use_fixed_reset_state_ids``：是否使用固定 reset 状态 ID（GRPO 推荐 True，PPO 默认为 False，旨在随机化）。

``algorithm.normalize_advantages``：是否对优势值归一化处理。

``algorithm.n_chunk_steps``：每个 rollout epoch 中的 chunk 数量（调用模型 predict 的次数）。

``algorithm.n_eval_chunk_steps``：评估模式下的 chunk 数量。

``algorithm.num_group_envs``：环境组数量（用于并行）。

``algorithm.rollout_epoch``：每个训练步骤前的 rollout 轮数。

``algorithm.reward_type``：奖励聚合层级（chunk_level、action_level）。

``algorithm.logprob_type``：对数概率的计算层级。

``algorithm.entropy_type``：熵的计算层级。

**length_params：**

``algorithm.length_params.max_new_token``：最大新增 token 数。  

``algorithm.length_params.max_length``：最大总序列长度。  

``algorithm.length_params.min_length``：最小序列长度。

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

``env.group_name``：环境 worker 组的逻辑名称。  

``env.channel.name``：进程间通信的共享内存通道名。  

``env.channel.queue_name``：观测缓冲区队列名。  

``env.channel.queue_size``：队列大小（0 表示不限制）。  

``env.enable_offload``：启用环境侧的下放以降低内存占用。

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

``rollout.channel.name``：共享内存通道（继承自 env）。  

``rollout.channel.queue_name``：动作缓冲区队列名。  

``rollout.channel.queue_size``：队列大小。  

``rollout.mode``：rollout 模式（collocate 表示**共享式**使用 GPU）。  

``rollout.backend``：模型后端（huggingface、vllm）。  

``rollout.pipeline_stage_num``：模型并行的流水线阶段数。

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

``actor.channel.name``：共享内存通道（继承自 env）。  

``actor.channel.queue_name``：回放缓冲区队列名。  

``actor.training_backend``：训练后端（分布式 FSDP）。  

``actor.micro_batch_size``：每张 GPU 的微批大小。  

``actor.global_batch_size``：全局批大小（跨所有 GPU）。  

``actor.enable_offload``：启用模型下放以降低内存占用。

**模型配置：**

``actor.model.model_name``：模型结构名（openvla_oft）。  

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

``actor.model.ckpt_path``：模型 checkpoint 路径。  

``actor.model.num_images_in_input``：输入的图像数量。  

``actor.model.use_wrist_image``：是否使用机器人末端手腕（wrist）上的摄像头拍摄的图像。  

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

基于环境的配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下示例以 Libero-10 为例说明环境关键参数。

路径为 

**环境类型**

.. code:: yaml

  simulator_type: libero
  task_suite_name: libero_10

``simulator_type``：模拟器类型（libero 表示 Libero 基准）。  

``task_suite_name``：任务集合（libero_10 表示 10 个任务的基准）。

**Episode 配置**

.. code:: yaml

  auto_reset: ${algorithm.auto_reset}
  ignore_terminations: ${algorithm.ignore_terminations}
  max_episode_steps: 512

``auto_reset``：episode 结束时是否自动重置（继承自 algorithm）。  

``ignore_terminations``：训练时是否忽略终止（继承自 algorithm）。  

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
  num_task: ${algorithm.num_group_envs}
  num_group: ${algorithm.num_group_envs}
  group_size: ${algorithm.group_size}
  use_fixed_reset_state_ids: ${algorithm.use_fixed_reset_state_ids}

``seed``：环境初始化随机种子（0 便于复现）。  

``num_task``：任务数量（继承自 algorithm.num_group_envs）。  

``num_group``：环境分组数量（继承自 algorithm.num_group_envs）。  

``group_size``：每个分组的环境数（继承自 algorithm.group_size）。  

``use_fixed_reset_state_ids``：是否使用固定 reset 状态（GRPO 为 True，PPO 默认 False）。

**输入配置**

.. code:: yaml

  use_wrist_image: False

``use_wrist_image``：是否使用机器人末端手腕（wrist）上的摄像头拍摄的图像。

**环境规模**

.. code:: yaml

  num_envs: ${multiply:${algorithm.group_size}, ${algorithm.num_group_envs}}

``num_envs``：总环境数（= group_size × num_group_envs）。

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

使用 Megatron+SGLang 添加新模型
=============================================

本文档基于 Megatron 实现，提供了在 RLinf 框架中添加新的 HuggingFace 模型的完整指南。  

前置条件
-------------

* 熟悉 **Megatron-LM** 分布式训练框架  
* 理解 **RLinf** 框架架构  
* 掌握 **PyTorch** 和分布式训练知识  

逐步实现
---------------------------

1. Megatron 模型配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如需添加新模型到 RLinf 中，请先确保使用的 Megatron-LM 版本已经支持该模型。

否则，您需要先在 Megatron-LM 中添加对新模型的支持。

参考官方指南：
`Megatron-LM <https://github.com/NVIDIA/Megatron-LM/>`__

**文件：** ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``  

当前 RLinf 框架中 Megatron-LM 训练框架中调用模型配置文件为 ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``。

修改 RLinf 传递给 Megatron-LM 的 ``model_provider_func``，以支持您需要进行训练的模型结构。  

.. code-block:: python

   def model_provider_func(self, pre_process, post_process):
       """模型依赖于流水线并行。"""
       use_te = HAVE_TE

       if self.mcore_gpt:
           model = MCoreGPTModel(
               config=self.transformer_config,
               transformer_layer_spec=get_specs(
                   self.spec_name,
                   self.transformer_config,
                   use_te,
               ),
               vocab_size=self._cfg.model.override_vocab_size,
               max_sequence_length=self._cfg.model.max_position_embeddings,
               pre_process=pre_process,
               post_process=post_process,
               parallel_output=True,
               share_embeddings_and_output_weights=self._cfg.model.share_embeddings_and_output_weights,
               position_embedding_type=self._cfg.model.position_embedding_type,
               rotary_percent=self._cfg.model.rotary_percentage,
               seq_len_interpolation_factor=self._cfg.model.seq_len_interpolation_factor,
               rotary_base=self._cfg.model.rotary_base,
           )

       else:
           from megatron.legacy.model.gpt_model import GPTModel
           config = build_config(ModelParallelConfig, self._cfg.model)
           setattr(config, 'hidden_size', self._cfg.model.hidden_size)

           model = GPTModel(
               config=config,
               num_tokentypes=0,
               parallel_output=True,
               pre_process=pre_process,
               post_process=post_process
           )
       return model

2. Megatron-LM 权重转换支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Megatron-LM 训练框架并不能直接从 hf 格式的 checkpoint 直接读取，需要进行权重转换操作。

您可以在 yaml 文件中通过配置 use_hf_ckpt 选项，启用是否在第一次 RLinf 的训练过程中从 huggingface 格式的 checkpoint 直接读取并转换为 Megatron-LM 格式的 checkpoint。

转换后，当前 Megatron-LM 格式 的 checkpoint 会保存在 ``save_path`` 目录下。

例如：

.. code-block:: yaml

   actor:
    group_name: "ActorGroup"
    training_backend: megatron
    mcore_gpt: True
    spec_name: decoder_gpt
    checkpoint_load_path: null
   megatron:
    use_hf_ckpt: True
    ckpt_convertor:
      model: DeepSeek-R1-Distill-Qwen-1.5B
      hf_model_path: ${rollout.model_dir}
      save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor

这样的方式会在 RLinf 的第一次训练过程中，进行一次 Megatron-LM 格式的 checkpoint 转换。但是，从 huggingface 格式的 checkpoint 转换到 Megatron-LM 格式的 checkpoint 是非常耗时的过程。

如果您之前已经转换过 Megatron-LM 格式 checkpoint, 您也可以直接在 yaml 文件中通过配置 ``checkpoint_load_path`` 选项，指定转换好的 Megatron-LM 格式 checkpoint 路径，在后续的训练过程中可以直接使用。

例如：

.. code-block:: yaml

   actor:
    group_name: "ActorGroup"
    training_backend: megatron
    mcore_gpt: True
    spec_name: decoder_gpt
    checkpoint_load_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor
   megatron:
    use_hf_ckpt: False
    ckpt_convertor:
      model: DeepSeek-R1-Distill-Qwen-1.5B
      hf_model_path: ${rollout.model_dir}
      save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor


这样转换好的 checkpoint 会直接被 RLinf 使用，无需再进行转换。

如果您有需要适配其他模型权重转换的需求，您可以直接在 `RLinf <https://github.com/RLinf/RLinf/issues>`__ 的 github 仓库中提出对应的 issue。

您也可以通过参考 ``toolkits/ckpt_convertor`` 中的文件，自主适配新的模型转换代码。

**工具：** ``toolkits/ckpt_convertor``  

当前 RLinf 中的 ckpt_convert 转换工具是先将 huggingface 格式的 checkpoint 转换为中间文件格式，然后再将中间文件格式转换为 Megatron-LM 格式的 checkpoint。

具体的接口在 ``toolkits/ckpt_convertor/convert_hf_to_mg.py`` 文件中。

.. code-block:: python

   def convert_hf_to_mg(
    hf_ckpt_path: str,
    ckpt_cfg: DictConfig,
    ):
    ...
    hf_to_middle_file(convert_config)
    # adjust to script's requirement
    convert_config.load_path = save_path

    middle_file_to_mg(convert_config)
    convert_config.load_path = load_path
    ...

RLinf 使用 hf_to_middle_file 方法将 huggingface 格式的 checkpoint 转换为中间文件格式。

然后再使用 middle_file_to_mg  方法将中间文件格式转换为 Megatron-LM 格式的 checkpoint。

如果您需要适配新的模型，那么您需要修改这两个方法，来对新模型中的权重进行对应的切分以及转换。

如果您已经适配了新的模型，我们特别欢迎您可以将 code 提交 pull request 贡献到我们的 `RLinf <https://github.com/RLinf/RLinf/pulls>`__ 仓库中，让更多的用户受益！

3. SGLang 模型配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 仅支持已经被 SGLang 适配过的模型。  

如果需要新模型，需要在 SGLang 中添加适配代码。

参考官方指南：  
`SGLang-support-new-model <https://docs.SGLang.ai/supported_models/support_new_models.html>`__  

4. 从 Megatron 同步权重到 SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 RLinf 中 MegatronActor 在每次开始训练前将当前的权重转换为同步到各个 SGLangActor 中。

其中就会涉及到对 MegatronActor 的权重格式转换到 SGLangActor 的权重格式。

权重转换分为两个阶段，第一个阶段是 MegatronActor 会将权重转换为 Sglang 格式并发送给各个 SGLangActor。

第二个阶段是 SGLangActor 会将接收到的权重进行加载并更新当前自己的权重。

**文件：** ``rlinf/utils/resharding/mcore_weight_reshard.py``  

在每次 rollout 前，**MegatronActor** 会使用 gather_and_reshard_model 将权重格式进行转换。

共享式下每个 MegatronActor 会一一对应给一个 SGLangActor 发送权重。  

首先会按照 MegatronActor 的并行配置重新进行权重的合并以及切分，然后对切分后的权重进行重命名，最后转换为 SGLangActor 的权重格式。

转换后的权重会通过 send 接口发送给对应的 SGLangActor。

.. code-block:: python

    def gather_and_reshard_model(self, model):
        ...
        # 按照 MegatronActor 的并行配置重新进行权重的合并以及切分
        if vp_size > 1:  # consolidate params across model chunks
            for idx, model_chunk in enumerate(model):
                for key, val in model_chunk.state_dict().items():
                    if "_extra_state" in key:
                        continue
                    if torch.is_tensor(val):
                        if "layers" in key:
                            key2 = rename_layer_num(
                                key,
                                get_layer_num(key) + idx * pp_size * layers_per_chunk,
                            )
                            tl_params[key2] = val
                        else:
                            model_level_params[key] = val
        else:
            for key, val in model[0].state_dict().items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if "decoder.layers" in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val

        if vp_size > 1 or reshard_pp_model:
            # gather layers across pp ranks
            gathered_params = {}
            for key, val in tl_params.items():
                weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
                torch.distributed.all_gather(weight_list, val, group=pp_group)
                for idx in range(pp_size):
                    layer_num = get_layer_num(key) + idx * layers_per_chunk
                    key2 = rename_layer_num(key, layer_num)
                    if not reshard_pp_model:  # Save only layers of 1 single PP stage
                        layers_start = layers_per_pp * pp_rank
                        layers_end = layers_per_pp * (pp_rank + 1) - 1
                        if layer_num >= layers_start and layer_num <= layers_end:
                            key2 = rename_layer_num(key, layer_num % layers_per_pp)
                            gathered_params[key2] = weight_list[idx]
                    else:
                        gathered_params[key2] = weight_list[idx]
            tl_params = gathered_params

        model_state_dict = model_level_params
        model_state_dict.update(tl_params)

        reshard_dtype = self.config.model_config.params_dtype

        # MegatronActor 对切分后的权重进行重命名
        if reshard_pp_model:
            model_state_dict = self.config.pp_reshard_fn(
                model_state_dict, pp_group, reshard_dtype
            )

        if reshard_tp_model:
            rank = torch.distributed.get_rank()
            group_index = rank // self.merge_factor
            subgroup_ranks = list(
                range(
                    group_index * self.merge_factor,
                    (group_index + 1) * self.merge_factor,
                )
            )
            tp_sub_group = self._get_tp_subgroup(subgroup_ranks)
            model_state_dict = self.config.tp_reshard_fn(
                model_state_dict, self.merge_factor, tp_sub_group
            )
        
        # 对切分后的权重进行重命名，最后转换为 SGLangActor 的权重格式
        if self.config.convert_fn is not None:
            model_state_dict = self.config.convert_fn(model_state_dict)
        ...

如果需要支持更多的模型，您需要新增 tp_reshard_fn 和 pp_reshard_fn 方法，来对新模型中的权重进行对应的切分以及转换， 新增 convert_fn 转换规则来适配对切分后的权重进行重命名。

以 qwen2.5 为例。

其中 MegatronActor 的权重的合并以及切分代码在 ``rlinf/utils/resharding/utils.py`` 文件中。

.. code-block:: python

    def tp_reshard_fn_qwen2_5(model_state_dict, merge_factor, tp_group):
        for k, v in model_state_dict.items():
            if (
                "rotary_pos_emb.inv_freq" in k
                or "linear_qkv.layer_norm_weight" in k
                or "mlp.linear_fc1.layer_norm_weight" in k
                or "final_layernorm.weight" in k
            ):
                model_state_dict[k] = v.clone()
                continue

            dim = 0
            if "self_attention.linear_proj.weight" in k or "mlp.linear_fc2.weight" in k:
                dim = 1
            model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
                v, dim, merge_factor, tp_group
            )
        return model_state_dict

    def pp_reshard_fn_qwen2_5(model_state_dict, pp_group, dtype):
        from megatron.core import parallel_state

        pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

        key = "decoder.final_layernorm.weight"
        tensor = _gather_pp_group_tensor_and_reshard(
            model_state_dict, key, pp_last_rank, pp_group, dtype
        )
        if tensor is not None:
            model_state_dict[key] = tensor.clone()

        key = "decoder.final_layernorm.bias"
        tensor = _gather_pp_group_tensor_and_reshard(
            model_state_dict, key, pp_last_rank, pp_group, dtype
        )
        if tensor is not None:
            model_state_dict[key] = tensor.clone()

        key = "embedding.word_embeddings.weight"
        tensor = _gather_pp_group_tensor_and_reshard(
            model_state_dict, key, pp_first_rank, pp_group, dtype
        )
        if tensor is not None:
            model_state_dict[key] = tensor.clone()

        key = "output_layer.weight"
        tensor = _gather_pp_group_tensor_and_reshard(
            model_state_dict, key, pp_last_rank, pp_group, dtype
        )
        if tensor is not None:
            model_state_dict[key] = tensor.clone()
        return model_state_dict

MegatronActor 对切分后的权重进行重命名代码在 ``rlinf/utils/convertor/utils.py`` 文件中。

.. code-block:: python

    class Qwen2_5Convertor(BaseConvertor):
        def build_rules(self) -> List[ConvertorRule]:
            LID = r"(?P<i>\d+)"
            WB = r"(?P<wb>weight|bias)"

            return [
                # embeddings
                ConvertorRule(
                    re.compile(r"embedding\.word_embeddings\.weight$"),
                    TransformType.SPLIT_NONE,
                    [r"model.embed_tokens.weight"],
                ),
                # final_layernorm
                ConvertorRule(
                    re.compile(r"decoder\.final_layernorm\.weight$"),
                    TransformType.SPLIT_NONE,
                    [r"model.norm.weight"],
                ),
                # lm_head
                ConvertorRule(
                    re.compile(r"output_layer\.weight$"),
                    TransformType.SPLIT_NONE,
                    [r"lm_head.weight"],
                ),
                # attn qkv norm
                ConvertorRule(
                    re.compile(
                        rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.layer_norm_weight$"
                    ),
                    TransformType.SPLIT_NONE,
                    [r"model.layers.\g<i>.input_layernorm.weight"],
                ),
                # attn qkv weights/bias
                ConvertorRule(
                    re.compile(
                        rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.{WB}$"
                    ),
                    TransformType.SPLIT_QKV,
                    [
                        r"model.layers.\g<i>.self_attn.q_proj.\g<wb>",
                        r"model.layers.\g<i>.self_attn.k_proj.\g<wb>",
                        r"model.layers.\g<i>.self_attn.v_proj.\g<wb>",
                    ],
                ),
                # attn o proj
                ConvertorRule(
                    re.compile(
                        rf"decoder\.layers\.{LID}\.self_attention\.linear_proj\.{WB}$"
                    ),
                    TransformType.SPLIT_NONE,
                    [r"model.layers.\g<i>.self_attn.o_proj.\g<wb>"],
                ),
                # mlp fc1
                ConvertorRule(
                    re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.{WB}$"),
                    TransformType.SPLIT_FC1,
                    [
                        r"model.layers.\g<i>.mlp.gate_proj.\g<wb>",
                        r"model.layers.\g<i>.mlp.up_proj.\g<wb>",
                    ],
                ),
                # mlp fc2
                ConvertorRule(
                    re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc2\.{WB}$"),
                    TransformType.SPLIT_NONE,
                    [r"model.layers.\g<i>.mlp.down_proj.\g<wb>"],
                ),
                # mlp norms
                ConvertorRule(
                    re.compile(
                        rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.layer_norm_weight$"
                    ),
                    TransformType.SPLIT_NONE,
                    [r"model.layers.\g<i>.post_attention_layernorm.weight"],
                ),
            ]

目前仅实现了 *Qwen-2.5* 系列。  

完成上述步骤后，MegatronActor 会将权重发送给 SglangActor，更新当前 SglangActor 的权重。

SglangActor 接收权重代码 ``rlinf/hybrid_engines/sglang/common/sgl_scheduler.py`` 文件中。

.. code-block:: python

    def sync_hf_weight(self, recv_req: SyncHFWeightInput):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED

        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )

        model = self.tp_worker.worker.model_runner.model

        if self.is_weight_offloaded:
            self.resume_memory_occupation(ResumeMemoryOccupationReqInput())
            self.is_weight_offloaded = False

        if colocate:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)

                model.load_weights([(name, new_weight)])
                del new_weight
        else:
            # disaggregate mode, recv tensor directly
            for name, tensor in state_dict.items():
                model.load_weights([(name, tensor)])
        self.flush_cache()
        return SyncHFWeightOutput()

若需要对 SglangActor 接收的权重进行特殊处理，可在此处进行修改。

若您对整个适配新模型的过程存在任何疑问以及问题，欢迎随时给我们 `RLinf <https://github.com/RLinf/RLinf/issue>`__ 提出对应的 issue，我们会尽快解答您的问题。

5. qwen2.5 系列模型演示
~~~~~~~~~~~~~~~~~~~~~~~~~~

完成上述步骤后，您就可以适配您的新模型到 RLinf 中，下面展示是 qwen2.5 系列模型演示的 yaml 配置文件。

您可以在适配好新模型后，参考这个 yaml 配置文件，进行相应的修改。

**文件：** ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``

设置 RLinf 使用的 Megatron 参数。  

.. code-block:: yaml

   # Megatron 参数
   model:
    precision: fp16
    add_bias_linear: False

    tensor_model_parallel_size: 2
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

   megatron:
    ddp_bucket_size: null
    distributed_backend: nccl # 支持 'nccl' 和 'gloo'
    distributed_timeout_minutes: 30
    ckpt_format: torch
    use_dist_ckpt: False
    tp_comm_bootstrap_backend: nccl
    tp_comm_overlap_cfg: null # tp_comm_overlap_cfg.yaml
    use_hf_ckpt: True # 若为 True，会将 hf 模型转换生成 megatron 检查点，并用于训练。

    ckpt_convertor: # ckpt 转换器配置
      model: DeepSeek-R1-Distill-Qwen-1.5B
      model_type: null # 若为 null，会根据 hf 模型配置自动设置
      hf_model_path: ${rollout.model_dir} # hf 模型路径
      save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor
      use_gpu_num : 0
      use_gpu_index: null
      process_num: 16 # 用于检查点转换的进程数
      tensor_model_parallel_size: ${actor.model.tensor_model_parallel_size}
      pipeline_model_parallel_size: ${actor.model.pipeline_model_parallel_size}

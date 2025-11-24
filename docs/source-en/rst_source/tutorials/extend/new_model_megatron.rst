Adding New Models with Megatron+SGLang
=============================================


This document is based on the Megatron implementation and provides a
comprehensive guide to adding new HuggingFace models in the RLinf
framework.

Prerequisites
-------------

* Familiarity with the **Megatron-LM** distributed training framework  
* Understanding of the **RLinf** framework architecture  
* Knowledge of **PyTorch** and distributed training

Step-by-Step Implementation
---------------------------

1. Megatron Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before adding a new model to RLinf, confirm that your Megatron-LM version already supports the model.

Otherwise, you’ll first need to add support for the new model in Megatron-LM. 

For details, refer to the official guide:
`Megatron-LM <https://github.com/NVIDIA/Megatron-LM/>`__

**File:** ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``  

In the current RLinf framework, the model configuration file used by the Megatron-LM training framework is ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``.

To support the model structure required for your training, modify the ``model_provider_func`` that RLinf passes to Megatron-LM.

.. code-block:: python

   def model_provider_func(self, pre_process, post_process):
       """Model depends on pipeline parallelism."""
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

2. Megatron-LM Weight Conversion Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Megatron-LM training framework cannot directly load HuggingFace format checkpoints; weight conversion is required.

You can configure the ``use_hf_ckpt`` option in the YAML file to enable automatic conversion from HuggingFace format checkpoints to Megatron-LM format checkpoints during the first RLinf training run. 

After conversion, the Megatron-LM format checkpoint will be saved in the ``save_path`` directory.

For example:

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

This approach will perform a one-time Megatron-LM format checkpoint conversion during RLinf's first training run. 

Converting from HuggingFace format checkpoints to Megatron-LM format checkpoints is a very time-consuming process. 

If you've already converted a Megatron-LM format checkpoint previously, you can directly specify the path to the converted Megatron-LM format checkpoint via the ``checkpoint_load_path`` option in the YAML file for subsequent training runs.

For example:

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


This way, RLinf will use the converted checkpoint directly without needing to convert it again.

If you need to adapt weight conversion for other models, you can submit an issue directly in the `RLinf <https://github.com/RLinf/RLinf/issues>`__ GitHub repository.

You can also adapt the new model conversion code yourself by referring to the files in ``toolkits/ckpt_convertor``.

**Tool:** ``toolkits/ckpt_convertor``  

RLinf's ckpt_convert tool first converts HuggingFace format checkpoints to an intermediate file format, then converts this intermediate format to Megatron-LM format checkpoints.

The specific interface is located in the ``toolkits/ckpt_convertor/convert_hf_to_mg.py`` file.

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

RLinf uses the ``hf_to_middle_file`` method to convert HuggingFace format checkpoints to an intermediate file format, then uses the ``middle_file_to_mg`` method to convert this intermediate format to Megatron-LM format checkpoints.

To adapt a new model, you will need to modify these two methods to handle the corresponding weight splitting and conversion for the model.

If you successfully adapt a new model, we particularly welcome you to submit a pull request with your code to our `RLinf <https://github.com/RLinf/RLinf/pulls>`__ repository—this way, more users can benefit!

3. SGLang Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf only supports models that have already been adapted by SGLang.

If you need a new model, add the adaptation code to SGLang itself. 

See the official guide:
`SGLang-support-new-model <https://docs.SGLang.ai/supported_models/support_new_models.html>`__.

4. Weight Sync from Megatron to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In RLinf, before each training run starts, the MegatronActor converts its current weights and syncs them to each SGLangActor.

This process involves converting the weight format from that of MegatronActor to SGLangActor.

The weight conversion has two stages:

1. The MegatronActor converts its weights to SGLang format and sends them to each SGLangActor.

2. Each SGLangActor loads the received weights and updates its own current weights.

**File:** ``rlinf/utils/resharding/mcore_weight_reshard.py``  

Before each rollout, the **MegatronActor** uses ``gather_and_reshard_model`` to convert the weight format. 

In collocated mode, each MegatronActor sends weights to a corresponding SGLangActor in a one-to-one manner.

First, the weights are re-sharded according to the MegatronActor's parallel configuration, then the sharded weights are renamed.
Finally, they are converted to SGLangActor's weight format. 

The converted weights are sent to the corresponding SGLangActor through the send interface.

.. code-block:: python

    def gather_and_reshard_model(self, model):
        ...
        # Merge and reshard weights according to MegatronActor's parallel configuration
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

        # MegatronActor renames the sharded weights
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
        
        # Rename sharded weights and finally convert to SGLangActor's weight format
        if self.config.convert_fn is not None:
            model_state_dict = self.config.convert_fn(model_state_dict)
        ...

To support more models, you need to add new ``tp_reshard_fn`` and ``pp_reshard_fn`` methods to handle the corresponding weight splitting and conversion for the new models, and add new ``convert_fn`` conversion rules to rename the sharded weights.

Take qwen2.5 as an example:

The code for weight re-sharding in MegatronActor is located in the ``rlinf/utils/resharding/utils.py`` file.

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

The code for MegatronActor to shard weights is in the ``rlinf/utils/convertor/utils.py`` file.

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

At present only the *Qwen-2.5* family is implemented.

After completing the above steps, MegatronActor will send the weights to SGLangActor, which will update its current weights. 

The code for SGLangActor to receive weights is in the ``rlinf/hybrid_engines/sglang/common/sgl_scheduler.py`` file.

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

Special handling of the weights received by the SGLangActor requires modifications here.

If you have any questions throughout the process of adapting new models, feel free to submit an issue to our RLinf `RLinf <https://github.com/RLinf/RLinf/issues>`__ repository.

We will address your questions as soon as possible.

5. Qwen2.5 Family Model Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you’ve completed the above steps, you can adapt your new model to RLinf.

Below is an example YAML configuration file for the qwen2.5 model family.

After adapting your new model, you can refer to this YAML configuration file and make appropriate modifications.

**File:** ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``  

Set Megatron parameters used by RLinf.

.. code-block:: yaml

   # Megatron parameters
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
    distributed_backend: nccl # Support 'nccl' and 'gloo'
    distributed_timeout_minutes: 30
    ckpt_format: torch
    use_dist_ckpt: False
    tp_comm_bootstrap_backend: nccl
    tp_comm_overlap_cfg: null # tp_comm_overlap_cfg.yaml
    use_hf_ckpt: True # if true, will transfer hf model to generate megatron checkpoint and use it for training.

    ckpt_convertor: # config for ckpt convertor
      model: DeepSeek-R1-Distill-Qwen-1.5B
      model_type: null # will be set by hf model's config if null
      hf_model_path: ${rollout.model_dir} # path to the hf model
      save_path: ${runner.output_dir}/${runner.experiment_name}/converted_ckpts/actor
      use_gpu_num : 0
      use_gpu_index: null
      process_num: 16 # number of processes to use for checkpointing
      tensor_model_parallel_size: ${actor.model.tensor_model_parallel_size}
      pipeline_model_parallel_size: ${actor.model.pipeline_model_parallel_size}
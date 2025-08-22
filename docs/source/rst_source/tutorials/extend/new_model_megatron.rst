.. Adding New Models with Megatron+SGLang
.. =============================================


.. This document is based on the Megatron implementation and provides a
.. comprehensive guide to adding new HuggingFace models in the RLinf
.. framework.

.. Prerequisites
.. -------------

.. * Familiarity with the **Megatron-LM** distributed training framework  
.. * Understanding of the **RLinf** framework architecture  
.. * Knowledge of **PyTorch** and distributed training

.. Step-by-Step Implementation
.. ---------------------------

.. 1. Megatron Model Configuration
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. **File:** ``rlinf/core/model_managers/megatron/megatron_model_manager.py``  
.. Modify the ``model_provider_func`` that RLinf passes to Megatron so
.. that specialised model structures are supported.

.. .. code-block:: python

..    def model_provider_func(self, pre_process, post_process):
..        """Model depends on pipeline parallelism."""
..        use_te = HAVE_TE

..        if self.mcore_gpt:
..            model = MCoreGPTModel(
..                config=self.transformer_config,
..                transformer_layer_spec=get_specs(
..                    self.spec_name,
..                    self.transformer_config,
..                    use_te,
..                ),
..                vocab_size=self._cfg.model.override_vocab_size,
..                max_sequence_length=self._cfg.model.max_position_embeddings,
..                pre_process=pre_process,
..                post_process=post_process,
..                parallel_output=True,
..                share_embeddings_and_output_weights=self._cfg.model.share_embeddings_and_output_weights,
..                position_embedding_type=self._cfg.model.position_embedding_type,
..                rotary_percent=self._cfg.model.rotary_percentage,
..                seq_len_interpolation_factor=self._cfg.model.seq_len_interpolation_factor,
..                rotary_base=self._cfg.model.rotary_base,
..            )

..        else:
..            from megatron.legacy.model.gpt_model import GPTModel
..            config = build_config(ModelParallelConfig, self._cfg.model)
..            setattr(config, 'hidden_size', self._cfg.model.hidden_size)

..            model = GPTModel(
..                config=config,
..                num_tokentypes=0,
..                parallel_output=True,
..                pre_process=pre_process,
..                post_process=post_process
..            )
..        return model


.. 2. Megatron Weight Conversion Support
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. **Tool:** ``tools/ckpt_convert``  
.. Provide weight-conversion scripts that match the modified model. The
.. example below converts DeepSeek checkpoints:

.. .. code-block:: bash

..    CKPT_PATH_HF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_hf
..    CKPT_PATH_MF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_middle_file
..    CKPT_PATH_MG=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_mg_tp1_pp2

..    TP_SIZE=1
..    PP_SIZE=2

..    # download (optional)
..    # modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' --local_dir "$CKPT_PATH_HF"

..    rm -rf $CKPT_PATH_MF
..    python $SCRIPT_PATH/convert_hf_to_middle_file.py \
..        --load-path $CKPT_PATH_HF \
..        --save-path $CKPT_PATH_MF \
..        --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
..        --use-gpu-num 0 \
..        --process-num 16

..    rm -rf $CKPT_PATH_MG
..    python $SCRIPT_PATH/convert_middle_file_to_mg.py \
..        --load-path $CKPT_PATH_MF \
..        --save-path $CKPT_PATH_MG \
..        --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
..        --tp-size $TP_SIZE \
..        --tpe-size 1 \
..        --ep-size 1 \
..        --pp-size $PP_SIZE \
..        --use-gpu-num 0 \
..        --process-num 16

..    rm -rf $CKPT_PATH_MF


.. 3. SGLang Model Configuration
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. RLinf only supports models that have already been adapted by SGLang.
.. If you need a new model, add the adaptation code to SGLang itself. See
.. the official guide:
.. `SGLang-support-new-model <https://docs.SGLang.ai/supported_models/support_new_models.html>`__.

.. 4. Weight Sync from Megatron to SGLang
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. **File:** ``rlinf/utils/resharding/utils.py``  
.. Before every rollout the **MegatronActor** converts the updated weights
.. into HuggingFace format and ships them to each SGLang instance. At
.. present only the *Qwen-2.5* family is implemented.

.. .. code-block:: python

..    @staticmethod
..    def convert_mega_qwen2_5_to_hf(model_state_dict: dict, config) -> dict:
..        new_statedict = {}
..        for name, param in model_state_dict.items():
..            transform_type, hf_names = TransformFunc.mega_name_qwen2_5_to_hf(name)
..            if transform_type == TransformType.SPLIT_QKV:
..                TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
..            elif transform_type == TransformType.SPLIT_QKV_BIAS:
..                TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
..            elif transform_type == TransformType.SPLIT_FC1:
..                TransformFunc.split_fc1(param, new_statedict, hf_names, config)
..            elif transform_type == TransformType.SPLIT_NONE:
..                TransformFunc.split_none(param, new_statedict, hf_names)
..            else:
..                raise NotImplementedError(f"Transform type {transform_type} not implemented")
..        return new_statedict


.. 5. Configuration File
.. ~~~~~~~~~~~~~~~~~~~~~

.. **File:** ``examples/math/qwen2.5/grpo-1.5b-megatron.yaml``  
.. Set Megatron parameters used by RLinf.

.. .. code-block:: yaml

..    # Megatron parameters
..    model:
..      precision: fp16
..      add_bias_linear: False
..      tensor_model_parallel_size: 2
..      pipeline_model_parallel_size: 1
..      activation: swiglu
..      sequence_parallel: True
..      recompute_method: block
..      recompute_granularity: full
..      recompute_num_layers: 20
..      seq_length: ${trainer.seq_length}
..      encoder_seq_length: ${trainer.seq_length}
..      normalization: rmsnorm
..      position_embedding_type: rope
..      bias_dropout_fusion: False
..      persist_layer_norm: False
..      bias_activation_fusion: False
..      attention_softmax_in_fp32: True
..      batch_p2p_comm: False
..      variable_seq_lengths: True
..      gradient_accumulation_fusion: False
..      moe_token_dispatcher_type: alltoall
..      use_cpu_initialization: False

..    optim:
..      optimizer: adam
..      bf16: False
..      fp16: True
..      lr: 2e-05
..      adam_beta1: 0.9
..      adam_beta2: 0.95
..      adam_eps: 1.0e-05
..      min_lr: 2.0e-6
..      weight_decay: 0.05
..      use_distributed_optimizer: True
..      overlap_grad_reduce: True
..      overlap_param_gather: True
..      optimizer_enable_pin: false
..      overlap_param_gather_with_optimizer_step: False
..      clip_grad: 1.0
..      loss_scale_window: 5

..    lr_sched:
..      lr_warmup_fraction: 0.01
..      lr_warmup_init: 0.0
..      lr_warmup_iters: 0
..      max_lr: 2.0e-5
..      min_lr: 0.0
..      lr_decay_style: constant
..      lr_decay_iters: 10

..    # Tokeniser
..    tokenizer:
..      tokenizer_model: /mnt/public/hf_models/DeepSeek-R1-Distill-Qwen-1.5B
..      use_fast: False
..      trust_remote_code: True
..      padding_side: 'right'

..    # Megatron settings
..    megatron:
..      ddp_bucket_size: null
..      distributed_backend: nccl          # 'nccl' or 'gloo'
..      distributed_timeout_minutes: 30
..      ckpt_format: torch
..      use_dist_ckpt: False
..      tp_comm_bootstrap_backend: nccl
..      tp_comm_overlap_cfg: null          # e.g. path to overlap YAML
..      use_hf_ckpt: True                  # convert HF weights to Megatron at start-up

..      # Checkpoint converter
..      ckpt:
..        model: DeepSeek-R1-Distill-Qwen-1.5B
..        model_type: null                 # filled from HF config if null
..        hf_model_path: ${generation.model_dir}
..        save_path: ${trainer.output_dir}/${trainer.experiment_name}/actor/megatron_ckpt_from_hf
..        use_gpu_num: 0
..        use_gpu_index: null
..        process_num: 16
..        tensor_model_parallel_size: ${actor.model.tensor_model_parallel_size}
..        pipeline_model_parallel_size: ${actor.model.pipeline_model_parallel_size}


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

**File:** ``rlinf/hybrid_engines/megatron/megatron_model_manager.py``  
Modify the ``model_provider_func`` that RLinf passes to Megatron so
that specialised model structures are supported.

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


2. Megatron Weight Conversion Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tool:** ``tools/ckpt_convert``  
Provide weight-conversion scripts that match the modified model. The
example below converts qwen2.5 checkpoints:

.. code-block:: bash

   CKPT_PATH_HF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_hf
   CKPT_PATH_MF=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_middle_file
   CKPT_PATH_MG=debug_ckpt_convert/DeepSeek-R1-Distill-Qwen-1.5B_mg_tp1_pp2

   TP_SIZE=1
   PP_SIZE=2

   # download (optional)
   # modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' --local_dir "$CKPT_PATH_HF"

   rm -rf $CKPT_PATH_MF
   python convert_hf_to_middle_file.py \
       --load-path $CKPT_PATH_HF \
       --save-path $CKPT_PATH_MF \
       --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
       --use-gpu-num 0 \
       --process-num 16

   rm -rf $CKPT_PATH_MG
   python convert_middle_file_to_mg.py \
       --load-path $CKPT_PATH_MF \
       --save-path $CKPT_PATH_MG \
       --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
       --tp-size $TP_SIZE \
       --tpe-size 1 \
       --ep-size 1 \
       --pp-size $PP_SIZE \
       --use-gpu-num 0 \
       --process-num 16

   rm -rf $CKPT_PATH_MF


3. SGLang Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf only supports models that have already been adapted by SGLang.
If you need a new model, add the adaptation code to SGLang itself. See
the official guide:
`SGLang-support-new-model <https://docs.SGLang.ai/supported_models/support_new_models.html>`__.

4. Weight Sync from Megatron to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``rlinf/utils/resharding/utils.py``  
Before every rollout the **MegatronActor** converts the updated weights
into HuggingFace format and ships them to each SGLang instance. At
present only the *Qwen-2.5* family is implemented.

.. code-block:: python

   @staticmethod
   def convert_mega_qwen2_5_to_hf(model_state_dict: dict, config) -> dict:
       new_statedict = {}
       for name, param in model_state_dict.items():
           transform_type, hf_names = TransformFunc.mega_name_qwen2_5_to_hf(name)
           if transform_type == TransformType.SPLIT_QKV:
               TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_QKV_BIAS:
               TransformFunc._split_gqa_tensor(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_FC1:
               TransformFunc.split_fc1(param, new_statedict, hf_names, config)
           elif transform_type == TransformType.SPLIT_NONE:
               TransformFunc.split_none(param, new_statedict, hf_names)
           else:
               raise NotImplementedError(f"Transform type {transform_type} not implemented")
       return new_statedict


5. Configuration File
~~~~~~~~~~~~~~~~~~~~~

**File:** ``examples/math/config/qwen2.5-1.5b-grpo-megatron.yaml``  
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